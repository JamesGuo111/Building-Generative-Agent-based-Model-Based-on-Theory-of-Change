# decision_schemas.py
from __future__ import annotations
from typing import Tuple, List, Dict, Any, Optional
import json
import re
import ast
import difflib

class SchemaError(ValueError):
    pass

# ---------- text cleanup & relaxed loading ----------

_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)

def _strip_code_fences(s: str) -> str:
    if s is None:
        return ""
    m = _CODE_FENCE_RE.search(str(s))
    return m.group(1).strip() if m else str(s).strip()

def _extract_json_object(s: str) -> Optional[str]:
    """
    Attempt to extract the outermost JSON object {...} from a block of text.
    This approach is more robust than regex alone and avoids greedy over-matching.
    """
    s = _strip_code_fences(s)
    start = s.find("{")
    if start < 0:
        return None

    depth = 0
    for i in range(start, len(s)):
        ch = s[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start:i+1]
    return None

def _loads_relaxed(obj_text: str) -> Dict[str, Any]:
    """
    Best-effort conversion of LLM output into a dict:
    - First try json.loads
    - Then try extracting {...} from surrounding text
    - Finally fall back to ast.literal_eval (to handle single quotes, trailing commas, etc.)
    """
    raw = _strip_code_fences(obj_text)

    candidates: List[str] = []
    if raw:
        candidates.append(raw)
    extracted = _extract_json_object(raw)
    if extracted and extracted != raw:
        candidates.append(extracted)

    last_err = None
    for c in candidates:
        try:
            obj = json.loads(c)
            if isinstance(obj, dict):
                return obj
        except Exception as e:
            last_err = e

        # Fallback for Python-dict-like strings (single quotes, trailing commas)
        try:
            obj = ast.literal_eval(c)
            if isinstance(obj, dict):
                return obj
        except Exception as e:
            last_err = e

    raise SchemaError(f"Invalid/Unrecoverable JSON: {last_err}")

def _norm_simple_token(s: Any) -> str:
    """
    Lightweight fault tolerance:
    - strip leading/trailing whitespace
    - remove common trailing punctuation (. ! ? ；：，,)
    - lowercase
    """
    if s is None:
        return ""
    out = str(s).strip()
    out = out.rstrip(" .!?;:，,、")
    out = out.strip().lower()
    return out

def _collapse_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def _extract_explanation_fallback(text: str) -> str:
    """
    When the explanation field is missing, conservatively extract a plausible
    explanation sentence from the raw text.
    """
    t = _strip_code_fences(text).strip()
    if not t:
        return ""
    # Common pattern: Answer: Yes\nExplanation: ...
    m = re.search(r"explanation\s*[:\-]\s*(.+)$", t, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    # Fallback: take everything after the first non-empty line
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if len(lines) >= 2:
        return " ".join(lines[1:]).strip()
    return ""

# ---------- Yes/No ----------

_YES_SET = {"yes", "y", "true", "1", "allow", "attend", "agree", "ok", "okay"}
_NO_SET  = {"no", "n", "false", "0", "deny", "not", "disagree"}

def _extract_yesno_from_text(text: str) -> Optional[str]:
    t = _norm_simple_token(text)
    if not t:
        return None

    # Prefer explicit word-boundary matches for yes/no
    m = re.search(r"\b(yes|no)\b", t, flags=re.IGNORECASE)
    if m:
        return "Yes" if m.group(1).lower() == "yes" else "No"

    # Relaxed check: first token
    head = t.split(" ", 1)[0]
    if head in _YES_SET:
        return "Yes"
    if head in _NO_SET:
        return "No"

    # Chinese fallback (rare but observed)
    if "是" == t or t.startswith("是 "):
        return "Yes"
    if "否" == t or t.startswith("否 "):
        return "No"
    if "可以" in t and "不可以" not in t:
        return "Yes"
    if "不" in t and ("可以" in t or "允许" in t):
        return "No"

    return None

def parse_yes_no(obj_text: str) -> Tuple[str, str]:
    """
    Parsing logic:
    - Standard: {"answer": "Yes"/"No", "explanation": "..."}
    - Tolerant: keys named choice/response; or not JSON at all (plain Yes/No text)
    """
    # 1) Try to parse as dict; fall back to plain-text handling on failure
    obj: Optional[Dict[str, Any]] = None
    try:
        obj = _loads_relaxed(obj_text)
    except SchemaError:
        obj = None

    if obj is not None:
        raw_ans = (
            obj.get("answer")
            or obj.get("choice")
            or obj.get("response")
            or obj.get("result")
        )
        ans = _extract_yesno_from_text(str(raw_ans)) if raw_ans is not None else None
        if ans is None:
            # Some models embed the answer inside a sentence
            ans = _extract_yesno_from_text(json.dumps(obj, ensure_ascii=False))
        explanation = obj.get("explanation")
        if explanation is None:
            explanation = _extract_explanation_fallback(obj_text)
        explanation = str(explanation).strip()
        if ans in ("Yes", "No"):
            return ans, explanation

    # 2) Plain-text fallback
    ans2 = _extract_yesno_from_text(obj_text)
    if ans2 in ("Yes", "No"):
        return ans2, _extract_explanation_fallback(obj_text)

    raise SchemaError("Could not recover a Yes/No answer.")

# ---------- Choice from options ----------

_PREFIX_RE = re.compile(
    r"^(i\s*choose|i\s*will\s*choose|my\s*choice\s*is|choice\s*[:\-]|answer\s*[:\-])\s*",
    re.IGNORECASE
)

def _normalize_choice_text(s: str) -> str:
    s = _strip_code_fences(s)
    s = s.strip().strip('"').strip("'")
    s = _PREFIX_RE.sub("", s).strip()
    # Remove bullet points / numbering
    s = re.sub(r"^[-*]\s+", "", s)
    s = re.sub(r"^\(?\d+\)?[).:\-]\s*", "", s)
    # Remove trailing punctuation
    s = s.rstrip(" .!?;:，,、")
    s = _collapse_ws(s).lower()
    return s

def _build_option_map(options: List[str]) -> Dict[str, str]:
    """
    normalized -> original
    """
    m: Dict[str, str] = {}
    for opt in options:
        key = _normalize_choice_text(opt)
        if key:
            m[key] = opt
    return m

def _recover_choice(raw_choice: str, options: List[str]) -> Optional[str]:
    if not raw_choice:
        return None
    opt_map = _build_option_map(options)
    norm = _normalize_choice_text(raw_choice)

    # 1) Exact normalized match
    if norm in opt_map:
        return opt_map[norm]

    # 2) Numeric index (1..n)
    if norm.isdigit():
        idx = int(norm)
        if 1 <= idx <= len(options):
            return options[idx - 1]

    # 3) Raw text contains exactly one option verbatim (safest)
    hits = []
    raw_full = _strip_code_fences(raw_choice)
    for opt in options:
        if opt and opt in raw_full:
            hits.append(opt)
    if len(hits) == 1:
        return hits[0]

    # 4) difflib fuzzy match (higher risk, only if very close)
    candidates = list(opt_map.keys())
    close = difflib.get_close_matches(norm, candidates, n=1, cutoff=0.86)
    if close:
        return opt_map[close[0]]

    return None

def parse_choice(obj_text: str, options: List[str]) -> Tuple[str, str]:
    """
    Parsing logic:
    - Standard: {"choice": "<one of options>", "explanation": "..."}
    - Tolerant: inconsistent key names; extra prefixes; casing/whitespace/punctuation differences;
      or even a single-line text response.
    Guarantees that the returned choice is one of the original strings in `options`.
    """
    if not options:
        raise SchemaError("No allowed options provided.")

    obj: Optional[Dict[str, Any]] = None
    try:
        obj = _loads_relaxed(obj_text)
    except SchemaError:
        obj = None

    if obj is not None:
        raw_choice = (
            obj.get("choice")
            or obj.get("answer")
            or obj.get("option")
            or obj.get("response")
            or obj.get("result")
        )
        recovered = _recover_choice(
            str(raw_choice) if raw_choice is not None else "", options
        )
        explanation = obj.get("explanation")
        if explanation is None:
            explanation = _extract_explanation_fallback(obj_text)
        explanation = str(explanation).strip()
        if recovered is not None:
            return recovered, explanation

    # Plain-text fallback: recover choice from full text
    recovered2 = _recover_choice(obj_text, options)
    if recovered2 is not None:
        return recovered2, _extract_explanation_fallback(obj_text)

    raise SchemaError("Could not recover a valid choice from options.")

# ---------- Score 0..100 ----------

def _extract_int_0_100(x: Any) -> Optional[int]:
    if x is None:
        return None
    # Direct int
    try:
        v = int(x)
        return max(0, min(100, v))
    except Exception:
        pass

    s = _strip_code_fences(str(x))
    # Find the first integer or decimal number
    m = re.search(r"(-?\d+(?:\.\d+)?)", s)
    if not m:
        return None
    try:
        v = float(m.group(1))
        v_int = int(round(v))
        return max(0, min(100, v_int))
    except Exception:
        return None

def parse_score(obj_text: str) -> Tuple[int, str]:
    """
    Parsing logic:
    - Standard: {"score": <int 0..100>, "explanation": "..."}
    - Tolerant: inconsistent key names; score as string; formats like 75/100 or 75.0;
      missing explanation.
    The final score is clipped to the range 0..100.
    """
    obj: Optional[Dict[str, Any]] = None
    try:
        obj = _loads_relaxed(obj_text)
    except SchemaError:
        obj = None

    if obj is not None:
        raw_score = (
            obj.get("score")
            or obj.get("answer")
            or obj.get("value")
            or obj.get("result")
        )
        score = _extract_int_0_100(raw_score)
        explanation = obj.get("explanation")
        if explanation is None:
            explanation = _extract_explanation_fallback(obj_text)
        explanation = str(explanation).strip()
        if score is not None:
            return score, explanation

    # Plain-text fallback
    score2 = _extract_int_0_100(obj_text)
    if score2 is not None:
        return score2, _extract_explanation_fallback(obj_text)

    raise SchemaError("Could not recover a score in 0..100.")
