# decision_cache.py
"""
This module provides a decision caching system for the model. Since agent attributes
are translated from numerical values into natural language via discrete buckets,
if two agents fall into exactly the same buckets for all relevant attributes, the
prompts sent to the LLM will be identical. In that case, if the same decision has
already been requested from the LLM before, the cached result can be reused instead
of calling the LLM again.

For each decision, the cache can store multiple sample answers (currently set to 3).
The cache will only be called for a decision after the decision have been queried from 
the LLM for three times, which helps mitigate LLM stochasticity and forms an empirical 
distribution. This idea follows the approach used in the earlier archetype-based work.
"""
from __future__ import annotations

import random
from typing import Dict, Tuple, List, Any, Optional

try:
    from .translator import Translator, AttrSpec
except ImportError:
    from translator import Translator, AttrSpec  # compatibility for running as a script

# ---- Cache schema version (increment when key structure or bucket definitions change,
# ---- to avoid mixing incompatible old caches) ----
SCHEMA_REV = 1

# ---- Attribute reference: which source (girl/hh/school) and which attribute name ----
AttrRef = Tuple[str, str]  # (source, attr_name)

# ---- For each decision type, the list of bucketed attributes included in the cache key
# ---- (should match or be a superset of those used in the prompt) ----
DECISION_ATTRS: Dict[str, List[AttrRef]] = {
    # Decision 1 — attendance
    "attendance_girl": [
        ("girl", "age"),
        ("girl", "status"),
        ("girl", "grade"),
        ("girl", "L"),
        ("girl", "esteem"),
        ("girl", "enjoyment"),
        ("hh",   "econ_barrier"),
        ("girl", "marriage"),
        ("hh",   "transport"),
        ("school", "quality_of_teaching"),
        ("school", "fairness"),
        ("school", "peer_support"),
        ("school", "safety"),
    ],
    "attendance_household": [
        ("girl", "age"),
        ("hh",   "econ_barrier"),
        ("hh",   "attitude"),
        ("hh",   "chores"),
        ("hh",   "transport"),
        ("girl", "marriage"),
        ("school", "quality_of_teaching"),
        ("school", "fairness"),
        ("school", "peer_support"),
        ("school", "safety"),
    ],

    # Decision 2 — transition
    "transition_girl": [
        ("girl", "age"),
        ("girl", "status"),
        ("girl", "grade"),
        ("girl", "L"),
        ("girl", "esteem"),
        ("girl", "enjoyment"),
        ("hh",   "econ_barrier"),
        ("girl", "employment_skills"),
        ("girl", "marriage"),
        ("hh",   "transport"),
        ("school", "quality_of_teaching"),
        ("school", "fairness"),
        ("school", "peer_support"),
        ("school", "safety"),
    ],
    "transition_household": [
        ("girl", "age"),
        ("hh",   "econ_barrier"),
        ("hh",   "attitude"),
        ("hh",   "chores"),
        ("hh",   "transport"),
        ("girl", "employment_skills"),
        ("girl", "marriage"),
        ("school", "quality_of_teaching"),
        ("school", "fairness"),
        ("school", "peer_support"),
        ("school", "safety"),
    ],

    # Decision 3 — self-esteem evaluation
    "self_esteem_eval": [
        ("girl", "status"),
        ("girl", "grade"),
        ("girl", "L"),
        ("girl", "enjoyment"),
        ("hh",   "econ_barrier"),
        ("girl", "employment_skills"),
        ("hh",   "attitude"),
        ("girl", "marriage"),
        ("school", "quality_of_teaching"),
        ("school", "fairness"),
        ("school", "peer_support"),
        ("school", "safety"),
        ("girl", "age"),
    ],

    # Decision 4 — household attitude evaluation
    "household_attitude_eval": [
        ("girl", "status"),
        ("girl", "grade"),
        ("hh",   "econ_barrier"),
        ("hh",   "chores"),
        ("girl", "L"),
        ("girl", "employment_skills"),
        ("girl", "esteem"),
        ("girl", "enjoyment"),
        ("girl", "marriage"),
        ("school", "quality_of_teaching"),
        ("school", "safety"),
        ("girl", "age"),
    ],
}


class DecisionCache:
    """
    Multi-sample cache:
    - For each key, collect N LLM outputs first (default: 3)
    - If fewer than N samples are collected, get_* returns None (forcing an LLM call)
    - Once N samples are collected, get_* randomly samples one from history
      (forming an empirical distribution)
    """
    def __init__(self, *, rng: random.Random | None = None, min_samples_to_use: int = 3, max_samples: int = 3) -> None:
        self.rng = rng or random.Random()
        self.min_samples_to_use = int(min_samples_to_use)
        self.max_samples = int(max_samples)

        # key -> list of samples
        self._yesno:  Dict[Tuple, List[Tuple[bool, str]]] = {}
        self._choice: Dict[Tuple, List[Tuple[str, str]]] = {}
        self._score:  Dict[Tuple, List[Tuple[int, str]]] = {}

    # ---------- key construction ----------
    def _find_value_with_alias(self, src: Dict[str, Any], spec: AttrSpec) -> Any:
        for k in (spec.name,) + spec.aliases:
            if k in src:
                return src[k]
        raise KeyError(f"Missing value for attribute '{spec.name}' (aliases: {spec.aliases})")

    def _bucket_index_of(self, tr: Translator, src: Dict[str, Any], attr_name: str) -> int:
        spec = tr._get_spec(attr_name)
        if spec is None:
            raise KeyError(f"Unknown attribute: {attr_name}")
        val = self._find_value_with_alias(src, spec)
        t = tr.translate_attr(spec.name, val)
        return int(t["bucket_index"])

    def _attr_sources(self, girl: Dict[str, Any], hh: Dict[str, Any], school: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        return {"girl": girl, "hh": hh, "school": school}

    def make_key(
        self,
        decision_type: str,
        *,
        model_id: str,
        country: str,
        tr: Translator,
        girl: Dict[str, Any],
        hh: Dict[str, Any],
        school: Dict[str, Any],
        extra_sig: str | None = None
    ) -> Tuple:
        if decision_type not in DECISION_ATTRS:
            raise KeyError(f"Unknown decision_type: {decision_type}")
        refs = DECISION_ATTRS[decision_type]
        srcs = self._attr_sources(girl, hh, school)

        buckets: List[int] = []
        for src_name, attr_name in refs:
            src_dict = srcs[src_name]
            bi = self._bucket_index_of(tr, src_dict, attr_name)
            buckets.append(bi)

        return (
            SCHEMA_REV,
            decision_type,
            str(model_id),
            str(country),
            str(extra_sig or ""),
            tuple(buckets),
        )

    # ---------- helpers ----------
    def _ready(self, pool: Optional[List[Tuple[Any, str]]]) -> bool:
        return pool is not None and len(pool) >= self.min_samples_to_use

    def _maybe_sample(self, pool: Optional[List[Tuple[Any, str]]]) -> Optional[Tuple[Any, str]]:
        if not self._ready(pool):
            return None
        return self.rng.choice(pool)

    def _maybe_append(self, pool: List[Tuple[Any, str]], item: Tuple[Any, str]) -> None:
        if len(pool) < self.max_samples:
            pool.append(item)

    # ---------- Yes/No ----------
    def get_yes_no(self, key: Tuple) -> Optional[Tuple[bool, str]]:
        return self._maybe_sample(self._yesno.get(key))  # type: ignore

    def put_yes_no(self, key: Tuple, answer: bool, explanation: str) -> None:
        pool = self._yesno.setdefault(key, [])
        self._maybe_append(pool, (bool(answer), str(explanation)))

    # ---------- Choice ----------
    def get_choice(self, key: Tuple) -> Optional[Tuple[str, str]]:
        return self._maybe_sample(self._choice.get(key))  # type: ignore

    def put_choice(self, key: Tuple, choice: str, explanation: str) -> None:
        pool = self._choice.setdefault(key, [])
        self._maybe_append(pool, (str(choice), str(explanation)))

    # ---------- Score ----------
    def get_score(self, key: Tuple) -> Optional[Tuple[int, str]]:
        out = self._maybe_sample(self._score.get(key))
        if out is None:
            return None
        s, expl = out
        return int(s), str(expl)

    def put_score(self, key: Tuple, score: int, explanation: str) -> None:
        pool = self._score.setdefault(key, [])
        self._maybe_append(pool, (int(score), str(explanation)))

    # ---------- statistics ----------
    def stats(self) -> Dict[str, int]:
        return {
            "yesno_keys": len(self._yesno),
            "choice_keys": len(self._choice),
            "score_keys": len(self._score),
            "yesno_samples": sum(len(v) for v in self._yesno.values()),
            "choice_samples": sum(len(v) for v in self._choice.values()),
            "score_samples": sum(len(v) for v in self._score.values()),
        }

    # ---------- Debug dump ----------
    def dump(self, *, max_keys_per_type: int = 12) -> str:
        """
        Print cache contents (truncated for readability).
        """
        lines: List[str] = []
        st = self.stats()
        lines.append("[DecisionCache] stats=" + str(st))

        def _fmt_key(k: Tuple) -> str:
            # k = (SCHEMA_REV, decision_type, model_id, country, extra_sig, buckets_tuple)
            try:
                return f"(rev={k[0]}, type={k[1]}, extra_sig={str(k[4])[:80]}, buckets_len={len(k[5])})"
            except Exception:
                return str(k)

        # yes/no
        lines.append("\n[DecisionCache] YES/NO (showing up to %d keys)" % max_keys_per_type)
        for i, (k, v) in enumerate(self._yesno.items()):
            if i >= max_keys_per_type:
                lines.append("... (truncated)")
                break
            lines.append(f"  {i+1}. key={_fmt_key(k)}  samples={v}")

        # choice
        lines.append("\n[DecisionCache] CHOICE (showing up to %d keys)" % max_keys_per_type)
        for i, (k, v) in enumerate(self._choice.items()):
            if i >= max_keys_per_type:
                lines.append("... (truncated)")
                break
            lines.append(f"  {i+1}. key={_fmt_key(k)}  samples={v}")

        # score
        lines.append("\n[DecisionCache] SCORE (showing up to %d keys)" % max_keys_per_type)
        for i, (k, v) in enumerate(self._score.items()):
            if i >= max_keys_per_type:
                lines.append("... (truncated)")
                break
            lines.append(f"  {i+1}. key={_fmt_key(k)}  samples={v}")

        return "\n".join(lines)

    def to_jsonable(self) -> Dict[str, Any]:
        def key_to_obj(k: Tuple) -> Dict[str, Any]:
            # (SCHEMA_REV, decision_type, model_id, country, extra_sig, buckets_tuple)
            return {
                "rev": k[0],
                "type": k[1],
                "model_id": k[2],
                "country": k[3],
                "extra_sig": k[4],
                "buckets": list(k[5]),
            }

        return {
            "schema_rev": SCHEMA_REV,
            "min_samples_to_use": self.min_samples_to_use,
            "max_samples": self.max_samples,
            "stats": self.stats(),
            "yesno": [
                {"key": key_to_obj(k), "samples": v}
                for k, v in self._yesno.items()
            ],
            "choice": [
                {"key": key_to_obj(k), "samples": v}
                for k, v in self._choice.items()
            ],
            "score": [
                {"key": key_to_obj(k), "samples": v}
                for k, v in self._score.items()
            ],
        }
