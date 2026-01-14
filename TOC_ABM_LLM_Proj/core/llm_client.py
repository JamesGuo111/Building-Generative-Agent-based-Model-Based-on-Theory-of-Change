# mvp/llm_client.py
from __future__ import annotations
from typing import List, Dict, Any, Optional
import json
import re
import time
import random

import httpx
import litellm
from litellm import completion  # pip install litellm

DEFAULT_SYSTEM_PROMPT = """You live in Ethiopia. You will be asked to estimate your actions mainly related to education.
You must estimate your actions based on:
1) the information you are given, and
2) what you know about the general population with similar attributes.

"There isn’t enough information" and "It is unclear" are not acceptable answers.

Even if the user prompt asks you to give a “Yes/No” answer, a “choice”, or a “number” directly,
you MUST still output your answer strictly as a single JSON object following the required schema below.

Do not include any text outside the JSON. Do not add explanations before or after the JSON.

For any task that asks you to choose from options:
- The allowed options will be shown in the user prompt.
- In the JSON, the value of "choice" MUST be EXACTLY the option text as shown (copy-paste the option text).
- Do NOT include the option number like "1)" / "2)".
- Do NOT paraphrase, translate, shorten, or add extra words.
- Keep punctuation and parentheses exactly the same.

Example:
If options are:
1) a, 2) b, 3) c (abcd)

Correct JSON:
{"choice":"c (abcd)","explanation":"This option best fits the situation."}

Incorrect JSON (has numbering):
{"choice":"3) c (abcd)","explanation":"This option best fits the situation."}

Incorrect JSON (paraphrased/modified text):
{"choice":"c","explanation":"This option best fits the situation."}

Valid output schemas:
- Asking for a "Yes/No" answer: {"answer": "Yes" | "No", "explanation": "<one short sentence>"}
- Asking for a choice from options: {"choice": "<one of the options text exactly as shown>", "explanation": "<one short sentence>"}
- Asking for a integer value: {"score": <integer 0-100>, "explanation": "<one short sentence>"}"""

class LLMError(RuntimeError):
    pass

def _extract_json(text: str, *, debug: bool = False) -> str:
    """
    Attempt to extract the first JSON object from the returned text.
    If the original text is already valid JSON, return it directly.
    This is meant to tolerate occasional extra prefixes or suffixes.
    """
    s = text.strip()
    if s.startswith("{") and s.endswith("}"):
        return s
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if m:
        out = m.group(0)
        if debug:
            print("[LLMClient] _extract_json: extracted JSON substring from non-JSON wrapper.")
            print("  raw(head) =", s[:180].replace("\n", "\\n"))
            print("  json(head)=", out[:180].replace("\n", "\\n"))
        return out
    if debug:
        print("[LLMClient] _extract_json: FAILED to find JSON object; returning raw stripped text.")
        print("  raw(head) =", s[:240].replace("\n", "\\n"))
    return s  # Fallback: let json.loads raise the exception

def _litellm_exc(name: str):
    """Safely retrieve litellm.exceptions.<Name>; return None if unavailable."""
    exc_mod = getattr(litellm, "exceptions", None)
    return getattr(exc_mod, name, None) if exc_mod else None

class LLMClient:
    """
    Minimal LiteLLM wrapper:
    - Only supports chat completions
    - Returns raw text (expected to be a JSON string)
    """
    def __init__(
        self,
        model: str = "gemini/gemini-2.5-flash-lite",
        *,
        temperature: float = 0.2,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_output_tokens: int = 256,
        timeout: int = 60,
        max_retries: int = 20,
        backoff_base: float = 0.8,
        backoff_max: float = 20.0,
        debug: bool = False,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.max_output_tokens = max_output_tokens
        self.timeout = timeout

        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max

        self.debug = bool(debug)

    def _should_retry(self, e: Exception) -> bool:
        # HTTP status codes
        if isinstance(e, httpx.HTTPStatusError):
            sc = e.response.status_code
            return sc in (429, 500, 502, 503, 504)

        # Broader httpx network/timeout base classes
        if isinstance(e, (httpx.TimeoutException, httpx.NetworkError)):
            return True

        # Certain DNS/socket errors may appear as OSError/RuntimeError
        if isinstance(e, OSError):
            # Common on macOS: errno=8 (nodename nor servname...)
            if getattr(e, "errno", None) in (8,):
                return True

        # litellm exception mapping
        retry_exc_names = [
            "InternalServerError",
            "ServiceUnavailableError",
            "RateLimitError",
            "Timeout",
            "APIConnectionError",
            "BadGatewayError",
            "GatewayTimeoutError",
        ]
        for name in retry_exc_names:
            cls = _litellm_exc(name)
            if cls is not None and isinstance(e, cls):
                return True

        # Fallback: match against error message substrings 
        msg = str(e).lower()
        retry_substrings = [
            "nodename nor servname provided",
            "temporary failure in name resolution",
            "name or service not known",
            "server disconnected",
            "connection reset by peer",
            "connection aborted",
        ]
        if any(s in msg for s in retry_substrings):
            return True

        # Original heuristic checks retained
        if "internal error encountered" in msg:
            return True
        if "status code: 500" in msg or "status code: 503" in msg or "status code: 502" in msg:
            return True
        if "rate limit" in msg or "too many requests" in msg:
            return True

        return False

    def _sleep_backoff(self, attempt: int) -> None:
        delay = min(self.backoff_max, self.backoff_base * (2 ** (attempt - 1)))
        delay *= (0.75 + 0.5 * random.random())
        time.sleep(delay)

    def chat_json(self, user_prompt: str) -> str:
        """
        Send system + user messages and return the content of the first choice.
        The content is expected to be JSON (or contain JSON); this method does not
        parse it and only returns the string.
        Automatically retries transient errors (429/5xx/timeouts) using
        exponential backoff with jitter.
        """
        last_err: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = completion(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_output_tokens,
                    timeout=self.timeout,
                )
                content = resp["choices"][0]["message"]["content"]
                if self.debug:
                    print("\n[LLMClient] raw content(head) =", str(content)[:260].replace("\n", "\\n"))
                return _extract_json(str(content), debug=self.debug)

            except Exception as e:
                last_err = e
                if attempt < self.max_retries and self._should_retry(e):
                    if self.debug:
                        print(f"[LLMClient] retryable error on attempt {attempt}: {e}")
                    self._sleep_backoff(attempt)
                    continue
                raise LLMError(f"LLM call failed: {e}") from e

        raise LLMError(f"LLM call failed: {last_err}") from last_err
