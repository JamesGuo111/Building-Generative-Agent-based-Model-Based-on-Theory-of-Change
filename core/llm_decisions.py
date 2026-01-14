# llm_decisions.py
# Six types of decision to be made by LLM; called in model.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional

try:
    from .prompt_builder import PromptBuilder
except ImportError:
    from prompt_builder import PromptBuilder

try:
    from .llm_client import LLMClient
except ImportError:
    from llm_client import LLMClient

try:
    from .decision_schemas import parse_yes_no, parse_choice, parse_score, SchemaError
except ImportError:
    from decision_schemas import parse_yes_no, parse_choice, parse_score, SchemaError

try:
    from .decision_cache import DecisionCache
except ImportError:
    from decision_cache import DecisionCache


class DecisionEngine:
    def __init__(
        self,
        client: LLMClient,
        builder: PromptBuilder,
        cache: DecisionCache | None = None,
        *,
        debug: bool = False,
        strict_schema: bool = True,
    ) -> None:
        self.client = client
        self.pb = builder
        self.cache = cache or DecisionCache()
        self.debug = bool(debug)
        self.strict_schema = bool(strict_schema)

    def _on_schema_error(self, where: str, txt: str, *, options: List[str] | None = None, prompt_head: str | None = None) -> None:
        print("\n[SchemaError]", where)
        if prompt_head is not None:
            print("  prompt(head) =", prompt_head[:350].replace("\n", "\\n"))
        if options is not None:
            print("  allowed options =", options)
        print("  model returned =", str(txt)[:500].replace("\n", "\\n"))
        if self.strict_schema:
            raise SchemaError(f"{where}: schema parse failed (see logs above).")

    # ---- Attendance ----
    def attendance_girl(self, girl: Dict[str, Any], hh: Dict[str, Any], school: Dict[str, Any]) -> Tuple[Optional[bool], str]:
        key = self.cache.make_key(
            "attendance_girl",
            model_id=self.client.model,
            country=self.pb.country,
            tr=self.pb.tr,
            girl=girl, hh=hh, school=school,
        )
        cached = self.cache.get_yes_no(key)
        if cached is not None:
            ans, expl = cached
            if self.debug:
                print("[cache hit] attendance_girl →", ans)
            return ans, expl

        prompt = self.pb.build_attendance_girl(girl, hh, school)
        txt = self.client.chat_json(prompt)
        if self.debug:
            print("\n[LLM] attendance_girl returned =", txt)
        try:
            ans_str, expl = parse_yes_no(txt)
            ans = (ans_str == "Yes")
            self.cache.put_yes_no(key, ans, expl)
            return ans, expl
        except SchemaError:
            self._on_schema_error("attendance_girl", txt, prompt_head=prompt)
            return None, "parse_error"

    def attendance_household(self, hh: Dict[str, Any], girl: Dict[str, Any], school: Dict[str, Any], girl_answer: bool) -> Tuple[Optional[bool], str]:
        extra_sig = f"girl_answer={'Yes' if girl_answer else 'No'}"
        key = self.cache.make_key(
            "attendance_household",
            model_id=self.client.model,
            country=self.pb.country,
            tr=self.pb.tr,
            girl=girl, hh=hh, school=school,
            extra_sig=extra_sig,
        )
        cached = self.cache.get_yes_no(key)
        if cached is not None:
            ans, expl = cached
            if self.debug:
                print("[cache hit] attendance_household →", ans)
            return ans, expl

        ga = "Yes" if girl_answer else "No"
        prompt = self.pb.build_attendance_household(hh, girl, school, girl_attendance_decision=ga)
        txt = self.client.chat_json(prompt)
        if self.debug:
            print("\n[LLM] attendance_household returned =", txt)
        try:
            ans_str, expl = parse_yes_no(txt)
            ans = (ans_str == "Yes")
            self.cache.put_yes_no(key, ans, expl)
            return ans, expl
        except SchemaError:
            self._on_schema_error("attendance_household", txt, prompt_head=prompt)
            return None, "parse_error"

    # ---- Transition ----
    def transition_girl(self, girl: Dict[str, Any], hh: Dict[str, Any], school: Dict[str, Any], options_readable: List[str]) -> Tuple[Optional[str], str]:
        opts_sig = " || ".join(options_readable)
        key = self.cache.make_key(
            "transition_girl",
            model_id=self.client.model,
            country=self.pb.country,
            tr=self.pb.tr,
            girl=girl, hh=hh, school=school,
            extra_sig=f"opts={opts_sig}",
        )
        cached = self.cache.get_choice(key)
        if cached is not None:
            choice, expl = cached
            if self.debug:
                print("[cache hit] transition_girl →", choice)
            return choice, expl

        prompt = self.pb.build_transition_girl(girl, hh, school, options_readable)
        if self.debug:
            print("\n[transition_girl] allowed options =", options_readable)
            print("[transition_girl] prompt(head) =", prompt[:320].replace("\n", "\\n"))

        txt = self.client.chat_json(prompt)
        if self.debug:
            print("[LLM] transition_girl returned =", txt)

        try:
            choice, expl = parse_choice(txt, options_readable)
            self.cache.put_choice(key, choice, expl)
            return choice, expl
        except SchemaError:
            self._on_schema_error("transition_girl", txt, options=options_readable, prompt_head=prompt)
            return None, "parse_error"

    def transition_household(self, hh: Dict[str, Any], girl: Dict[str, Any], school: Dict[str, Any], options_readable: List[str], girl_choice: str) -> Tuple[Optional[str], str]:
        opts_sig = " || ".join(options_readable)
        extra_sig = f"opts={opts_sig} | girl_choice={girl_choice.strip().rstrip('.')}"
        key = self.cache.make_key(
            "transition_household",
            model_id=self.client.model,
            country=self.pb.country,
            tr=self.pb.tr,
            girl=girl, hh=hh, school=school,
            extra_sig=extra_sig,
        )
        cached = self.cache.get_choice(key)
        if cached is not None:
            choice, expl = cached
            if self.debug:
                print("[cache hit] transition_household →", choice)
            return choice, expl

        prompt = self.pb.build_transition_household(hh, girl, school, options_readable, girl_transition_decision=girl_choice)
        if self.debug:
            print("\n[transition_household] allowed options =", options_readable)
            print("[transition_household] girl_choice =", girl_choice)
            print("[transition_household] prompt(head) =", prompt[:320].replace("\n", "\\n"))

        txt = self.client.chat_json(prompt)
        if self.debug:
            print("[LLM] transition_household returned =", txt)

        try:
            choice, expl = parse_choice(txt, options_readable)
            self.cache.put_choice(key, choice, expl)
            return choice, expl
        except SchemaError:
            self._on_schema_error("transition_household", txt, options=options_readable, prompt_head=prompt)
            return None, "parse_error"

    # ---- Evaluations ----
    def eval_self_esteem(self, girl: Dict[str, Any], hh: Dict[str, Any], school: Dict[str, Any]) -> Tuple[Optional[int], str]:
        key = self.cache.make_key(
            "self_esteem_eval",
            model_id=self.client.model,
            country=self.pb.country,
            tr=self.pb.tr,
            girl=girl, hh=hh, school=school,
        )
        cached = self.cache.get_score(key)
        if cached is not None:
            score, expl = cached
            if self.debug:
                print("[cache hit] self_esteem_eval →", score)
            return score, expl

        prompt = self.pb.build_self_esteem_eval(girl, hh, school)
        txt = self.client.chat_json(prompt)
        if self.debug:
            print("\n[LLM] self_esteem_eval returned =", txt)
        try:
            score, expl = parse_score(txt)
            self.cache.put_score(key, score, expl)
            return score, expl
        except SchemaError:
            self._on_schema_error("self_esteem_eval", txt, prompt_head=prompt)
            return None, "parse_error"

    def eval_household_attitude(self, hh: Dict[str, Any], girl: Dict[str, Any], school: Dict[str, Any]) -> Tuple[Optional[int], str]:
        key = self.cache.make_key(
            "household_attitude_eval",
            model_id=self.client.model,
            country=self.pb.country,
            tr=self.pb.tr,
            girl=girl, hh=hh, school=school,
        )
        cached = self.cache.get_score(key)
        if cached is not None:
            score, expl = cached
            if self.debug:
                print("[cache hit] household_attitude_eval →", score)
            return score, expl

        prompt = self.pb.build_household_attitude_eval(hh, girl, school)
        txt = self.client.chat_json(prompt)
        if self.debug:
            print("\n[LLM] household_attitude_eval returned =", txt)
        try:
            score, expl = parse_score(txt)
            self.cache.put_score(key, score, expl)
            return score, expl
        except SchemaError:
            self._on_schema_error("household_attitude_eval", txt, prompt_head=prompt)
            return None, "parse_error"
