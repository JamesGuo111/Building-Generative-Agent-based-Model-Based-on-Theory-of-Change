# mvp/prompt_builder.py
# Define the user prompts for each LLM decision, based on natural-language translations of the agent’s attribute values after bucketing.
from __future__ import annotations
from typing import Dict, Any, Iterable, List

try:
    from .translator import Translator, AttrSpec
except ImportError:
    from translator import Translator, AttrSpec

def _strip_trailing_punct(s: str) -> str:
    return s.strip().rstrip(" .!?")

def _age_phrase(age_text: str, lowercase: bool = True) -> str:
    label = age_text.split("(")[0].strip()
    return label.lower() if lowercase else label

def _join_options(options: Iterable[str]) -> str:
    opts = [str(o).strip() for o in options if str(o).strip()]
    return ", ".join(opts)

def _enumerate_options(options: Iterable[str]) -> str:
    opts = [str(o).strip() for o in options if str(o).strip()]
    return ", ".join(f"{i}) {o}" for i, o in enumerate(opts, 1))

def _options_block(options: Iterable[str]) -> str:
    """
    Render allowed options without numbering (IMPORTANT for schema consistency).
    Each option must appear exactly as-is in the prompt so the model can copy it.
    """
    opts = [str(o).strip() for o in options if str(o).strip()]
    return "\n".join(f"- {o}" for o in opts)



class PromptBuilder:

    def __init__(self, country: str = "Ethiopia", translator: Translator | None = None):
        self.country = country
        self.tr = translator or Translator()

    def build_attendance_girl(self, girl: Dict[str, Any], hh: Dict[str, Any], school: Dict[str, Any]) -> str:
        age = self._txt([girl], "age")
        status = self._txt([girl], "status")
        grade = self._txt([girl], "grade")
        L = self._sent([girl], "L")
        esteem = self._sent([girl], "esteem")
        enjoyment = self._sent([girl], "enjoyment")
        econ = self._sent([hh], "econ_barrier")
        marriage = self._txt([girl], "marriage")
        transport = self._sent([hh], "transport")
        Q = self._sent([school], "quality_of_teaching")
        fairness = self._sent([school], "fairness")
        peer = self._sent([school], "peer_support")
        safety = self._sent([school], "safety")

        age_in_phrase = _age_phrase(age, lowercase=True)

        header = (
            f"You are a girl in {age_in_phrase} living in {self.country}. "
            f"You are currently {status} and {grade} grade."
        )
        body = (
            f" {_strip_trailing_punct(L)}. {_strip_trailing_punct(esteem)}. "
            f"{_strip_trailing_punct(enjoyment)}. {_strip_trailing_punct(econ)}. "
            f"You are {marriage}. {_strip_trailing_punct(transport)}. "
            f"{_strip_trailing_punct(Q)}. {_strip_trailing_punct(fairness)}. "
            f"{_strip_trailing_punct(peer)}. {_strip_trailing_punct(safety)}."
        )
        ask = (
            "Give these factors, do you choose to attend the school for next two weeks?\n"
            "\"There isn’t enough information\" or \"It is unclear\" are not acceptable answers. "
            "Give a \"Yes\" or \"No\" answer. "
            "Give one sentence to explain your choice."
        )
        return f"{header}{body}\n\n{ask}"

    def build_attendance_household(
        self,
        hh: Dict[str, Any],
        girl: Dict[str, Any],
        school: Dict[str, Any],
        girl_attendance_decision: str
    ) -> str:
        age = self._txt([girl], "age")
        econ = self._sent([hh], "econ_barrier")
        attitude = self._sent([hh], "attitude")
        chore = self._txt([hh], "chores")
        transport = self._txt([hh], "transport")
        marriage = self._txt([girl], "marriage")
        Q = self._sent([school], "quality_of_teaching")
        fairness = self._sent([school], "fairness")
        peer = self._sent([school], "peer_support")
        safety = self._sent([school], "safety")

        age_in_phrase = _age_phrase(age, lowercase=True)

        header = (
            f"You are the parent/caregiver of a girl in {age_in_phrase} living in {self.country}. "
            f"{_strip_trailing_punct(econ)}. {_strip_trailing_punct(attitude)}. "
            f"{_strip_trailing_punct(chore)}. {_strip_trailing_punct(transport)}. "
            f"Your child is {marriage}. {_strip_trailing_punct(Q)}. {_strip_trailing_punct(fairness)}. "
            f"{_strip_trailing_punct(peer)}. {_strip_trailing_punct(safety)}."
        )
        ask = (
            "Give these factors, do you allow your child to attend the school for next two weeks? "
            f"Your child’s decision on attendance is {girl_attendance_decision}. "
            "\"There isn’t enough information\" or \"It is unclear\" are not acceptable answers. "
            "Give a \"Yes\" or \"No\" answer. "
            "Give one sentence to explain your choice."
        )
        return f"{header}\n\n{ask}"

    def build_transition_girl(
        self,
        girl: Dict[str, Any],
        hh: Dict[str, Any],
        school: Dict[str, Any],
        options: List[str]
    ) -> str:
        age = self._txt([girl], "age")
        status = self._txt([girl], "status")
        grade = self._txt([girl], "grade")
        L = self._sent([girl], "L")
        esteem = self._sent([girl], "esteem")
        enjoyment = self._sent([girl], "enjoyment")
        econ = self._sent([hh], "econ_barrier")
        employment = self._txt([girl], "employment_skills")
        marriage = self._txt([girl], "marriage")
        transport = self._sent([hh], "transport")
        Q = self._sent([school], "quality_of_teaching")
        fairness = self._sent([school], "fairness")
        peer = self._sent([school], "peer_support")
        safety = self._sent([school], "safety")

        age_in_phrase = _age_phrase(age, lowercase=True)
        opts = _options_block(options)

        header = (
            f"You are a girl in {age_in_phrase} living in {self.country}. "
            f"You are currently {status} and {grade} grade. "
            f"{_strip_trailing_punct(L)}. {_strip_trailing_punct(esteem)}. "
            f"{_strip_trailing_punct(enjoyment)}. {_strip_trailing_punct(econ)}. "
            f"You {_strip_trailing_punct(employment)}. You are {marriage}. "
            f"{_strip_trailing_punct(transport)}. {_strip_trailing_punct(Q)}. "
            f"{_strip_trailing_punct(fairness)}. {_strip_trailing_punct(peer)}. {_strip_trailing_punct(safety)}."
        )
        ask = (
            "Given these factors, what will you choose to do for next year?\n"
            "Choose ONE option from the list below:\n"
            f"{opts}\n"
            "\"There isn’t enough information\" or \"It is unclear\" are not acceptable answers. "
            "State your option directly without words like \"I choose…\". "
            "Give one sentence to explain your choice."
        )

        return f"{header}\n\n{ask}"

    def build_transition_household(
        self,
        hh: Dict[str, Any],
        girl: Dict[str, Any],
        school: Dict[str, Any],
        options: List[str],
        girl_transition_decision: str
    ) -> str:
        age = self._txt([girl], "age")
        econ = self._sent([hh], "econ_barrier")
        attitude = self._sent([hh], "attitude")
        chore = self._txt([hh], "chores")
        transport = self._txt([hh], "transport")
        employment = self._txt([girl], "employment_skills")
        marriage = self._txt([girl], "marriage")
        Q = self._sent([school], "quality_of_teaching")
        fairness = self._sent([school], "fairness")
        peer = self._sent([school], "peer_support")
        safety = self._sent([school], "safety")

        age_in_phrase = _age_phrase(age, lowercase=True)
        opts = _options_block(options)

        header = (
            f"You are the parent/caregiver of a girl in {age_in_phrase} living in {self.country}. "
            f"{_strip_trailing_punct(econ)}. {_strip_trailing_punct(attitude)}. "
            f"{_strip_trailing_punct(chore)}. {_strip_trailing_punct(transport)}. "
            f"Your child {_strip_trailing_punct(employment)}. Your child is {marriage}. "
            f"{_strip_trailing_punct(Q)}. {_strip_trailing_punct(fairness)}. "
            f"{_strip_trailing_punct(peer)}. {_strip_trailing_punct(safety)}."
        )
        ask = (
            "Given these factors, what will you choose to do for next year?\n"
            f"Your child’s decision on this is {girl_transition_decision}.\n"
            "Choose ONE option from the list below:\n"
            f"{opts}\n"
            "\"There isn’t enough information\" or \"It is unclear\" are not acceptable answers. "
            "State your option directly without words like \"I choose…\". "
            "Give one sentence to explain your choice."
        )

        return f"{header}\n\n{ask}"

    def build_self_esteem_eval(self, girl: Dict[str, Any], hh: Dict[str, Any], school: Dict[str, Any]) -> str:
        age = self._txt([girl], "age")
        status = self._txt([girl], "status")
        grade = self._txt([girl], "grade")
        L = self._sent([girl], "L")
        enjoyment = self._sent([girl], "enjoyment")
        econ = self._sent([hh], "econ_barrier")
        employment = self._txt([girl], "employment_skills")
        attitude_2nd = self._sent([hh], "attitude")
        marriage = self._txt([girl], "marriage")
        Q = self._sent([school], "quality_of_teaching")
        fairness = self._sent([school], "fairness")
        peer = self._sent([school], "peer_support")
        safety = self._sent([school], "safety")

        age_in_phrase = _age_phrase(age, lowercase=True)

        header = (
            f"You are a girl in {age_in_phrase} living in {self.country}. "
            f"You are currently {status} and {grade} grade. "
            f"{_strip_trailing_punct(L)}. {_strip_trailing_punct(enjoyment)}. {_strip_trailing_punct(econ)}. "
            f"You {_strip_trailing_punct(employment)}. "
            f"Your parents/caregiver’s attitude toward education (in second person description): "
            f"{_strip_trailing_punct(attitude_2nd)}. "
            f"You are {marriage}. {_strip_trailing_punct(Q)}. {_strip_trailing_punct(fairness)}. "
            f"{_strip_trailing_punct(peer)}. {_strip_trailing_punct(safety)}."
        )

        rubric = (
            "Given these factors, evaluate your level of self-esteem on a scale of 0-100, following the criteria below:\n"
            "0-25\n"
            "You often feel nervous and anxious when asked to speak in front of others, and you believe that doing well on a test is just a matter of luck rather than your own ability. "
            "Your family makes most of the important decisions for you — whether you attend school, when you will marry, and even how much time you can spend with friends. "
            "At school, you frequently feel lonely and unsupported.\n"
            "26-50\n"
            "You sometimes get nervous about reading in front of others and still doubt your own role in success, often thinking luck plays a big part. "
            "Your family strongly influences your choices, such as school attendance and time with friends, though you may have a small say. "
            "At school, you occasionally feel lonely or disconnected from peers.\n"
            "51-75\n"
            "You feel more comfortable in front of others, though you may still worry at times. "
            "You can recognize your effort in doing well, though you sometimes attribute success to luck. "
            "Your family continues to guide important decisions, such as education and friendships, but you also have growing input. "
            "At school, you can feel included but still experience moments of loneliness.\n"
            "76-100\n"
            "You feel confident reading in front of others and believe your achievements come from your own abilities and effort. "
            "You have a stronger voice in personal decisions, with your family offering guidance rather than full control over your schooling, friendships, or future. "
            "At school, you feel connected and supported, rarely experiencing loneliness.\n\n"
            "Return an integer from 0-100 that best fit with your evaluation. "
            "\"There isn’t enough information\" or \"It is unclear\" are not acceptable answers. "
            "Directly give out the number without saying words like \"I mark myself …\". "
            "Give one sentence to explain your evaluation."
        )
        return f"{header}\n\n{rubric}"

    def build_household_attitude_eval(self, hh: Dict[str, Any], girl: Dict[str, Any], school: Dict[str, Any]) -> str:
        age = self._txt([girl], "age")
        status = self._txt([girl], "status")
        grade = self._txt([girl], "grade")
        econ = self._sent([hh], "econ_barrier")
        chore = self._sent([hh], "chores")
        L_2nd = self._sent([girl], "L")
        employment = self._txt([girl], "employment_skills")
        esteem_2nd = self._sent([girl], "esteem")
        enjoyment_2nd = self._sent([girl], "enjoyment")
        marriage = self._txt([girl], "marriage")
        Q = self._sent([school], "quality_of_teaching")
        safety = self._sent([school], "safety")

        age_in_phrase = _age_phrase(age, lowercase=True)

        header = (
            f"You are the parent/caregiver of a girl in {age_in_phrase} living in {self.country}. "
            f"Your child is currently {status} and {grade} grade. "
            f"{_strip_trailing_punct(econ)}. {_strip_trailing_punct(chore)}. "
            f"Your child’s level of learning proficiency is (in second person description): {_strip_trailing_punct(L_2nd)}. "
            f"Your child {_strip_trailing_punct(employment)}. "
            f"Your child’s level of self-esteem is (in second person description): {_strip_trailing_punct(esteem_2nd)}. "
            f"Your child’s level of enjoyment is (in second person description): {_strip_trailing_punct(enjoyment_2nd)}. "
            f"Your child is {marriage}. {_strip_trailing_punct(Q)}. {_strip_trailing_punct(safety)}."
        )

        rubric = (
            "Given these factors, evaluate your attitude toward your child’s education on a scale of 0-100, following the criteria below:\n"
            "0-25\n"
            "You see no value in a girl’s education and refuse to cover any school-related costs.\n"
            "26-50\n"
            "You think education may have some value, but you do not believe it is necessary for a girl to attend school and not want to take the financial burden.\n"
            "51-75\n"
            "You agree a girl should go to school, but you are hesitant to bear the financial burden.\n"
            "76-100\n"
            "You believe a girl’s education is meaningful and necessary, and you are fully willing to cover school-related costs.\n\n"
            "Return an integer from 0-100 that best fit with your evaluation. "
            "\"There isn’t enough information\" or \"It is unclear\" are not acceptable answers. "
            "Directly give out the number without saying words like \"I mark myself …\". "
            "Give one sentence to explain your evaluation."
        )
        return f"{header}\n\n{rubric}"

    # Internal: translation / value access utilities

    def _txt(self, sources: List[Dict[str, Any]], attr: str) -> str:
        t = self._translate_from_sources(sources, attr)
        return t["text"]

    def _sent(self, sources: List[Dict[str, Any]], attr: str) -> str:
        return _strip_trailing_punct(self._txt(sources, attr))

    def _translate_from_sources(self, sources: List[Dict[str, Any]], attr: str) -> Dict[str, Any]:
        spec: AttrSpec | None = self.tr._get_spec(attr)
        if spec is None:
            raise KeyError(f"Unknown attribute: {attr}")

        keys = (spec.name,) + spec.aliases
        for src in sources:
            for k in keys:
                if k in src:
                    return self.tr.translate_attr(spec.name, src[k])

        raise KeyError(f"Missing value for attribute '{spec.name}' (aliases: {spec.aliases}) in provided sources.")
