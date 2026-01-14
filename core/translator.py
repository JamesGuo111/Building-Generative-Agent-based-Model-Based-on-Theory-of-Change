# translator.py
"""
Attribute Translator (0–100 scale)
----------------------------------
Purpose:
- Internally, the ABM stores attributes using a unified 0–100 scale (or a small set of discrete integers).
- When sending inputs to the LLM, numeric values are translated into natural language based on predefined bucketing rules.
- At the same time, the translator returns the bucket index (bucket_index: 1, 2, …) and the bucket range (bucket_range),
  which are later used to construct multi-dimensional decision cache keys.

Features:
- No third-party dependencies; uses only the Python standard library.
- Does not perform any rescaling or clamping: the caller must ensure that values are already within valid ranges
  (e.g., 0–100 for attributes, age 6–18, grade 1–12).

Usage overview:
    from mvp.translator import Translator
    tr = Translator()

    # Translate a single attribute (value already in 0–100 or a discrete integer range)
    res = tr.translate_attr("enjoyment", 78)
    # res = {
    #   "attr": "enjoyment", "value": 78,
    #   "bucket_index": 4, "bucket_range": (76, 100),
    #   "text": "You always love and enjoy school so much."
    # }

    # Translate multiple attribute groups at once (e.g., girl / household / school)
    res_all = tr.translate_group(
        {"L": 43, "esteem": 62, "enjoy": 78, "e_skill": 15, "grade": 9, "status": 1, "age": 14, "marriage": 0},
        group="girl"
    )

Notes:
- All intervals are closed intervals [min, max].
- If a value falls outside the defined intervals, it is safely assigned to the last bucket.
"""


from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

Number = Union[int, float]

# Tool: General Categorization Tool
@dataclass(frozen=True)
class RangeBucket:
    """Continuous bucket: [min, max] (closed interval) + description text"""
    min_incl: int
    max_incl: int
    text: str

@dataclass(frozen=True)
class ChoiceBucket:
    """Discrete value → description text"""
    value: int
    text: str

@dataclass(frozen=True)
class AttrSpec:
    """
    Bucketing specification for a single attribute:
    - kind="range": uses a list of RangeBucket
    - kind="choice": uses a list of ChoiceBucket
    - aliases: alternative names for the attribute
      (e.g., "enjoy" ↔ "enjoyment"; "e_skill" ↔ "employment_skills")
    """
    name: str
    kind: str  # "range" or "choice"
    range_buckets: Optional[List[RangeBucket]] = None
    choice_buckets: Optional[List[ChoiceBucket]] = None
    aliases: Tuple[str, ...] = ()

# Specification definitions (from the provided mapping table)
SPECS: Dict[str, AttrSpec] = {}

def _register(spec: AttrSpec):
    SPECS[spec.name] = spec
    for a in spec.aliases:
        SPECS[a] = spec

# ===== Girls =====
# Learning Proficiency
_register(AttrSpec(
    name="L",
    kind="range",
    range_buckets=[
        RangeBucket(0, 25,   "You are a non-learner: someone who shows little or no engagement with learning, making minimal effort to acquire or apply new knowledge and skills."),
        RangeBucket(26, 50,  "You are an emergent learner: someone who is beginning to develop understanding, relying on guidance and support while exploring how to apply knowledge and skills."),
        RangeBucket(51, 75,  "You are an established learner: someone who demonstrates a dependable grasp of knowledge and skills, applying them accurately and effectively in routine or familiar situations."),
        RangeBucket(76, 100, "You are a proficient learner: someone who confidently applies knowledge and skills with accuracy, adaptability, and understanding across different contexts."),
    ],
    aliases=("learning_proficiency",),
))


# Esteem
_register(AttrSpec(
    name="esteem",
    kind="range",
    range_buckets=[
        RangeBucket(0, 25, "You often feel nervous and anxious when asked to speak in front of others, and you believe that doing well on a test is just a matter of luck rather than your own ability. Your family makes most of the important decisions for you — whether you attend school, when you will marry, and even how much time you can spend with friends. At school, you frequently feel lonely and unsupported."),
        RangeBucket(26, 50, "You sometimes get nervous about reading in front of others and still doubt your own role in success, often thinking luck plays a big part. Your family strongly influences your choices, such as school attendance and time with friends, though you may have a small say. At school, you occasionally feel lonely or disconnected from peers."),
        RangeBucket(51, 75, "You feel more comfortable in front of others, though you may still worry at times. You can recognize your effort in doing well, though you sometimes attribute success to luck. Your family continues to guide important decisions, such as education and friendships, but you also have growing input. At school, you can feel included but still experience moments of loneliness."),
        RangeBucket(76, 100, "You feel confident reading in front of others and believe your achievements come from your own abilities and effort. You have a stronger voice in personal decisions, with your family offering guidance rather than full control over your schooling, friendships, or future. At school, you feel connected and supported, rarely experiencing loneliness."),
    ],
))

# Enjoyment
_register(AttrSpec(
    name="enjoyment",
    kind="range",
    range_buckets=[
        RangeBucket(0, 25,   "You hate school"),
        RangeBucket(26, 50,  "You do not enjoy school, but also not hate it."),
        RangeBucket(51, 75,  "You sometimes enjoy and like school."),
        RangeBucket(76, 100, "You always love and enjoy school so much."),
    ],
    aliases=("enjoy",),
))

# Employment-skills
_register(AttrSpec(
    name="employment_skills",
    kind="range",
    range_buckets=[
        RangeBucket(0, 33, "do not have any practical employment skills and lack experience with tasks related to work or income-generating activities."),
        RangeBucket(34, 66, "have basic employment skills, such as assisting with simple tasks, following instructions, or performing routine work under supervision."),
        RangeBucket(67, 100, "have developed employment skills, including the ability to perform tasks independently, solve common work-related problems, and contribute reliably to income-generating activities."),
    ],
    aliases=("e_skill", "employment-skills"),
))


# Grade (1-12)
_register(AttrSpec(
    name="grade",
    kind="range",
    range_buckets=[
        RangeBucket(1, 4,   "Lower Primary (grades 1-4)"),
        RangeBucket(5, 8,   "Upper Primary (grades 5-8)"),
        RangeBucket(9, 10,  "Lower Secondary (grades 9-10)"),
        RangeBucket(11, 12, "Upper Secondary (grades 11-12)"),
    ],
))

# Status（Discrete choice）
_register(AttrSpec(
    name="status",
    kind="choice",
    choice_buckets=[
        ChoiceBucket(0, "dropped out of school"),
        ChoiceBucket(1, "enrolled in school"),
        ChoiceBucket(2, "in Technical and Vocational Education and Training"),
        ChoiceBucket(3, "employed"),
    ],
))

# Age（6-18 into 3 buckets)
_register(AttrSpec(
    name="age",
    kind="range",
    range_buckets=[
        RangeBucket(6, 10,  "childhood (6-10)"),
        RangeBucket(11, 14, "Early adolescence (11-14)"),
        RangeBucket(15, 18, "Late adolescence (15-18)"),
    ],
))

# Marriage（Discrete choice）
_register(AttrSpec(
    name="marriage",
    kind="choice",
    choice_buckets=[
        ChoiceBucket(0, "not married"),
        ChoiceBucket(1, "married"),
    ],
))

# ===== Households =====

# Econ-barrier
_register(AttrSpec(
    name="econ_barrier",
    kind="range",
    range_buckets=[
        RangeBucket(0, 40,  "Your family is poor; you struggle to cover basic daily needs, and paying for school-related costs is very difficult."),
        RangeBucket(41, 70, "You can meet basic needs and some school-related expenses, but higher costs for education seems to be unacceptable."),
        RangeBucket(71, 100,"You are financially secure, and covering education is well within your family’s capacity."),
    ],
))

# Attitude
_register(AttrSpec(
    name="attitude",
    kind="range",
    range_buckets=[
        RangeBucket(0, 25,   "You see no value in a girl’s education and refuse to cover any school-related costs."),
        RangeBucket(26, 50,  "You think education may have some value, but you do not believe it is necessary for a girl to attend school and not want to take the financial burden."),
        RangeBucket(51, 75,  "You agree a girl should go to school, but you are hesitant to bear the financial burden."),
        RangeBucket(76, 100, "You believe a girl’s education is meaningful and necessary, and you are fully willing to cover school-related costs."),
    ],
))

# Transport
_register(AttrSpec(
    name="transport",
    kind="range",
    range_buckets=[
        RangeBucket(0, 30,  "Getting to school is extremely difficult with high costs."),
        RangeBucket(31, 70, "Getting to school is not easy and will cost some time and money"),
        RangeBucket(71, 100,"Getting to school is easy."),
    ],
))

# Chores
_register(AttrSpec(
    name="chores",
    kind="range",
    range_buckets=[
        RangeBucket(0, 33,   "Heavy chore burdens every week"),
        RangeBucket(34, 66,  "Moderate chore burdens every week"),
        RangeBucket(67, 100, "Light chore burdens every week"),
    ],
    aliases=("chore",),
))


# ===== School =====
# Quality of teaching
_register(AttrSpec(
    name="quality_of_teaching",
    kind="range",
    range_buckets=[
        RangeBucket(0, 40,  "The school provides limited instruction, with teachers often unprepared or lacking the skills to engage students effectively."),
        RangeBucket(41, 70, "The school delivers adequate instruction, with teachers covering core material but offering limited support for deeper understanding."),
        RangeBucket(71, 100,"The school provides strong and effective instruction, with teachers well-prepared, supportive, and able to engage students in meaningful learning."),
    ],
    aliases=("Q", "school_Q"),
))

# Fairness
_register(AttrSpec(
    name="fairness",
    kind="range",
    range_buckets=[
        RangeBucket(0, 40,  "The school treats students unequally, with favoritism or bias strongly affecting opportunities and outcomes."),
        RangeBucket(41, 70, "The school provides a generally fair environment, though some students still experience unequal treatment or opportunities."),
        RangeBucket(71, 100,"The school consistently treats students equally, ensuring fair opportunities and outcomes for everyone."),
    ],
    aliases=("fair", "school_fair"),
))

_register(AttrSpec(
    name="peer_support",
    kind="range",
    range_buckets=[
        RangeBucket(0, 33, "Students rarely challenge unfair gender behaviors, seldom recognize or report violence, and show limited understanding of disability or inclusion."),
        RangeBucket(34, 66, "Students occasionally challenge unfair gender behaviors, recognize violence but inconsistently report it, and show basic awareness of disability or inclusion without actively supporting affected peers."),
        RangeBucket(67, 100, "Students actively challenge inequitable gender behaviors, boys and girls recognize and report violence and abuse, and they show understanding and acceptance of disability while supporting each other."),
    ],
    aliases=("peer_sup", "school_peer"),
))


# Safety
_register(AttrSpec(
    name="safety",
    kind="range",
    range_buckets=[
        RangeBucket(0, 40,  "The school environment is unsafe, with frequent risks of violence, harassment, or accidents, and little protection for students."),
        RangeBucket(41, 70, "The school is generally safe, though some risks of violence, harassment, or accidents remain and protections are uneven."),
        RangeBucket(71, 100,"The school provides a secure environment, with strong measures to prevent violence, harassment, and accidents, ensuring students feel protected."),
    ],
    aliases=("Safety", "school_Safety"),
))

# Translator Imlementation
class Translator:
    """
    Attribute translator (assumes inputs are in the 0–100 range or discrete integers,
    and converts them into bucketed natural-language descriptions for inclusion in LLM prompts):
    - translate_attr(name, value): translate a single attribute and return its text along with bucket information
    - translate_group(state_dict, group=...): translate a group of attributes ("girl" / "household" / "school")
    - format_for_prompt(result_dict): format the translated results into text lines suitable for prompt assembly
    """


    GROUP_ATTRS = {
        "girl": ("L", "esteem", "enjoyment", "employment_skills", "grade", "status", "age", "marriage"),
        "household": ("econ_barrier", "attitude", "transport", "chores"),
        "school": ("quality_of_teaching", "fairness", "peer_support", "safety"),
    }

    def __init__(self) -> None:
        self._specs = SPECS

    def translate_attr(self, name: str, value: Number) -> Dict[str, Any]:
        """
        Translate a single attribute (no rescaling or clamping is performed).
        Returns:
        {
            "attr": standardized attribute name,
            "value": original value,
            "bucket_index": bucket index (starting from 1; for choice types, follows definition order),
            "bucket_range": (min, max) or None (None for choice types),
            "text": natural-language description
        }
        """
        spec = self._get_spec(name)
        if spec is None:
            raise KeyError(f"Unknown attribute: {name}")

        if spec.kind == "range":
            v = float(value)
            b_idx, bucket = self._pick_range_bucket(v, spec.range_buckets or [])
            return {
                "attr": spec.name,
                "value": v,
                "bucket_index": b_idx,
                "bucket_range": (bucket.min_incl, bucket.max_incl),
                "text": bucket.text,
            }

        elif spec.kind == "choice":
            v_int = int(round(float(value)))
            b_idx, bucket = self._pick_choice_bucket(v_int, spec.choice_buckets or [])
            return {
                "attr": spec.name,
                "value": v_int,
                "bucket_index": b_idx,
                "bucket_range": None,
                "text": bucket.text,
            }

        else:
            raise ValueError(f"Unknown spec.kind: {spec.kind}")

    def translate_group(self, state: Dict[str, Number], *, group: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        Translate a group of attributes: takes {attr: value} as input and returns
        {attr: translated result dictionary}.

        - If group ('girl' | 'household' | 'school') is provided, only commonly used
          attributes for that group are translated (missing or unknown attributes are ignored).
        - If group is not provided, all keys in the state dictionary are translated
          (attribute aliases are recognized).
        """
        results: Dict[str, Dict[str, Any]] = {}
        names = tuple(state.keys()) if group is None else self.GROUP_ATTRS.get(group, ())
        for raw_name in names:
            spec = self._get_spec(raw_name)
            if spec is None:
                continue
            if raw_name not in state:
                val = None
                for alias in (spec.name,) + spec.aliases:
                    if alias in state:
                        val = state[alias]
                        break
                if val is None:
                    continue
            else:
                val = state[raw_name]

            results[spec.name] = self.translate_attr(spec.name, val)
        return results

    @staticmethod
    def format_for_prompt(translated: Dict[str, Dict[str, Any]]) -> str:
        lines = []
        for attr in sorted(translated.keys()):
            lines.append(f"- {attr.replace('_', ' ').title()}: {translated[attr]['text']}")
        return "\n".join(lines)

    # Internal Tool

    def _get_spec(self, name: str) -> Optional[AttrSpec]:
        return self._specs.get(name)

    @staticmethod
    def _pick_range_bucket(v: float, buckets: List[RangeBucket]) -> Tuple[int, RangeBucket]:
        for i, b in enumerate(buckets, start=1):
            if b.min_incl <= v <= b.max_incl:
                return i, b
        # Fall back: if no bucket matches, assign to the last one.
        return len(buckets), buckets[-1]

    @staticmethod
    def _pick_choice_bucket(v: int, buckets: List[ChoiceBucket]) -> Tuple[int, ChoiceBucket]:
        for i, b in enumerate(buckets, start=1):
            if v == b.value:
                return i, b
        # Fallback: if no bucket matches, assign to the first one.
        return 1, buckets[0]
