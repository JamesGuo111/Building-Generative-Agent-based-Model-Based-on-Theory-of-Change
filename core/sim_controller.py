# sim_controller.py
"""
This module exists to support Theory of Change (ToC) analysis within the model.
The core goal is to examine whether a given causal chain holds, or to test alternative
causal mechanisms.

In an ABM, the effects of interventions must be explicitly controlled by the modeler.
For example, how much an intervention improves school teaching quality must be
manually parameterized. This module defines how such parameters evolve over time,
including growth patterns, decay rules, thresholds for transitions, and initial settings.

By running simulations under different parameter evolution schemes, the modeler can
observe final outcomes and assess whether the intended causal chain is supported.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple, Any, Union

from .utils import stage_of_grade

Number = float

def _clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    if x < lo: return lo
    if x > hi: return hi
    return x

def _clamp01(x: float) -> float:
    if x < 0.0: return 0.0
    if x > 1.0: return 1.0
    return x

ScheduleFn = Callable[[Number, int, Any, "SchoolModel"], Number]
# Convention: fn(current_value, tick, obj, model) -> new_value
# - obj: Girl / Household / string "school"
# - Returned values are clamped to 0..100

# ---- Common schedule helpers (linear / exponential / constant) ----
def sched_constant(v: float) -> ScheduleFn:
    def _f(current: Number, t: int, obj: Any, model: "SchoolModel") -> Number:
        return v
    return _f

def sched_linear_add(delta_per_tick: float, lo: float = 0.0, hi: float = 100.0) -> ScheduleFn:
    def _f(current: Number, t: int, obj: Any, model: "SchoolModel") -> Number:
        return _clamp(current + delta_per_tick, lo, hi)
    return _f

def sched_linear_path(start: float, slope: float, lo: float = 0.0, hi: float = 100.0) -> ScheduleFn:
    """绝对值：value(t) = start + slope * t（忽略 current）。"""
    def _f(current: Number, t: int, obj: Any, model: "SchoolModel") -> Number:
        return _clamp(start + slope * t, lo, hi)
    return _f

def sched_exp_multiply(rate_per_tick: float, lo: float = 0.0, hi: float = 100.0) -> ScheduleFn:
    """乘法增长：new = current * (1 + rate)。"""
    def _f(current: Number, t: int, obj: Any, model: "SchoolModel") -> Number:
        return _clamp(current * (1.0 + rate_per_tick), lo, hi)
    return _f

# ---- Learning process & threshold policies ----
@dataclass
class LearningPolicy:
    gain_per_qual: float = 0.00002
    p_fail_at_qual0: float = 0.10
    fail_penalty: float = 1.0
    absent_decay: float = 0.01

    skill_gain_base: float = 0.6 
    skill_secondary_multiplier: float = 1.2
    skill_use_peer: bool = True            

    def update_learning(self, model: "SchoolModel", g: Any) -> None:
        # ---- 1) attend => increase L ----
        qual = (float(model.school_Q) + float(model.school_fair) + float(model.school_Safety)) / 3.0
        base_gain = self.gain_per_qual * qual
        g.L = _clamp(g.L + base_gain * 100.0)

        # ---- 2) update e_skill（only if agent has e_skill）----
        self.update_employment_skill(model, g)


    def update_employment_skill(self, model: "SchoolModel", g: Any) -> None:
        if not hasattr(g, "e_skill"):
            return

        q = float(getattr(model, "school_Q", 50.0))
        if self.skill_use_peer:
            peer = float(getattr(model, "school_peer", q))
            context = (q + peer) / 2.0
        else:
            context = q

        stg = stage_of_grade(int(getattr(g, "grade", 1)))
        stage_mul = self.skill_secondary_multiplier if stg in ("LS", "US") else 1.0

        context_factor = 0.5 + 0.5 * (context / 100.0)

        gain = self.skill_gain_base * context_factor * stage_mul
        g.e_skill = _clamp(float(g.e_skill) + gain)

    def update_absence(self, model: "SchoolModel", g: Any) -> None:
        g.L = _clamp(g.L - self.absent_decay * 100.0)


@dataclass
class TransitionPolicy:
    base_general: Union[float, Callable[[int, "SchoolModel"], float]] = 0.66
    base_exam: Union[float, Callable[[int, "SchoolModel"], float]] = 0.45
    attendance_weight: Union[float, Callable[[int, "SchoolModel"], float]] = 1.0
    job_find_base: Union[float, Callable[[int, "SchoolModel"], float]] = 0.15

    def _resolve_prob01(self, v: Union[float, Callable[[int, "SchoolModel"], float]], t: int, model: "SchoolModel") -> float:
        out = v(t, model) if callable(v) else float(v)
        return _clamp01(out)


    def _resolve_weight(self, v: Union[float, Callable[[int, "SchoolModel"], float]], t: int, model: "SchoolModel") -> float:
        out = v(t, model) if callable(v) else float(v)
        if out < 0.0: out = 0.0
        if out > 5.0: out = 5.0
        return float(out)

    def get_base_general(self, t: int, model: "SchoolModel") -> float:
        return self._resolve_prob01(self.base_general, t, model)

    def get_base_exam(self, t: int, model: "SchoolModel") -> float:
        return self._resolve_prob01(self.base_exam, t, model)

    def get_attendance_weight(self, t: int, model: "SchoolModel") -> float:
        return self._resolve_weight(self.attendance_weight, t, model)

    def get_job_find_base(self, t: int, model: "SchoolModel") -> float:
        return self._resolve_prob01(self.job_find_base, t, model)


@dataclass
class ParamSchedules:
    """
    Define parameter evolution ovretime
    Fields are grouped by entity type and attribute name.
    - girl: enjoyment / employment_skills（=e_skill）/ L etc.
    - hh  : econ_barrier(=budget/econ) / chores / transport
    - school: quality_of_teaching / fairness / peer_support / safety
    """
    girl: Dict[str, ScheduleFn] = field(default_factory=dict)
    household: Dict[str, ScheduleFn] = field(default_factory=dict)
    school: Dict[str, ScheduleFn] = field(default_factory=dict)

    _ALIASES = {
        "girl": {
            "enjoyment": "enjoy",
            "enjoy": "enjoy",
            "employment_skills": "e_skill",
            "e_skill": "e_skill",
            "L": "L",
        },
        "household": {
            "econ_barrier": "econ_barrier",
            "econ": "econ_barrier",
            "budget": "econ_barrier",
            "chores": "chores",
            "chore": "chores",
            "transport": "transport",
        },
        "school": {
            "quality_of_teaching": "school_Q",
            "Q": "school_Q",
            "school_Q": "school_Q",
            "fairness": "school_fair",
            "school_fair": "school_fair",
            "peer_support": "school_peer",
            "school_peer": "school_peer",
            "safety": "school_Safety",
            "school_Safety": "school_Safety",
        },
    }

    @staticmethod
    def _norm(group: str, attr: str) -> str:
        m = ParamSchedules._ALIASES.get(group, {})
        return m.get(attr, attr)

    def apply(self, model: "SchoolModel") -> None:
        t = getattr(model, "ticks", 0)

        # ---- School-level ----
        for attr, fn in self.school.items():
            a = self._norm("school", attr)
            cur = float(getattr(model, a))
            new = _clamp(fn(cur, t, "school", model))
            setattr(model, a, new)

        # ---- Households ----
        if self.household:
            for hh in model.households:
                for attr, fn in self.household.items():
                    a = self._norm("household", attr)
                    cur = float(getattr(hh, a))
                    new = _clamp(fn(cur, t, hh, model))
                    setattr(hh, a, new)

        # ---- Girls ----
        if self.girl:
            for g in model.girls:
                for attr, fn in self.girl.items():
                    a = self._norm("girl", attr)
                    cur = float(getattr(g, a))
                    new = _clamp(fn(cur, t, g, model))
                    setattr(g, a, new)

@dataclass
class InitPolicy:
    """
    Initialization policy for school, household, and girl attributes.
    """
    def init_school(self, model: "SchoolModel") -> Tuple[float, float, float, float]:
        rng = model.random
        school_Q      = 75
        school_fair   = 75
        school_Safety = 75
        school_peer   = 70
        return (school_Q, school_fair, school_Safety, school_peer)

    def init_household(self, model: "SchoolModel") -> Dict[str, float]:
        rng = model.random
        return {
            "econ_barrier":  rng.random() * 43,
            "attitude":      rng.random() * 66,
            "transport":     rng.randint(0,33) if rng.random() < 0.45 else rng.randint(34,100),
            "chores":        rng.randint(0,33) if rng.random() < 0.25 else rng.randint(34,100),
        }

    def init_girl(self, model: "SchoolModel", hh: Any) -> Dict[str, float]:
        rng = model.random
        grade = self.sample_grade_intervention(rng)
        return {
            "grade": grade,
            "age": 6 + (grade - 1),
            "status": model.ST_ENROLLED,
            "L":  15 + rng.random() * 35,
            "esteem":  rng.random() * 70,
            "enjoy":  rng.random() * 80,
            "e_skill": rng.random() * 50,
            "marriage": rng.randint(0, 1),
        }
    
    @staticmethod
    def sample_grade_intervention(rng) -> int:
        # Intervention (Baseline) girls distribution (Table 6)
        grades  = (4, 5, 6, 7, 8)
        weights = (18.9, 23.9, 24.1, 18.6, 14.3)  # percent weights

        r = rng.random() * sum(weights)
        cum = 0.0
        for g, w in zip(grades, weights):
            cum += w
            if r < cum:
                return g
        return grades[-1]  # safety fallback


# ---- Simulation Controller ----
@dataclass
class SimulationController:
    learning: "LearningPolicy" = field(default_factory=lambda: LearningPolicy())
    transitions: TransitionPolicy = field(default_factory=TransitionPolicy)
    schedules: "ParamSchedules" = field(default_factory=lambda: ParamSchedules())
    init: "InitPolicy" = field(default_factory=lambda: InitPolicy())

    def attach(self, model: "SchoolModel") -> None:
        pass

    def pre_step(self, model: "SchoolModel") -> None:
        self.schedules.apply(model)

    def update_learning(self, model: "SchoolModel", g: Any) -> None:
        self.learning.update_learning(model, g)

    def update_absence(self, model: "SchoolModel", g: Any) -> None:
        self.learning.update_absence(model, g)

    def transition_base_general(self, model: "SchoolModel") -> float:
        return self.transitions.get_base_general(model.ticks, model)

    def transition_base_exam(self, model: "SchoolModel") -> float:
        return self.transitions.get_base_exam(model.ticks, model)

    def transition_attendance_weight(self, model: "SchoolModel") -> float:
        return self.transitions.get_attendance_weight(model.ticks, model)

    def job_find_base(self, model: "SchoolModel") -> float:
        return self.transitions.get_job_find_base(model.ticks, model)
