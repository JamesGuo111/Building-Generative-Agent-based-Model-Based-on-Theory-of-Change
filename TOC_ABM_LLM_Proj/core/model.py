# mvp/model.py
# ABM主体实现
from __future__ import annotations
from typing import List, Tuple, Dict, Optional
from mesa import Model
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
import random

from .agents import Girl, Household, SchoolTile
from .utils import clamp01, mean, weighted_pick, stage_of_grade, dist_point_to_rect

from .llm_client import LLMClient, DEFAULT_SYSTEM_PROMPT
from .llm_decisions import DecisionEngine
from .prompt_builder import PromptBuilder
from .decision_cache import DecisionCache

from .sim_controller import SimulationController, LearningPolicy, TransitionPolicy, ParamSchedules


# ===== Transition options: internal code ↔ readable label =====
TRANSITION_CODE_TO_READABLE = {
    "progress":        "progress to next grade",
    "remain":          "remain in the same grade",
    "dropout":         "drop out of school",
    "tvet":            "enter TVET (Technical and Vocational Education and Training)",
    "employment":      "go find a job",
    "re-enrol":        "re-enrol in school",
    "stay-out":        "stay out of school",
}

class SchoolModel(Model):
    ST_DROPPED = 0
    ST_ENROLLED = 1
    ST_TVET = 2
    ST_EMP = 3
    ST_GRAD = 4

    def __init__(self,
                 grid_width=51, grid_height=51,
                 school_width=15, school_height=9,
                 num_pairs=120,
                 init_in_school_pct=40,
                 seed: int | None = None,
                 year_ticks=24, period_4ticks=4,
                 L_absent_decay=0.01,
                 job_find_base=0.15,
                 decisions: str = "llm",
                 llm_model: str = "gemini/gemini-2.5-flash-lite",
                 country: str = "Ethiopia",
                 controller: SimulationController | None = None,
                 # debug toggles
                 debug: bool = True,
                 debug_transition_limit: int = 999999,
                 strict_transition_mapping: bool = True,
                 strict_schema: bool = True,
                 ):
        super().__init__()
        self.random = random.Random(seed)

        self.debug = bool(debug)
        self.debug_transition_limit = int(debug_transition_limit)
        self.strict_transition_mapping = bool(strict_transition_mapping)
        self.strict_schema = bool(strict_schema)

        self.grid_w = grid_width
        self.grid_h = grid_height
        self.grid = MultiGrid(self.grid_w, self.grid_h, torus=False)

        self.school_left   = (self.grid_w // 2) - (school_width // 2)
        self.school_right  = self.school_left + school_width - 1
        self.school_bottom = (self.grid_h // 2) - (school_height // 2)
        self.school_top    = self.school_bottom + school_height - 1

        # Controller
        if controller is None:
            learning = LearningPolicy(absent_decay=L_absent_decay)
            transitions = TransitionPolicy(
                base_general=0.50, 
                base_exam=0.35,     
                attendance_weight=1.0, 
                job_find_base=job_find_base,
            )
            schedules = ParamSchedules()
            self.ctrl = SimulationController(learning=learning, transitions=transitions, schedules=schedules)
        else:
            self.ctrl = controller
        self.ctrl.attach(self)

        self.school_Q, self.school_fair, self.school_Safety, self.school_peer = self.ctrl.init.init_school(self)

        self.num_pairs = num_pairs
        self.init_in_school_pct = init_in_school_pct
        self.year_ticks = year_ticks
        self.period_4ticks = period_4ticks

        self.households: List[Household] = []
        self.girls: List[Girl] = []
        self.school_tiles: List[SchoolTile] = []

        self._setup_school_tiles()
        self._setup_households()
        self._setup_girls()
        self._initial_place_some_in_school()
        self.ticks = 0

        self.datacollector = DataCollector(
            model_reporters={
                "Enrolled":   lambda m: sum(1 for g in m.girls if g.status == m.ST_ENROLLED),
                "Dropped":    lambda m: sum(1 for g in m.girls if g.status == m.ST_DROPPED),
                "TVET":       lambda m: sum(1 for g in m.girls if g.status == m.ST_TVET),
                "Employed":   lambda m: sum(1 for g in m.girls if g.status == m.ST_EMP),
                "Graduated":  lambda m: sum(1 for g in m.girls if g.status == m.ST_GRAD),
                "SuccessfulTransition": lambda m: sum(1 for g in m.girls if g.status in (m.ST_TVET, m.ST_EMP, m.ST_GRAD)),
                "InSchoolNow": lambda m: m.girls_in_school_now(),
            }
        )
        self.datacollector.collect(self)

        # LLM Components & Cache
        self.decisions = decisions
        if self.decisions == "llm":
            client = LLMClient(model=llm_model, temperature=0.2, system_prompt=DEFAULT_SYSTEM_PROMPT, debug=self.debug)
            builder = PromptBuilder(country=country)
            self.cache = DecisionCache(rng=self.random, min_samples_to_use=3, max_samples=3)
            self.decider = DecisionEngine(
                client, builder, cache=self.cache,
                debug=self.debug,
                strict_schema=self.strict_schema,
            )
        else:
            self.decider = None
            self.cache = DecisionCache()

    # Initialization
    def _is_school_cell(self, x: int, y: int) -> bool:
        return (self.school_left <= x <= self.school_right) and (self.school_bottom <= y <= self.school_top)

    def _setup_school_tiles(self):
        uid = 0
        for x in range(self.grid_w):
            for y in range(self.grid_h):
                if self._is_school_cell(x, y):
                    tile = SchoolTile(unique_id=f"S-{uid}", model=self)
                    uid += 1
                    self.school_tiles.append(tile)
                    self.grid.place_agent(tile, (x, y))

    def _random_non_school_empty_cell(self) -> Tuple[int, int]:
        while True:
            x = self.random.randrange(self.grid_w)
            y = self.random.randrange(self.grid_h)
            if not self._is_school_cell(x, y) and self._cell_is_empty_of_household(x, y):
                return x, y

    def _cell_is_empty_of_household(self, x: int, y: int) -> bool:
        cell_agents = self.grid.get_cell_list_contents((x, y))
        return not any(isinstance(a, Household) for a in cell_agents)

    def _random_school_cell(self) -> Tuple[int, int]:
        x = self.random.randrange(self.school_left, self.school_right + 1)
        y = self.random.randrange(self.school_bottom, self.school_top + 1)
        return x, y

    def _setup_households(self):
        for i in range(self.num_pairs):
            x, y = self._random_non_school_empty_cell()
            hh = Household(unique_id=f"HH-{i}", model=self, x=x, y=y)
            init_vals = self.ctrl.init.init_household(self)
            hh.econ_barrier = float(init_vals["econ_barrier"])
            hh.attitude     = float(init_vals["attitude"])
            hh.transport    = float(init_vals["transport"])
            hh.chores       = float(init_vals["chores"])
            self.households.append(hh)
            self.grid.place_agent(hh, (x, y))

    def _setup_girls(self):
        for i, hh in enumerate(self.households):
            g = Girl(unique_id=f"G-{i}", model=self, household_id=hh.unique_id, home_xy=(hh.x, hh.y))
            init_vals = self.ctrl.init.init_girl(self, hh)
            g.grade   = int(init_vals["grade"])
            g.age     = int(init_vals["age"])
            g.status  = int(init_vals["status"])
            g.L       = float(init_vals["L"])
            g.esteem  = float(init_vals["esteem"])
            g.enjoy   = float(init_vals["enjoy"])
            g.e_skill = float(init_vals["e_skill"])
            g.marriage = int(init_vals["marriage"])
            self.girls.append(g)
            self.grid.place_agent(g, (hh.x, hh.y))

    def _initial_place_some_in_school(self):
        k = round(self.init_in_school_pct * len(self.girls) / 100.0)
        idx = list(range(len(self.girls)))
        self.random.shuffle(idx)
        for i in idx[:k]:
            g = self.girls[i]
            x, y = self._random_school_cell()
            self.grid.move_agent(g, (x, y))
            g.attending = True
            g.attend_weeks += 1

    # Metrics
    def girls_in_school_now(self) -> int:
        cnt = 0
        for g in self.girls:
            x, y = g.pos
            if g.status == self.ST_ENROLLED and self._is_school_cell(x, y):
                cnt += 1
        return cnt

    def avg_distance_to_school(self) -> float:
        ds = []
        for g in self.girls:
            x, y = g.pos
            ds.append(dist_point_to_rect(x, y, self.school_left, self.school_right, self.school_bottom, self.school_top))
        return (sum(ds) / len(ds)) if ds else 0.0

    def summary(self) -> Dict[str, float]:
        statuses = [g.status for g in self.girls]
        return dict(
            ticks=self.ticks,
            girls_total=len(self.girls),
            enrolled=sum(1 for s in statuses if s == self.ST_ENROLLED),
            dropped=sum(1 for s in statuses if s == self.ST_DROPPED),
            tvet=sum(1 for s in statuses if s == self.ST_TVET),
            employed=sum(1 for s in statuses if s == self.ST_EMP),
            graduated=sum(1 for s in statuses if s == self.ST_GRAD),
            successful_transition=sum(1 for s in statuses if s in (self.ST_TVET, self.ST_EMP, self.ST_GRAD)),
            girls_in_school_now=self.girls_in_school_now(),
            avg_distance_to_school=round(self.avg_distance_to_school(), 3),
        )

    # ---------- Transition options ----------
    def _transition_option_codes(self, g: Girl) -> list[str]:
        if g.status in (self.ST_TVET, self.ST_EMP, self.ST_GRAD):
            return ["remain"]

        if g.status == self.ST_DROPPED:
            return ["re-enrol", "stay-out"]

        stg = stage_of_grade(g.grade)
        if stg == "LP":
            return ["progress", "remain", "dropout"]
        if stg == "UP":
            # REMOVED: move-secondary (progress already handles grade 8 -> 9 via exam threshold)
            return ["progress", "remain", "dropout"]
        if stg == "LS":
            return ["progress", "tvet", "employment", "remain", "dropout"]
        return ["progress", "tvet", "employment", "remain", "dropout"]

    def _code_to_readable(self, code: str, g: Girl) -> str:
        if code == "progress" and g.status == self.ST_ENROLLED and g.grade == 12:
            return "graduate from upper secondary school"
        label = TRANSITION_CODE_TO_READABLE.get(code, code)
        return label

    def transition_options_readable(self, g: Girl) -> list[str]:
        return [self._code_to_readable(c, g) for c in self._transition_option_codes(g)]

    def _readable_to_code_exact(self, readable: str, g: Girl) -> str:
        if not isinstance(readable, str):
            readable = str(readable)
        r = readable.strip().rstrip(".")

        codes = self._transition_option_codes(g)
        mapping = {self._code_to_readable(c, g): c for c in codes}

        out = mapping.get(r, None)
        if out is None:
            msg = (
                f"[TransitionMappingError] readable='{r}' not in mapping keys.\n"
                f"  available readable keys={list(mapping.keys())}\n"
                f"  codes={codes}\n"
                f"  girl(status={g.status}, grade={g.grade}, age={g.age})"
            )
            print(msg)
            if self.strict_transition_mapping:
                raise ValueError(msg)
            return r
        return out

    # ------------------------- go / step -------------------------
    def _progress_success_prob(self, g: Girl) -> float:
        """
        transition rule:
        p = base + (attendance_rate * school_quality01 * attendance_weight)
        - attendance_rate = g.attend_weeks / year_ticks
        - school_quality01 = school_Q / 100
        - base: 0.50 normal, 0.35 exam grades
        """
        EXAM_GRADES = {8, 10}

        base = self.ctrl.transition_base_exam(self) if g.grade in EXAM_GRADES else self.ctrl.transition_base_general(self)
        attendance_rate = 0.0 if self.year_ticks <= 0 else (g.attend_weeks / float(100))
        if attendance_rate < 0.0: attendance_rate = 0.0
        if attendance_rate > 1.0: attendance_rate = 1.0

        quality01 = float(self.school_Q) / 100.0
        if quality01 < 0.0: quality01 = 0.0
        if quality01 > 1.0: quality01 = 1.0

        w = float(self.ctrl.transition_attendance_weight(self))
        p = base + attendance_rate * quality01 * w

        if p < 0.0: p = 0.0
        if p > 1.0: p = 1.0
        return float(p)


    def _school_state_dict(self) -> Dict[str, float]:
        return {
            "quality_of_teaching": float(self.school_Q),
            "fairness": float(self.school_fair),
            "peer_support": float(self.school_peer),
            "safety": float(self.school_Safety),
        }

    def dump_decision_cache(self) -> None:
        print("\n" + self.cache.dump(max_keys_per_type=12))

    def run(self, steps: int) -> None:
        for _ in range(int(steps)):
            self.step()
        if self.debug:
            print("\n[Simulation finished] dumping decision cache...")
            self.dump_decision_cache()

    def step(self):
        if self.decisions != "llm":
            raise RuntimeError("Current code only supports 'llm' decisions. Set decisions='llm'.")

        self.ctrl.pre_step(self)
        school_state = self._school_state_dict()

        # 1) Attendance
        for g in self.girls:
            if g.status == self.ST_ENROLLED:
                hh = self._find_household(g.hid)
                g_dec, _ = self.decider.attendance_girl(g.to_dict(), hh.to_dict(), school_state)
                h_dec, _ = self.decider.attendance_household(hh.to_dict(), g.to_dict(), school_state, girl_answer=bool(g_dec))
                g.attending = bool(g_dec and h_dec)
            else:
                g.attending = False

        # 2) Position & Ticking
        for g in self.girls:
            if g.status == self.ST_ENROLLED:
                if g.attending:
                    if not self._is_school_cell(*g.pos):
                        self.grid.move_agent(g, self._random_school_cell())
                    g.attend_weeks += 1
                else:
                    if self._is_school_cell(*g.pos):
                        self.grid.move_agent(g, (g.home_x, g.home_y))
            else:
                if self._is_school_cell(*g.pos):
                    self.grid.move_agent(g, (g.home_x, g.home_y))

        # 3) Attending/Absence outcome
        for g in self.girls:
            if g.status != self.ST_ENROLLED:
                continue
            if g.attending:
                self.ctrl.update_learning(self, g)
            else:
                self.ctrl.update_absence(self, g)

        # 4) Every four tick: esteem and attitude evaluation
        if (self.ticks > 0) and ((self.ticks % self.period_4ticks) == 0):
            for hh in self.households:
                gg = self._girl_of_household(hh)
                if gg is None or gg.status == self.ST_GRAD:
                    continue
                score, _ = self.decider.eval_household_attitude(hh.to_dict(), gg.to_dict(), school_state)
                if score is not None:
                    hh.attitude = float(score)

            for g in self.girls:
                if g.status == self.ST_GRAD:
                    continue
                hh = self._find_household(g.hid)
                score, _ = self.decider.eval_self_esteem(g.to_dict(), hh.to_dict(), school_state)
                if score is not None:
                    g.esteem = float(score)

        # 5) each year：Transition
        if (self.ticks > 0) and ((self.ticks % self.year_ticks) == 0):
            if self.debug:
                print("\n========================")
                print(f"[YEAR-END TRANSITION] tick={self.ticks}")
                print("========================")

            # Count number of agents offered tvet and employment
            offered_counts: Dict[str, int] = {"tvet": 0, "employment": 0} 
            total_considered = 0

            # count household choice distribution
            executed_choice_counts: Dict[str, int] = {}
            printed = 0

            for g in self.girls:
                if g.status in (self.ST_TVET, self.ST_EMP, self.ST_GRAD):
                    g.attend_weeks = 0
                    g.age += 1
                    continue

                hh = self._find_household(g.hid)
                option_codes = self._transition_option_codes(g)
                options_readable = self.transition_options_readable(g)

                total_considered += 1
                if "tvet" in option_codes:
                    offered_counts["tvet"] += 1
                if "employment" in option_codes:
                    offered_counts["employment"] += 1

                if self.debug and printed < self.debug_transition_limit:
                    print(f"\n[Agent {g.unique_id}] status={g.status} grade={g.grade} age={g.age} stage={stage_of_grade(g.grade)}")
                    print("  option_codes    =", option_codes)
                    print("  options_readable=", options_readable)

                g_choice, _ = self.decider.transition_girl(g.to_dict(), hh.to_dict(), school_state, options_readable)
                if self.debug and printed < self.debug_transition_limit:
                    print("  girl_choice =", g_choice)

                h_choice = None
                if g_choice is not None:
                    h_choice, _ = self.decider.transition_household(
                        hh.to_dict(), g.to_dict(), school_state, options_readable, girl_choice=g_choice
                    )

                if self.debug and printed < self.debug_transition_limit:
                    print("  household_choice =", h_choice)

                if h_choice is not None:
                    # strict readable->code mapping
                    code = self._readable_to_code_exact(h_choice, g)  
                    executed_choice_counts[code] = executed_choice_counts.get(code, 0) + 1
                    self._apply_transition_choice(g, h_choice)

                g.attend_weeks = 0
                g.age += 1
                printed += 1

            if self.debug:
                print("\n[Transition offered counts] (how many agents even SEE tvet/employment options)")
                print("  total_considered =", total_considered)
                print("  offered_counts   =", offered_counts)
                print("\n[Transition executed choice counts] (what actually got applied)")
                print("  executed_choice_counts =", executed_choice_counts)
                print("\n[Cache stats after year-end] =", self.cache.stats())

        self.ticks += 1
        self.datacollector.collect(self)

    # Helper
    def _apply_transition_choice(self, g: Girl, choice: str):
        code = self._readable_to_code_exact(choice, g)
        stg = stage_of_grade(g.grade)

        if g.status == self.ST_DROPPED:
            if code == "re-enrol":
                g.status = self.ST_ENROLLED
                if g.drop_grade > 0:
                    g.grade = g.drop_grade
                return
            if code == "stay-out":
                return
            raise ValueError(f"[Unknown dropped-choice] code={code} from choice={choice}")

        if code == "dropout":
            g.drop_grade = g.grade
            g.status = self.ST_DROPPED
            g.attending = False
            self.grid.move_agent(g, (g.home_x, g.home_y))
            return

        if code == "remain":
            return

        if code == "tvet":
            g.status = self.ST_TVET
            g.attending = False
            self.grid.move_agent(g, (g.home_x, g.home_y))
            return

        if code == "employment":
            p = max(0.0, min(1.0, self.ctrl.job_find_base(self) + 0.5 * (g.e_skill / 100.0)))
            if self.random.random() < p:
                g.status = self.ST_EMP
            else:
                g.drop_grade = g.grade
                g.status = self.ST_DROPPED
            g.attending = False
            self.grid.move_agent(g, (g.home_x, g.home_y))
            return

        if code == "progress":
            # grade 12 -> graduate
            if g.status == self.ST_ENROLLED and g.grade == 12:
                g.status = self.ST_GRAD
                g.attending = False
                self.grid.move_agent(g, (g.home_x, g.home_y))
                return

            # Possibility determines progress success
            p = self._progress_success_prob(g)
            if self.random.random() < p:
                g.grade += 1
            return

        raise ValueError(
            f"[Unknown transition code] code={code} choice(raw)={choice} "
            f"girl(status={g.status}, grade={g.grade}, age={g.age})"
        )

    # Helper
    def _find_household(self, hid: str) -> Household:
        for hh in self.households:
            if hh.unique_id == hid:
                return hh
        raise RuntimeError("Household not found")

    def _girl_of_household(self, hh: Household) -> Optional[Girl]:
        for g in self.girls:
            if g.hid == hh.unique_id:
                return g
        return None
