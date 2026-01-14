# tests/run_simulation.py
from __future__ import annotations
import os, sys, csv, math
from collections import defaultdict, Counter
from typing import Dict, Any, List

# Add the repository root directory to sys.path to enable imports like "from mvp import XXX"
CURR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CURR)
if ROOT not in sys.path:
    sys.path.append(ROOT)

# Load .env file (optional)
try:
    from dotenv import load_dotenv, find_dotenv
    env_file = find_dotenv(usecwd=True)
    if env_file:
        load_dotenv(env_file)
        print(f"[env] loaded .env from: {env_file}")
except Exception:
    pass

# --- Import internal modules ---
from core import model as model_module
from core.utils import stage_of_grade

# --- Counting LLM client (suppresses per-call printing to avoid console spam) ---
from core.llm_client import LLMClient
class CountingSilentLLMClient(LLMClient):
    calls = {
        "attendance_girl": 0,
        "attendance_household": 0,
        "eval_self_esteem": 0,
        "eval_household_attitude": 0,
        "transition_girl": 0,
        "transition_household": 0,
        "unknown": 0,
    }

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("timeout", 30)
        kwargs.setdefault("max_output_tokens", 128)
        super().__init__(*args, **kwargs)

    @staticmethod
    def _infer_decision(user_prompt: str) -> str:
        u = user_prompt
        if "You are a girl in" in u and "do you choose to attend the school for next two weeks?" in u:
            return "attendance_girl"
        if "You are the parent/caregiver" in u and "do you allow your child to attend the school for next two weeks?" in u:
            return "attendance_household"
        if "evaluate your level of self-esteem on a scale of 0-100" in u:
            return "eval_self_esteem"
        if "evaluate your attitude toward your child’s education on a scale of 0-100" in u:
            return "eval_household_attitude"
        if "what will you choose to do for next year" in u and "You are a girl in" in u:
            return "transition_girl"
        if "what will you choose to do for next year" in u and "You are the parent/caregiver" in u:
            return "transition_household"
        return "unknown"

    def chat_json(self, user_prompt: str) -> str:
        dec = self._infer_decision(user_prompt)
        self.calls[dec] = self.calls.get(dec, 0) + 1
        return super().chat_json(user_prompt)

model_module.LLMClient = CountingSilentLLMClient

# --- Simulation Controller ---
from core.sim_controller import (
    SimulationController, ParamSchedules, LearningPolicy, TransitionPolicy,
    sched_linear_add, sched_constant
)

def make_demo_controller() -> SimulationController:
    schedules = ParamSchedules(
        # school={
        #     "quality_of_teaching": lambda cur, t, obj, model: cur + 0.5 + model.random.random() * 0.2,
        #     "fairness":            lambda cur, t, obj, model: cur + 0.5 + model.random.random() * 0.2,
        #     "safety":              lambda cur, t, obj, model: cur + 0.5 + model.random.random() * 0.2,
        #     "peer_support":        lambda cur, t, obj, model: cur + 0.55 + model.random.random() * 0.2,
        # },
        household={
            "econ_barrier": lambda cur, t, obj, model: cur + 0.5,
            # "transport":    lambda cur, t, obj, model: cur + 0.5,
            # "chores":       lambda cur, t, obj, model: cur + 0.5,
        },
        # girl={
        #     "enjoyment": lambda cur, t, obj, model: cur + model.random.random() * 0.5,
        #     "e_skill":   lambda cur, t, obj, model: cur + 0.4,
        # },
    )

    transitions = TransitionPolicy(
        base_general=0.66,
        base_exam=0.45,
        attendance_weight=1.0,
        job_find_base=(lambda t, m: 0.15 if t < 60 else 0.18),
    )

    learning = LearningPolicy(
        gain_per_qual = 0.00002,
        absent_decay = 0.01
    )
    return SimulationController(learning=learning, transitions=transitions, schedules=schedules)

# --- Statistics / logger: injected into the decider to record attendance and transitions ---
class DecisionTap:
    """A thin wrapper around the DecisionEngine interface to record key decisions without modifying model code."""
    def __init__(self, model):
        self.model = model
        self.decider = model.decider

        # Per-tick attendance intention statistics (girl & household & both)
        self.attendance_counts = defaultdict(lambda: {"girl_yes": 0, "hh_yes": 0, "both_yes": 0})

        # Household transition choice counts per “year tick” (aggregated by code)
        self.transition_counts = defaultdict(Counter)
        # Row-level transition records (tick + text + partial context)
        self.transition_rows: List[Dict[str, Any]] = []

        # Preserve original methods
        self._orig_att_girl = self.decider.attendance_girl
        self._orig_att_hh   = self.decider.attendance_household
        self._orig_tr_girl  = self.decider.transition_girl
        self._orig_tr_hh    = self.decider.transition_household

        # Replace with wrapped versions
        self.decider.attendance_girl = self._wrap_attendance_girl
        self.decider.attendance_household = self._wrap_attendance_household
        self.decider.transition_girl = self._wrap_transition_girl
        self.decider.transition_household = self._wrap_transition_household

        # Temporary cache of the current girl's choice text, used to pair with household choices
        self._last_girl_choice_text_by_tick: Dict[int, str] = {}

    @staticmethod
    def _norm_choice_to_code(text: str) -> str:
        r = (text or "").strip().rstrip(".")
        # Normalize to internal codes for robust aggregation
        if r in ("progress to next grade",): return "progress"
        if r in ("remain in the same grade",): return "remain"
        if r in ("drop out of school",): return "dropout"
        if r.startswith("enter TVET"): return "tvet"
        if r == "go find a job": return "employment"
        if r == "re-enrol in school": return "re-enrol"
        if r == "stay out of school": return "stay-out"
        return r or "unknown"

    # ---- wrappers ----
    def _wrap_attendance_girl(self, girl: Dict[str, Any], hh: Dict[str, Any], school: Dict[str, Any]):
        ans, expl = self._orig_att_girl(girl, hh, school)
        t = self.model.ticks
        if ans is True: 
            self.attendance_counts[t]["girl_yes"] += 1
        return ans, expl

    def _wrap_attendance_household(self, hh: Dict[str, Any], girl: Dict[str, Any], school: Dict[str, Any], girl_answer: bool):
        ans, expl = self._orig_att_hh(hh, girl, school, girl_answer)
        t = self.model.ticks
        if ans is True:
            self.attendance_counts[t]["hh_yes"] += 1
            if girl_answer:
                self.attendance_counts[t]["both_yes"] += 1
        return ans, expl

    def _wrap_transition_girl(self, girl: Dict[str, Any], hh: Dict[str, Any], school: Dict[str, Any], options_readable: List[str]):
        choice, expl = self._orig_tr_girl(girl, hh, school, options_readable)
        t = self.model.ticks
        self._last_girl_choice_text_by_tick[t] = (choice or "").strip().rstrip(".")
        return choice, expl

    def _wrap_transition_household(self, hh: Dict[str, Any], girl: Dict[str, Any], school: Dict[str, Any],
                                   options_readable: List[str], girl_choice: str):
        choice, expl = self._orig_tr_hh(hh, girl, school, options_readable, girl_choice)
        t = self.model.ticks
        code = self._norm_choice_to_code(choice or "")
        self.transition_counts[t][code] += 1

        self.transition_rows.append({
            "tick": t,
            "girl_choice": self._last_girl_choice_text_by_tick.get(t, ""),
            "household_choice": (choice or "").strip().rstrip("."),
            "girl_age": int(girl.get("age", -1)),
            "girl_grade": int(girl.get("grade", -1)),
            "girl_status": int(girl.get("status", -1)),
            "girl_L": float(girl.get("L", float("nan"))),
            "girl_esteem": float(girl.get("esteem", float("nan"))),
            "girl_enjoyment": float(girl.get("enjoyment", float("nan"))),
            "hh_econ": float(hh.get("econ_barrier", float("nan"))),
            "hh_transport": float(hh.get("transport", float("nan"))),
            "hh_chores": float(hh.get("chores", float("nan"))),
        })
        return choice, expl

# Stats tools
def _mean(xs):
    xs = [float(x) for x in xs if x is not None and not math.isnan(float(x))]
    return (sum(xs) / len(xs)) if xs else 0.0

def attendance_by_stage(model) -> Dict[str, int]:
    by_stage = {"LP":0, "UP":0, "LS":0, "US":0}
    totals   = {"LP":0, "UP":0, "LS":0, "US":0}
    for g in model.girls:
        stg = stage_of_grade(int(g.grade))
        totals[stg] += 1
        if getattr(g, "attending", False) and g.status == model.ST_ENROLLED:
            by_stage[stg] += 1
    return {"attending": by_stage, "totals": totals}

def safe_mkdir(p: str):
    os.makedirs(p, exist_ok=True)

def main():
    # Parameters
    NUM_PAIRS = int(os.getenv("NUM_PAIRS", "1000"))
    TICKS     = int(os.getenv("TICKS", str(24*6)))
    YEAR_TICKS= int(os.getenv("YEAR_TICKS", "24"))
    PERIOD_4  = int(os.getenv("PERIOD_4TICKS", "4"))
    SEED      = int(os.getenv("SEED", "3"))
    LLM_MODEL = os.getenv("LLM_MODEL", "gemini/gemini-2.5-flash-lite")
    COUNTRY   = os.getenv("COUNTRY", "Ethiopia")

    # Build sim controller
    controller = make_demo_controller()

    # Build school model
    print("[info] creating SchoolModel ...")
    try:
        m = model_module.SchoolModel(
            num_pairs=NUM_PAIRS,
            init_in_school_pct=40,
            seed=SEED,
            year_ticks=YEAR_TICKS,
            period_4ticks=PERIOD_4,
            decisions="llm",
            llm_model=LLM_MODEL,
            country=COUNTRY,
            controller=controller,  
        )
    except TypeError:
        print("[warn] SchoolModel.__init__ does not accept a controller parameter: running in no-controller mode "
        "(dynamic parameter changes and dynamic threshold policies will not take effect).")
        m = model_module.SchoolModel(
            num_pairs=NUM_PAIRS,
            init_in_school_pct=40,
            seed=SEED,
            year_ticks=YEAR_TICKS,
            period_4ticks=PERIOD_4,
            decisions="llm",
            llm_model=LLM_MODEL,
            country=COUNTRY,
        )

    tap = DecisionTap(m)

    # ------- Output directories -------
    out_dir = os.path.join(ROOT, "outputs")
    safe_mkdir(out_dir)
    f_ts   = os.path.join(out_dir, "sim_timeseries.csv")
    f_tr_c = os.path.join(out_dir, "transition_counts.csv")
    f_tr_r = os.path.join(out_dir, "transition_rows.csv")
    f_att  = os.path.join(out_dir, "attendance_by_stage.csv")

    # ------- Main loop -------
    print(f"[info] running simulation: pairs={NUM_PAIRS}, ticks={TICKS} ...")
    timeseries_rows: List[Dict[str, Any]] = []
    attstage_rows: List[Dict[str, Any]] = []

    for _ in range(TICKS):
        # Run one step
        print(f"Running tick {m.ticks+1}/{TICKS} ...")
        m.step()

        # --- Aggregate overall statistics for the current tick ---
        s = m.summary()  # ticks, enrolled, dropped, tvet, employed, girls_in_school_now, avg_distance_to_school, etc.

        # Attendance intention counts (girl / household / both).
        # Note: the tap records counts during the LLM decision phase of this tick.
        att_cnt = tap.attendance_counts.get(
            m.ticks - 1, {"girl_yes": 0, "hh_yes": 0, "both_yes": 0}
        )  # Decisions occur inside step(); use the previous tick index.

        # Averages (L / esteem / household attitude)
        avg_L        = _mean(g.L for g in m.girls)
        avg_esteem   = _mean(g.esteem for g in m.girls)
        avg_enjoy    = _mean(g.enjoy for g in m.girls)
        avg_e_skill  = _mean(g.e_skill for g in m.girls)
        avg_grade    = _mean(g.grade for g in m.girls)
        avg_age      = _mean(g.age for g in m.girls)
        avg_status   = _mean(g.status for g in m.girls)
        avg_marriage = _mean(g.marriage for g in m.girls)  # Mean of 0/1 = marriage rate
        avg_attend_weeks = _mean(getattr(g, "attend_weeks", 0) for g in m.girls)

        # Attending rate among enrolled girls (more intuitive)
        enrolled_cnt = sum(1 for g in m.girls if g.status == m.ST_ENROLLED)
        attending_cnt = sum(
            1 for g in m.girls
            if g.status == m.ST_ENROLLED and getattr(g, "attending", False)
        )
        attending_rate = (attending_cnt / enrolled_cnt) if enrolled_cnt else 0.0

        # ---------- Averages: households ----------
        avg_econ_barrier = _mean(hh.econ_barrier for hh in m.households)
        avg_attitude     = _mean(hh.attitude for hh in m.households)
        avg_transport    = _mean(hh.transport for hh in m.households)
        avg_chores       = _mean(hh.chores for hh in m.households)

        timeseries_rows.append({
            "tick": s["ticks"],
            "girls_total": s["girls_total"],
            "enrolled": s["enrolled"],
            "dropped": s["dropped"],
            "tvet": s["tvet"],
            "employed": s["employed"],
            "in_school_now": s["girls_in_school_now"],
            "attend_girl_yes": att_cnt["girl_yes"],
            "attend_hh_yes": att_cnt["hh_yes"],
            "attend_both_yes": att_cnt["both_yes"],

            # ----- girl averages -----
            "avg_L": round(avg_L, 3),
            "avg_esteem": round(avg_esteem, 3),
            "avg_enjoy": round(avg_enjoy, 3),
            "avg_e_skill": round(avg_e_skill, 3),
            "avg_attend_weeks": round(avg_attend_weeks, 3),
            "attending_rate_enrolled": round(attending_rate, 4),
            "avg_grade": round(avg_grade, 3),
            "avg_age": round(avg_age, 3),
            "avg_status": round(avg_status, 3),
            "married_rate": round(avg_marriage, 4),

            # ----- household averages -----
            "avg_household_attitude": round(avg_attitude, 3),
            "avg_household_econ_barrier": round(avg_econ_barrier, 3),
            "avg_household_transport": round(avg_transport, 3),
            "avg_household_chores": round(avg_chores, 3),

            # ----- school -----
            "school_Q": round(float(m.school_Q), 3),
            "school_fair": round(float(m.school_fair), 3),
            "school_Safety": round(float(m.school_Safety), 3),
            "school_peer": round(float(m.school_peer), 3),
            "avg_distance_to_school": s["avg_distance_to_school"],
        })


        # stage attendance
        st = attendance_by_stage(m)
        row_att = {
            "tick": s["ticks"],
            "LP_attend": st["attending"]["LP"], "LP_total": st["totals"]["LP"],
            "UP_attend": st["attending"]["UP"], "UP_total": st["totals"]["UP"],
            "LS_attend": st["attending"]["LS"], "LS_total": st["totals"]["LS"],
            "US_attend": st["attending"]["US"], "US_total": st["totals"]["US"],
        }
        attstage_rows.append(row_att)

    # ------- CSV：timeseries -------
    with open(f_ts, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(timeseries_rows[0].keys()))
        writer.writeheader()
        writer.writerows(timeseries_rows)

    # ------- CSV: transition counts -------
    # Collect all transition codes that ever appeared, to ensure a stable column order
    all_codes = set()
    for t, counter in tap.transition_counts.items():
        all_codes.update(counter.keys())
    # Recommended column order (any missing codes will be appended automatically)
    preferred = ["progress", "remain", "dropout", "tvet", "employment", "re-enrol", "stay-out"]
    ordered_codes = preferred + [c for c in sorted(all_codes) if c not in preferred]
    # Write file
    if tap.transition_counts:
        with open(f_tr_c, "w", newline="", encoding="utf-8") as f:
            cols = ["tick"] + ordered_codes
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            for t in sorted(tap.transition_counts.keys()):
                row = {"tick": t}
                for c in ordered_codes:
                    row[c] = tap.transition_counts[t].get(c, 0)
                writer.writerow(row)

    # ------- CSV：transition rows -------
    if tap.transition_rows:
        with open(f_tr_r, "w", newline="", encoding="utf-8") as f:
            cols = list(tap.transition_rows[0].keys())
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            writer.writerows(tap.transition_rows)

    # ------- CSV：attendance_by_stage -------
    with open(f_att, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(attstage_rows[0].keys()))
        writer.writeheader()
        writer.writerows(attstage_rows)

    # ------- LLM call count summary -------
    print("\n[LLM call counts]")
    for k, v in CountingSilentLLMClient.calls.items():
        print(f"  {k:>24}: {v}")

    import json

    cache_json = os.path.join(out_dir, "decision_cache.json")
    with open(cache_json, "w", encoding="utf-8") as f:
        json.dump(m.cache.to_jsonable(), f, ensure_ascii=False, indent=2)

    cache_txt = os.path.join(out_dir, "decision_cache_dump.txt")
    with open(cache_txt, "w", encoding="utf-8") as f:
        f.write(m.cache.dump(max_keys_per_type=999999)) 

    print("\n[cache outputs]")
    print("  decision_cache.json ->", cache_json)
    print("  decision_cache_dump.txt ->", cache_txt)


    print("\n[outputs]")
    print(f"  timeseries         -> {f_ts}")
    print(f"  transition counts  -> {f_tr_c if tap.transition_counts else '(no yearly transitions logged)'}")
    print(f"  transition rows    -> {f_tr_r if tap.transition_rows else '(no yearly transitions logged)'}")
    print(f"  attendance by stage-> {f_att}")
    print("\nDone.")

if __name__ == "__main__":
    main()
