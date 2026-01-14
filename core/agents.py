# agents.py
# Define all agents' attributes
from __future__ import annotations
from mesa import Agent

class Household(Agent):
    def __init__(self, unique_id, model, x: int, y: int):
        super().__init__(unique_id, model)
        self.x = x
        self.y = y
        self.econ_barrier = 0.0
        self.attitude     = 0.0
        self.transport    = 0.0
        self.chores       = 0.0

    def to_dict(self) -> dict:
        return {
            "econ_barrier": float(self.econ_barrier),
            "attitude": float(self.attitude),
            "transport": float(self.transport),
            "chores": float(self.chores),
        }

class Girl(Agent):
    def __init__(self, unique_id, model, household_id: int, home_xy: tuple[int, int]):
        super().__init__(unique_id, model)
        self.hid = household_id
        self.home_x, self.home_y = home_xy
        self.attending = False
        self.attend_weeks = 0
        self.drop_grade = 0
        self.L = 0.0
        self.esteem = 0.0
        self.enjoy = 0.0
        self.e_skill = 0.0
        self.grade = 1
        self.age = 6
        self.marriage = 0  # 0/1
        self.status = 1    # 0=dropped 1=enrolled 2=tvet 3=employed 4=graduated

    def to_dict(self) -> dict:
        return {
            "L": float(self.L),
            "esteem": float(self.esteem),
            "enjoyment": float(self.enjoy), 
            "employment_skills": float(self.e_skill), 
            "grade": int(self.grade),
            "age": int(self.age),
            "marriage": int(self.marriage),
            "status": int(self.status),
        }

