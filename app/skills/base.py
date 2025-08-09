from typing import Dict, Any, List

class Skill:
    name: str
    description: str
    parameters: Dict[str, Any]

    def run(self, **kwargs):
        raise NotImplementedError

class SkillRegistry:
    def __init__(self):
        self.skills: Dict[str, Skill] = {}

    def register(self, skill: Skill):
        self.skills[skill.name] = skill

    def execute(self, name: str, **kwargs):
        if name in self.skills:
            return self.skills[name].run(**kwargs)
        raise ValueError(f"Skill {name} not found")
