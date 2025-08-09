from typing import List, Dict, Any
from pydantic import BaseModel

class Action(BaseModel):
    type: str
    target: str
    parameters: Dict[str, Any]

class EneResponse(BaseModel):
    speech: str
    intent: str
    emotion: str
    actions: List[Action]
    memory_write: List[str]
    ask_confirmation: bool

class DecisionRequest(BaseModel):
    context: str
    recent_events: List[str] = []
    objectives: List[str] = []
