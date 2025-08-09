from pydantic import BaseModel, Field
from typing import List, Literal, Optional

Role = Literal["user", "assistant", "system"]

class ChatMessage(BaseModel):
    role: Role
    content: str

class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Identifiant de la conversation")
    message: str = Field(..., description="Message utilisateur")
    system: Optional[str] = "You are a concise assistant."
    temperature: float = 0.7
    max_tokens: int = 256

class ChatResponse(BaseModel):
    session_id: str
    reply: str
    usage_tokens: Optional[int] = None
    history: List[ChatMessage]
