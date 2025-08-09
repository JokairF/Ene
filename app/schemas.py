from pydantic import BaseModel, Field, conint, confloat
from typing import List, Literal, Optional, Dict, Any

# --- Rôles (inchangé) ---
Role = Literal["user", "assistant", "system"]

class ChatMessage(BaseModel):
    role: Role
    content: str = Field(..., description="Texte du message")

# --- Personnalité / Style ---
Personality = Literal["ene", "takane", "neutral"]
ReplyStyle = Literal["concise", "balanced", "immersive"]

class GenerationControls(BaseModel):
    temperature: confloat(ge=0.0, le=2.0) = 0.7
    top_p: confloat(ge=0.0, le=1.0) = 1.0
    presence_penalty: confloat(ge=-2.0, le=2.0) = 0.0
    frequency_penalty: confloat(ge=-2.0, le=2.0) = 0.0
    max_tokens: conint(gt=0, le=4096) = 256

class StyleControls(BaseModel):
    personality: Personality = Field(
        default="ene",
        description="Personnalité à injecter dans le system prompt."
    )
    reply_style: ReplyStyle = Field(
        default="balanced",
        description="Tendance générale de la réponse."
    )
    min_words: conint(ge=0, le=2000) = Field(
        default=0,
        description="Nombre minimum de mots souhaité (anti-short). 0 = désactivé."
    )

# --- Requête principale (compatible avec ton ancienne) ---
class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Identifiant de la conversation")
    message: str = Field(..., description="Message utilisateur")

    # Ancien champ conservé (tu peux toujours surcharger côté client)
    system: Optional[str] = Field(
        default="You are a concise assistant.",
        description="Complément de règles système spécifique à la requête."
    )

    # Nouveau : contrôles (facultatifs). Si tu ne les envoies pas, les defaults s'appliquent.
    gen: GenerationControls = GenerationControls()
    style: StyleControls = StyleControls()

    # Option(s) futures (placeholder) : passage de métadonnées front -> back
    metadata: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    session_id: str
    reply: str
    usage_tokens: Optional[int] = None
    history: List[ChatMessage]

# --- (Optionnel) Typage pratique pour le streaming SSE ---
class StreamChunk(BaseModel):
    event: Literal["token", "done", "error"]
    data: str
