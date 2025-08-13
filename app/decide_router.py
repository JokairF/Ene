from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, model_validator, field_validator
from typing import List, Dict, Any
import json

from app.llm import LocalLLM
from app.persona.system_json_struct import build_system_prompt, INTENTS, EMOTIONS, fewshot_history
from app import config

# Instance globale (idéalement à partager depuis config.py)
llm_instance = LocalLLM(model_path=config.MODEL_PATH)

router = APIRouter(prefix="/decide", tags=["Ene Decide"])

class ActionModel(BaseModel):
    type: str
    target: str
    parameters: Dict[str, Any] = Field(default_factory=dict)

class DecideRequest(BaseModel):
    message: str
    context: str = ""
    freedom: str = "l1"  # l1 / l2 / l3

class DecideResponse(BaseModel):
    speech: str = ""
    intent: str = "informer"       # défaut FR valide
    emotion: str = "curiosité"     # défaut FR valide
    actions: List[ActionModel] = Field(default_factory=list)
    memory_write: List[str] = Field(default_factory=list)
    ask_confirmation: bool = False

    @field_validator("intent")
    def intent_in_allowed(cls, v):
        if v not in INTENTS:
            raise ValueError(f"intent '{v}' not in allowed set: {INTENTS}")
        return v

    @field_validator("emotion")
    def emotion_in_allowed(cls, v):
        if v not in EMOTIONS:
            raise ValueError(f"emotion '{v}' not in allowed set: {EMOTIONS}")
        return v
    
# petit mapping de secours si le modèle émet encore de l'anglais
INTENT_MAP_EN2FR = {
    "give_info": "informer",
    "ask_info": "commenter",   # à ajuster si tu ajoutes 'demander_information' dans tes INTENTS
    "greeting": "commenter",
    "joke": "taquiner",
    "tease": "taquiner",
    "comfort": "rassurer",
    "farewell": "commenter",
    "skill_action": "suggérer_action"
}
EMOTION_MAP_EN2FR = {
    "neutral": "curiosité",
    "happy": "joie",
    "playful": "joie",
    "curious": "curiosité",
    "annoyed": "colère",
    "embarrassed": "tristesse",
    "sympathetic": "compassion"
}

@router.post("", response_model=DecideResponse)
def decide(req: DecideRequest):
    system_prompt = build_system_prompt(req.freedom)
    user_prompt = f"Contexte: {req.context}\nUtilisateur: {req.message}\nRéponds en JSON strict."
    # 👉 Injecte ici les few-shots dans l'historique
    history = fewshot_history()

    raw = llm_instance.generate_json(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        history=history,           # <= important
        max_tokens=350,
        temperature=1.0
    )

    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(data, dict):
            raise ValueError("Le modèle n'a pas renvoyé un JSON objet.")

        # Normalisation douce si le modèle répond avec anglais
        if "intent" in data and data["intent"] not in INTENTS:
            data["intent"] = INTENT_MAP_EN2FR.get(data["intent"], "informer")
        if "emotion" in data and data["emotion"] not in EMOTIONS:
            data["emotion"] = EMOTION_MAP_EN2FR.get(data["emotion"], "curiosité")

        # Garantir les clés minimales
        data.setdefault("speech", "")
        data.setdefault("intent", "informer")
        data.setdefault("emotion", "curiosité")
        data.setdefault("actions", [])
        data.setdefault("memory_write", [])
        data.setdefault("ask_confirmation", False)

        return DecideResponse(**data)

    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Réponse non valide : {e}")