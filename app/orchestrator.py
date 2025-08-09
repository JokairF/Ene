from .llm import LocalLLM
from .prompts import ENE_SYSTEM_PROMPT, FEW_SHOTS
from .models import DecisionRequest, EneResponse
from .memory.store import MemoryStore
import json
import time

class Orchestrator:
    def __init__(self, llm: LocalLLM, memory: MemoryStore | None = None):
        self.llm = llm
        self.memory = memory

    def _build_user_input(self, req: DecisionRequest, retrieved_context: str) -> str:
        base = f"{req.context}\nRecent events: {req.recent_events}\nObjectives: {req.objectives}"
        if retrieved_context:
            return base + f"\n[Retrieved context]\n{retrieved_context}"
        return base

    def _write_memories(self, memories: list[str]):
        if not (self.memory and memories):
            return
        # Politique simple: tout va en episodic avec tags=auto; tu pourras affiner (importance/récence)
        items = [{"text": m, "metadata": {"source": "ene_runtime", "type": "autolog", "tags": ["conversation"]}} for m in memories]
        self.memory.bulk_add("episodic", items)

    def decide(self, req: DecisionRequest) -> EneResponse:
        # 1) Récupération contexte
        retrieved = ""
        if self.memory:
            # On requête avec le contexte utilisateur pur (meilleur signal)
            retrieved = self.memory.get_context(query=req.context, top_k_each=2)

        # 2) Prompt final
        user_input = self._build_user_input(req, retrieved)

        # 3) Appel LLM
        raw_output = self.llm.generate(ENE_SYSTEM_PROMPT, FEW_SHOTS, user_input)

        # 4) Parsing & fallback
        try:
            data = json.loads(raw_output)
        except json.JSONDecodeError:
            # Fallback robuste si JSON cassé
            data = {
                "speech": "Oops, glitch mémoire~ je réessaie si tu veux ?",
                "intent": "annoncer_problème",
                "emotion": "prudence",
                "actions": [],
                "memory_write": [],
                "ask_confirmation": True
            }

        # 5) Écriture mémoire (episodic simple)
        try:
            self._write_memories(data.get("memory_write", []))
        except Exception:
            pass  # On ne fait pas planter la décision pour la mémoire

        return EneResponse(**data)
