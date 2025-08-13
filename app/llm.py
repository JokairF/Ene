from __future__ import annotations
from typing import List, Dict, Any, Iterable
from llama_cpp import Llama
import json

def _as_pair(m) -> Dict[str, str]:
    if isinstance(m, dict):
        return {"role": m.get("role", ""), "content": m.get("content", "")}
    return {"role": getattr(m, "role", ""), "content": getattr(m, "content", "")}

class LocalLLM:
    def __init__(self, model_path: str):
        self.llm = Llama(
            model_path=model_path,
            # Valeurs prudentes CPU, ajuste si besoin
            n_ctx=4096,
            n_batch=512,
            n_threads=8,
            n_gpu_layers=0,   # CPU
            chat_format="mistral-instruct",
            verbose=False,
        )

    def chat(
        self,
        *,
        system: str | None,
        history: List[Dict[str, str] | Any],
        user_msg: str,
        max_tokens: int = 256,
        temperature: float = 0.6,
        stream: bool = False,
        **kw,
    ):
        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.extend([_as_pair(m) for m in history])
        messages.append({"role": "user", "content": user_msg})

        args = dict(messages=messages, max_tokens=max_tokens, temperature=temperature)
        if stream:
            return self.llm.create_chat_completion(stream=True, **args)
        return self.llm.create_chat_completion(**args)

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 350,
        temperature: float = 0.9,
        top_p: float = 0.95,
        repetition_penalty: float = 1.1,
        history: List[Dict[str, str]] | None = None,
    ) -> dict | str:
        """
        Génère une réponse et tente de la parser en JSON.
        Retourne un dict si parse OK, sinon la chaîne brute.
        """
        if history is None:
            history = []

        raw = self.chat(
            system=system_prompt,
            history=history,
            user_msg=user_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            response_format={"type":"json_object"}
        )

        # Selon llama_cpp, le retour est un dict avec content
        if isinstance(raw, dict):
            try:
                text = raw["choices"][0]["message"]["content"]
            except Exception:
                text = str(raw)
        else:
            text = str(raw)

        # Tentative de parse JSON stricte
        try:
            return json.loads(text)
        except Exception:
            # Tentative de nettoyage minimal (en cas de texte parasite avant/après)
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(text[start:end+1])
                except Exception:
                    pass
            return text
