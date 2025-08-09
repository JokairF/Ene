from __future__ import annotations
from typing import List, Dict, Any, Iterable
from llama_cpp import Llama

def _as_pair(m) -> Dict[str, str]:
    if isinstance(m, dict):
        return {"role": m.get("role",""), "content": m.get("content","")}
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
        history: List[Dict[str,str] | Any],
        user_msg: str,
        max_tokens: int = 256,
        temperature: float = 0.6,
        stream: bool = False,
        **kw,
    ):
        messages: List[Dict[str,str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.extend([_as_pair(m) for m in history])
        messages.append({"role": "user", "content": user_msg})

        args = dict(messages=messages, max_tokens=max_tokens, temperature=temperature)
        if stream:
            return self.llm.create_chat_completion(stream=True, **args)
        return self.llm.create_chat_completion(**args)
