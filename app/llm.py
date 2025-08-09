# llm.py
from llama_cpp import Llama

class LocalLLM:
    def __init__(self, model_path: str):
        self.llm = Llama(
            model_path=model_path,
            n_ctx=32768,
            n_batch=1024,
            n_threads=8,
            n_gpu_layers=0,     # CPU pour l’instant
            verbose=False,
        )

    def chat(
        self,
        system: str | None,
        history: list,           # liste d'objets avec .role et .content
        user_msg: str,
        *,
        max_tokens: int = 256,
        temperature: float = 0.7,
        stream: bool = False,
    ):
        # Construit les messages pour l’API chat
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        for m in history:
            messages.append({"role": m.role, "content": m.content})
        messages.append({"role": "user", "content": user_msg})

        if stream:
            return self.llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
            )

        out = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return out["choices"][0]["message"]["content"]
