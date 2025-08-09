# app/llm.py
from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Optional, Union

from llama_cpp import Llama


def _get_role(m) -> str:
    return (m.get("role") if isinstance(m, dict) else getattr(m, "role", "")) or ""


def _get_content(m) -> str:
    return (m.get("content") if isinstance(m, dict) else getattr(m, "content", "")) or ""


def _normalize_history(history: Optional[Iterable[Any]]) -> List[Dict[str, str]]:
    conv: List[Dict[str, str]] = []
    if not history:
        return conv
    for m in history:
        role = _get_role(m)
        content = _get_content(m)
        if role and content:
            conv.append({"role": role, "content": content})
    return conv


class LocalLLM:
    """
    Mince wrapper autour llama.cpp qui:
      - normalise l'historique (dicts ou objets)
      - retourne toujours un format OpenAI-like (dict)
      - supporte le streaming (yield events)
      - expose des paramètres via variables d'env
    """

    def __init__(self, model_path: str, **overrides):
        # Valeurs sûres par défaut pour CPU (tu peux ajuster)
        n_ctx = int(os.getenv("LLM_N_CTX", overrides.get("n_ctx", 4096)))
        n_batch = int(os.getenv("LLM_N_BATCH", overrides.get("n_batch", 256)))
        n_threads = int(os.getenv("LLM_N_THREADS", overrides.get("n_threads", os.cpu_count() or 8)))
        n_gpu_layers = int(os.getenv("LLM_N_GPU_LAYERS", overrides.get("n_gpu_layers", -1)))  # -1 = CPU
        verbose = bool(int(os.getenv("LLM_VERBOSE", int(overrides.get("verbose", False)))))

        # NB: un n_ctx à 32768 sur CPU Q4_K_M peut exploser la RAM/latence.
        # 4096 / 8192 sont plus réalistes sur CPU.
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_batch=n_batch,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            verbose=verbose,
        )

    def _build_messages(
        self,
        system: Optional[str],
        history: Optional[Iterable[Any]],
        user_msg: str,
    ) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.extend(_normalize_history(history))
        messages.append({"role": "user", "content": user_msg})
        return messages

    def chat(
        self,
        system: Optional[str],
        history: Optional[Iterable[Any]],
        user_msg: str,
        *,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: Optional[float] = None,
        stream: bool = False,
        stop: Optional[Iterable[str]] = None,
        **_kwargs: Any,  # future-proof (ignore params non supportés par llama.cpp)
    ) -> Union[Dict[str, Any], Iterable[Dict[str, Any]]]:
        """
        Appelle llama.cpp en mode chat.

        Retour:
          - non-stream: dict OpenAI-like (incluant .choices[].message.content)
          - stream: itérable d'events dict (choices[0].delta.content)
        """
        messages = self._build_messages(system, history, user_msg)

        # Prépare les kwargs acceptés par llama.cpp
        call_kwargs: Dict[str, Any] = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if top_p is not None:
            call_kwargs["top_p"] = top_p
        if stop:
            call_kwargs["stop"] = list(stop)

        if stream:
            # llama-cpp renvoie déjà des events OpenAI-like; on les propage tels quels.
            return self.llm.create_chat_completion(stream=True, **call_kwargs)

        # Non-stream: on retourne la réponse complète (dict)
        res = self.llm.create_chat_completion(stream=False, **call_kwargs)
        # On s'assure que le format contient bien choices[].message.content
        # (llama-cpp le fait déjà, mais on garde ça robuste)
        if isinstance(res, dict) and "choices" in res:
            return res
        # fallback: on uniformise
        text = ""
        try:
            text = str(res)
        except Exception:
            text = ""
        return {"choices": [{"message": {"role": "assistant", "content": text}}]}
