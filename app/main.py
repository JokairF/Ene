from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from typing import Optional, Dict, Any, Iterable
import os
import logging

from app.llm import LocalLLM
from app.chat import ChatSessions
from app.schemas import ChatRequest, ChatResponse, ChatMessage

# ----- Logger -----
logger = logging.getLogger("uvicorn.error")

app = FastAPI(title="Ene API")

# --- CORS (origines configurables via ENV, fallback pour dev local) ---
DEFAULT_ORIGINS = [
    "http://localhost:5173", "http://127.0.0.1:5173",
    "http://localhost:3000", "http://127.0.0.1:3000"
]
ENV_ORIGINS = os.getenv("CORS_ALLOW_ORIGINS", "")
ALLOW_ORIGINS = [o.strip() for o in ENV_ORIGINS.split(",") if o.strip()] or DEFAULT_ORIGINS

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- LLM (CPU pour l’instant) ---
llm = LocalLLM(model_path="E:/models/Mistral-7B-Instruct-v0.2-GGUF/mistral-7b-instruct-v0.2.Q4_K_M.gguf")

# --- Sessions (mémoire court terme) ---
sessions = ChatSessions(max_turns=30)

# ------------------ Persona / System prompt ------------------

DEFAULT_PERSONALITY = "ene"  # 'takane' | 'neutral'
DEFAULT_LANG = "fr"

def persona_prompt(personality: Optional[str], lang: str = DEFAULT_LANG) -> str:
    p = (personality or DEFAULT_PERSONALITY).lower()
    if p in ("ene", "ene/takane", "takane"):
        return (
            "Rôle & Voix (FR uniquement) : Tu es *Ene* (alias Takane Enomoto). "
            "Personnalité : espiègle, énergique, taquine mais compatissante. "
            "Tu appelles l'utilisateur « Maître » sur un ton joueur. "
            "Style : familier, punchy, petites onomatopées (« Heh~ », « Ahaha~ »), "
            "analogies informatiques quand c’est pertinent, parfois un clin d’œil méta. "
            "Évite les explications techniques hors rôle ; si nécessaire, propose de basculer en mode technique. "
            "Langue : réponds *exclusivement en français*, sans mélanger d’anglais."
        )
    return (
        "Tu es une assistante conversationnelle chaleureuse et utile. "
        "Langue : réponds exclusivement en français."
    )

def style_directives(reply_style: Optional[str]) -> str:
    rs = (reply_style or "balanced").lower()
    if rs == "immersive":
        return ("Réponses un peu développées, concrètes, avec une touche d'humour ; "
                "reste centrée sur la demande.")
    if rs == "concise":
        return "Réponses courtes et percutantes, sans digression."
    return "Réponses claires, utiles, avec une touche de personnalité."

def build_system_prompt(user_system: Optional[str], personality: Optional[str], reply_style: Optional[str]) -> str:
    base = persona_prompt(personality)
    style = style_directives(reply_style)
    parts = [
        base,
        f"Directives de style : {style}",
        "IMPORTANT : Réponds en français uniquement. Reste strictement en personnage."
    ]
    if user_system and user_system.strip():
        parts.append(f"Règles supplémentaires:\n{user_system.strip()}")
    return "\n\n".join(parts)

# ------------------ Compactage de contexte ------------------

COMPACT_AFTER = 24       # seuil d’activation du résumé
COMPACT_KEEP_LAST = 10   # N derniers messages conservés bruts

def compact_history_if_needed(session_id: str) -> None:
    hist = sessions.history(session_id)
    if len(hist) <= COMPACT_AFTER:
        return

    head = hist[:-COMPACT_KEEP_LAST]
    tail = hist[-COMPACT_KEEP_LAST:]
    head_text = "\n".join([f"{m.role.upper()}: {m.content}" for m in head])[:6000]  # garde court

    try:
        summary_out = llm.chat(
            system=("Tu es un agent de résumé. En 3-5 puces, synthétise ce dialogue : "
                    "préférences utilisateur, faits persistants, décisions/intentions, TODO. Style concis."),
            history=[],
            user_msg=head_text,
            temperature=0.2,
            max_tokens=256,
        )
        summary = (summary_out.get("choices", [{}])[0].get("text") or "").strip() or "Résumé indisponible."
    except Exception as e:
        logger.error(f"Summary failed: {type(e).__name__}: {e}")
        summary = "Résumé automatique indisponible (fallback)."

    sessions.reset(session_id)
    sessions.append(session_id, ChatMessage(role="system", content=f"[Contexte condensé]\n{summary}"))
    for m in tail:
        sessions.append(session_id, m)

# ------------------ Helpers génération + LLM safe call ------------------

def get_gen(req: ChatRequest) -> Dict[str, Any]:
    """
    Récupère les paramètres de génération compatibles avec LocalLLM.chat().
    On filtre pour éviter les erreurs 'unexpected keyword argument'.
    """
    allowed_keys = {"temperature", "max_tokens", "stream"}

    # anciens champs à la racine
    base = {
        "temperature": getattr(req, "temperature", 0.7),
        "max_tokens": getattr(req, "max_tokens", 256),
    }

    # nouveaux éventuels dans req.gen
    gen = getattr(req, "gen", None)
    if gen and isinstance(gen, dict):
        for k, v in gen.items():
            if k in allowed_keys:
                base[k] = v
    elif gen and not isinstance(gen, dict):
        for k in allowed_keys:
            if hasattr(gen, k):
                base[k] = getattr(gen, k)

    return base

def get_style(req: ChatRequest) -> Dict[str, Any]:
    style = getattr(req, "style", None)
    return dict(
        personality=(getattr(req, "personality", None) or getattr(style, "personality", None) or DEFAULT_PERSONALITY),
        reply_style=(getattr(style, "reply_style", None) or "balanced"),
        min_words=int(getattr(style, "min_words", 0) or 0),
    )

def enforce_min_words(text: str, min_words: int) -> str:
    if min_words <= 0:
        return text
    words = text.split()
    if len(words) >= min_words:
        return text
    try:
        more = llm.chat(
            system="Tu étoffes un texte sans divaguer, en gardant le même ton.",
            history=[{"role": "assistant", "content": text}],
            user_msg=f"Développe légèrement pour atteindre ~{min_words} mots utiles.",
            temperature=0.6,
            max_tokens=min(256, max(64, (min_words - len(words)) * 4)),
        )["choices"][0]["text"].strip()
        return (text + ("\n\n" if text else "") + more).strip()
    except Exception as e:
        logger.error(f"enforce_min_words failed: {type(e).__name__}: {e}")
        return text

def _to_safe_history(hist):
    # certains wrappers n’aiment pas les objets Pydantic
    return [{"role": m.role, "content": m.content} for m in hist]

def _call_llm_chat(llm_obj, system_prompt, hist_before, user_msg, **gen):
    safe_hist = [{"role": m.role, "content": m.content} for m in hist_before]

    # Tentative 1 : normal + hint langue
    user_msg_1 = _prefix_lang(user_msg, DEFAULT_LANG)
    try:
        return llm_obj.chat(
            system=system_prompt,
            history=safe_hist,
            user_msg=user_msg_1,
            **gen,
        )
    except Exception as e1:
        logger.error(f"llm.chat failed with system=: {type(e1).__name__}: {e1}")

    # Tentative 2 : fallback (on préfixe aussi le contenu)
    try:
        # On ré-injecte le system prompt + rappel FR dans le message
        prefixed = f"{system_prompt}\n\n[Consigne] Réponds en français.\n\n[Utilisateur]: {user_msg}"
        fallback_gen = {k: v for k, v in gen.items() if k in ("temperature", "max_tokens", "stream")}
        return llm_obj.chat(
            system=system_prompt,   # on garde system= pour ta classe LocalLLM
            history=safe_hist,
            user_msg=prefixed,
            **fallback_gen,
        )
    except Exception as e2:
        logger.exception("llm.chat failed fallback")
        raise e2

def _prefix_lang(user_msg: str, lang: str = DEFAULT_LANG) -> str:
    # Hint explicite collé au message utilisateur (efficace sur certains modèles)
    if lang.lower().startswith("fr"):
        return f"Réponds en français.\n\n{user_msg}"
    return user_msg

def _extract_text(out) -> str:
    """Retourne le texte final d'une réponse non-stream (str | dict multi formats)."""
    if isinstance(out, str):
        return out.strip()
    if isinstance(out, dict):
        # format OpenAI-like
        choices = out.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0] or {}
            # llama.cpp final parfois "text" directement
            if isinstance(first.get("text"), str):
                return first["text"].strip()
            # OpenAI: message.content
            msg = first.get("message") or {}
            if isinstance(msg.get("content"), str):
                return msg["content"].strip()
            # fallback si top-level delta (peu probable en non-stream)
            delta = first.get("delta") or {}
            if isinstance(delta.get("content"), str):
                return delta["content"].strip()
        # autres variantes top-level
        for k in ("text", "content", "reply"):
            v = out.get(k)
            if isinstance(v, str):
                return v.strip()
    return ""

def _extract_token(ev) -> str:
    """Retourne un token pour le streaming (événements str | dict)."""
    if isinstance(ev, str):
        return ev
    if not isinstance(ev, dict):
        return ""
    choices = ev.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0] or {}
        delta = first.get("delta") or first.get("message") or {}
        tok = delta.get("content")
        if isinstance(tok, str):
            return tok
        # certaines implémentations mettent "text" directement sur choice
        if isinstance(first.get("text"), str):
            return first["text"]
        if isinstance(first.get("content"), str):
            return first["content"]
    # top-level secours
    for k in ("token", "text", "content"):
        v = ev.get(k)
        if isinstance(v, str):
            return v
    return ""

def _extract_usage(out) -> Optional[int]:
    """Retourne total_tokens si présent, sinon None. Tolère str ou dict."""
    if isinstance(out, dict):
        usage = out.get("usage") or {}
        # variantes possibles selon wrappers
        for k in ("total_tokens", "total", "tokens"):
            v = usage.get(k)
            if isinstance(v, int):
                return v
    return None


# ------------------ Endpoints ------------------

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    """
    Appel non-streaming : renvoie la réponse complète + historique.
    """
    style = get_style(req)
    gen = get_gen(req)
    system_prompt = build_system_prompt(req.system, style["personality"], style["reply_style"])

    hist_before = sessions.history(req.session_id)
    sessions.append(req.session_id, ChatMessage(role="user", content=req.message))

    try:
        out = _call_llm_chat(llm, system_prompt, hist_before, req.message, **gen)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {type(e).__name__}: {e}")

    reply = _extract_text(out)  # <--- CHANGEMENT
    reply = enforce_min_words(reply, style["min_words"])

    sessions.append(req.session_id, ChatMessage(role="assistant", content=reply))
    compact_history_if_needed(req.session_id)

    return ChatResponse(
        session_id=req.session_id,
        reply=reply,
        usage_tokens=_extract_usage(out),
        history=sessions.history(req.session_id),
    )

@app.post("/chat/reset/{session_id}")
def reset_chat(session_id: str):
    sessions.reset(session_id)
    return {"ok": True}

@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """
    SSE streaming : tokens au fil de l’eau + stockage de la réponse finale.
    """
    style = get_style(req)
    gen = get_gen(req)
    system_prompt = build_system_prompt(req.system, style["personality"], style["reply_style"])

    hist_before = sessions.history(req.session_id)
    user_msg = req.message
    sessions.append(req.session_id, ChatMessage(role="user", content=user_msg))

    def token_gen() -> Iterable[Dict[str, str]]:
        buffer = []
        try:
            for ev in _call_llm_chat(llm, system_prompt, hist_before, user_msg, stream=True, **gen):
                tok = _extract_token(ev)  # <--- CHANGEMENT
                if tok:
                    buffer.append(tok)
                    yield {"event": "token", "data": tok}
        except Exception as e:
            yield {"event": "error", "data": f"LLM error: {type(e).__name__}: {e}"}
        finally:
            final = "".join(buffer).strip()
            if final:
                sessions.append(req.session_id, ChatMessage(role="assistant", content=final))
                compact_history_if_needed(req.session_id)
            yield {"event": "done", "data": "1"}

    return EventSourceResponse(token_gen(), ping=15000)
