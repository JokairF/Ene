# app/main.py
from __future__ import annotations
import os, logging
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

from app.schemas import ChatRequest, ChatResponse, ChatMessage
from app.chat import ChatSessions
from app.llm import LocalLLM

from app.utils.io import (
    to_safe_history, extract_text, extract_token, extract_usage,
    truncate, is_first_turn, prefix_lang,
)

from app.decide_router import router as decide_router
from app import config

from app.persona.prompt import build_system_prompt, force_french_and_persona
from app.persona.bible  import preload_bible, get_bible_snippet

# --------------------- App & CORS ---------------------
logger = logging.getLogger("uvicorn.error")
app = FastAPI(title="Ene API")
app.include_router(decide_router)

DEFAULT_ORIGINS = [
    "http://localhost:5173","http://127.0.0.1:5173",
    "http://localhost:3000","http://127.0.0.1:3000",
    "http://localhost:4200","http://127.0.0.1:4200",
    "http://localhost:8000","http://127.0.0.1:8000",
]
ALLOW_ORIGINS = [o.strip() for o in os.getenv("CORS_ALLOW_ORIGINS","").split(",") if o.strip()] or DEFAULT_ORIGINS

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------- Runtime ---------------------
DEFAULT_LANG    = "fr"
MAX_MSG_CHARS   = 1200
MAX_USER_CHARS  = 2000
COMPACT_AFTER   = 24
COMPACT_KEEP_LAST = 10

llm = LocalLLM(model_path=config.MODEL_PATH)
sessions = ChatSessions(max_turns=30)

# --------------------- Génération ---------------------
def _val(obj, key: str, default=None):
    if obj is None: return default
    if isinstance(obj, dict): return obj.get(key, default)
    return getattr(obj, key, default)

def get_gen(req: ChatRequest) -> Dict[str, Any]:
    base = {"temperature": getattr(req, "temperature", 0.6),
            "max_tokens":  getattr(req, "max_tokens", 256)}
    gen = getattr(req, "gen", None)
    for k in ("temperature","max_tokens","stream"):
        v = _val(gen, k, None)
        if v is not None: base[k] = v
    return base

def get_style(req: ChatRequest) -> Dict[str, Any]:
    style = getattr(req, "style", None)
    personality = getattr(req, "personality", None) or _val(style, "personality", "ene")
    reply_style = _val(style, "reply_style", "balanced")
    min_words   = int(_val(style, "min_words", getattr(req, "min_words", 0)) or 0)
    return {"personality": personality, "reply_style": reply_style, "min_words": min_words}

def enforce_min_words(text: str, min_words: int) -> str:
    if min_words <= 0 or len(text.split()) >= min_words:
        return text
    return text  # simple (peut être réactivé plus tard pour re-prompt)

# --------------------- Compactage contexte ---------------------
def compact_history_if_needed(session_id: str) -> None:
    hist = sessions.history(session_id)
    if len(hist) <= COMPACT_AFTER:
        return

    head, tail = hist[:-COMPACT_KEEP_LAST], hist[-COMPACT_KEEP_LAST:]
    head_txt = "\n".join([
        f"{(m.role if hasattr(m,'role') else m.get('role','')).upper()}: "
        f"{(m.content if hasattr(m,'content') else m.get('content',''))}"
        for m in head
    ])[:6000]

    try:
        out = llm.chat(
            system=("Tu es un agent de résumé. En 3–5 puces FR, synthétise préférences, "
                    "faits persistants, décisions/intentions, TODO. Style concis."),
            history=[], user_msg=head_txt, temperature=0.2, max_tokens=256
        )
        summary = extract_text(out) or "Résumé indisponible."
    except Exception:
        summary = "Résumé automatique indisponible (fallback)."

    sessions.reset(session_id)
    sessions.append(session_id, ChatMessage(role="system", content=f"[Contexte condensé]\n{summary}"))
    for m in tail:
        sessions.append(session_id, m)

# --------------------- Préparation requête ---------------------
def prepare(session_id: str, req: ChatRequest) -> Dict[str, Any]:
    style = get_style(req)
    gen   = get_gen(req)
    first = is_first_turn(sessions.history(session_id))

    # On ignore req.system par défaut
    allow_user_system = os.getenv("ALLOW_USER_SYSTEM","0") == "1"
    user_system = req.system if (allow_user_system and (req.system or "").strip()) else ""

    system = build_system_prompt(
        user_system=user_system,
        personality=style["personality"],
        reply_style=style["reply_style"],
        first_turn=first,
        bible_snippet=get_bible_snippet()
    )
    hist_before = sessions.history(session_id)
    user_msg_for_llm = prefix_lang(truncate(req.message, MAX_USER_CHARS), DEFAULT_LANG)

    return {
        "style": style,
        "gen": gen,
        "system": system,
        "hist_before": hist_before,
        "user_msg_for_llm": user_msg_for_llm
    }

# --------------------- LLM call ---------------------
def call_llm(system: str, hist_before: List[Any], user_msg: str, **gen):
    safe_hist = to_safe_history(hist_before, max_len=MAX_MSG_CHARS)
    try:
        return llm.chat(system=system, history=safe_hist, user_msg=user_msg, **gen)
    except Exception:
        # fallback : fusionne (FR + persona)
        fused = f"{system}\n\n[Consigne] Réponds en français.\n\n[Utilisateur]: {user_msg}"
        clean_gen = {k: v for k, v in gen.items() if k in ("temperature","max_tokens","stream")}
        return llm.chat(system=system, history=safe_hist, user_msg=fused, **clean_gen)

# --------------------- Startup ---------------------
@app.on_event("startup")
def _startup():
    preload_bible()  # charge/condense au démarrage
    logger.info("Ene API ready (bible preloaded).")

# --------------------- Routes sous /api ---------------------
router = APIRouter(prefix="/api")

@router.get("/health")
def health():
    return {"ok": True}

@router.get("/debug/system-prompt")
def debug_system_prompt(session_id: str = ""):
    # Permet de visualiser le prompt final injecté
    ctx = prepare(session_id or "debug", ChatRequest(session_id="debug", message="(test)"))
    return {"len": len(ctx["system"]), "preview": ctx["system"][:1200]}

@router.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    ctx = prepare(req.session_id, req)
    sessions.append(req.session_id, ChatMessage(role="user", content=req.message))
    try:
        out = call_llm(ctx["system"], ctx["hist_before"], ctx["user_msg_for_llm"], **ctx["gen"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {type(e).__name__}: {e}")
    reply = force_french_and_persona(extract_text(out))
    reply = enforce_min_words(reply, ctx["style"]["min_words"])
    sessions.append(req.session_id, ChatMessage(role="assistant", content=reply))
    compact_history_if_needed(req.session_id)
    return ChatResponse(
        session_id=req.session_id,
        reply=reply,
        usage_tokens=extract_usage(out),
        history=sessions.history(req.session_id),
    )

@router.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    ctx = prepare(req.session_id, req)
    sessions.append(req.session_id, ChatMessage(role="user", content=req.message))

    def gen():
        buf = []
        try:
            for ev in call_llm(ctx["system"], ctx["hist_before"], ctx["user_msg_for_llm"], stream=True, **ctx["gen"]):
                tok = extract_token(ev)
                if tok:
                    buf.append(tok)
                    yield {"event": "token", "data": tok}
        except Exception as e:
            yield {"event": "error", "data": f"LLM error: {type(e).__name__}: {e}"}
        finally:
            final = force_french_and_persona("".join(buf).strip())
            if final:
                sessions.append(req.session_id, ChatMessage(role="assistant", content=final))
                compact_history_if_needed(req.session_id)
            yield {"event": "done", "data": "1"}

    return EventSourceResponse(gen(), ping=15000)

@router.post("/chat/reset/{session_id}")
def reset_chat(session_id: str):
    sessions.reset(session_id)
    return {"ok": True}

# Monte le router /api
app.include_router(router)
