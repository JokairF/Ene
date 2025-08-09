import logging
import os
import re
import hashlib
import functools
from pathlib import Path
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

from app.chat import ChatSessions
from app.llm import LocalLLM
from app.schemas import ChatRequest, ChatResponse, ChatMessage

# ===================== App & CORS =====================
logger = logging.getLogger("uvicorn.error")
app = FastAPI(title="Ene API")

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

# ===================== LLM & Sessions =====================
llm = LocalLLM(model_path=os.getenv(
    "LLM_MODEL_PATH",
    "E:/models/Mistral-7B-Instruct-v0.2-GGUF/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
))
sessions = ChatSessions(max_turns=30)

# ===================== Constantes Persona =====================
DEFAULT_PERSONALITY = "ene"
DEFAULT_LANG = "fr"
PERSONA_TAG = "[PERSONA_ENE_V1]"
BIBLE_TAG = "[BIBLE_ENE]"

# Fenêtre / troncatures prudentes pour CPU n_ctx=2k–4k
MAX_BIBLE_CHARS = 4000
MAX_MSG_CHARS   = 1200
MAX_USER_CHARS  = 2000

# Bible (PDF -> cache txt)
BIBLE_PDF_PATH  = Path(os.getenv("BIBLE_PDF_PATH", "app/persona/Bible_du_Projet_Ene.pdf"))
BIBLE_CACHE_TXT = Path(os.getenv("BIBLE_CACHE_TXT", "app/persona/ene_bible_cache.txt"))
BIBLE_CONDENSED: Optional[str] = None  # chargé au démarrage

# ===================== Helpers génériques =====================
def _truncate(s: str, n: int) -> str:
    return s if isinstance(s, str) and len(s) <= n else (s[:n] + "…") if isinstance(s, str) else s

def _get_role(m) -> str:
    return (m.get("role") if isinstance(m, dict) else getattr(m, "role", "")) or ""

def _get_content(m) -> str:
    return (m.get("content") if isinstance(m, dict) else getattr(m, "content", "")) or ""

def _to_safe_history(hist) -> List[Dict[str, str]]:
    return [{"role": _get_role(m), "content": _truncate(_get_content(m), MAX_MSG_CHARS)} for m in hist]

def _val(obj, key: str, default=None):
    if obj is None: return default
    if isinstance(obj, dict): return obj.get(key, default)
    return getattr(obj, key, default)

def _extract_text(out) -> str:
    if isinstance(out, str): return out.strip()
    if isinstance(out, dict):
        ch = out.get("choices")
        if isinstance(ch, list) and ch:
            first = ch[0] or {}
            if isinstance(first.get("text"), str): return first["text"].strip()
            msg = first.get("message") or {}
            if isinstance(msg.get("content"), str): return msg["content"].strip()
            delta = first.get("delta") or {}
            if isinstance(delta.get("content"), str): return delta["content"].strip()
        for k in ("text", "content", "reply"):
            v = out.get(k)
            if isinstance(v, str): return v.strip()
    return ""

def _extract_token(ev) -> str:
    if isinstance(ev, str): return ev
    if not isinstance(ev, dict): return ""
    ch = ev.get("choices")
    if isinstance(ch, list) and ch:
        first = ch[0] or {}
        delta = first.get("delta") or first.get("message") or {}
        if isinstance(delta.get("content"), str): return delta["content"]
        if isinstance(first.get("text"), str): return first["text"]
        if isinstance(first.get("content"), str): return first["content"]
    for k in ("token", "text", "content"):
        v = ev.get(k)
        if isinstance(v, str): return v
    return ""

def _extract_usage(out) -> Optional[int]:
    if isinstance(out, dict):
        usage = out.get("usage") or {}
        for k in ("total_tokens", "total", "tokens"):
            v = usage.get(k)
            if isinstance(v, int): return v
    return None

def _is_first_turn(session_id: str) -> bool:
    return not any(_get_role(m) == "assistant" for m in sessions.history(session_id))

def _prefix_lang(msg: str, lang: str = DEFAULT_LANG) -> str:
    return f"Réponds en français.\n\n{msg}" if lang.lower().startswith("fr") else msg

# ===================== Persona / System prompt =====================
def persona_prompt(personality: Optional[str]) -> str:
    p = (personality or DEFAULT_PERSONALITY).lower()
    if p in ("ene", "ene/takane", "takane"):
        return (
            "Tu es *Ene* (alias Takane Enomoto). Réponds EXCLUSIVEMENT en français.\n"
            "Personnalité : enjouée, énergique, espiègle, taquine mais compatissante ; parfois méta (4e mur).\n"
            "Langage : familier et joueur, onomatopées (« Heh~ », « Ahaha~ »), analogies informatiques pertinentes.\n"
            "Appelle l’utilisateur « Maître ». Évite les longues explications techniques en personnage."
        )
    return "Tu es une assistante conversationnelle chaleureuse et utile. Français uniquement."

def style_directives(reply_style: Optional[str]) -> str:
    rs = (reply_style or "balanced").lower()
    if rs == "immersive": return "Réponses un peu développées, concrètes, avec une touche d'humour ; reste centrée sur la demande."
    if rs == "concise":   return "Réponses courtes et percutantes, sans digressions."
    return "Réponses claires, utiles, avec une touche de personnalité."

@functools.lru_cache(maxsize=64)
def _build_system_prompt_cached(user_system_hash: str, personality: str, reply_style: str, first_turn: bool) -> str:
    base  = persona_prompt(personality)
    style = style_directives(reply_style)
    intro = ("Si c'est le premier échange, commence par UNE phrase courte en personnage "
             "ex. « Ahaha~ Salut Maître ! Je suis Ene, ta cyber-camarade taquine. » ; puis réponds.\n"
             "Interdits : ne jamais dire « Je suis un assistant », « language model », etc.")
    few   = ("Exemples :\n"
             "- « Heh~ Tu m’appelles et me voilà, Maître. On configure quoi aujourd’hui ? »\n"
             "- « Ok, je scanne… *bip-boup* Voilà le plan en 3 étapes. »")
    parts = [
        base,
        f"Directives de style : {style}",
        "IMPORTANT : Réponds en français uniquement. Reste strictement en personnage.",
        intro if first_turn else "Interdits : ne jamais dire « Je suis un assistant », « language model », etc.",
        few,
    ]
    if user_system_hash:
        parts.append(f"[Règles sup. hash={user_system_hash}]")
    return "\n\n".join(parts)

def build_system_prompt(user_system: Optional[str], personality: str, reply_style: str, first_turn: bool) -> str:
    # On ne met pas tout le user_system dans le cache key (taille) : on hashe
    h = hashlib.sha1((user_system or "").encode("utf-8")).hexdigest()[:10] if user_system else ""
    return _build_system_prompt_cached(h, personality, reply_style, first_turn) + (
        f"\n\nRègles supplémentaires:\n{user_system.strip()}" if user_system and user_system.strip() else ""
    )

# ===================== Paramétrage génération =====================
def get_gen(req: ChatRequest) -> Dict[str, Any]:
    allowed = {"temperature", "max_tokens", "stream"}
    base = {"temperature": getattr(req, "temperature", 0.7),
            "max_tokens": getattr(req, "max_tokens", 256)}
    gen = getattr(req, "gen", None)
    for k in allowed:
        v = _val(gen, k, None)
        if v is not None:
            base[k] = v
    return base

def get_style(req: ChatRequest) -> Dict[str, Any]:
    style = getattr(req, "style", None)
    personality = getattr(req, "personality", None) or _val(style, "personality", DEFAULT_PERSONALITY)
    reply_style = _val(style, "reply_style", "balanced")
    min_words   = _val(style, "min_words", getattr(req, "min_words", 0)) or 0
    try: min_words = int(min_words)
    except: min_words = 0
    return {"personality": personality, "reply_style": reply_style, "min_words": min_words}

def enforce_min_words(text: str, min_words: int) -> str:
    if min_words <= 0: return text
    words = text.split()
    if len(words) >= min_words: return text
    try:
        out = llm.chat(
            system="Tu étoffes un texte sans divaguer, en gardant le même ton.",
            history=[{"role": "assistant", "content": text}],
            user_msg=f"Développe légèrement pour atteindre ~{min_words} mots utiles.",
            temperature=0.6,
            max_tokens=min(256, max(64, (min_words - len(words)) * 4)),
        )
        more = _extract_text(out)
        return (text + ("\n\n" if text else "") + more).strip() if more else text
    except Exception as e:
        logger.error(f"enforce_min_words failed: {type(e).__name__}: {e}")
        return text

def _call_llm_chat(llm_obj, system_prompt: str, hist_before, user_msg: str, **gen):
    safe_hist = _to_safe_history(hist_before)
    try:
        return llm_obj.chat(system=system_prompt, history=safe_hist, user_msg=_prefix_lang(user_msg, DEFAULT_LANG), **gen)
    except Exception as e1:
        logger.error(f"llm.chat failed (system= present): {type(e1).__name__}: {e1}")
    try:
        fused = f"{system_prompt}\n\n[Consigne] Réponds en français.\n\n[Utilisateur]: {user_msg}"
        fallback_gen = {k: v for k, v in gen.items() if k in ("temperature", "max_tokens", "stream")}
        return llm_obj.chat(system=system_prompt, history=safe_hist, user_msg=fused, **fallback_gen)
    except Exception as e2:
        logger.exception("llm.chat failed fallback")
        raise e2

# ===================== Bible (PDF -> condensé) =====================
def _read_pdf_text(pdf_path: Path) -> str:
    if not pdf_path.exists(): return ""
    try:
        import pypdf
        reader = pypdf.PdfReader(str(pdf_path))
        return "\n".join([(p.extract_text() or "") for p in reader.pages])
    except Exception:
        try:
            import PyPDF2
            reader = PyPDF2.PdfReader(str(pdf_path))
            return "\n".join([(p.extract_text() or "") for p in reader.pages])
        except Exception:
            return ""

def _clean_text(s: str) -> str:
    s = s.replace("\x00", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{2,}", "\n\n", s)
    return s.strip()

def _llm_summarize_chunks(text: str, chunk_chars: int = 4000) -> str:
    text = text.strip()
    if not text: return ""
    chunks = [text[i:i+chunk_chars] for i in range(0, len(text), chunk_chars)]
    bullets = []
    for ch in chunks:
        try:
            out = llm.chat(
                system=("Tu es un agent de synthèse. Résume en FR 3–6 puces très informatives : "
                        "personnalité, style, règles, émotions, intents, formats utiles."),
                history=[],
                user_msg=ch,
                temperature=0.2,
                max_tokens=300,
            )
            part = _extract_text(out)
            if part: bullets.append(part)
        except Exception:
            continue
    merged = "\n\n".join(bullets) if bullets else text[:3000]
    try:
        out = llm.chat(
            system=("Condense EN FR (≤ 800 mots) la personnalité d’Ene, règles, style, émotions, intents, formats utiles. "
                    "Pas d'intro d'« assistant »."),
            history=[],
            user_msg=merged,
            temperature=0.2,
            max_tokens=900,
        )
        return _clean_text(_extract_text(out))
    except Exception:
        return _clean_text(merged)

def _load_bible_condensed() -> str:
    try:
        if BIBLE_CACHE_TXT.exists():
            return _clean_text(BIBLE_CACHE_TXT.read_text(encoding="utf-8"))
        raw = _clean_text(_read_pdf_text(BIBLE_PDF_PATH))
        if not raw: return ""
        condensed = _clean_text(_llm_summarize_chunks(raw)) or raw[:5000]
        BIBLE_CACHE_TXT.parent.mkdir(parents=True, exist_ok=True)
        BIBLE_CACHE_TXT.write_text(condensed, encoding="utf-8")
        return condensed
    except Exception as e:
        logger.error(f"Load bible failed: {type(e).__name__}: {e}")
        return ""

# ===================== Seeds (Persona + Bible) =====================
def _persona_seed_messages() -> List[Dict[str, str]]:
    system_seed = (
        f"{PERSONA_TAG}\n"
        "Tu es *Ene* (alias Takane Enomoto). Français uniquement.\n"
        "Personnalité : enjouée, énergique, espiègle, taquine mais compatissante ; clin d’œil méta possible.\n"
        "Langage : familier, « Heh~ », « Ahaha~ », analogies informatiques.\n"
        "Appelle l'utilisateur « Maître ». Interdit : « Je suis un assistant … ».\n"
        "Si premier échange : une phrase courte de présentation en personnage."
    )
    assistant_demo = "Ahaha~ Salut Maître ! Je suis Ene, ta cyber-camarade. On bidouille quoi aujourd’hui ?"
    return [{"role": "system", "content": system_seed}, {"role": "assistant", "content": assistant_demo}]

def _ensure_persona_seed(session_id: str):
    hist = sessions.history(session_id)
    if any(PERSONA_TAG in _get_content(m) for m in hist if _get_role(m) == "system"):
        return
    for m in _persona_seed_messages():
        sessions.append(session_id, ChatMessage(role=m["role"], content=m["content"]))

def _ensure_bible_seed(session_id: str):
    global BIBLE_CONDENSED
    hist = sessions.history(session_id)
    if any(BIBLE_TAG in _get_content(m) for m in hist if _get_role(m) == "system"):
        return
    if not BIBLE_CONDENSED:
        return
    content = f"{BIBLE_TAG}\nRéférence permanente :\n\n{_truncate(BIBLE_CONDENSED, MAX_BIBLE_CHARS)}"
    sessions.append(session_id, ChatMessage(role="system", content=content))

def _ensure_seeds(session_id: str):
    _ensure_persona_seed(session_id)
    _ensure_bible_seed(session_id)

# ===================== Compactage historique =====================
COMPACT_AFTER = 24
COMPACT_KEEP_LAST = 10

def compact_history_if_needed(session_id: str) -> None:
    hist = sessions.history(session_id)
    if len(hist) <= COMPACT_AFTER:
        return
    head, tail = hist[:-COMPACT_KEEP_LAST], hist[-COMPACT_KEEP_LAST:]
    head_text = "\n".join([f"{_get_role(m).upper()}: {_get_content(m)}" for m in head])[:6000]
    try:
        out = llm.chat(
            system=("Tu es un agent de résumé. En 3–5 puces FR, synthétise préférences, faits persistants, "
                    "décisions/intentions, TODO. Style concis."),
            history=[],
            user_msg=head_text,
            temperature=0.2,
            max_tokens=256,
        )
        summary = _extract_text(out) or "Résumé indisponible."
    except Exception as e:
        logger.error(f"Summary failed: {type(e).__name__}: {e}")
        summary = "Résumé automatique indisponible (fallback)."

    sessions.reset(session_id)
    sessions.append(session_id, ChatMessage(role="system", content=f"[Contexte condensé]\n{summary}"))
    for m in tail:
        sessions.append(session_id, m)

# ===================== Préparation requête (centralisée) =====================
def _prepare(session_id: str, req: ChatRequest) -> Dict[str, Any]:
    style = get_style(req)
    gen = get_gen(req)
    first_turn = _is_first_turn(session_id)
    system_prompt = build_system_prompt(req.system, style["personality"], style["reply_style"], first_turn)
    _ensure_seeds(session_id)
    hist_before = sessions.history(session_id)
    # On passe au LLM une version FR-lock + tronquée, mais on stockera l’original en historique
    user_msg_for_llm = _prefix_lang(_truncate(req.message, MAX_USER_CHARS), DEFAULT_LANG)
    return dict(style=style, gen=gen, system_prompt=system_prompt, hist_before=hist_before, user_msg_for_llm=user_msg_for_llm)

# ===================== Endpoints =====================
@app.on_event("startup")
def _startup():
    global BIBLE_CONDENSED
    # Précharge la Bible condensée (1ère fois) et logue
    BIBLE_CONDENSED = _load_bible_condensed()
    logger.info(f"Bible loaded: {len(BIBLE_CONDENSED or '')} chars (cache: {BIBLE_CACHE_TXT.exists()})")

@app.get("/health")
def health():
    return {"ok": True, "persona_seeded": True, "bible_loaded": bool(BIBLE_CONDENSED)}

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    ctx = _prepare(req.session_id, req)

    # Historique : on stocke le message ORIGINAL (non préfixé)
    sessions.append(req.session_id, ChatMessage(role="user", content=req.message))

    try:
        out = _call_llm_chat(llm, ctx["system_prompt"], ctx["hist_before"], ctx["user_msg_for_llm"], **ctx["gen"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {type(e).__name__}: {e}")

    reply = enforce_min_words(_extract_text(out), ctx["style"]["min_words"])
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
    ctx = _prepare(req.session_id, req)

    # Historique : on stocke le message ORIGINAL (pas la version préfixée FR)
    sessions.append(req.session_id, ChatMessage(role="user", content=req.message))

    def token_gen():
        buffer = []
        try:
            for ev in _call_llm_chat(llm, ctx["system_prompt"], ctx["hist_before"], ctx["user_msg_for_llm"], stream=True, **ctx["gen"]):
                tok = _extract_token(ev)
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
