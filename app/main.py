from fastapi import FastAPI, HTTPException
from app.llm import LocalLLM
from app.chat import ChatSessions
from app.schemas import ChatRequest, ChatResponse, ChatMessage
from sse_starlette.sse import EventSourceResponse

app = FastAPI(title="Ene API")

# --- LLM CPU (tu gardes ton chemin) ---
llm = LocalLLM(model_path="E:/models/Mistral-7B-Instruct-v0.2-GGUF/mistral-7b-instruct-v0.2.Q4_K_M.gguf")

# --- Sessions de chat ---
sessions = ChatSessions(max_turns=30)

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    # Historique courant
    hist = sessions.history(req.session_id)

    # Ajoute le message user
    sessions.append(req.session_id, ChatMessage(role="user", content=req.message))

    # Reprend l'historique mis à jour
    hist = sessions.history(req.session_id)

    # Appel LLM
    try:
        out = llm.chat(
            system=req.system,
            history=hist[:-1],  # tout sauf le dernier déjà ajouté
            user_msg=req.message,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {e}")

    reply = out["choices"][0]["text"].strip()

    # Ajoute la réponse assistant
    sessions.append(req.session_id, ChatMessage(role="assistant", content=reply))

    return ChatResponse(
        session_id=req.session_id,
        reply=reply,
        usage_tokens=out.get("usage", {}).get("total_tokens"),
        history=sessions.history(req.session_id),
    )

@app.post("/chat/reset/{session_id}")
def reset_chat(session_id: str):
    sessions.reset(session_id)
    return {"ok": True}

@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    hist = sessions.history(req.session_id)

    def token_gen():
        for ev in llm.chat(
            system=req.system,
            history=hist,
            user_msg=req.message,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            stream=True,
        ):
            # llama-cpp renvoie delta/content selon versions
            choice = ev["choices"][0]
            delta = choice.get("delta") or choice.get("message") or {}
            tok = delta.get("content", "")
            if tok:
                yield {"event": "token", "data": tok}

    # Ajoute les messages au démarrage pour garder l’historique
    sessions.append(req.session_id, ChatMessage(role="user", content=req.message))
    return EventSourceResponse(token_gen())