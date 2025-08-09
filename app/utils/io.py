from __future__ import annotations
from typing import Any, Dict, List, Optional

def truncate(s: str, n: int) -> str:
    return s if isinstance(s, str) and len(s) <= n else (s[:n] + "…") if isinstance(s, str) else s

def to_safe_history(hist, max_len: int = 1200) -> List[Dict[str,str]]:
    safe = []
    for m in hist:
        if isinstance(m, dict):
            r, c = m.get("role",""), m.get("content","")
        else:
            r, c = getattr(m,"role",""), getattr(m,"content","")
        safe.append({"role": r, "content": truncate(c, max_len)})
    return safe

def extract_text(out) -> str:
    # str | dict(openai-like) | llama.cpp dict
    if isinstance(out, str): return out.strip()
    if isinstance(out, dict):
        ch = out.get("choices")
        if isinstance(ch, list) and ch:
            first = (ch[0] or {})
            if isinstance(first.get("text"), str): return first["text"].strip()
            msg = first.get("message") or {}
            if isinstance(msg.get("content"), str): return msg["content"].strip()
            delta = first.get("delta") or {}
            if isinstance(delta.get("content"), str): return delta["content"].strip()
        for k in ("text","content","reply"):
            v = out.get(k)
            if isinstance(v,str): return v.strip()
    return ""

def extract_token(ev) -> str:
    if isinstance(ev, str): return ev
    if not isinstance(ev, dict): return ""
    ch = ev.get("choices")
    if isinstance(ch, list) and ch:
        first = (ch[0] or {})
        delta = first.get("delta") or first.get("message") or {}
        if isinstance(delta.get("content"), str): return delta["content"]
        if isinstance(first.get("text"), str): return first["text"]
        if isinstance(first.get("content"), str): return first["content"]
    for k in ("token","text","content"):
        v = ev.get(k)
        if isinstance(v,str): return v
    return ""

def extract_usage(out) -> Optional[int]:
    if isinstance(out, dict):
        u = out.get("usage") or {}
        for k in ("total_tokens","total","tokens"):
            v = u.get(k)
            if isinstance(v,int): return v
    return None

def is_first_turn(hist) -> bool:
    for m in hist:
        role = (m.get("role") if isinstance(m,dict) else getattr(m,"role","")) or ""
        if role == "assistant": return False
    return True

def prefix_lang(msg: str, lang: str = "fr") -> str:
    return f"Réponds en français.\n\n{msg}" if (lang or "").lower().startswith("fr") else msg
