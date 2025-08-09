from __future__ import annotations
import os, re
from pathlib import Path
from typing import Optional
from app.utils.io import truncate

BIBLE_PDF_PATH  = Path(os.getenv("BIBLE_PDF_PATH",  "app/persona/Bible_du_Projet_Ene.pdf"))
BIBLE_CACHE_TXT = Path(os.getenv("BIBLE_CACHE_TXT", "app/persona/ene_bible_cache.txt"))

_BIBLE: Optional[str] = None

def _read_pdf_text(pdf: Path) -> str:
    if not pdf.exists(): return ""
    try:
        import pypdf
        r = pypdf.PdfReader(str(pdf))
        return "\n".join([(p.extract_text() or "") for p in r.pages])
    except Exception:
        try:
            import PyPDF2
            r = PyPDF2.PdfReader(str(pdf))
            return "\n".join([(p.extract_text() or "") for p in r.pages])
        except Exception:
            return ""

def _clean(s: str) -> str:
    s = s.replace("\x00"," ")
    s = re.sub(r"[ \t]+"," ", s)
    s = re.sub(r"\n{2,}","\n\n", s)
    return s.strip()

def _summarize_with_llm(text: str) -> str:
    # Résumé minimal : si pas d’LLM ou erreur, coupe simplement.
    # Tu peux brancher llm ici si tu veux une vraie synthèse.
    return _clean(text)[:6000]

def preload_bible():
    global _BIBLE
    if _BIBLE is not None: return
    if BIBLE_CACHE_TXT.exists():
        _BIBLE = _clean(BIBLE_CACHE_TXT.read_text(encoding="utf-8"))
        return
    raw = _clean(_read_pdf_text(BIBLE_PDF_PATH))
    if not raw:
        _BIBLE = ""
        return
    condensed = _summarize_with_llm(raw)
    BIBLE_CACHE_TXT.parent.mkdir(parents=True, exist_ok=True)
    BIBLE_CACHE_TXT.write_text(condensed, encoding="utf-8")
    _BIBLE = condensed

def get_bible_snippet(max_chars: int = 3500) -> str:
    preload_bible()
    return truncate(_BIBLE or "", max_chars)
