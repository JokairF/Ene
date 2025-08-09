from typing import Dict, List, Optional
import chromadb
from chromadb.utils import embedding_functions
from .embeddings import LocalEmbedder
import time
import uuid

class MemoryStore:
    def __init__(self, persist_dir: str = "./chroma", model_name: str = "BAAI/bge-small-en-v1.5"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.embedder = LocalEmbedder(model_name)
        # On gère nous-mêmes les embeddings pour éviter les dépendances lourdes dans Chroma
        self.collections = {
            "episodic": self.client.get_or_create_collection("episodic"),
            "semantic": self.client.get_or_create_collection("semantic"),
            "procedural": self.client.get_or_create_collection("procedural")
        }

    def _ensure_id(self, metadata: Dict) -> str:
        return metadata.get("id") or str(uuid.uuid4())

    def add(self, mem_type: str, text: str, metadata: Dict):
        assert mem_type in self.collections, f"Type mémoire inconnu: {mem_type}"
        _id = self._ensure_id(metadata)
        embedding = self.embedder.embed([text])[0]
        meta = {"ts": time.time(), **metadata}
        self.collections[mem_type].add(
            documents=[text],
            metadatas=[meta],
            embeddings=[embedding],
            ids=[_id]
        )
        return _id

    def bulk_add(self, mem_type: str, items: List[Dict]):
        # items: [{"text": "...", "metadata": {...}}, ...]
        texts = [it["text"] for it in items]
        metas = [{"ts": time.time(), **it.get("metadata", {})} for it in items]
        ids = [self._ensure_id(m) for m in metas]
        embs = self.embedder.embed(texts)
        self.collections[mem_type].add(
            documents=texts,
            metadatas=metas,
            embeddings=list(embs),
            ids=ids
        )
        return ids

    def query(self, mem_type: str, query: str, top_k: int = 5):
        assert mem_type in self.collections, f"Type mémoire inconnu: {mem_type}"
        q_emb = self.embedder.embed([query])[0]
        return self.collections[mem_type].query(
            query_embeddings=[q_emb],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

    def get_context(self, query: str, top_k_each: int = 2) -> str:
        """
        Récupère un petit contexte mixte: episodic + semantic + procedural
        """
        ctx_parts = []
        for mem_type in ["episodic", "semantic", "procedural"]:
            res = self.query(mem_type, query, top_k=top_k_each)
            docs = res.get("documents", [[]])[0]
            metas = res.get("metadatas", [[]])[0]
            if not docs:
                continue
            # Tri léger récence+distance si dispo
            distances = res.get("distances", [[]])[0] or [0.0] * len(docs)
            scored = list(zip(docs, metas, distances))
            # tri asc sur distance, desc sur récence (ts)
            scored.sort(key=lambda x: (x[2], -(x[1].get("ts", 0))))
            for d, m, dist in scored[:top_k_each]:
                tag = m.get("tags") or m.get("type") or mem_type
                ctx_parts.append(f"[{mem_type}:{tag}] {d}")
        return "\n".join(ctx_parts[: (top_k_each * 3)])
