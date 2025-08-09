from typing import List
from fastembed import TextEmbedding

class LocalEmbedder:
    """
    FastEmbed: CPU/GPU-friendly, léger, offline.
    Modèles dispo: 'BAAI/bge-small-en-v1.5', 'sentence-transformers/all-MiniLM-L6-v2', etc.
    Par défaut: bge-small (qualité/vitesse ok).
    """
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model = TextEmbedding(model_name)

    def embed(self, texts: List[str]) -> List[List[float]]:
        # FastEmbed renvoie un générateur -> conversion en liste
        return list(self.model.embed(texts))
