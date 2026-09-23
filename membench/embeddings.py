"""Local embeddings, so the self-hosted systems use the same model as the
hosted one and nobody pays for an API.

nomic-embed-text-v1.5 is asymmetric. Documents and queries take different
prefixes, and forgetting them is the single easiest way to make a retrieval
benchmark look broken.
"""
from __future__ import annotations

import threading

from .config import Config

_model = None
_lock = threading.Lock()


def _load(cfg: Config):
    global _model
    with _lock:
        if _model is None:
            from sentence_transformers import SentenceTransformer

            # trust_remote_code is required: nomic ships its own model class
            _model = SentenceTransformer(cfg.embed_model, trust_remote_code=True)
    return _model


class LocalEmbedder:
    """Minimal embedder the adapters can hand to their own vector stores."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.dims = cfg.embed_dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        m = _load(self.cfg)
        prefixed = [f"{self.cfg.embed_doc_prefix}{t}" for t in texts]
        return [v.tolist() for v in m.encode(prefixed, normalize_embeddings=True, show_progress_bar=False)]

    def embed_query(self, text: str) -> list[float]:
        m = _load(self.cfg)
        v = m.encode([f"{self.cfg.embed_query_prefix}{text}"], normalize_embeddings=True, show_progress_bar=False)[0]
        return v.tolist()

    # aliases various libraries expect
    embed = embed_documents
    encode = embed_documents
