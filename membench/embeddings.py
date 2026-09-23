"""The shared embedding model, loaded once and used by every self-hosted system.

nomic-embed-text-v1.5 through transformers, following nomic's own recipe: mean
pool over the attention mask, layer norm, truncate to the target width, then L2
normalise. sentence-transformers would be the obvious way to do this and it is
deliberately not used: it imports sklearn.metrics, whose compiled
_pairwise_distances_reduction extension is blocked by Application Control on
some Windows machines, and an embedding model is not worth a dependency that
can refuse to load.

The model is asymmetric. Documents and queries take different prefixes, and
forgetting them is the easiest way to make a retrieval benchmark look broken.

  https://huggingface.co/nomic-ai/nomic-embed-text-v1.5
"""
from __future__ import annotations

import threading

from .config import Config

_lock = threading.Lock()
_loaded: dict[str, tuple] = {}

BATCH = 32
MAX_TOKENS = 512


def _load(cfg: Config):
    """Tokenizer and model, once per process. Loading is slow and the harness
    asks for an embedder from several places."""
    with _lock:
        if cfg.embed_model not in _loaded:
            import torch
            from transformers import AutoModel, AutoTokenizer

            tok = AutoTokenizer.from_pretrained(cfg.embed_model)
            # nomic ships its own model class, so this has to be allowed to run
            model = AutoModel.from_pretrained(cfg.embed_model, trust_remote_code=True)
            model.eval()
            _loaded[cfg.embed_model] = (tok, model, torch)
    return _loaded[cfg.embed_model]


def _encode(cfg: Config, texts: list[str]) -> list[list[float]]:
    tok, model, torch = _load(cfg)
    out: list[list[float]] = []
    for i in range(0, len(texts), BATCH):
        batch = texts[i : i + BATCH]
        enc = tok(batch, padding=True, truncation=True, max_length=MAX_TOKENS, return_tensors="pt")
        with torch.no_grad():
            hidden = model(**enc).last_hidden_state

        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        # layer norm then truncate is what makes the shorter widths usable;
        # at the full width it is simply the published recipe
        pooled = torch.nn.functional.layer_norm(pooled, normalized_shape=(pooled.shape[1],))
        pooled = pooled[:, : cfg.embed_dim]
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        out.extend(pooled.tolist())
    return out


class LocalEmbedder:
    """Plain embedder for adapters that take one."""

    def __init__(self, cfg: Config, prefixes: bool = True):
        self.cfg = cfg
        self.dims = cfg.embed_dim
        self.prefixes = prefixes

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self.prefixes:
            texts = [f"{self.cfg.embed_doc_prefix}{t}" for t in texts]
        return _encode(self.cfg, list(texts))

    def embed_query(self, text: str) -> list[float]:
        if self.prefixes:
            text = f"{self.cfg.embed_query_prefix}{text}"
        return _encode(self.cfg, [text])[0]

    # some libraries expect these names
    embed = embed_documents
    encode = embed_documents

    def warmup(self) -> None:
        """Pull the weights and run one pass, so the first measured write is not
        also the download."""
        self.embed_query("warmup")


def langchain_embeddings(cfg: Config, prefixes: bool = True):
    """The same embedder wearing LangChain's Embeddings interface.

    Built in a function because langchain_core cannot be imported until it is
    installed, and an import error at module scope would take down adapters
    that have nothing to do with LangChain.
    """
    from langchain_core.embeddings import Embeddings

    inner = LocalEmbedder(cfg, prefixes=prefixes)

    class SharedEmbeddings(Embeddings):
        dims = inner.dims

        def embed_documents(self, texts):
            return inner.embed_documents(list(texts))

        def embed_query(self, text):
            return inner.embed_query(text)

    return SharedEmbeddings()
