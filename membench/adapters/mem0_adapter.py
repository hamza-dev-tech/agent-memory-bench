"""Mem0, open source and self-hosted.

Everything runs in this process. No Docker and no server: the vector store is
Qdrant in local mode, which qdrant-client embeds and which mem0 already lists
as a hard dependency, so running this costs no extra install. State is a
directory on disk.

Four things worth knowing if you copy this adapter:

* Mem0 is not somewhere you put text. Every add() hands the turn to an LLM that
  decides which facts to keep and whether each one adds to, updates or
  contradicts something stored, so one turn is one or more model calls. That is
  most of the ingest-per-turn number, so nothing here batches, defers or
  threads writes to make the column look smaller.
* search() changed shape between releases. Through 0.1.x the entity went in as
  user_id=... with a limit=... cap; from 2.x both live in filters={"user_id":
  ...} with top_k=..., and passing user_id at the top level now raises
  ValueError. requirements.txt says mem0ai>=0.1.50, which resolves to either,
  so the call below binds to whatever signature is installed rather than to a
  guess about the version. https://docs.mem0.ai/migration/oss-v2-to-v3
* Mem0 owns its embedding calls, so the "search_document: " and "search_query:
  " prefixes nomic-embed-text-v1.5 expects are not applied here the way the
  GoodMem adapter applies them. Injecting them would mean tuning one system and
  not the others. Same weights, no prefixes, and the write-up says so.
* Local Qdrant locks its directory, so two runs pointed at the same path will
  not both start. Override the location with MEMBENCH_MEM0_DIR.

Config shape and the provider keys used below:
https://docs.mem0.ai/open-source/python-quickstart
"""
from __future__ import annotations

import inspect
import os
from pathlib import Path

from .base import Chunk, MemorySystem, SearchResult, timed


class Mem0System(MemorySystem):
    key = "mem0"
    label = "Mem0 (open source)"
    track = "baseline"

    def __init__(self, cfg):
        self.cfg = cfg
        self.mem = None
        self._search_params: set[str] = set()
        override = os.environ.get("MEMBENCH_MEM0_DIR")
        self._dir = Path(override) if override else Path(cfg.results_dir) / cfg.run_id / "mem0"
        self.config_notes = (
            f"self-hosted in process, qdrant local mode, {cfg.embed_model} via "
            f"sentence-transformers, {cfg.llm_model} for fact extraction, no reranker, "
            "nomic prefixes not applied because mem0 embeds internally"
        )

    # -- lifecycle ---------------------------------------------------------------

    def setup(self) -> None:
        from mem0 import Memory  # imported late so the package stays optional

        # posthog fires on every operation and would land in the latency column
        os.environ.setdefault("MEM0_TELEMETRY", "false")
        self._dir.mkdir(parents=True, exist_ok=True)

        self.mem = Memory.from_config({
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": "membench",
                    "path": str(self._dir / "qdrant"),
                    # defaults to 1536, and the wrong value here fails at the
                    # first upsert rather than at startup
                    "embedding_model_dims": self.cfg.embed_dim,
                },
            },
            "llm": {
                "provider": "openai",
                "config": {
                    "model": self.cfg.llm_model,
                    "api_key": self.cfg.require_llm_key(),
                    # mem0's groq provider drives the groq sdk and its own default
                    # endpoint; the openai provider takes the exact base url that
                    # Config names, which is what the other systems are pointed at
                    "openai_base_url": self.cfg.llm_base_url,
                    "temperature": self.cfg.llm_temperature,
                    # max_tokens is deliberately left at mem0's own default. The
                    # 256 in Config caps the shared answering model; applying it to
                    # fact extraction would truncate the JSON mem0 parses and lose
                    # memories for a reason that is not mem0's fault.
                },
            },
            "embedder": {
                "provider": "huggingface",
                "config": {
                    "model": self.cfg.embed_model,
                    "embedding_dims": self.cfg.embed_dim,
                    # reaches SentenceTransformer as a kwarg. nomic ships its own
                    # model class, exactly as LocalEmbedder has to handle.
                    "model_kwargs": {"trust_remote_code": True},
                },
            },
            "history_db_path": str(self._dir / "history.db"),
        })

        # read once; search() is called per probe and signature work is not free
        self._search_params = set(inspect.signature(self.mem.search).parameters)
        try:
            from importlib.metadata import version
            self.version = f"mem0ai {version('mem0ai')}"
        except Exception:
            self.version = "mem0ai unknown"

    def teardown(self) -> None:
        # local qdrant keeps the directory locked until its client is closed, so
        # a second system in the same run can reuse the path. Reaching through
        # vector_store is internal, hence the guard.
        client = getattr(getattr(self.mem, "vector_store", None), "client", None)
        if client is not None:
            try:
                client.close()
            except Exception:
                pass
        self.mem = None

    def reset(self, run_key: str) -> None:
        # the runner passes run_key as the user key to add() and search() too, so
        # this drops exactly this conversation and leaves other runs sharing the
        # directory untouched
        self.mem.delete_all(user_id=run_key)

    # -- write -------------------------------------------------------------------

    def add(self, user_key: str, text: str, meta: dict) -> float:
        # as_memory_text() already carries the timestamp and the speaker name, so
        # the turn goes in verbatim. Role stays "user" because LoCoMo speakers are
        # people rather than an agent, and mem0 reads role when it attributes facts.
        payload = {k: v for k, v in meta.items() if v is not None}
        with timed() as t:
            self.mem.add(
                [{"role": "user", "content": text}],
                user_id=user_key,
                metadata=payload,
                infer=True,  # the extraction pass is the thing being measured
            )
        return t[0]

    def flush(self, user_key: str) -> None:
        """Nothing to wait for. Mem0 extracts and upserts inside add(), so the
        cost is already in the ingest number rather than hiding in a queue."""

    # -- read --------------------------------------------------------------------

    def search(self, user_key: str, query: str, k: int) -> SearchResult:
        kwargs: dict = {}
        if "user_id" in self._search_params:
            kwargs["user_id"] = user_key  # 0.1.x
        else:
            kwargs["filters"] = {"user_id": user_key}  # 2.x and later
        kwargs["top_k" if "top_k" in self._search_params else "limit"] = k

        with timed() as t:
            found = self.mem.search(query, **kwargs)

        # 0.1.x returned a bare list before settling on {"results": [...]}
        items = found.get("results", []) if isinstance(found, dict) else (found or [])
        chunks = [
            Chunk(text=it["memory"], score=it.get("score"), source=it.get("id"))
            for it in items[:k]
            if it.get("memory")
        ]
        return SearchResult(chunks=chunks, latency_s=t[0], raw=None)
