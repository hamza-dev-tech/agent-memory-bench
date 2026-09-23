"""Mem0, open source and self-hosted.

Everything runs in this process. No Docker and no server: the vector store is
Qdrant in local mode, which qdrant-client embeds and which mem0 already lists
as a hard dependency, so running this costs no extra install. State is a
directory on disk.

Things a copier will trip over:

* Mem0 is not somewhere you put text. Every add() hands the turn to an LLM that
  decides which facts to keep and whether each one adds to, updates or
  contradicts something already stored, so a turn costs at least one model
  call. That is most of the ingest-per-turn number, so nothing here batches,
  defers or threads writes to make the column look smaller. The extraction call
  asks for a JSON object back; Groq serves json_object mode on every model it
  hosts, so the shared endpoint is enough for it.
* search() changed shape between releases. Through 0.1.x the entity went in as
  user_id=... with a limit=... cap; from 2.x both live in filters={"user_id":
  ...} with top_k=..., and passing user_id at the top level now raises
  ValueError. add() and delete_all() kept their top-level user_id in both.
  requirements.txt floors mem0ai at 0.1.50 and a fresh install lands on 2.1.0,
  so the call below binds to whichever signature is installed rather than to a
  guess about the version. https://docs.mem0.ai/migration/oss-v2-to-v3
* Mem0 owns its embedding calls and hands every text to embed_query, writing or
  searching alike, so there is no side to attach nomic's asymmetric prefixes to
  the way the GoodMem and Letta adapters do. It runs without them. Same weights,
  no prefixes, and the write-up says so.
* Its huggingface embedder loads the model through sentence-transformers, which
  will not import on a machine where Application Control blocks sklearn's
  compiled extensions. The langchain provider takes an Embeddings object
  instead, which is how the shared embedder gets in.
* The openai provider drops the base url you configured if OPENROUTER_API_KEY
  is set in the environment, which would run mem0 against a different endpoint
  than everything else in the table without saying anything. setup() refuses to
  start in that case.
* Local Qdrant locks its directory, so two runs pointed at the same path will
  not both start. Override the location with MEMBENCH_MEM0_DIR.

Argument names checked against the 2.x sources, since the docs do not spell all
of them out:
  https://github.com/mem0ai/mem0/blob/main/mem0/llms/openai.py
  https://github.com/mem0ai/mem0/blob/main/mem0/embeddings/langchain.py
  https://github.com/mem0ai/mem0/blob/main/mem0/vector_stores/qdrant.py
  https://docs.mem0.ai/open-source/configuration
"""
from __future__ import annotations

import inspect
import os
from pathlib import Path

from ..embeddings import langchain_embeddings
from .base import Chunk, MemorySystem, SearchResult, timed


def _memory_class():
    """Import mem0 late, and not before telemetry is off.

    mem0.memory.telemetry reads MEM0_TELEMETRY once, at import, so setting it
    afterwards does nothing. posthog fires on every operation and its network
    time would land in the latency column.
    """
    os.environ.setdefault("MEM0_TELEMETRY", "false")
    try:
        from mem0 import Memory
    except ImportError as e:
        raise RuntimeError(
            "mem0 is not installed. Run: pip install 'mem0ai>=2.0'. It pulls "
            "qdrant-client in with it, so the local vector store needs nothing "
            f"else. Failed on: {e}"
        ) from e
    return Memory


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
            f"self-hosted in process, qdrant local mode, {cfg.embed_model} via transformers "
            f"through mem0's langchain embedder, {cfg.llm_model} for fact extraction at mem0's "
            f"own token cap, no reranker, nomic prefixes not applied because mem0 embeds "
            f"internally"
        )
        # the runner reads version off the system before setup() runs
        try:
            from importlib.metadata import version
            self.version = f"mem0ai {version('mem0ai')}"
        except Exception:
            self.version = "mem0ai unknown"

    # -- lifecycle ---------------------------------------------------------------

    def setup(self) -> None:
        Memory = _memory_class()

        if os.environ.get("OPENROUTER_API_KEY"):
            raise RuntimeError(
                "OPENROUTER_API_KEY is set. mem0's openai provider prefers it over the "
                "base url configured here, so mem0 would extract memories on a different "
                "endpoint than every other system in the table. Unset it for the run."
            )

        self._dir.mkdir(parents=True, exist_ok=True)
        config = {
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
                # The huggingface provider would load the model through
                # sentence-transformers, which cannot import on this machine.
                # The langchain provider takes an Embeddings instance, so mem0
                # gets the same weights the other self-hosted systems use.
                "provider": "langchain",
                "config": {
                    "model": langchain_embeddings(self.cfg, prefixes=False),
                    "embedding_dims": self.cfg.embed_dim,
                },
            },
            "history_db_path": str(self._dir / "history.db"),
        }

        try:
            self.mem = Memory.from_config(config)
        except Exception as e:
            raise RuntimeError(
                f"mem0 would not start on {self._dir}: {e}. If another run is using "
                "that path, the local Qdrant still holds the directory lock; point "
                "MEMBENCH_MEM0_DIR somewhere else."
            ) from e

        # read once; search() runs per probe and signature work is not free
        self._search_params = set(inspect.signature(self.mem.search).parameters)

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
        # rerank defaults to False in 2.x and there is no reranker in the config,
        # so this is mem0's plain vector search, as the baseline track wants

        with timed() as t:
            found = self.mem.search(query, **kwargs)

        # 0.1.x on an older config version returned a bare list
        items = found.get("results", []) if isinstance(found, dict) else (found or [])
        chunks = [
            Chunk(text=it["memory"], score=it.get("score"), source=it.get("id"))
            for it in items[:k]
            if it.get("memory")
        ]
        return SearchResult(chunks=chunks, latency_s=t[0], raw=None)
