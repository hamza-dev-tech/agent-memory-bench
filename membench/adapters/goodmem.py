"""GoodMem.

Runs against a hosted instance. Two spaces are configured on the server side:
one matching the shared baseline (nomic-embed-text-v1.5, no reranker) and one
with the vendor's recommended reranker, so both tracks come from the same box.

Two things worth knowing if you copy this adapter:

* The baseline space embeds with nomic-embed-text-v1.5, which is asymmetric:
  documents want a "search_document: " prefix and queries a "search_query: "
  one. Skip them and recall drops for a reason that has nothing to do with
  GoodMem. The recommended space handles prefixes itself, so we do not add
  them there.
* retrieve() will happily write the answer for you if you pass an llm_id. We
  do not, because every system here has to be judged on what it retrieves,
  not on how well it writes. Passing fetch_memory_content and reading
  retrieved_item gives the raw chunks instead.

Spaces are shared, not per-run, so reset() deletes what this harness wrote and
leaves anything else alone.
"""
from __future__ import annotations

import os
import time

from .base import Chunk, MemorySystem, SearchResult, timed

POLL_INTERVAL_S = 1.0
INDEX_TIMEOUT_S = 900


class GoodMemSystem(MemorySystem):
    key = "goodmem"

    def __init__(self, cfg, track: str = "baseline"):
        self.cfg = cfg
        self.track = track
        self.label = "GoodMem" if track == "baseline" else "GoodMem (recommended config)"
        self.client = None
        self._written: list[str] = []
        self._space = cfg.goodmem_default_space if track == "baseline" else cfg.goodmem_optimized_space
        self.config_notes = (
            "hosted instance, nomic-embed-text-v1.5 via Fireworks, no reranker"
            if track == "baseline"
            else "hosted instance, vendor's recommended embedder, reranker and LLM"
        )

    # -- lifecycle ---------------------------------------------------------------

    def setup(self) -> None:
        from goodmem import Goodmem  # imported late so the package is optional

        key = os.environ.get("GOODMEM_API_KEY")
        if not key:
            raise RuntimeError("GOODMEM_API_KEY is not set")
        self.client = Goodmem(base_url=self.cfg.goodmem_base_url, api_key=key, timeout=120.0)
        try:
            from importlib.metadata import version
            self.version = f"sdk {version('goodmem')}"
        except Exception:
            self.version = "sdk unknown"

    def teardown(self) -> None:
        if self.client:
            self.client.close()
            self.client = None

    def reset(self, run_key: str) -> None:
        for mid in self._written:
            try:
                self.client.memories.delete(id=mid)
            except Exception:
                pass  # already gone, or the space was cleared by hand
        self._written = []

    # -- write -------------------------------------------------------------------

    def _doc(self, text: str) -> str:
        return f"{self.cfg.embed_doc_prefix}{text}" if self.track == "baseline" else text

    def add(self, user_key: str, text: str, meta: dict) -> float:
        with timed() as t:
            memory = self.client.memories.create(
                space_id=self._space,
                content_type="text/plain",
                original_content=self._doc(text),
            )
        self._written.append(memory.memory_id)
        return t[0]

    def flush(self, user_key: str) -> None:
        """Wait for server-side chunking and embedding to finish.

        Checking the last write only would be wrong: these are queued and can
        complete out of order.
        """
        deadline = time.monotonic() + INDEX_TIMEOUT_S
        pending = list(self._written)
        while pending and time.monotonic() < deadline:
            still: list[str] = []
            for mid in pending:
                m = self.client.memories.get(id=mid)
                if m.processing_status == "FAILED":
                    raise RuntimeError(f"GoodMem failed to index {mid}")
                if m.processing_status != "COMPLETED":
                    still.append(mid)
            pending = still
            if pending:
                time.sleep(POLL_INTERVAL_S)
        if pending:
            raise TimeoutError(f"{len(pending)} memories still indexing after {INDEX_TIMEOUT_S}s")

    # -- read --------------------------------------------------------------------

    def search(self, user_key: str, query: str, k: int) -> SearchResult:
        message = f"{self.cfg.embed_query_prefix}{query}" if self.track == "baseline" else query
        kwargs = dict(
            message=message,
            space_ids=[self._space],
            stream=False,
            fetch_memory=True,
            fetch_memory_content=False,
            requested_size=k,
        )
        if self.track != "baseline":
            # the reranker is the point of this track; the llm_id stays off so
            # we still read chunks rather than a generated answer
            kwargs["reranker_id"] = self.cfg.goodmem_reranker_id

        with timed() as t:
            events = self.client.memories.retrieve(**kwargs)

        chunks: list[Chunk] = []
        for e in events:
            if getattr(e, "status", None):
                raise RuntimeError(e.status.message)
            item = getattr(e, "retrieved_item", None)
            if not item:
                continue
            # retrieved_item.chunk wraps the chunk record: .chunk.chunk.chunk_text
            inner = getattr(item, "chunk", None)
            record = getattr(inner, "chunk", None) if inner else None
            text = getattr(record, "chunk_text", None)
            if not text:
                continue
            score = getattr(inner, "score", None) or getattr(record, "score", None)
            chunks.append(Chunk(text=self._strip_prefix(text), score=score, source=getattr(record, "memory_id", None)))

        return SearchResult(chunks=chunks[:k], latency_s=t[0], raw=None)

    def _strip_prefix(self, text: str) -> str:
        """The embedding prefix is an artefact of how nomic wants its input. It
        should not end up in the context window the answering model sees."""
        p = self.cfg.embed_doc_prefix
        return text[len(p):].lstrip() if text.startswith(p) else text
