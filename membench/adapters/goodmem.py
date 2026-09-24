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
BATCH_SIZE = 100


def _batched(items: list[str], size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


# Retrieval reports itself through status events rather than exceptions, and
# most of them are fatal to a measurement even though the call returned. Only
# these two are not, so this is a whitelist: a code nobody has classified stops
# the run rather than quietly shrinking a recall number.
#
# The one worth naming is RERANKING_FAILED. The reranker is the entire
# difference between the two GoodMem tracks, so if it fails and the chunks come
# back anyway, the recommended row would be baseline retrieval published under
# a heading that says otherwise.
_BENIGN_STATUS = {
    # summarisation is deliberately off: the harness grades retrieved text, so
    # letting GoodMem write the answer is the one thing it must not do
    "FEATURE_DISABLED",
    "LLM_CAPABILITY_INFERRED",
}


def _is_benign(status) -> bool:
    code = str(getattr(status, "code", "") or "")
    if code not in _BENIGN_STATUS:
        return False
    if code == "FEATURE_DISABLED":
        # only summarisation is allowed to be the disabled feature; anything
        # else being off is a change to what is being measured
        details = getattr(status, "details", None) or {}
        feature = str(details.get("feature", "")).lower()
        return "summar" in feature or "abstract" in feature
    return True


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
        """Empty the space, and check that it emptied.

        This used to delete self._written, which is an in-memory list. A fresh
        process therefore had nothing to delete and left whatever was already
        in the space untouched. Several smoke runs and one failed attempt
        later, the space held more than one copy of the same conversation, and
        GoodMem was the only system in the table searching a polluted index:
        duplicates crowd a top-k with repeats of one fact and push distinct
        facts out of it, so its recall was measured against a handicap none of
        the others had. Every other adapter drops its whole scope, not its own
        bookkeeping: mem0 delete_all, Letta deletes the agent, graphiti
        clear_data.

        The verification is the point. A delete that silently fails leaves
        exactly the state that caused this, and nothing downstream can see it.
        """
        from goodmem.models import BatchDeleteMemorySelectorRequest

        existing = self._space_memory_ids()
        for batch in _batched(existing, BATCH_SIZE):
            self.client.memories.batch_delete(
                requests=[BatchDeleteMemorySelectorRequest(memory_id=m) for m in batch]
            )
        self._written = []

        left = self._space_memory_ids()
        if left:
            raise RuntimeError(
                f"{len(left)} memories are still in space {self._space} after deleting "
                f"{len(existing)}. Retrieval would run against someone else's data, so "
                "this stops here rather than publishing a number measured on it."
            )

    def _space_memory_ids(self) -> list[str]:
        """Every memory id in the space, following pagination to the end.

        max_items is deliberately absent: a cap here would silently leave the
        tail of a large space in place, which is the bug this method exists to
        close.
        """
        return [m.memory_id for m in self.client.memories.list(space_id=self._space)]

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
        complete out of order. Polling one id at a time would be several hundred
        round trips per conversation, so this asks in batches and only re-asks
        about the ones still working.
        """
        deadline = time.monotonic() + INDEX_TIMEOUT_S
        pending = list(self._written)
        while pending and time.monotonic() < deadline:
            still: list[str] = []
            for batch in _batched(pending, BATCH_SIZE):
                for result in self.client.memories.batch_get(memory_ids=batch).results:
                    memory = result.memory
                    if not result.success or memory is None:
                        # a read that failed says nothing about indexing, so ask again
                        if result.memory_id:
                            still.append(result.memory_id)
                        continue
                    if memory.processing_status == "FAILED":
                        raise RuntimeError(f"GoodMem failed to index {memory.memory_id}")
                    if memory.processing_status != "COMPLETED":
                        still.append(memory.memory_id)
            pending = still
            if pending:
                time.sleep(POLL_INTERVAL_S)
        if pending:
            raise TimeoutError(f"{len(pending)} memories still indexing after {INDEX_TIMEOUT_S}s")

        # Second guard, on the other side of the write. reset() proved the space
        # was empty before; this proves nothing else arrived while we filled it.
        # A space holding more than this conversation retrieves other runs'
        # chunks into the context window, and the resulting recall number looks
        # entirely normal.
        in_space = len(self._space_memory_ids())
        if in_space != len(self._written):
            raise RuntimeError(
                f"space {self._space} holds {in_space} memories but this run wrote "
                f"{len(self._written)}. Something else is writing to it, and retrieval "
                "would mix the two."
            )

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
            status = getattr(e, "status", None)
            if status is not None and not _is_benign(status):
                raise RuntimeError(f"{getattr(status, 'code', '?')}: {status.message}")
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
