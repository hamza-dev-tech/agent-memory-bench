"""The contract every memory system implements.

The whole point of the benchmark is that this file is the only thing the
runner knows about. A system is a black box that can be reset, written to,
and queried, and every one of them gets the same conversations, the same
questions and the same answering model. If a system also wants to generate
the answer itself (GoodMem can, Letta can), we do not let it: we take its
retrieved text and hand that to the shared model, otherwise we would be
comparing four different answer-writers instead of four memory layers.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class Chunk:
    """One piece of text a system handed back, in the order it ranked it."""
    text: str
    score: float | None = None
    source: str | None = None


@dataclass
class SearchResult:
    chunks: list[Chunk]
    latency_s: float
    raw: dict | None = None

    @property
    def context(self) -> str:
        return "\n".join(c.text.strip() for c in self.chunks if c.text and c.text.strip())


@dataclass
class IngestStats:
    turns: int = 0
    seconds: float = 0.0
    failures: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def per_turn_s(self) -> float:
        return self.seconds / self.turns if self.turns else 0.0


@contextmanager
def timed():
    """Wall clock around a call. Yields a one-element list so the caller can
    read the elapsed time after the block: `with timed() as t: ...; t[0]`."""
    holder = [0.0]
    start = time.perf_counter()
    try:
        yield holder
    finally:
        holder[0] = time.perf_counter() - start


class MemorySystem(ABC):
    """A memory backend under test.

    Implementations live in membench/adapters/. Keep them thin: no retries
    that the other systems do not also get, no prompt engineering, no caching.
    Anything clever you add here has to be added to all five or the numbers
    stop meaning anything.
    """

    #: short machine name, e.g. "mem0"
    key: str = ""
    #: how it appears in the results table, e.g. "Mem0 (open source)"
    label: str = ""
    #: "baseline" (shared models, defaults) or "recommended" (vendor's own config)
    track: str = "baseline"

    #: filled in by the adapter so the report can state exactly what ran
    version: str = "unknown"
    config_notes: str = ""

    def setup(self) -> None:
        """Start containers, create clients, pull models. Called once."""

    @abstractmethod
    def reset(self, run_key: str) -> None:
        """Drop all memory for this run. Every conversation starts empty, so a
        system cannot score by remembering an earlier conversation's facts."""

    @abstractmethod
    def add(self, user_key: str, text: str, meta: dict) -> float:
        """Write one conversation turn. Returns seconds spent in the call.

        `meta` carries `speaker`, `session`, `timestamp` and `dia_id` so a
        system that models time or speakers can use them. Systems that cannot
        are free to ignore it, and the write-up says which did what.
        """

    def flush(self, user_key: str) -> None:
        """Wait until writes are searchable. Only needed by systems that index
        asynchronously. A system that indexes in the background would otherwise
        look artificially fast on ingest and artificially bad on recall."""

    @abstractmethod
    def search(self, user_key: str, query: str, k: int) -> SearchResult:
        """Return the text this system would put in front of an agent."""

    def teardown(self) -> None:
        """Close clients, stop containers."""

    # -- helpers shared by adapters ------------------------------------------------

    def describe(self) -> dict:
        return {
            "key": self.key,
            "label": self.label,
            "track": self.track,
            "version": self.version,
            "config_notes": self.config_notes,
        }

    def __repr__(self) -> str:  # pragma: no cover
        return f"<{type(self).__name__} {self.key} track={self.track}>"
