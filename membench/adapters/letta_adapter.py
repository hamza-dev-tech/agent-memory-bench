"""Letta (formerly MemGPT), the SQLite line.

Letta is an agent runtime rather than a retrieval library, so the unit of memory
is an agent. This makes one agent per conversation, writes every turn into that
agent's archival memory, and deletes the agent on the next reset.

Running it at all took some deciding, and the write-up has to carry the result:

* 0.16.8 is the newest release with a REST server, and it cannot run here.
  Its server/db.py builds an asyncpg engine unconditionally and never reads the
  SQLITE value sitting in its own settings, so with no Postgres it dies on a
  refused connection to localhost:5432, wearing a "generator didn't stop after
  athrow()" as a disguise. Its ORM is SQLite-ready, picking CommonVector over
  pgvector; only the async engine never got the branch. Everything from 0.12.1
  up is the same, and 0.32.x is Letta Code with no server in it.
* **0.11.7, released 2025-09-04, is the last release that runs without
  Postgres**, so it is what this adapter targets, and the results table says so
  next to the number. That is a year older than the rest of the table. Point a
  Postgres at 0.16.8 and the fair comparison is a route rename away: archival
  memory became passages, and the three calls below move with it.

      python -m venv .venv-letta                      not the harness venv
      .venv-letta/Scripts/pip install "letta==0.11.7" sqlite-vec
      set OPENAI_API_KEY=...                          resolves the model handle
      .venv-letta/Scripts/python scripts/patch_letta.py
      .venv-letta/Scripts/letta server --port 8283

  Its own environment because letta pins the openai SDK a long way back from
  where the harness sits, and sqlite-vec because 0.11.7 imports it at module
  scope from its ORM and does not declare it, the same shape of hole 0.16.8 has
  around asyncpg.

Other things a copier will trip over:

* The REST calls here are made directly rather than through letta-client. The
  SDK is pinned to the server: 0.11.7 wants 0.1.307, and the 1.x line talks to
  routes this server does not have. Three endpoints do not justify holding a
  package hostage to that.
* The agent is never messaged. Sending a turn as a message would hand the
  decision about what is worth remembering to the agent's own LLM, and reading
  the reply back would be Letta writing the answer instead of the shared model.
  The price is that the thing Letta is known for, an agent curating its own
  memory, is not what gets measured, and that belongs in the write-up.
* Archival search returns text, a timestamp and tags. No id and no score, so
  Chunk.score and Chunk.source are both None and no ranking of its own survives
  into the log.
* Letta embeds in its own process, so it cannot be handed the shared embedder
  object. An ollama handle would be a different build of nomic and would turn
  the table into a comparison of two embedders as well as five memory systems.
  The harness serves the real weights on a local endpoint instead and the agent
  is created pointing at it; Letta's openai embedding path is an AsyncOpenAI
  client, which appends /embeddings to whatever base url it is given. See
  membench/embed_server.py.

Routes and payloads were read off the running server's own openapi.json rather
than from the docs, which describe a later version.
"""
from __future__ import annotations

import os
import re
import uuid

import httpx

from ..embed_server import EmbedServer
from .base import Chunk, MemorySystem, SearchResult, timed

DEFAULT_BASE_URL = "http://localhost:8283"
# 0.12.0 moved archival memory to /passages and made Postgres mandatory
LAST_SQLITE_RELEASE = (0, 11)


class LettaSystem(MemorySystem):
    key = "letta"
    label = "Letta"
    track = "baseline"

    def __init__(self, cfg):
        self.cfg = cfg
        self.http: httpx.Client | None = None
        self.agent_id: str | None = None
        self.base_url = os.environ.get("LETTA_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
        self._embeddings = EmbedServer(cfg)
        self._embed_url = ""

        # Only used to create the agent, never to answer anything, so the
        # provider only has to resolve. The server reads the matching key from
        # its own environment.
        self.model = os.environ.get("LETTA_MODEL", f"openai/{cfg.llm_model}")
        # nomic is asymmetric and nothing downstream adds the prefixes, so the
        # adapter adds them here exactly as the GoodMem baseline does.
        self._prefixed = "nomic" in cfg.embed_model.lower()

        # replaced in setup() with what the server reports, which is the version
        # that actually ran
        self.version = "letta server"
        self.config_notes = self._notes("not contacted yet")

    def _notes(self, version: str) -> str:
        return (
            f"self-hosted {version} server at {self.base_url} on SQLite, archival memory "
            f"only. Turns go in with archival-memory and come back with its search; the "
            f"agent is never messaged, so nothing it generates reaches the grader. "
            f"Embeddings are {self.cfg.embed_model}, the shared weights, served to the "
            f"server over a local endpoint because it embeds in process. Agent model "
            f"handle {self.model}, needed to create the agent and unused for retrieval. "
            f"Archival search returns no scores. Running at all needs one line changed in "
            f"the installed package, see scripts/patch_letta.py: a Postgres server_default "
            f"of now() is stored literally by SQLite and nothing can be written without "
            f"it. It is a column default and touches nothing about what Letta stores or "
            f"retrieves."
        )

    # -- lifecycle ---------------------------------------------------------------

    def setup(self) -> None:
        self.http = httpx.Client(base_url=self.base_url, timeout=120.0)
        try:
            health = self.http.get("/v1/health/")
            health.raise_for_status()
            version = str(health.json().get("version") or "unknown")
        except Exception as e:
            raise RuntimeError(
                f"no Letta server at {self.base_url}: {e}. The top of this file has the "
                "commands to start one, or point LETTA_BASE_URL at one already running."
            ) from e

        parts = re.findall(r"\d+", version)[:2]
        if len(parts) == 2 and tuple(int(p) for p in parts) > LAST_SQLITE_RELEASE:
            raise RuntimeError(
                f"this server is Letta {version}, and the routes below are the 0.11.x ones. "
                "From 0.12.0 archival memory became /passages and Postgres became "
                "mandatory. Run 0.11.7, or update the three calls in this adapter and "
                "point the server at a Postgres."
            )

        self.version = f"letta server {version}"
        self.config_notes = self._notes(version)
        # started after the health check so a missing server fails on the clear
        # error rather than after a model load
        self._embed_url = self._embeddings.start()

    def teardown(self) -> None:
        self._drop_agent()
        if self.http:
            self.http.close()
            self.http = None
        self._embeddings.stop()

    def reset(self, run_key: str) -> None:
        self._drop_agent()
        r = self.http.post("/v1/agents/", json={
            # the suffix keeps a resumed run from being confused with the attempt
            # it is replacing, since run_key is stable across resumes
            "name": f"membench-{run_key}-{uuid.uuid4().hex[:8]}",
            "description": "agent-memory-bench run, safe to delete",
            "model": self.model,
            "embedding_config": {
                "embedding_endpoint_type": "openai",
                "embedding_endpoint": self._embed_url,
                "embedding_model": self.cfg.embed_model.split("/")[-1],
                "embedding_dim": self.cfg.embed_dim,
            },
            # the agent never reasons, so the base toolset is setup cost for nothing
            "include_base_tools": False,
            "tags": ["membench"],
        })
        if r.status_code >= 400:
            raise RuntimeError(f"could not create the agent ({r.status_code}): {r.text[:300]}")
        self.agent_id = r.json()["id"]

    def _drop_agent(self) -> None:
        if not (self.http and self.agent_id):
            return
        try:
            self.http.delete(f"/v1/agents/{self.agent_id}")
        except Exception:
            pass  # already gone, or the server was cleared by hand
        self.agent_id = None

    # -- write -------------------------------------------------------------------

    def add(self, user_key: str, text: str, meta: dict) -> float:
        # meta is not passed on: as_memory_text already puts the timestamp and
        # the speaker in the text, and tagging passages would be filtering that
        # no other system in the table gets.
        body = f"{self.cfg.embed_doc_prefix}{text}" if self._prefixed else text
        with timed() as t:
            r = self.http.post(f"/v1/agents/{self.agent_id}/archival-memory", json={"text": body})
        if r.status_code >= 400:
            raise RuntimeError(f"archival insert failed ({r.status_code}): {r.text[:200]}")
        return t[0]

    def flush(self, user_key: str) -> None:
        """Nothing to wait for. The insert embeds before it returns, so a passage
        is searchable by the time the call comes back."""

    # -- read --------------------------------------------------------------------

    def search(self, user_key: str, query: str, k: int) -> SearchResult:
        q = f"{self.cfg.embed_query_prefix}{query}" if self._prefixed else query
        with timed() as t:
            r = self.http.get(
                f"/v1/agents/{self.agent_id}/archival-memory/search",
                params={"query": q, "top_k": k},
            )
        if r.status_code >= 400:
            raise RuntimeError(f"archival search failed ({r.status_code}): {r.text[:200]}")

        results = r.json().get("results") or []
        chunks = [
            Chunk(text=self._strip_prefix(item["content"]), score=None, source=None)
            for item in results[:k]
            if item.get("content")
        ]
        return SearchResult(chunks=chunks, latency_s=t[0], raw=None)

    def _strip_prefix(self, text: str) -> str:
        """The prefix is an artefact of how nomic wants its input and should not
        reach the context window the answering model sees."""
        p = self.cfg.embed_doc_prefix
        return text[len(p):].lstrip() if text.startswith(p) else text
