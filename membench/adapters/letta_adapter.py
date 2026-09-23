"""Letta (formerly MemGPT).

Letta is an agent runtime rather than a retrieval library, so the unit of memory
is an agent. This makes one agent per conversation, writes every turn into that
agent's archival memory, and deletes the agent on the next reset.

Things a copier will trip over:

* It needs a server and this box has no Docker. With no Postgres URI configured
  the server keeps everything in SQLite, so no container is involved:

      python -m venv .venv-letta                      not the harness venv, see below
      .venv-letta/Scripts/pip install "letta==0.16.8" asyncpg
      set GROQ_API_KEY=...                            resolves the model handle
      .venv-letta/Scripts/letta server --port 8283

  Its own environment on purpose. 0.16.8 resolves the openai SDK back to the
  2.x line, and the harness runs 3.x, so installing the server next to mem0 and
  langchain-openai means a downgrade underneath two of the systems being
  measured. The server is a separate process reached over HTTP, so nothing is
  lost by keeping it apart; the harness itself only needs letta-client.

  asyncpg is not a typo. 0.16.8 imports it at module scope from its ORM base,
  so the server will not boot without it even though SQLite is doing the work
  and no Postgres exists. It is not in the package's own dependencies.

  Pin that version. The `letta` name on PyPI is now Letta Code, a terminal agent
  that installs its own `letta` command and contains no REST server; 0.16.8 is
  the last release that still carries `letta server`. It listens on
  http://localhost:8283, and LETTA_BASE_URL points this adapter somewhere else.
* The model handle is resolved when the agent is created and a miss is fatal,
  so the server has to have been started with GROQ_API_KEY set. Letta registers
  models as "<provider>/<model id>" and looks the handle up by exact string,
  and Groq's own id already contains a slash, so the handle has two.
* Embeddings are the awkward part. Letta computes them inside its own server
  process, so it cannot be handed the shared LocalEmbedder object. Passing a
  handle like ollama/nomic-embed-text would put it on a different build of the
  model than everything else, which quietly turns the results table into a
  comparison of two embedders as well as five memory systems. Instead the
  harness serves the exact shared weights on a local endpoint and the agent is
  created with an embedding_config pointing at it: same weights, same width,
  same prefixes as the GoodMem baseline. See membench/embed_server.py.
* The agent is never messaged. Sending a turn as a message would hand the
  decision about what is worth remembering to the agent's own LLM, and reading
  the reply back would be Letta writing the answer instead of the shared model.
  passages.create writes the turn verbatim and passages.search reads passages
  back, so only retrieved text reaches the grader. The price is that the thing
  Letta is known for, an agent curating its own memory, is not what gets
  measured, and that belongs in the write-up. Archival search also returns no
  similarity scores, so Chunk.score is None.

Client calls are letta-client 1.12.1 (requirements.txt says >=0.1.0, which
resolves there today; the 0.1.x line laid the resources out differently, so pin
it if you care about reproducing a number). Routes and response shapes checked
against the 0.16.8 server source.
https://docs.letta.com/api/python/resources/agents/subresources/passages
https://github.com/letta-ai/letta/blob/archive/letta/server/rest_api/routers/v1/agents.py
"""
from __future__ import annotations

import os
import uuid

from ..embed_server import EmbedServer
from .base import Chunk, MemorySystem, SearchResult, timed

DEFAULT_BASE_URL = "http://localhost:8283"


class LettaSystem(MemorySystem):
    key = "letta"
    label = "Letta"
    track = "baseline"

    def __init__(self, cfg):
        self.cfg = cfg
        self.client = None
        self.agent_id: str | None = None
        self.base_url = os.environ.get("LETTA_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
        self._embeddings = EmbedServer(cfg)
        self._embed_url = ""

        # Groq's own id for this model already contains a slash, so the handle
        # comes out with two of them. That is what the server stores.
        self.model = os.environ.get("LETTA_MODEL", f"groq/{cfg.llm_model}")
        # nomic is asymmetric and nothing downstream adds the prefixes, so the
        # adapter adds them here exactly as the GoodMem baseline does.
        self._prefixed = "nomic" in cfg.embed_model.lower()

        # The runner reads version and config_notes before setup() runs, so both
        # have to resolve without touching the server.
        try:
            from importlib.metadata import version
            self.version = f"letta-client {version('letta-client')}"
        except Exception:
            self.version = "letta-client unknown"

        self.config_notes = (
            f"self-hosted server at {self.base_url}, archival memory only. Turns go in with "
            f"passages.create and come back with passages.search; the agent is never messaged, "
            f"so nothing it generates reaches the grader. Embeddings are {cfg.embed_model}, the "
            f"shared weights, served to the Letta server over a local endpoint because it embeds "
            f"in process. Agent model handle {self.model}, needed to create the agent and unused "
            f"for retrieval. "
            f"Archival search is embedding similarity and returns no scores."
        )

    # -- lifecycle ---------------------------------------------------------------

    def setup(self) -> None:
        try:
            from letta_client import Letta  # imported late so the package is optional
        except ImportError as e:
            raise RuntimeError(
                "letta-client is not installed. Run: pip install 'letta-client>=1.0'. "
                "The server it talks to is a separate install, see the top of this file. "
                f"Failed on: {e}"
            ) from e

        # This SDK retries twice by default. No other adapter here retries, and
        # a retry buys reliability with latency that the table would not show.
        # api_key is left to the client, which reads LETTA_API_KEY; a local
        # server started without --secure wants none.
        self.client = Letta(base_url=self.base_url, max_retries=0, timeout=120.0)
        try:
            self.client.health()
        except Exception as e:
            raise RuntimeError(
                f"no Letta server at {self.base_url}: {e}. Start one with "
                "'pip install letta==0.16.8' then 'letta server --port 8283', or point "
                "LETTA_BASE_URL at one that is already running."
            ) from e

        # started after the health check so a missing server fails on the clear
        # error rather than after a model load
        self._embed_url = self._embeddings.start()

    def teardown(self) -> None:
        self._drop_agent()
        if self.client:
            self.client.close()
            self.client = None
        self._embeddings.stop()

    def reset(self, run_key: str) -> None:
        self._drop_agent()
        # The suffix keeps a resumed run from being confused with the attempt it
        # is replacing, since run_key is stable across resumes.
        agent = self.client.agents.create(
            name=f"membench-{run_key}-{uuid.uuid4().hex[:8]}",
            description="agent-memory-bench run, safe to delete",
            model=self.model,
            embedding_config={
                "embedding_endpoint_type": "openai",
                "embedding_endpoint": self._embed_url,
                "embedding_model": self.cfg.embed_model.split("/")[-1],
                "embedding_dim": self.cfg.embed_dim,
            },
            # The agent never reasons, so the base toolset is setup cost for nothing.
            include_base_tools=False,
            tags=["membench"],
        )
        self.agent_id = agent.id

    def _drop_agent(self) -> None:
        if not (self.client and self.agent_id):
            return
        try:
            self.client.agents.delete(self.agent_id)
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
            self.client.agents.passages.create(self.agent_id, text=body)
        return t[0]

    def flush(self, user_key: str) -> None:
        """Nothing to wait for. The insert embeds before it returns, so a passage
        is searchable by the time the call comes back."""

    # -- read --------------------------------------------------------------------

    def search(self, user_key: str, query: str, k: int) -> SearchResult:
        q = f"{self.cfg.embed_query_prefix}{query}" if self._prefixed else query
        with timed() as t:
            resp = self.client.agents.passages.search(self.agent_id, query=q, top_k=k)
        # A result carries id, content, timestamp and tags, and no score.
        chunks = [
            Chunk(text=self._strip_prefix(r.content), score=None, source=r.id)
            for r in resp.results
            if r.content
        ]
        return SearchResult(chunks=chunks[:k], latency_s=t[0], raw=None)

    def _strip_prefix(self, text: str) -> str:
        """The prefix is an artefact of how nomic wants its input and should not
        reach the context window the answering model sees."""
        p = self.cfg.embed_doc_prefix
        return text[len(p):].lstrip() if text.startswith(p) else text
