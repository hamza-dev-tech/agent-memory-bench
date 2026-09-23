"""LangMem.

A library rather than a service, so there is no container to start and no URL
to configure. Memory lives in a LangGraph BaseStore, and InMemoryStore keeps it
in this process for the length of the run.

Four things worth knowing if you copy this adapter:

* LangMem ships two APIs that look interchangeable. The memory tools
  (create_manage_memory_tool, create_search_memory_tool) are LangChain tools an
  agent decides to call, which would put a second agent between the conversation
  and the memory and turn this into a measurement of that agent. The background
  manager is the one to use when code rather than an agent decides what gets
  written, so that is what runs here. Reads come back from the store's own
  semantic search, which is what the search tool would have returned anyway.
* The store calls embed_documents when it indexes and embed_query when it
  searches, and that split is the only reason nomic's asymmetric prefixes land
  on the right side. Hand the store a bare function instead of an Embeddings
  object and langgraph wraps it in EmbeddingsLambda, whose embed_query is
  embed_documents([text])[0], so every query picks up the document prefix.
* The index config defaults to fields ["$"], so the text being embedded is the
  whole stored value dumped as JSON, braces and key names included, not just the
  memory sentence. That is langgraph's default and it stays: pointing it at
  content.content would be tuning this system and not the others.
* Every write is an LLM call. The manager searches the store first, then
  extracts with tool calling through trustcall and decides between an insert and
  an update, so LangMem's ingest time is mostly shared-model time. A model that
  cannot call tools fails every write. The puts land before invoke() returns,
  which is why flush() has nothing to wait on.

InMemoryStore has no documented way to empty itself, so reset() builds a new
store and a new manager around it. Nothing survives the process.

Checked against langmem 0.0.30. requirements.txt only floors it at 0.0.10, but
0.0.30 pins langgraph >=0.6,<2 itself, which is tighter than the >=0.2.50 floor
in that file, so a fresh install still lands on a matching pair.
  https://langchain-ai.github.io/langmem/reference/memory/
  https://langchain-ai.github.io/langmem/guides/dynamically_configure_namespaces/
  https://github.com/langchain-ai/langgraph/blob/main/libs/checkpoint/langgraph/store/base/embed.py
"""
from __future__ import annotations

import json
from types import SimpleNamespace

from ..embeddings import LocalEmbedder
from .base import Chunk, MemorySystem, SearchResult, timed

NAMESPACE_ROOT = "membench"


def _imports():
    """Every third-party import in one place, so a missing install says what to
    install instead of raising ImportError from whichever method ran first."""
    try:
        from langchain_core.embeddings import Embeddings
        from langchain_core.messages import HumanMessage
        from langchain_openai import ChatOpenAI
        from langgraph.store.memory import InMemoryStore
        from langmem import create_memory_store_manager
    except ImportError as e:
        raise RuntimeError(
            "LangMem is not installed. Run: pip install 'langmem>=0.0.30'. "
            "It pulls langgraph, langchain-core and langchain-openai with it, "
            f"so nothing else is needed. Failed on: {e}"
        ) from e
    return SimpleNamespace(
        Embeddings=Embeddings,
        HumanMessage=HumanMessage,
        ChatOpenAI=ChatOpenAI,
        InMemoryStore=InMemoryStore,
        create_manager=create_memory_store_manager,
    )


def _shared_embeddings(cfg, base):
    """LocalEmbedder in the Embeddings shape the store expects.

    Subclassing matters here. See the note in the module docstring about what
    langgraph does to a bare callable.
    """
    inner = LocalEmbedder(cfg)

    class _Shared(base):
        def embed_documents(self, texts):
            return inner.embed_documents(list(texts))

        def embed_query(self, text):
            return inner.embed_query(text)

    return _Shared()


def _memory_text(value) -> str:
    """Pull the sentence back out of a stored memory.

    The manager writes {"kind": <schema name>, "content": <model dump>}, and the
    default Memory schema dumps to {"content": "..."}. A custom schema would dump
    to something else, which goes on as JSON rather than being dropped.
    """
    if not isinstance(value, dict):
        return str(value)
    content = value.get("content", value)
    if isinstance(content, dict) and isinstance(content.get("content"), str):
        return content["content"]
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)


class LangMemSystem(MemorySystem):
    key = "langmem"
    label = "LangMem"
    track = "baseline"

    def __init__(self, cfg):
        self.cfg = cfg
        self.store = None
        self.manager = None
        self._lm = None
        self._model = None
        self._embed = None
        self.config_notes = (
            "in process, langgraph InMemoryStore with semantic search on "
            f"{cfg.embed_model}, default index fields so the embedded text is the "
            "stored value as JSON, writes through create_memory_store_manager (the "
            "background manager, not the agent memory tools) extracting with "
            f"{cfg.llm_model}, reads through store.search, defaults everywhere else"
        )
        # the runner reads version off the system before setup() runs
        try:
            from importlib.metadata import version

            self.version = f"langmem {version('langmem')}, langgraph {version('langgraph')}"
        except Exception:
            self.version = "langmem unknown"

    # -- lifecycle ---------------------------------------------------------------

    def setup(self) -> None:
        self._lm = _imports()
        # a model string would be resolved straight to OpenAI, so the manager gets
        # an instance pointed at the same Groq endpoint every other system uses.
        # max_retries stays at the client default: the shared answering client
        # waits out 429s too, so this is not an advantage LangMem gets alone.
        self._model = self._lm.ChatOpenAI(
            model=self.cfg.llm_model,
            base_url=self.cfg.llm_base_url,
            api_key=self.cfg.require_llm_key(),
            temperature=self.cfg.llm_temperature,
        )
        self._embed = _shared_embeddings(self.cfg, self._lm.Embeddings)

    def teardown(self) -> None:
        self.store = None
        self.manager = None

    def reset(self, run_key: str) -> None:
        self.store = self._lm.InMemoryStore(
            index={"dims": self.cfg.embed_dim, "embed": self._embed}
        )
        # {user_id} is filled per call from config["configurable"], so one manager
        # serves the whole run and conversations still cannot see each other: the
        # namespace differs by run_key. The reference documents store= and a sync
        # invoke(); the quickstart shows the langgraph-entrypoint shape instead,
        # which needs a graph this harness has no use for.
        self.manager = self._lm.create_manager(
            self._model,
            namespace=(NAMESPACE_ROOT, "{user_id}"),
            store=self.store,
        )

    # -- write -------------------------------------------------------------------

    def add(self, user_key: str, text: str, meta: dict) -> float:
        # meta goes unused: LoCoMo turns already carry the speaker and the session
        # date in the text, and a memory here is a sentence with no metadata field
        # to hang them off.
        with timed() as t:
            self.manager.invoke(
                {"messages": [self._lm.HumanMessage(content=text)]},
                config={"configurable": {"user_id": user_key}},
            )
        return t[0]

    def flush(self, user_key: str) -> None:
        """Nothing to wait for. The manager extracts and puts inside invoke(), so
        the cost is already counted in ingest rather than hiding in a queue."""

    # -- read --------------------------------------------------------------------

    def search(self, user_key: str, query: str, k: int) -> SearchResult:
        with timed() as t:
            items = self.store.search((NAMESPACE_ROOT, user_key), query=query, limit=k)

        chunks: list[Chunk] = []
        for item in items:
            text = _memory_text(item.value)
            if text.strip():
                chunks.append(Chunk(text=text, score=item.score, source=item.key))
        return SearchResult(chunks=chunks[:k], latency_s=t[0], raw=None)
