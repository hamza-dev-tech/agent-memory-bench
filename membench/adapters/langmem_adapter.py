"""LangMem.

A library, not a service, so there is nothing to start and no container to run.
Memory lives in a LangGraph BaseStore and InMemoryStore keeps it in process for
the length of the run.

Things worth knowing if you copy this:

* LangMem ships two APIs that look interchangeable. The memory tools
  (create_manage_memory_tool / create_search_memory_tool) are LangChain tools an
  agent decides to call, which would put a second agent between the conversation
  and the memory and make this a test of that agent. The background manager
  (create_memory_store_manager) is the one the docs use when code, not an agent,
  decides what gets written, so that is what runs here. Reads come back from the
  store's own semantic search, which is what an agent would get from the search
  tool anyway.
* The store calls embed_documents when it indexes and embed_query when it
  searches. That distinction is the only reason nomic's asymmetric prefixes end
  up on the right side. Hand the store a bare function instead of an Embeddings
  object and langgraph wraps it so that queries go through the document path and
  pick up the document prefix.
* Every write is an LLM call: the manager extracts memories with tool calling
  (trustcall) and reads the store first to decide between an insert and an
  update. Point it at a model that cannot call tools and every write fails.
  Writes are synchronous, so flush() stays the base class no-op.
* InMemoryStore has no documented way to empty itself, so reset() builds a new
  store and a new manager around it. Nothing survives the process.

API checked against langmem 0.0.30, which is what the >=0.0.10 floor in
requirements.txt resolves to and which wants langgraph >= 0.6:
  https://langchain-ai.github.io/langmem/reference/memory/
  https://langchain-ai.github.io/langmem/background_quickstart/
  https://github.com/langchain-ai/langgraph/blob/main/libs/checkpoint/langgraph/store/base/embed.py
"""
from __future__ import annotations

import json

from ..embeddings import LocalEmbedder
from .base import Chunk, MemorySystem, SearchResult, timed

NAMESPACE_ROOT = "membench"


def _shared_embeddings(cfg):
    """LocalEmbedder in the Embeddings shape langgraph's store expects.

    Built inside a function so langchain_core stays an optional install.
    """
    from langchain_core.embeddings import Embeddings

    inner = LocalEmbedder(cfg)

    class _Shared(Embeddings):
        def embed_documents(self, texts):
            return inner.embed_documents(list(texts))

        def embed_query(self, text):
            return inner.embed_query(text)

    return _Shared()


def _memory_text(value) -> str:
    """Pull the text back out of a stored memory.

    The manager writes {"kind": <schema name>, "content": <model dump>} and the
    default Memory schema dumps to {"content": "..."}. Anything else goes on as
    JSON rather than being silently dropped.
    """
    if not isinstance(value, dict):
        return str(value)
    content = value.get("content", value)
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        inner = content.get("content")
        if isinstance(inner, str):
            return inner
        return json.dumps(content, ensure_ascii=False)
    return str(content)


class LangMemSystem(MemorySystem):
    key = "langmem"
    label = "LangMem"
    track = "baseline"

    def __init__(self, cfg):
        self.cfg = cfg
        self.store = None
        self.manager = None
        self._model = None
        self._embed = None
        self.config_notes = (
            "in process, langgraph InMemoryStore with semantic search on "
            "nomic-embed-text-v1.5, writes through create_memory_store_manager "
            "(the background manager, not the agent memory tools) extracting with "
            f"{cfg.llm_model}, reads through store.search, defaults everywhere else"
        )
        try:
            from importlib.metadata import version

            self.version = f"langmem {version('langmem')}, langgraph {version('langgraph')}"
        except Exception:
            self.version = "langmem unknown"

    # -- lifecycle ---------------------------------------------------------------

    def setup(self) -> None:
        from langchain_openai import ChatOpenAI

        # langmem would resolve a model string straight to OpenAI, so it gets an
        # instance pointed at the same Groq endpoint every other system answers
        # from. max_retries is the library default: free tier 429s are a fact of
        # this harness, the shared client waits them out too.
        self._model = ChatOpenAI(
            model=self.cfg.llm_model,
            base_url=self.cfg.llm_base_url,
            api_key=self.cfg.require_llm_key(),
            temperature=self.cfg.llm_temperature,
        )
        self._embed = _shared_embeddings(self.cfg)

    def teardown(self) -> None:
        self.store = None
        self.manager = None

    def reset(self, run_key: str) -> None:
        from langgraph.store.memory import InMemoryStore
        from langmem import create_memory_store_manager

        self.store = InMemoryStore(index={"dims": self.cfg.embed_dim, "embed": self._embed})
        # {user_id} is filled from config["configurable"] on every call, so one
        # manager can serve every conversation in the run without leaking across
        # them: the namespace differs per run_key.
        self.manager = create_memory_store_manager(
            self._model,
            namespace=(NAMESPACE_ROOT, "{user_id}"),
            store=self.store,
        )

    # -- write -------------------------------------------------------------------

    def add(self, user_key: str, text: str, meta: dict) -> float:
        from langchain_core.messages import HumanMessage

        # meta is ignored on purpose: the dataset already writes the speaker and
        # the session date into the turn text, and LangMem has no separate field
        # to put them in.
        with timed() as t:
            self.manager.invoke(
                {"messages": [HumanMessage(content=text)]},
                config={"configurable": {"user_id": user_key}},
            )
        return t[0]

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
