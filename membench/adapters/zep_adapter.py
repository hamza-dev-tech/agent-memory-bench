"""Zep's open source engine, Graphiti.

Not a vector store with extra steps. Every write runs several LLM passes that
pull entities and relations out of the turn and reconcile them against what is
already in the graph, so ingest is minutes here where the other systems take
seconds. That cost belongs to Graphiti, so it gets measured rather than worked
around.

Signatures below were read from the graphiti-core 0.30.2 wheel, which is what
requirements.txt (graphiti-core>=0.7.0) resolves to today.

Things a copier will trip over:

* No Docker on this machine, so the store is Kuzu, an embedded file with no
  server. 0.30 deprecates that driver and prints a warning pointing at Neo4j,
  so MEMBENCH_GRAPH_DB=neo4j switches to bolt for anyone who wants the
  supported path.
* graphiti_core.embedder.client reads EMBEDDING_DIM once, at import, and
  search.py uses it as the width of the vector it falls back to when a query
  embeds to nothing. setup() sets it before the first graphiti import for that
  reason; set it afterwards and the fallback is 1024 wide against 768 wide
  data.
* GraphitiClients is a pydantic model with arbitrary_types_allowed, so the
  embedder has to be a real EmbedderClient subclass. A duck typed object fails
  validation inside the Graphiti constructor.
* The embedder interface has a one item call and a batch call, and nomic wants
  a different prefix for each side. On the path this benchmark measures the
  split lines up: stored edge facts go through create_batch and get the
  document prefix, the probe goes through create and gets the query prefix.
  Entity dedupe embeds its lookup names through create_batch as well, so that
  comparison is document against document, which is at least self consistent.
* A cross encoder has to be passed or the constructor builds an OpenAI one and
  demands OPENAI_API_KEY. The one passed here points at Groq and is never
  called: graphiti.search() runs EDGE_HYBRID_SEARCH_RRF, which reranks by
  reciprocal rank fusion and has no cross encoder step. Nothing is reranked.

Install:

    pip install "graphiti-core[kuzu]"

For Neo4j instead, no Docker, on Windows: take Community from
https://neo4j.com/deployment-center/, unzip, then

    bin\\neo4j-admin dbms set-initial-password secretpw
    bin\\neo4j console

    set MEMBENCH_GRAPH_DB=neo4j
    set NEO4J_URI=bolt://localhost:7687
    set NEO4J_USER=neo4j
    set NEO4J_PASSWORD=secretpw
"""
from __future__ import annotations

import asyncio
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from ..config import Config
from ..embeddings import LocalEmbedder
from .base import Chunk, MemorySystem, SearchResult, timed

LOCOMO_STAMP = "%I:%M %p on %d %B, %Y"
INSTALL = 'pip install "graphiti-core[kuzu]"'


def _when(stamp: str | None) -> datetime:
    """LoCoMo dates a session as '1:56 pm on 8 May, 2023'. Graphiti stamps every
    edge it extracts with this and the temporal probes are graded on it, so a
    quiet fallback to now() costs recall."""
    try:
        return datetime.strptime(stamp or "", LOCOMO_STAMP).replace(tzinfo=timezone.utc)
    except ValueError:
        return datetime.now(timezone.utc)


def _make_embedder(cfg: Config):
    """The benchmark's nomic model behind graphiti's interface.

    Built in a function rather than at module level because EmbedderClient
    cannot be subclassed until graphiti-core is installed, and an import error
    at module level would take the adapter registry down with it.
    """
    from graphiti_core.embedder.client import EmbedderClient

    class SharedEmbedder(EmbedderClient):
        def __init__(self):
            self.inner = LocalEmbedder(cfg)

        async def create(self, input_data) -> list[float]:
            # graphiti calls this as create(input_data=[text]); the interface
            # also allows a bare string
            text = input_data if isinstance(input_data, str) else list(input_data)[0]
            # runs inline and blocks the loop. A thread pool would hand this one
            # system encoding throughput that the other self-hosted adapters,
            # which share the same single model, do not get.
            return self.inner.embed_query(text)

        async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
            return self.inner.embed_documents(list(input_data_list))

    return SharedEmbedder()


class ZepGraphitiSystem(MemorySystem):
    key = "zep"
    label = "Zep (Graphiti)"
    track = "baseline"

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.graphiti = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self.backend = os.environ.get("MEMBENCH_GRAPH_DB", "kuzu").strip().lower()

        override = os.environ.get("MEMBENCH_ZEP_DB")
        default_db = Path(cfg.results_dir) / cfg.run_id / "graphiti.kuzu"
        self._db = Path(override) if override else default_db

        # graphiti's default, json_schema, is what Groq documents for
        # openai/gpt-oss-120b. Point MEMBENCH_LLM at a model Groq does not list
        # there and every extraction call comes back 400, which is what the
        # json_object fallback is for.
        # https://console.groq.com/docs/structured-outputs
        self._json_mode = os.environ.get("MEMBENCH_GRAPHITI_JSON_MODE", "json_schema")

        # read here, not in setup(): the runner snapshots version before setup()
        # runs, so anything filled in later is missing from the results table
        try:
            from importlib.metadata import version
            self.version = f"graphiti-core {version('graphiti-core')}"
        except Exception:
            self.version = "graphiti-core unknown"

        self.config_notes = (
            f"self-hosted on {self.backend}, entity extraction with {cfg.llm_model} via Groq, "
            f"{cfg.embed_model} via transformers, stock EDGE_HYBRID_SEARCH_RRF, "
            "no reranker"
        )

    # -- lifecycle ---------------------------------------------------------------

    def _driver(self):
        if self.backend == "neo4j":
            from graphiti_core.driver.neo4j_driver import Neo4jDriver

            password = os.environ.get("NEO4J_PASSWORD")
            if not password:
                raise RuntimeError(
                    "MEMBENCH_GRAPH_DB=neo4j needs NEO4J_PASSWORD. The top of this file has "
                    "the commands to start a server and set one."
                )
            # this does not open a connection, so a dead server shows up at the
            # first query rather than here
            return Neo4jDriver(
                os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
                os.environ.get("NEO4J_USER", "neo4j"),
                password,
            )

        try:
            from graphiti_core.driver.kuzu_driver import KuzuDriver
        except ImportError as e:
            raise RuntimeError(f"the kuzu backend is missing. Run: {INSTALL}") from e

        self._db.parent.mkdir(parents=True, exist_ok=True)
        # a file rather than ':memory:', so a run that dies leaves a graph to look at
        driver = KuzuDriver(db=str(self._db))
        # GraphDriver declares _database but only the Neo4j and FalkorDB drivers
        # assign it, so the first call that carries a group_id dies on
        # AttributeError inside graphiti. Kuzu is one embedded file and its
        # clone() returns self, so the attribute is only ever compared against;
        # the empty string is what get_default_group_id returns for this
        # provider. Group ids still scope the nodes.
        if not hasattr(driver, "_database"):
            driver._database = ""
        return driver

    def _run(self, coro):
        """Graphiti is async and the harness is not. One loop for the whole run,
        because the drivers hold connections bound to the loop that made them."""
        return self.loop.run_until_complete(coro)

    def setup(self) -> None:
        # both of these have to be set before graphiti_core is first imported.
        # EMBEDDING_DIM is read at import; the second stops the library posting
        # an install event, which is an outbound call nobody asked this
        # benchmark to make.
        os.environ["EMBEDDING_DIM"] = str(self.cfg.embed_dim)
        os.environ.setdefault("GRAPHITI_TELEMETRY_ENABLED", "false")

        try:
            from graphiti_core import Graphiti
            from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
            from graphiti_core.llm_client.config import LLMConfig
            from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
        except ImportError as e:
            raise RuntimeError(f"graphiti-core is not installed. Run: {INSTALL}") from e

        llm = LLMConfig(
            api_key=self.cfg.require_llm_key(),
            model=self.cfg.llm_model,
            base_url=self.cfg.llm_base_url,
            temperature=self.cfg.llm_temperature,
            # max_tokens stays at graphiti's own default of 16384. The 256 in
            # Config caps the shared answering model; applied here it would cut
            # every extraction off mid-JSON and lose the turn.
        )

        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.graphiti = Graphiti(
            graph_driver=self._driver(),
            llm_client=OpenAIGenericClient(config=llm, structured_output_mode=self._json_mode),
            embedder=_make_embedder(self.cfg),
            cross_encoder=OpenAIRerankerClient(config=llm),
        )
        try:
            self._run(self.graphiti.build_indices_and_constraints())
        except Exception as e:
            raise RuntimeError(
                f"the {self.backend} graph store is not reachable ({e}). The top of this file "
                "has the commands to start one."
            ) from e

        self._build_fulltext_indices()

    def teardown(self) -> None:
        if self.graphiti is not None:
            self._run(self.graphiti.close())
            self.graphiti = None
        if self.loop is not None:
            self.loop.close()
            asyncio.set_event_loop(None)
            self.loop = None

    def _gid(self, key: str) -> str:
        # group_id partitions the graph, and graphiti rejects anything outside
        # [A-Za-z0-9_-] (helpers.validate_group_id). Run keys carry dots.
        return re.sub(r"[^A-Za-z0-9_-]", "-", key)

    def reset(self, run_key: str) -> None:
        from graphiti_core.utils.maintenance.graph_data_operations import clear_data

        # drops this conversation's partition and leaves anything else in the
        # same file alone
        self._run(clear_data(self.graphiti.driver, group_ids=[self._gid(run_key)]))

    # -- write -------------------------------------------------------------------

    def add(self, user_key: str, text: str, meta: dict) -> float:
        with timed() as t:
            self._run(self.graphiti.add_episode(
                name=meta.get("dia_id") or "turn",
                episode_body=text,
                source_description=f"{meta.get('session', '')} {meta.get('speaker', '')}".strip(),
                reference_time=_when(meta.get("timestamp")),
                group_id=self._gid(user_key),
            ))
        return t[0]

    def flush(self, user_key: str) -> None:
        """add_episode extracts, resolves and writes before it returns, which is
        also why it is the slowest ingest in the table, so there is no queue to
        drain. Kuzu's text indices are the exception: they are snapshots, so
        they are rebuilt here, once, after the whole conversation is in."""
        if self.backend == "kuzu":
            self._build_fulltext_indices(rebuild=True)

    # -- kuzu text indices -----------------------------------------------------------

    def _build_fulltext_indices(self, rebuild: bool = False) -> None:
        """Create the text indices graphiti searches but never makes on Kuzu.

        Two separate holes in graphiti-core 0.30. KuzuDriver's
        build_indices_and_constraints() is a no-op whose comment says the
        indices come from setup_schema(), and setup_schema() only creates
        tables, so every hybrid search raises "Table RelatesToNode_ doesn't have
        an index with name edge_name_and_fact" and the turn is lost. Nothing
        loads Kuzu's FTS extension either. The statements come from graphiti's
        own get_fulltext_indices() rather than a copy, so they follow the
        library rather than drift from it.

        Rebuilding matters as much as creating. A Kuzu text index is a snapshot
        of the table at the moment it was built and does not follow later
        writes, so an index made at setup finds nothing that was ingested
        afterwards and the BM25 half of the hybrid search silently contributes
        nothing. flush() rebuilds once, after the conversation is in, which is
        the only point where the graph is complete and no probe has run.
        """
        if self.backend != "kuzu":
            return  # neo4j creates its own on build_indices_and_constraints

        import kuzu
        from graphiti_core.driver.driver import GraphProvider
        from graphiti_core.graph_queries import get_fulltext_indices

        conn = kuzu.Connection(self.graphiti.driver.db)
        try:
            # the extension download is cached under ~/.kuzu after the first run
            for stmt in ("INSTALL FTS", "LOAD FTS"):
                conn.execute(stmt)

            for stmt in get_fulltext_indices(GraphProvider.KUZU):
                table, index = self._index_target(stmt)
                if not (table and index):
                    # skipping quietly would leave a stale index behind and the
                    # BM25 half of every search would contribute nothing
                    raise RuntimeError(
                        f"cannot read the table and index name out of graphiti's own "
                        f"statement, so the index cannot be rebuilt: {stmt!r}"
                    )
                if rebuild:
                    try:
                        conn.execute(f"CALL DROP_FTS_INDEX('{table}', '{index}')")
                    except Exception:
                        pass  # never created, which is the same end state
                try:
                    conn.execute(stmt)
                except Exception as e:
                    # already there on the first pass is fine; anything else is not
                    if "already exists" not in str(e).lower():
                        raise RuntimeError(
                            f"could not build the {index or '?'} text index on Kuzu: {e}"
                        ) from e
        finally:
            conn.close()

    @staticmethod
    def _index_target(stmt: str) -> tuple[str | None, str | None]:
        """Pull the table and index name out of a CREATE_FTS_INDEX call."""
        m = re.search(r"CREATE_FTS_INDEX\(\s*'([^']+)'\s*,\s*'([^']+)'", stmt)
        return (m.group(1), m.group(2)) if m else (None, None)

    # -- read --------------------------------------------------------------------

    def search(self, user_key: str, query: str, k: int) -> SearchResult:
        with timed() as t:
            edges = self._run(
                self.graphiti.search(query=query, group_ids=[self._gid(user_key)], num_results=k)
            )

        chunks: list[Chunk] = []
        for e in edges:
            # The edge fact is what Graphiti puts in front of an agent, and it
            # carries a date, which matters because a lot of the probes ask when
            # something happened. valid_at is when the fact became true;
            # reference_time is the episode it came from.
            when = e.valid_at or e.reference_time
            text = f"[{when.date()}] {e.fact}" if when else e.fact
            # search() returns edges ranked but drops the fusion score, so there
            # is nothing honest to put in score
            chunks.append(Chunk(text=text, score=None, source=e.uuid))
        return SearchResult(chunks=chunks[:k], latency_s=t[0], raw=None)
