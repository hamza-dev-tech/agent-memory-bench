"""Harness tests that touch no network.

A fake memory system and a fake model are enough to check the parts that could
silently corrupt a result: dataset selection, grading, and whether the runner
really asks each system the same thing.

    python -m pytest tests/ -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from membench import dataset
from membench.adapters.base import Chunk, MemorySystem, SearchResult
from membench.config import CONFIG
from membench.judge import Verdict, normalise, string_match
from membench.metrics import ProbeRecord, SystemSummary, approx_tokens
from membench.runner import Runner


class FakeSystem(MemorySystem):
    """Stores turns in a list and returns the ones sharing a word with the query."""

    key = "fake"
    label = "Fake"
    track = "baseline"

    def __init__(self):
        self.store: dict[str, list[str]] = {}
        self.flushed = 0

    def reset(self, run_key):
        self.store[run_key] = []

    def add(self, user_key, text, meta):
        self.store.setdefault(user_key, []).append(text)
        return 0.001

    def flush(self, user_key):
        self.flushed += 1

    def search(self, user_key, query, k):
        words = set(normalise(query).split())
        scored = [
            (len(words & set(normalise(t).split())), t) for t in self.store.get(user_key, [])
        ]
        scored.sort(reverse=True)
        return SearchResult([Chunk(t, float(s)) for s, t in scored[:k] if s], 0.002)


class FakeChat:
    """Answers with the first chunk, or refuses when nothing came back."""

    def __init__(self):
        self.calls = 0

    def complete(self, system, user, max_tokens=None):
        from membench.llm import Completion

        self.calls += 1
        notes = user.split("NOTES\n", 1)[-1].split("\n\nQUESTION", 1)[0]
        text = "NO ANSWER" if notes.startswith("(no notes") else notes.splitlines()[0][:60]
        return Completion(text=text, prompt_tokens=len(user) // 4, completion_tokens=5, latency_s=0.01, model="fake")


class FakeJudge:
    def __init__(self):
        self.chat = FakeChat()

    def grade(self, question, gold, answer, adversarial):
        abstained = answer.strip() == "NO ANSWER"
        if adversarial:
            return Verdict(abstained, abstained, None, False, "adversarial", abstained)
        hit = string_match(answer, gold)
        return Verdict(hit, hit, hit, False, "fake judge", abstained)


def test_dataset_is_deterministic():
    a = dataset.build(CONFIG.dataset_path, 2, 20, 4, seed=1)
    b = dataset.build(CONFIG.dataset_path, 2, 20, 4, seed=1)
    assert [p.probe_id for c in a for p in c.probes] == [p.probe_id for c in b for p in c.probes]
    c = dataset.build(CONFIG.dataset_path, 2, 20, 4, seed=2)
    assert [p.question for x in a for p in x.probes] != [p.question for x in c for p in x.probes]


def test_turns_carry_speaker_and_date():
    convs = dataset.build(CONFIG.dataset_path, 1, 5, 1, CONFIG.seed)
    t = convs[0].turns[0]
    line = t.as_memory_text()
    assert t.speaker in line and t.timestamp in line


def test_string_match_is_forgiving_but_not_silly():
    assert string_match("7 May 2023", "the 7 May 2023")
    assert string_match("Psychology, counseling", "psychology")
    assert not string_match("Enterprise", "Pro")
    assert not string_match("", "Pro")


def test_adversarial_probe_rewards_refusal():
    j = FakeJudge()
    assert j.grade("q", "", "NO ANSWER", adversarial=True).correct
    assert not j.grade("q", "", "Some confident nonsense", adversarial=True).correct


def test_summary_counts_hallucinations_separately():
    s = SystemSummary(system="x", label="X", track="baseline")
    base = dict(
        system="x", track="baseline", conversation_id="c", category="single-hop",
        question="q", gold="g", answer="a", by_string=True, by_judge=True, disagreed=False,
        why="", search_latency_s=0.01, answer_latency_s=0.01, context_chars=10,
        context_chunks=1, memory_tokens=3, prompt_tokens=100,
    )
    s.add(ProbeRecord(probe_id="1", adversarial=False, correct=True, abstained=False, **base))
    s.add(ProbeRecord(probe_id="2", adversarial=True, correct=False, abstained=False, **base))
    assert s.answerable == 1 and s.correct == 1
    assert s.adversarial == 1 and s.hallucinated == 1
    assert s.row()["Recall"] == "100.0%"
    assert s.row()["Refused adversarial"] == "0.0%"


def test_runner_end_to_end(tmp_path):
    convs = dataset.build(CONFIG.dataset_path, 1, 6, 2, CONFIG.seed)
    convs[0].turns = convs[0].turns[:80]
    runner = Runner(CONFIG, FakeChat(), FakeJudge(), tmp_path / "probes.jsonl")
    summary = runner.run_system(FakeSystem(), convs)
    assert summary.probes == len(convs[0].probes)
    assert summary.ingest_turns == 80
    assert (tmp_path / "probes.jsonl").exists()


def test_resume_skips_graded_probes(tmp_path):
    convs = dataset.build(CONFIG.dataset_path, 1, 4, 1, CONFIG.seed)
    convs[0].turns = convs[0].turns[:40]
    log = tmp_path / "probes.jsonl"
    system = FakeSystem()
    first = Runner(CONFIG, FakeChat(), FakeJudge(), log).run_system(system, convs)
    again = Runner(CONFIG, FakeChat(), FakeJudge(), log).run_system(FakeSystem(), convs)
    assert first.probes > 0
    assert again.probes == 0  # everything already graded


def test_token_estimate_is_monotonic():
    assert approx_tokens("") == 0
    assert approx_tokens("a" * 400) > approx_tokens("a" * 40)


def test_embed_server_speaks_openai(monkeypatch):
    """The endpoint Letta is pointed at, with the model stubbed out.

    Loading nomic here would turn a fast test file into a 500MB download, and
    what is worth checking is the HTTP shape, not the arithmetic.
    """
    import json
    import urllib.request

    from membench import embed_server

    class StubEmbedder:
        def __init__(self, cfg, prefixes=True):
            self.seen = []

        def warmup(self):
            pass

        def embed_documents(self, texts):
            self.seen.extend(texts)
            return [[float(len(t)), 0.5] for t in texts]

    monkeypatch.setattr(embed_server, "LocalEmbedder", StubEmbedder)
    server = embed_server.EmbedServer(CONFIG)
    url = server.start()
    try:
        req = urllib.request.Request(
            f"{url}/embeddings",
            data=json.dumps({"input": ["one", "three"], "model": "x"}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            body = json.load(r)

        assert [d["index"] for d in body["data"]] == [0, 1]
        assert body["data"][0]["embedding"] == [3.0, 0.5]
        assert body["object"] == "list"

        # a bare string is the other half of the OpenAI schema
        req = urllib.request.Request(
            f"{url}/embeddings",
            data=json.dumps({"input": "seven!!"}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            assert len(json.load(r)["data"]) == 1
    finally:
        server.stop()


def test_a_system_that_cannot_ingest_is_not_graded(tmp_path):
    """The guard that keeps a broken run out of the results table.

    The failure this is written against: mem0's every write was rejected by the
    model endpoint, the store stayed empty, and the harness went on to grade
    probes against it. Refusing on every answerable question and correctly
    refusing the adversarial one produces a row of plausible numbers off nothing
    at all, which is the exact shape of a result nobody should publish.
    """
    import dataclasses

    import pytest

    class BrokenSystem(FakeSystem):
        key = "broken"
        label = "Broken"

        def add(self, user_key, text, meta):
            raise RuntimeError("413 request too large")

    cfg = dataclasses.replace(CONFIG, ingest_abort_after=10 ** 6)  # exercise the rate check
    convs = dataset.build(CONFIG.dataset_path, 1, 4, 1, CONFIG.seed)
    convs[0].turns = convs[0].turns[:40]
    log = tmp_path / "probes.jsonl"

    with pytest.raises(RuntimeError, match="lost 40 of 40 turns"):
        Runner(cfg, FakeChat(), FakeJudge(), log).run_system(BrokenSystem(), convs)

    graded = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert not [r for r in graded if r["kind"] == "probe"]


def test_ingest_gives_up_early_when_every_turn_fails(tmp_path):
    import pytest

    class BrokenSystem(FakeSystem):
        key = "broken"
        label = "Broken"
        attempts = 0

        def add(self, user_key, text, meta):
            type(self).attempts += 1
            raise RuntimeError("no such model")

    convs = dataset.build(CONFIG.dataset_path, 1, 4, 1, CONFIG.seed)
    with pytest.raises(RuntimeError, match="configuration rather than luck"):
        Runner(CONFIG, FakeChat(), FakeJudge(), tmp_path / "probes.jsonl").run_system(
            BrokenSystem(), convs
        )
    # stopped at the threshold instead of walking all 419 turns
    assert BrokenSystem.attempts == CONFIG.ingest_abort_after


def test_kuzu_index_names_parse_out_of_graphitis_own_statements():
    """The rebuild is only as good as this parse.

    graphiti generates the CREATE_FTS_INDEX calls; the adapter has to read the
    table and index back out of them to drop the stale ones first. If that ever
    silently returned nothing, the indices would never be rebuilt, the BM25 half
    of every hybrid search would quietly contribute nothing, and Zep's numbers
    would be wrong in a way no error message would mention.
    """
    from membench.adapters.zep_adapter import ZepGraphitiSystem as Z

    assert Z._index_target(
        "CALL CREATE_FTS_INDEX('RelatesToNode_', 'edge_name_and_fact', ['name', 'fact']);"
    ) == ("RelatesToNode_", "edge_name_and_fact")
    assert Z._index_target("CALL CREATE_FTS_INDEX( 'Entity' , 'node_name_and_summary' , ['name'])") == (
        "Entity",
        "node_name_and_summary",
    )
    assert Z._index_target("CREATE FULLTEXT INDEX edge_name_and_fact") == (None, None)


def test_the_article_is_either_a_draft_or_has_no_placeholders():
    """The failure this whole repo exists to prevent, as an assertion.

    docs/ARTICLE.md is written before the numbers exist, so every measured
    value is a <<TK>> marker. The danger is obvious: a marker that nobody
    notices becomes a number that nobody measured, which is precisely how the
    post this replaces came to be. So the file may carry markers while it says
    DRAFT at the top, and may drop the DRAFT line only once every marker is
    gone. There is no third state.
    """
    article = ROOT / "docs" / "ARTICLE.md"
    if not article.exists():
        return

    text = article.read_text(encoding="utf-8")
    is_draft = text.lstrip().startswith("# DRAFT")
    placeholders = text.count("<<TK")

    if is_draft:
        return
    assert placeholders == 0, (
        f"{article.name} no longer says DRAFT but still has {placeholders} "
        "unfilled placeholder(s). Fill them from the run log or put the DRAFT "
        "line back."
    )


def test_two_tracks_of_one_system_do_not_collide_on_resume(tmp_path):
    """The bug that made the vendor-recommended run silently not happen.

    Two tracks of the same product share a system key on purpose. Resume used
    to remember (system, probe_id), so the second track found every probe
    already graded, skipped the conversation, and wrote nothing. Since the
    report is rebuilt from probe records, the track then vanished from the
    table with no error anywhere, which is indistinguishable from never having
    asked for it.
    """
    convs = dataset.build(CONFIG.dataset_path, 1, 4, 1, CONFIG.seed)
    convs[0].turns = convs[0].turns[:40]
    log = tmp_path / "probes.jsonl"

    class Recommended(FakeSystem):
        track = "recommended"

    first = Runner(CONFIG, FakeChat(), FakeJudge(), log).run_system(FakeSystem(), convs)
    second = Runner(CONFIG, FakeChat(), FakeJudge(), log).run_system(Recommended(), convs)
    assert first.probes > 0
    assert second.probes == first.probes, "the second track was skipped as already graded"

    # and a genuine resume of the same track still skips
    again = Runner(CONFIG, FakeChat(), FakeJudge(), log).run_system(Recommended(), convs)
    assert again.probes == 0


def test_two_tracks_never_share_a_results_row(tmp_path):
    """A row of real numbers under the wrong heading.

    Summaries used to be keyed on the system alone, and the label came from
    whichever summary line was written last. So when GoodMem's recommended
    track graded nothing, the baseline's 120 probes were rebuilt into a single
    row wearing the recommended label: full figures, correct to four
    significant digits, describing a run that never happened. The baseline row
    was gone. Nothing about the table looked wrong, which is the whole problem.
    """
    from membench.metrics import rebuild_summaries

    log = tmp_path / "probes.jsonl"
    with open(log, "w", encoding="utf-8") as f:
        for i in range(3):
            f.write(json.dumps({
                "kind": "probe", "ts": 0.0, "system": "goodmem", "track": "baseline",
                "conversation_id": "c", "probe_id": f"c:{i}", "category": "single-hop",
                "adversarial": False, "question": "q", "gold": "g", "answer": "g",
                "correct": True, "by_string": True, "by_judge": True, "disagreed": False,
                "why": "", "abstained": False, "search_latency_s": 0.01,
                "answer_latency_s": 0.1, "context_chars": 10, "context_chunks": 1,
                "memory_tokens": 10, "prompt_tokens": 20,
            }) + "\n")
        # written last, and describing nothing: the recommended track never graded
        f.write(json.dumps({
            "kind": "summary", "ts": 1.0, "system": "goodmem", "track": "recommended",
            "label": "GoodMem (recommended config)", "version": "x", "config_notes": "",
        }) + "\n")

    rows = rebuild_summaries(log)
    assert len(rows) == 1, "an empty track must not become a row"
    assert rows[0].track == "baseline"
    assert rows[0].probes == 3


def test_only_summarisation_being_off_is_a_benign_goodmem_status():
    """GoodMem reports through status events, not exceptions, so the call can
    succeed while the measurement is void.

    Treating every status as fatal killed all 120 probes of the recommended
    track over a notice that summarisation was off, which is exactly how this
    harness is meant to be configured. Treating none as fatal is far worse:
    RERANKING_FAILED is the whole difference between GoodMem's two tracks, so
    chunks arriving after it would be baseline retrieval published under a
    heading that says recommended.
    """
    from membench.adapters.goodmem import _is_benign

    class S:
        def __init__(self, code, details=None):
            self.code, self.details, self.message = code, details or {}, "m"

    assert _is_benign(S("FEATURE_DISABLED", {"feature": "summarization"}))
    assert _is_benign(S("LLM_CAPABILITY_INFERRED"))

    assert not _is_benign(S("RERANKING_FAILED"))
    assert not _is_benign(S("VECTOR_SEARCH_PARTIAL"))
    assert not _is_benign(S("EMBEDDER_FAILED"))
    assert not _is_benign(S("RATE_LIMITED"))
    # a disabled feature that is not summarisation changes what is measured
    assert not _is_benign(S("FEATURE_DISABLED", {"feature": "reranking"}))
    # and anything nobody has classified stops the run
    assert not _is_benign(S("SOME_CODE_ADDED_NEXT_RELEASE"))
