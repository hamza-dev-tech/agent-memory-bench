"""Harness tests that touch no network.

A fake memory system and a fake model are enough to check the parts that could
silently corrupt a result: dataset selection, grading, and whether the runner
really asks each system the same thing.

    python -m pytest tests/ -q
"""
from __future__ import annotations

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
