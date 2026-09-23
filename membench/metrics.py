"""Per-probe records and the aggregation behind the results table."""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field, asdict


def pct(n: int, d: int) -> float:
    return round(100.0 * n / d, 1) if d else 0.0


def approx_tokens(text: str) -> int:
    """Rough token count for the injected memory. Tokenizers differ, and this
    number is only ever compared between systems using the same estimator."""
    return max(0, round(len(text) / 4))


def ms(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    i = min(len(s) - 1, int(round(q * (len(s) - 1))))
    return round(s[i] * 1000, 1)


@dataclass
class ProbeRecord:
    system: str
    track: str
    conversation_id: str
    probe_id: str
    category: str
    adversarial: bool
    question: str
    gold: str
    answer: str
    correct: bool
    by_string: bool
    by_judge: bool | None
    disagreed: bool
    why: str
    abstained: bool
    search_latency_s: float
    answer_latency_s: float
    context_chars: int
    context_chunks: int
    memory_tokens: int
    prompt_tokens: int

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class SystemSummary:
    """One row of the table, plus everything needed to defend that row."""

    system: str
    label: str
    track: str
    version: str = "unknown"
    config_notes: str = ""
    probes: int = 0
    answerable: int = 0
    correct: int = 0
    adversarial: int = 0
    adversarial_abstained: int = 0
    hallucinated: int = 0
    judge_disagreements: int = 0
    ingest_turns: int = 0
    ingest_seconds: float = 0.0
    ingest_failures: int = 0
    search_latencies: list[float] = field(default_factory=list)
    context_chars: list[int] = field(default_factory=list)
    memory_tokens: list[int] = field(default_factory=list)
    prompt_tokens: list[int] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def add(self, r: ProbeRecord) -> None:
        self.probes += 1
        self.search_latencies.append(r.search_latency_s)
        self.context_chars.append(r.context_chars)
        self.memory_tokens.append(r.memory_tokens)
        self.prompt_tokens.append(r.prompt_tokens)
        if r.disagreed:
            self.judge_disagreements += 1
        if r.adversarial:
            # no answer exists, so the only right move is to refuse
            self.adversarial += 1
            if r.abstained:
                self.adversarial_abstained += 1
            else:
                self.hallucinated += 1
        else:
            self.answerable += 1
            if r.correct:
                self.correct += 1

    def row(self) -> dict:
        med = lambda xs: int(statistics.median(xs)) if xs else 0
        return {
            "System": self.label,
            "Track": self.track,
            "Recall": f"{pct(self.correct, self.answerable)}%",
            "Refused adversarial": f"{pct(self.adversarial_abstained, self.adversarial)}%" if self.adversarial else "n/a",
            "Search p50": f"{ms(self.search_latencies, 0.5)} ms",
            "Search p95": f"{ms(self.search_latencies, 0.95)} ms",
            "Memory tokens": med(self.memory_tokens),
            "Ingest/turn": f"{round(self.ingest_seconds / self.ingest_turns, 3)} s" if self.ingest_turns else "n/a",
            "Judge disagreements": self.judge_disagreements,
        }

    def as_dict(self) -> dict:
        d = asdict(self)
        d["derived"] = {
            "recall_pct": pct(self.correct, self.answerable),
            "adversarial_refusal_pct": pct(self.adversarial_abstained, self.adversarial),
            "search_p50_ms": ms(self.search_latencies, 0.5),
            "search_p95_ms": ms(self.search_latencies, 0.95),
            "ingest_per_turn_s": round(self.ingest_seconds / self.ingest_turns, 4) if self.ingest_turns else None,
            "median_context_chars": int(statistics.median(self.context_chars)) if self.context_chars else 0,
            "median_memory_tokens": int(statistics.median(self.memory_tokens)) if self.memory_tokens else 0,
            "median_prompt_tokens": int(statistics.median(self.prompt_tokens)) if self.prompt_tokens else 0,
        }
        return d
