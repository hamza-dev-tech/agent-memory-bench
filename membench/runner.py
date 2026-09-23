"""Runs one memory system over the workload and grades what comes back.

Order matters: ingest a whole conversation, flush, then ask. Asking while a
system is still indexing punishes whoever indexes asynchronously, which is a
property of the harness rather than of the product.

Everything is written to a JSONL log as it happens. Free-tier rate limits mean
a full run takes hours and something will eventually fail; the log lets you
resume instead of starting again.
"""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

from .adapters.base import MemorySystem
from .config import Config
from .dataset import Conversation
from .judge import Judge
from .llm import Chat, ANSWER_SYSTEM, answer_prompt
from .metrics import ProbeRecord, SystemSummary, approx_tokens


class Runner:
    def __init__(self, cfg: Config, chat: Chat, judge: Judge, log_path: Path):
        self.cfg = cfg
        self.chat = chat
        self.judge = judge
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._done = self._load_done()

    def _load_done(self) -> set[tuple[str, str]]:
        """(system, probe_id) pairs already graded, so a resumed run skips them."""
        done: set[tuple[str, str]] = set()
        if self.log_path.exists():
            with open(self.log_path, encoding="utf-8") as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if r.get("kind") == "probe":
                        done.add((r["system"], r["probe_id"]))
        return done

    def _write(self, kind: str, payload: dict) -> None:
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"kind": kind, "ts": time.time(), **payload}, ensure_ascii=False) + "\n")

    def run_system(self, system: MemorySystem, conversations: list[Conversation]) -> SystemSummary:
        summary = SystemSummary(
            system=system.key, label=system.label, track=system.track,
            version=system.version, config_notes=system.config_notes,
        )
        print(f"\n=== {system.label} [{system.track}]")
        system.setup()
        try:
            for conv in conversations:
                self._one_conversation(system, conv, summary)
        finally:
            system.teardown()
        self._write("summary", summary.as_dict())
        return summary

    def _one_conversation(self, system: MemorySystem, conv: Conversation, summary: SystemSummary) -> None:
        run_key = f"{self.cfg.run_id}-{system.key}-{conv.conversation_id}"
        pending = [p for p in conv.probes if (system.key, p.probe_id) not in self._done]
        if not pending:
            print(f"  {conv.conversation_id}: already graded, skipping")
            return

        print(f"  {conv.conversation_id}: {len(conv.turns)} turns, {len(pending)} probes")
        system.reset(run_key)

        t0 = time.perf_counter()
        failures = 0
        for i, turn in enumerate(conv.turns, 1):
            try:
                system.add(run_key, turn.as_memory_text(), {
                    "speaker": turn.speaker,
                    "session": turn.session,
                    "timestamp": turn.timestamp,
                    "dia_id": turn.dia_id,
                })
            except Exception as e:  # one bad turn should not kill a 600 turn run
                failures += 1
                summary.errors.append(f"add {turn.dia_id}: {e}")
                if failures > 20:
                    raise
            if i % 100 == 0:
                print(f"    ingested {i}/{len(conv.turns)}")
        system.flush(run_key)
        ingest_s = time.perf_counter() - t0

        summary.ingest_turns += len(conv.turns)
        summary.ingest_seconds += ingest_s
        summary.ingest_failures += failures
        self._write("ingest", {
            "system": system.key, "conversation": conv.conversation_id,
            "turns": len(conv.turns), "seconds": round(ingest_s, 2), "failures": failures,
        })
        print(f"    ingest took {ingest_s / 60:.1f} min ({failures} failures)")

        for n, probe in enumerate(pending, 1):
            try:
                rec = self._one_probe(system, conv, probe, run_key)
            except Exception as e:
                summary.errors.append(f"probe {probe.probe_id}: {e}")
                self._write("error", {
                    "system": system.key, "probe_id": probe.probe_id,
                    "error": str(e), "trace": traceback.format_exc()[-800:],
                })
                continue
            summary.add(rec)
            self._done.add((system.key, probe.probe_id))
            self._write("probe", rec.as_dict())
            mark = "ok " if rec.correct else "MISS"
            print(f"    [{n}/{len(pending)}] {mark} {probe.category_name:<12} {probe.question[:58]}")

    def _one_probe(self, system: MemorySystem, conv: Conversation, probe, run_key: str) -> ProbeRecord:
        found = system.search(run_key, probe.question, self.cfg.top_k)
        completion = self.chat.complete(ANSWER_SYSTEM, answer_prompt(found.context, probe.question))
        verdict = self.judge.grade(probe.question, probe.gold, completion.text, probe.is_adversarial)
        return ProbeRecord(
            system=system.key,
            track=system.track,
            conversation_id=conv.conversation_id,
            probe_id=probe.probe_id,
            category=probe.category_name,
            adversarial=probe.is_adversarial,
            question=probe.question,
            gold=probe.gold,
            answer=completion.text,
            correct=verdict.correct,
            by_string=verdict.by_string,
            by_judge=verdict.by_judge,
            disagreed=verdict.disagreed,
            why=verdict.why,
            abstained=verdict.abstained,
            search_latency_s=found.latency_s,
            answer_latency_s=completion.latency_s,
            context_chars=len(found.context),
            context_chunks=len(found.chunks),
            memory_tokens=approx_tokens(found.context),
            prompt_tokens=completion.prompt_tokens,
        )
