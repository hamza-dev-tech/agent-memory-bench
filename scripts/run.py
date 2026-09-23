#!/usr/bin/env python
"""Run the benchmark.

    python scripts/run.py --systems goodmem mem0 --conversations 1 --probes 20

Start small. A full run over two conversations is a few thousand writes per
system and free-tier rate limits make that an afternoon, not a coffee break.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from membench import dataset, report
from membench.config import CONFIG
from membench.judge import Judge
from membench.llm import Chat
from membench.metrics import ProbeRecord
from membench.runner import Runner


def build_system(name: str, cfg):
    if name == "goodmem":
        from membench.adapters.goodmem import GoodMemSystem
        return GoodMemSystem(cfg, track="baseline")
    if name == "goodmem-recommended":
        from membench.adapters.goodmem import GoodMemSystem
        return GoodMemSystem(cfg, track="recommended")
    if name == "mem0":
        from membench.adapters.mem0_adapter import Mem0System
        return Mem0System(cfg)
    if name == "langmem":
        from membench.adapters.langmem_adapter import LangMemSystem
        return LangMemSystem(cfg)
    if name == "letta":
        from membench.adapters.letta_adapter import LettaSystem
        return LettaSystem(cfg)
    if name in ("zep", "graphiti"):
        from membench.adapters.zep_adapter import ZepGraphitiSystem
        return ZepGraphitiSystem(cfg)
    raise SystemExit(f"unknown system: {name}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+", default=list(CONFIG.systems))
    ap.add_argument("--conversations", type=int, default=CONFIG.conversations)
    ap.add_argument("--probes", type=int, default=CONFIG.probes)
    ap.add_argument("--adversarial", type=int, default=CONFIG.adversarial_probes)
    ap.add_argument("--top-k", type=int, default=CONFIG.top_k)
    ap.add_argument("--run-id", default=CONFIG.run_id)
    ap.add_argument("--max-turns", type=int, default=0,
                    help="truncate each conversation, for smoke tests only; recall will be meaningless")
    args = ap.parse_args()

    cfg = CONFIG
    cfg.conversations, cfg.probes = args.conversations, args.probes
    cfg.adversarial_probes, cfg.top_k, cfg.run_id = args.adversarial, args.top_k, args.run_id

    convs = dataset.build(cfg.dataset_path, cfg.conversations, cfg.probes, cfg.adversarial_probes, cfg.seed)
    if args.max_turns:
        for c in convs:
            c.turns = c.turns[: args.max_turns]
        print(f"!! smoke mode: {args.max_turns} turns per conversation, numbers are not comparable")
    info = dataset.summarise(convs)
    info["truncated_to_turns"] = args.max_turns or None
    print("workload:", json.dumps(info))

    out_dir = cfg.results_dir / cfg.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    chat, judge = Chat(cfg), Judge(cfg)
    runner = Runner(cfg, chat, judge, out_dir / "probes.jsonl")

    summaries = []
    for name in args.systems:
        system = build_system(name, cfg)
        try:
            summaries.append(runner.run_system(system, convs))
        except Exception as e:
            print(f"!! {name} failed: {e}")

    records = []
    with open(out_dir / "probes.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("kind") == "probe":
                r.pop("kind", None), r.pop("ts", None)
                records.append(ProbeRecord(**r))

    path = report.write(cfg, summaries, info, out_dir)
    review = report.write_review_csv(records, out_dir)
    print(f"\nwrote {path}")
    if review:
        print(f"wrote {review}: read these before publishing anything")
    print(f"llm calls: {chat.calls + judge.chat.calls}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
