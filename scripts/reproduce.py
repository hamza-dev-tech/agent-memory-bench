#!/usr/bin/env python
"""Re-run the published 2026-09-23 benchmark with the settings it used.

    python scripts/reproduce.py --check                 # offline: is the workload identical?
    python scripts/reproduce.py                         # the open-source systems
    python scripts/reproduce.py --systems mem0 letta    # a subset

Two things decide whether a rerun lands on the published table: the workload
(which conversation, which probes) and the models. Both are fixed here, so a
rerun only differs in what it is meant to measure, the memory systems.

The published run also has two GoodMem rows. They ran on an instance the vendor
supplied, so they are not in the default list: add `goodmem` (and
`goodmem-recommended`) only with your own GoodMem settings in .env.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PUBLISHED = ROOT / "results" / "run-2026-09-23"

# what the published run used, from its RESULTS.md
WORKLOAD = ["--conversations", "1", "--probes", "120", "--adversarial", "28", "--top-k", "10"]
MODEL_ENV = {
    "MEMBENCH_LLM": "gpt-4o-mini",
    "MEMBENCH_JUDGE": "gpt-4o-mini",
    "MEMBENCH_LLM_BASE_URL": "https://api.openai.com/v1",
    "MEMBENCH_LLM_KEY_ENV": "OPENAI_API_KEY",
}
OPEN_SOURCE = ["mem0", "langmem", "zep", "letta"]
# sha256 of data/locomo10.json as fetched for the published run
LOCOMO_SHA256 = "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4"


def check_workload() -> bool:
    """True when this checkout would grade exactly the published probes."""
    from membench import dataset
    from membench.config import CONFIG

    if not CONFIG.dataset_path.exists():
        print(f"missing {CONFIG.dataset_path}: run scripts/fetch_locomo.py first")
        return False

    digest = hashlib.sha256(CONFIG.dataset_path.read_bytes()).hexdigest()
    same_data = digest == LOCOMO_SHA256
    print(f"dataset sha256 {'matches' if same_data else 'DIFFERS: ' + digest}")

    convs = dataset.build(CONFIG.dataset_path, 1, 120, 28, CONFIG.seed)
    # ids are positions within a conversation, so compare the questions too
    selected = {(p.probe_id, p.question) for c in convs for p in c.probes}
    graded = set()
    with open(PUBLISHED / "probes.jsonl", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("kind") == "probe" and rec.get("system") == "mem0":
                graded.add((rec["probe_id"], rec["question"]))
    same_probes = selected == graded
    print(f"probe selection: {len(selected & graded)} of {len(graded)} published probes selected"
          + ("" if same_probes else ", NOT the same set"))
    return same_data and same_probes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--systems", nargs="+", default=OPEN_SOURCE)
    ap.add_argument("--check", action="store_true", help="verify the workload, run nothing, spend nothing")
    ap.add_argument("--run-id", default=f"reproduce-{dt.date.today().isoformat()}")
    args = ap.parse_args()

    if not check_workload():
        print("the workload does not match the published run; stopping before any model call")
        return 1
    if args.check:
        print("workload is identical to the published run")
        return 0

    if not os.environ.get(MODEL_ENV["MEMBENCH_LLM_KEY_ENV"]):
        print(f"{MODEL_ENV['MEMBENCH_LLM_KEY_ENV']} is not set. The published run used "
              f"{MODEL_ENV['MEMBENCH_LLM']} on {MODEL_ENV['MEMBENCH_LLM_BASE_URL']}.")
        return 1

    env = {**os.environ, **MODEL_ENV}
    cmd = [sys.executable, str(ROOT / "scripts" / "run.py"), "--systems", *args.systems,
           *WORKLOAD, "--run-id", args.run_id]
    print("running:", " ".join(cmd[1:]))
    print("models:", ", ".join(f"{k}={v}" for k, v in MODEL_ENV.items()))
    return subprocess.call(cmd, env=env, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
