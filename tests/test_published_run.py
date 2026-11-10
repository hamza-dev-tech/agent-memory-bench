"""The published run has to stay reproducible from the instructions we give.

The 2026-09-23 table came from `--conversations 1 --probes 120 --adversarial 28`.
The README and the write-up used to drop `--adversarial 28`, which silently falls
back to the default of 10 and draws a different probe set: 102 of the 120
questions overlap, so anyone following them could never land on our numbers.
These tests pin the selection to the probes that were actually graded.
"""
from __future__ import annotations

import json
import sys
from dataclasses import fields
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from membench import dataset
from membench.config import CONFIG

PUBLISHED = ROOT / "results" / "run-2026-09-23"
# the exact workload flags of the published run
CONVERSATIONS, PROBES, ADVERSARIAL = 1, 120, 28


def _published_probes(system: str) -> set[tuple[str, str]]:
    """(probe_id, question) pairs. Ids alone are positions, so any 120 probes
    would share them; the question is what pins the selection."""
    ids = set()
    with open(PUBLISHED / "probes.jsonl", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("kind") == "probe" and rec.get("system") == system and rec.get("track") == "baseline":
                ids.add((rec["probe_id"], rec["question"]))
    return ids


needs_dataset = pytest.mark.skipif(
    not CONFIG.dataset_path.exists(), reason="run scripts/fetch_locomo.py first"
)


@needs_dataset
def test_published_flags_select_exactly_the_graded_probes():
    convs = dataset.build(CONFIG.dataset_path, CONVERSATIONS, PROBES, ADVERSARIAL, CONFIG.seed)
    selected = {(p.probe_id, p.question) for c in convs for p in c.probes}
    for system in ("mem0", "langmem", "zep", "letta", "goodmem"):
        assert selected == _published_probes(system), system


@needs_dataset
def test_the_default_adversarial_count_does_not_reproduce_it():
    # Guards the old instructions: if this ever starts passing, the defaults
    # changed and the README needs a second look.
    # Read the declared default, not CONFIG.adversarial_probes: other tests set
    # fields on the shared CONFIG object while they run.
    default_adv = next(f.default for f in fields(type(CONFIG)) if f.name == "adversarial_probes")
    convs = dataset.build(CONFIG.dataset_path, CONVERSATIONS, PROBES, default_adv, CONFIG.seed)
    selected = {(p.probe_id, p.question) for c in convs for p in c.probes}
    assert selected != _published_probes("mem0")


def test_the_documented_command_carries_every_workload_flag():
    flags = f"--conversations {CONVERSATIONS} --probes {PROBES} --adversarial {ADVERSARIAL}"
    for doc in ("README.md", "docs/ARTICLE.md"):
        text = (ROOT / doc).read_text(encoding="utf-8")
        assert flags in text, f"{doc} does not give the published workload flags"
