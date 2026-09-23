"""Writes results.json and RESULTS.md from the summaries."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .config import Config
from .metrics import ProbeRecord, SystemSummary

COLUMNS = [
    "System", "Track", "Recall", "Refused adversarial",
    "Search p50", "Search p95", "Memory tokens", "Ingest/turn", "Judge disagreements",
]


def markdown_table(rows: list[dict]) -> str:
    head = "| " + " | ".join(COLUMNS) + " |"
    rule = "| " + " | ".join("---" for _ in COLUMNS) + " |"
    body = ["| " + " | ".join(str(r.get(c, "")) for c in COLUMNS) + " |" for r in rows]
    return "\n".join([head, rule, *body])


def write(cfg: Config, summaries: list[SystemSummary], dataset_info: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    payload = {
        "run_id": cfg.run_id,
        "generated_at": stamp,
        "config": cfg.as_dict(),
        "dataset": dataset_info,
        "systems": [s.as_dict() for s in summaries],
    }
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    rows = [s.row() for s in summaries]
    disagreements = sum(s.judge_disagreements for s in summaries)
    md = f"""# Results

Run `{cfg.run_id}`, {stamp}.

{markdown_table(rows)}

## What the columns mean

- **Recall**: share of answerable probes the shared model got right, using only
  what that system retrieved. Graded by an LLM judge at temperature 0, with a
  normalised string match as a second opinion.
- **Refused adversarial**: LoCoMo category 5 questions have no answer in the
  conversation. Refusing is correct; answering is a hallucination.
- **Search p50 / p95**: time in the memory system's own search call. The
  answering model is not included, since it is the same for everyone.
- **Memory tokens**: median size of the injected memory, estimated the same way
  for every system. Cheaper is better at equal recall.
- **Ingest/turn**: wall clock per conversation turn written, including any
  extraction the system does.
- **Judge disagreements**: probes where the judge and the string match differed.
  These are the ones read by hand; see `review.csv`.

## Setup

- Answering model: `{cfg.llm_model}` at temperature {cfg.llm_temperature}, same for every system.
- Embeddings: `{cfg.embed_model}`, run locally for the self-hosted systems and on
  the vendor's own hosting for GoodMem. Same model, different host.
- Workload: {dataset_info.get('conversations')} LoCoMo conversations,
  {dataset_info.get('turns')} turns, {dataset_info.get('probes')} probes
  ({dataset_info.get('probes_by_category')}).
- Retrieval depth: top {cfg.top_k} for every system.

{disagreements} probe(s) needed a human look.

Raw per-probe records are in `probes.jsonl`; every number above can be recomputed from it.
"""
    (out_dir / "RESULTS.md").write_text(md, encoding="utf-8")
    return out_dir / "RESULTS.md"


def write_review_csv(records: list[ProbeRecord], out_dir: Path) -> Path | None:
    """The probes where the graders disagreed, for a human to settle."""
    rows = [r for r in records if r.disagreed]
    if not rows:
        return None
    import csv

    path = out_dir / "review.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["system", "probe_id", "category", "question", "gold", "answer", "string_match", "judge", "why"])
        for r in rows:
            w.writerow([r.system, r.probe_id, r.category, r.question, r.gold, r.answer, r.by_string, r.by_judge, r.why])
    return path
