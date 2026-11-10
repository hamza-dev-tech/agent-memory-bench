#!/usr/bin/env python
"""Has any measured system shipped a release since the published run?

    python scripts/check_versions.py
    python scripts/check_versions.py --markdown report.md

The versions that were measured come from the run log itself (the summary
record each system writes), not from a hand-kept list, so this can only
disagree with the table if the table is out of date. The latest versions come
from PyPI. Standard library only, so it runs in CI without installing the
memory systems.

A newer release does not mean the table is wrong, only that it describes an
older version, which is worth knowing before anyone quotes it.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "results" / "run-2026-09-23" / "probes.jsonl"

# system key in the log -> the PyPI package its version string names
PACKAGES = {
    "mem0": "mem0ai",
    "langmem": "langmem",
    "letta": "letta",
    "zep": "graphiti-core",
    "goodmem": "goodmem",
}


def measured_versions(log: Path = LOG) -> dict[str, str]:
    """{system: version} from the baseline summary records of a run log."""
    out: dict[str, str] = {}
    with open(log, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("kind") != "summary" or rec.get("track") != "baseline":
                continue
            system = rec.get("system")
            if system in PACKAGES:
                # e.g. "mem0ai 2.1.0", "letta server 0.11.7", "langmem 0.0.30, langgraph 1.2.12"
                m = re.search(r"(\d+(?:\.\d+)+)", str(rec.get("version", "")))
                if m:
                    out[system] = m.group(1)
    return out


def parse(v: str) -> tuple[int, ...]:
    """Numeric release parts only: '0.11.7' -> (0, 11, 7), '1.0.0rc1' -> (1, 0, 0)."""
    return tuple(int(p) for p in re.findall(r"\d+", v.split("+")[0].split("rc")[0].split("a")[0].split("b")[0])[:4])


def latest_on_pypi(package: str) -> str:
    with urllib.request.urlopen(f"https://pypi.org/pypi/{package}/json", timeout=30) as r:
        return json.load(r)["info"]["version"]


def compare(measured: dict[str, str], latest: dict[str, str]) -> list[dict]:
    rows = []
    for system, was in sorted(measured.items()):
        now = latest.get(system)
        rows.append({
            "system": system,
            "package": PACKAGES[system],
            "measured": was,
            "latest": now or "unknown",
            "newer": bool(now) and parse(now) > parse(was),
        })
    return rows


def render(rows: list[dict]) -> str:
    newer = [r for r in rows if r["newer"]]
    lines = [
        "Measured systems with a release newer than the published run "
        "(`results/run-2026-09-23`). The table still describes the versions it measured.",
        "",
        "| System | Package | Measured | Latest |",
        "| --- | --- | --- | --- |",
    ]
    for r in newer:
        lines.append(f"| {r['system']} | `{r['package']}` | {r['measured']} | {r['latest']} |")
    lines += ["", "Rerun with `python scripts/reproduce.py --systems "
              + " ".join(r["system"] for r in newer if r["system"] != "goodmem")
              + "` to refresh these rows. Opened by `.github/workflows/version-watch.yml`."]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--markdown", type=Path, help="write the issue body here when something is newer")
    args = ap.parse_args()

    measured = measured_versions()
    latest = {}
    for system in measured:
        try:
            latest[system] = latest_on_pypi(PACKAGES[system])
        except Exception as e:  # one unreachable package should not hide the others
            print(f"!! could not read {PACKAGES[system]} from PyPI: {e}")
    rows = compare(measured, latest)
    for r in rows:
        flag = "NEWER" if r["newer"] else "same"
        print(f"{r['system']:8} {r['package']:14} measured {r['measured']:9} latest {r['latest']:9} {flag}")

    any_newer = any(r["newer"] for r in rows)
    if any_newer and args.markdown:
        args.markdown.write_text(render(rows), encoding="utf-8")
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as f:
            f.write(f"newer={'true' if any_newer else 'false'}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
