#!/usr/bin/env python
"""Draw the recall chart at the top of the README from a run log.

    python scripts/chart.py                              # the published run
    python scripts/chart.py results/run-2026-11-30       # a rerun

Reads the summary record each system writes, so the chart cannot drift from
the log. Writes docs/results-light.svg and docs/results-dark.svg, one per
GitHub theme, which the README picks between with <picture>. Standard library
only.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parent.parent
PUBLISHED = ROOT / "results" / "run-2026-09-23"
OUT = ROOT / "docs"

# What the vendors publish for themselves, each on its own method. Drawn as a
# band, not as bars, because it is not the same measurement.
VENDOR_LOW, VENDOR_HIGH = 80.0, 93.0

THEMES = {
    "light": {"text": "#1f2328", "muted": "#59636e", "grid": "#d1d9e0",
              "bar": "#3d5afe", "band": "#f2a900", "band_text": "#8a5a00"},
    "dark": {"text": "#e6edf3", "muted": "#9198a1", "grid": "#3d444d",
             "bar": "#7c8dff", "band": "#e3b341", "band_text": "#e3b341"},
}

FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"


def name(rec: dict) -> str:
    """The row name the README table uses."""
    if rec["system"] == "goodmem":
        return "GoodMem (vendor config)" if rec["track"] == "recommended" else "GoodMem (defaults)"
    base = {"mem0": "Mem0", "langmem": "LangMem", "letta": "Letta", "zep": "Zep / Graphiti"}.get(
        rec["system"], rec.get("label", rec["system"]))
    m = re.search(r"(\d+(?:\.\d+)+)", str(rec.get("version", "")))
    return f"{base} {m.group(1)}" if m else base


def rows(run: Path) -> list[tuple[str, float]]:
    out = []
    with open(run / "probes.jsonl", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("kind") == "summary" and rec.get("answerable"):
                out.append((name(rec), 100.0 * rec["correct"] / rec["answerable"]))
    return sorted(out, key=lambda r: -r[1])


def svg(data: list[tuple[str, float]], run: Path, t: dict) -> str:
    w, label_w, plot_w = 760, 200, 500
    x0 = label_w
    top, row_h, bar_h = 72, 36, 20
    bottom = top + row_h * len(data)
    h = bottom + 62

    def x(pct: float) -> float:
        return x0 + plot_w * pct / 100.0

    p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
         f'font-family="{FONT}" role="img" aria-label="Recall by memory system">']
    p.append(f'<text x="0" y="22" font-size="17" font-weight="600" fill="{t["text"]}">'
             'Recall when every system is measured the same way</text>')
    p.append(f'<text x="0" y="44" font-size="13" fill="{t["muted"]}">'
             'Share of answerable probes answered correctly. Same conversation, same questions, '
             'same answering model.</text>')

    # the vendors' own figures, as a band behind the bars
    p.append(f'<rect x="{x(VENDOR_LOW):.1f}" y="{top - 8}" width="{x(VENDOR_HIGH) - x(VENDOR_LOW):.1f}" '
             f'height="{bottom - top + 8}" fill="{t["band"]}" fill-opacity="0.18" rx="3"/>')
    mid = (x(VENDOR_LOW) + x(VENDOR_HIGH)) / 2
    p.append(f'<text x="{mid:.1f}" y="{top - 14}" font-size="12" text-anchor="middle" '
             f'fill="{t["band_text"]}">vendor-published, {VENDOR_LOW:.0f}-{VENDOR_HIGH:.0f}%</text>')

    for pct in (0, 25, 50, 75, 100):
        gx = x(pct)
        p.append(f'<line x1="{gx:.1f}" y1="{top - 8}" x2="{gx:.1f}" y2="{bottom}" '
                 f'stroke="{t["grid"]}" stroke-width="1"/>')
        p.append(f'<text x="{gx:.1f}" y="{bottom + 18}" font-size="12" text-anchor="middle" '
                 f'fill="{t["muted"]}">{pct}%</text>')

    for i, (label, pct) in enumerate(data):
        cy = top + i * row_h + row_h / 2
        p.append(f'<text x="{x0 - 12}" y="{cy + 4.5:.1f}" font-size="13.5" text-anchor="end" '
                 f'fill="{t["text"]}">{escape(label)}</text>')
        p.append(f'<rect x="{x0}" y="{cy - bar_h / 2:.1f}" width="{x(pct) - x0:.1f}" height="{bar_h}" '
                 f'rx="3" fill="{t["bar"]}"/>')
        p.append(f'<text x="{x(pct) + 8:.1f}" y="{cy + 4.5:.1f}" font-size="13.5" font-weight="600" '
                 f'fill="{t["text"]}">{pct:.1f}%</text>')

    src = f"results/{run.name}/probes.jsonl"
    p.append(f'<text x="0" y="{h - 8}" font-size="12" fill="{t["muted"]}">'
             f'gpt-4o-mini at temperature 0, top 10 retrieved. Drawn from {escape(src)} '
             'by scripts/chart.py.</text>')
    p.append("</svg>")
    return "\n".join(p) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?", type=Path, default=PUBLISHED)
    args = ap.parse_args()
    run = args.run if args.run.is_absolute() else Path.cwd() / args.run
    data = rows(run)
    if not data:
        print(f"no summary records in {run / 'probes.jsonl'}")
        return 1
    for theme, colours in THEMES.items():
        path = OUT / f"results-{theme}.svg"
        path.write_text(svg(data, run, colours), encoding="utf-8")
        print(f"wrote {path.relative_to(ROOT)}")
    for label, pct in data:
        print(f"  {label:26} {pct:5.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
