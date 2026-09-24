#!/usr/bin/env python
"""Put a band around recall instead of pretending to a single number.

    python scripts/adjudicate.py results/run-2026-09-23/probes.jsonl

Reading the 125 probes where the two graders disagreed turned up two habits in
the LLM judge, both of which hit every system the same way:

* It accepts dates that are simply wrong. Gold "The Friday before 22 October
  2023" against an answer of 2023-10-13 was marked correct; the 22nd was a
  Sunday, so the Friday before it is the 20th, and the 13th is a week out.
* It rules inconsistently on partial answers. "A horse painting" for a gold of
  "Horse, sunset, sunrise" was accepted. "Charlotte's Web" for a gold of
  "Nothing is Impossible", "Charlotte's Web" was rejected. One of three passed
  and one of two failed.

Neither changes the ranking, because the judge is equally generous to
everybody. Both make a single recall figure sound more precise than it is.

So this recomputes recall two ways and reports the gap:

  lenient   what the judge said, which is the number in RESULTS.md
  strict    the judge, minus probes it got demonstrably or arguably wrong:
            a relative date resolved to the wrong day, or a multi-item gold
            answered with only some of its items

The truth is inside that band. Publishing the band, and saying what is in the
gap, is the honest version of a number that cannot be exact.

Both rules are mechanical. No model is asked anything here, so the adjudication
itself cannot drift between runs.
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

WEEKDAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}
MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}

# "The Friday before 22 October 2023", "the sunday before 25 May 2023"
RELATIVE_DAY = re.compile(
    r"\bthe\s+(" + "|".join(WEEKDAYS) + r")\s+before\s+(\d{1,2})\s+(" + "|".join(MONTHS) + r")\s+(\d{4})",
    re.I,
)


def _gold_dates(gold: str) -> list[date] | None:
    """The exact day a "the <weekday> before <date>" gold refers to.

    None when the gold is not that shape, which includes the vaguer ones like
    "two weekends before": those are left to the judge rather than guessed at,
    since a rule that has to invent an interpretation is not a check.
    """
    m = RELATIVE_DAY.search(gold or "")
    if not m:
        return None
    weekday, day, month, year = m.group(1).lower(), int(m.group(2)), m.group(3).lower(), int(m.group(4))
    anchor = date(year, MONTHS[month], day)
    back = (anchor.weekday() - WEEKDAYS[weekday]) % 7 or 7
    return [anchor - timedelta(days=back)]


def _answer_dates(answer: str) -> set[date]:
    """Every date the answer states, in any of the shapes the model uses."""
    out: set[date] = set()
    text = answer or ""
    for y, mo, d in re.findall(r"(\d{4})-(\d{1,2})-(\d{1,2})", text):
        try:
            out.add(date(int(y), int(mo), int(d)))
        except ValueError:
            pass
    pattern = r"(\d{1,2})\s+(" + "|".join(MONTHS) + r")[,\s]+(\d{4})"
    for d, mo, y in re.findall(pattern, text, re.I):
        try:
            out.add(date(int(y), MONTHS[mo.lower()], int(d)))
        except ValueError:
            pass
    pattern = r"(" + "|".join(MONTHS) + r")\s+(\d{1,2})[,\s]+(\d{4})"
    for mo, d, y in re.findall(pattern, text, re.I):
        try:
            out.add(date(int(y), MONTHS[mo.lower()], int(d)))
        except ValueError:
            pass
    return out


def _words(text: str) -> set[str]:
    """Word stems, so read/reading and paint/painting are the same word.

    Crude on purpose: four characters is enough to survive English inflection
    without a stemmer, and a rule that needs a library is a rule nobody
    re-checks.
    """
    return {w[:4] for w in re.findall(r"[a-z0-9']+", (text or "").lower()) if len(w) > 2}


# a part carrying a pronoun or a copula is a clause, not a list item
PROSE = re.compile(r"\b(is|are|was|were|be|she|he|they|her|his|their|it|that|who)\b", re.I)


def _gold_items(gold: str) -> list[str]:
    """A multi-item gold split into its items, or [] if it is not a list.

    Commas only. Splitting on "and" turns prose into demands the gold never
    made: "Read a book and paint" is one answer phrased with a conjunction and
    "Grateful and thankful for her family" is a single sentiment, and an
    earlier version of this rule marked both incomplete.

    Short parts, none of them clauses, for the same reason: "Yes, she is
    supportive" is a sentence that happens to contain a comma.
    """
    text = (gold or "").strip().rstrip(".")
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) < 2:
        return []
    if any(len(p.split()) > 3 or PROSE.search(p) for p in parts):
        return []
    return parts


def contested(rec: dict) -> str | None:
    """Why this probe's verdict is arguable, or None if it is not."""
    if rec.get("adversarial") or not rec.get("correct"):
        return None

    gold, answer = rec.get("gold", ""), rec.get("answer", "")

    wanted = _gold_dates(gold)
    if wanted:
        got = _answer_dates(answer)
        if got and not (got & set(wanted)):
            return "date"

    items = _gold_items(gold)
    if items:
        have = _words(answer)
        missing = [i for i in items if not (_words(i) & have)]
        if missing:
            return "partial"
    return None


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "results/run-2026-09-23/probes.jsonl")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    probes = [r for r in rows if r.get("kind") == "probe"]

    stats: dict[tuple[str, str], dict] = defaultdict(
        lambda: {"answerable": 0, "lenient": 0, "date": 0, "partial": 0}
    )
    examples: dict[str, list] = {"date": [], "partial": []}

    for r in probes:
        key = (r["system"], r.get("track", ""))
        if r.get("adversarial"):
            continue
        s = stats[key]
        s["answerable"] += 1
        if not r.get("correct"):
            continue
        s["lenient"] += 1
        why = contested(r)
        if why:
            s[why] += 1
            if len(examples[why]) < 4:
                examples[why].append((r["gold"], r["answer"]))

    print(f"{'system':28s} {'lenient':>9s} {'strict':>9s} {'band':>8s}   contested")
    for (system, track), s in sorted(stats.items()):
        n, lenient = s["answerable"], s["lenient"]
        strict = lenient - s["date"] - s["partial"]
        label = f"{system} ({track})" if track != "baseline" else system
        print(
            f"{label:28s} {100 * lenient / n:8.1f}% {100 * strict / n:8.1f}% "
            f"{100 * (lenient - strict) / n:7.1f}pt   {s['date']} date, {s['partial']} partial"
        )

    for why, rows_ in examples.items():
        print(f"\n{why}:")
        for gold, answer in rows_:
            print(f"   gold   {gold[:66]}")
            print(f"   answer {answer[:66]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
