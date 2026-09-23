#!/usr/bin/env python
"""Make Letta 0.11.7's archival memory writable on SQLite.

    .venv-letta/Scripts/python scripts/patch_letta.py

One line, and the run cannot happen without it. letta/orm/archives_agents.py
declares

    created_at: Mapped[datetime] = mapped_column(..., server_default="now()")

which is a Postgres function passed as a raw string. SQLite has no now(), so it
stores the six characters and the next read fails on

    Invalid isoformat string: 'now()'

That is the first archival insert, so nothing can be written at all. sqlalchemy's
func.now() is the same intent expressed in a way the dialect resolves: it renders
now() on Postgres and CURRENT_TIMESTAMP on SQLite.

The patch touches a column default and nothing that decides what Letta stores,
retrieves or ranks, but it is still a change to the system under test, so it
lives in a script that says what it does rather than in somebody's shell
history, and the results table carries a note. Re-running it is harmless.

Run it with the Letta virtualenv's interpreter, not the harness one.
"""
from __future__ import annotations

import sys
from pathlib import Path

TARGET = Path("letta/orm/archives_agents.py")
OLD = 'server_default="now()"'
NEW = "server_default=func.now()"
IMPORT = "from sqlalchemy import func"


def find_installed() -> Path:
    for entry in sys.path:
        candidate = Path(entry) / TARGET
        if candidate.is_file():
            return candidate
    raise SystemExit(
        "letta is not installed in this interpreter. Run this with the Letta "
        "virtualenv: .venv-letta/Scripts/python scripts/patch_letta.py"
    )


def main() -> int:
    path = find_installed()
    src = path.read_text(encoding="utf-8")

    if OLD not in src:
        if NEW in src:
            print(f"already patched: {path}")
            return 0
        raise SystemExit(
            f"{path} does not contain {OLD!r}. This patch is written against letta "
            "0.11.7; check whether the installed version still needs it."
        )

    src = src.replace(OLD, NEW)
    if IMPORT not in src:
        # after the last existing import, so the file still reads top to bottom
        lines = src.splitlines()
        last = max(i for i, ln in enumerate(lines) if ln.startswith(("import ", "from ")))
        lines.insert(last + 1, IMPORT)
        src = "\n".join(lines) + "\n"

    path.write_text(src, encoding="utf-8")
    print(f"patched {path}\n  {OLD}  ->  {NEW}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
