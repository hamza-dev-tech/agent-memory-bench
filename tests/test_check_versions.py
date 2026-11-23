"""The version watch, offline: no PyPI calls."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import check_versions as cv


def test_measured_versions_come_from_the_published_log():
    assert cv.measured_versions() == {
        "goodmem": "0.1.34",
        "langmem": "0.0.30",
        "letta": "0.11.7",
        "mem0": "2.1.0",
        "zep": "0.30.2",
    }


def test_version_order_is_numeric_not_textual():
    assert cv.parse("0.11.10") > cv.parse("0.11.7")
    assert cv.parse("2.1.0") == cv.parse("2.1.0rc1")
    assert cv.parse("1.0") > cv.parse("0.99.9")


def test_only_newer_releases_are_flagged_and_reported():
    rows = cv.compare(
        {"mem0": "2.1.0", "letta": "0.11.7", "zep": "0.30.2"},
        {"mem0": "2.2.0", "letta": "0.11.7"},
    )
    by = {r["system"]: r for r in rows}
    assert by["mem0"]["newer"] and not by["letta"]["newer"]
    assert by["zep"]["latest"] == "unknown" and not by["zep"]["newer"]
    body = cv.render(rows)
    assert "| mem0 | `mem0ai` | 2.1.0 | 2.2.0 |" in body
    assert "letta" not in body.split("Rerun")[0]
    assert "Letta Code" not in body


def test_letta_is_read_from_the_server_line_not_letta_code():
    f = [{"yanked": False}]
    releases = {
        "0.11.7": f, "0.16.7": f, "0.16.8": f,
        "0.17.0rc1": f,                   # pre-release: skipped
        "0.16.9": [{"yanked": True}],     # yanked: skipped
        "0.20.0": [],                     # no files: skipped
        "0.32.0": f, "0.34.12": f,        # Letta Code: over the line
    }
    assert cv.newest_below(releases, cv.SERVER_BELOW["letta"]) == "0.16.8"
    assert cv.newest_below({"0.33.0": f}, cv.SERVER_BELOW["letta"]) is None
    body = cv.render(cv.compare({"letta": "0.11.7"}, {"letta": "0.16.8"}))
    assert "| letta | `letta` | 0.11.7 | 0.16.8 |" in body and "Letta Code" in body
