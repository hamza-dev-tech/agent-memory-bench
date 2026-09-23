#!/usr/bin/env python
"""Download the LoCoMo conversations into data/.

LoCoMo is published by Snap Research alongside the paper "Evaluating Very
Long-Term Conversational Memory of LLM Agents" (Maharana et al., 2024).
"""
from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
DEST = Path(__file__).resolve().parent.parent / "data" / "locomo10.json"


def main() -> int:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if DEST.exists():
        print(f"{DEST} already here ({DEST.stat().st_size / 1e6:.1f} MB)")
        return 0
    print(f"downloading {URL}")
    urllib.request.urlretrieve(URL, DEST)
    digest = hashlib.sha256(DEST.read_bytes()).hexdigest()
    print(f"saved {DEST} ({DEST.stat().st_size / 1e6:.1f} MB)")
    print(f"sha256 {digest}")
    print("record that hash in the write-up so the workload is pinned")
    return 0


if __name__ == "__main__":
    sys.exit(main())
