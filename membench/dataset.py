"""LoCoMo, turned into conversations to ingest and probes to ask.

LoCoMo (Maharana et al., 2024) is a public set of very long multi-session
conversations with human-written question and answer pairs, which is exactly
the shape of the problem an agent memory layer claims to solve. Using it
instead of a private dataset means anyone can rerun this and get the same
questions.

Shape of the file (data/locomo10.json), verified against the published data:
    [ { "sample_id": str,
        "conversation": { "speaker_a": str, "speaker_b": str,
                          "session_1_date_time": "1:56 pm on 8 May, 2023",
                          "session_1": [ {"speaker","dia_id","text"}, ... ],
                          ... },
        "qa": [ {"question","answer","evidence":["D1:3"],"category":int}, ... ] } ]

Categories in LoCoMo: 1 multi-hop, 2 temporal, 3 open-domain knowledge,
4 single-hop, 5 adversarial (the answer is deliberately not in the
conversation). We keep a fixed slice of category 5 because a memory system
that always produces an answer should be penalised for it.
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

ADVERSARIAL_CATEGORY = 5
CATEGORY_NAMES = {
    1: "multi-hop",
    2: "temporal",
    3: "open-domain",
    4: "single-hop",
    5: "adversarial",
}


@dataclass
class Turn:
    speaker: str
    text: str
    dia_id: str
    session: str
    timestamp: str

    def as_memory_text(self) -> str:
        """What gets written into the memory system.

        The speaker and the session date are part of the fact. Strip them and
        every system loses on the temporal questions for a reason that has
        nothing to do with its memory.
        """
        return f"[{self.timestamp}] {self.speaker}: {self.text}"


@dataclass
class Probe:
    probe_id: str
    conversation_id: str
    question: str
    gold: str
    category: int
    evidence: list[str]

    @property
    def is_adversarial(self) -> bool:
        return self.category == ADVERSARIAL_CATEGORY

    @property
    def category_name(self) -> str:
        return CATEGORY_NAMES.get(self.category, str(self.category))


@dataclass
class Conversation:
    conversation_id: str
    speakers: tuple[str, str]
    turns: list[Turn]
    probes: list[Probe]

    @property
    def sessions(self) -> int:
        return len({t.session for t in self.turns})


def _session_sort_key(name: str) -> int:
    m = re.search(r"(\d+)", name)
    return int(m.group(1)) if m else 0


def load_locomo(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python scripts/fetch_locomo.py"
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build(
    path: Path,
    n_conversations: int,
    n_probes: int,
    n_adversarial: int,
    seed: int,
    max_turns: int = 0,
) -> list[Conversation]:
    """Deterministic selection: same seed, same conversations, same probes.

    max_turns is for smoke tests. It cuts each conversation short, and the
    probes are then drawn from what the shortened conversation can actually
    answer. Cutting the turns and leaving the probes alone is how a smoke run
    comes back all misses: the evidence for most questions sits in a later
    session, so the run proves nothing except that the code did not crash.
    LoCoMo tags every question with the dia_ids it rests on, which is what
    makes the narrower pool possible.
    """
    raw = load_locomo(path)
    rng = random.Random(seed)

    # Sort by id first so the sample never depends on file ordering.
    raw = sorted(raw, key=lambda c: str(c.get("sample_id", "")))
    chosen = raw[:n_conversations]

    per_conv = max(1, n_probes // max(1, len(chosen)))
    per_conv_adv = max(0, n_adversarial // max(1, len(chosen)))

    out: list[Conversation] = []
    for conv in chosen:
        cid = str(conv.get("sample_id") or f"conv{len(out) + 1}")
        c = conv["conversation"]
        turns: list[Turn] = []
        for key in sorted([k for k in c if re.fullmatch(r"session_\d+", k)], key=_session_sort_key):
            stamp = c.get(f"{key}_date_time", "")
            for t in c[key]:
                text = (t.get("text") or "").strip()
                if not text:
                    continue  # a few turns are image-only captions
                turns.append(
                    Turn(
                        speaker=t.get("speaker", "?"),
                        text=text,
                        dia_id=t.get("dia_id", ""),
                        session=key,
                        timestamp=stamp,
                    )
                )

        qa = conv.get("qa", [])
        if max_turns:
            turns = turns[:max_turns]
            in_window = {t.dia_id for t in turns}
            # adversarial questions carry no evidence and survive any cut: the
            # right answer is a refusal, which a short conversation can give
            qa = [
                q for q in qa
                if q.get("category") == ADVERSARIAL_CATEGORY
                or (q.get("evidence") and in_window.issuperset(q["evidence"]))
            ]

        normal = [q for q in qa if q.get("category") != ADVERSARIAL_CATEGORY and q.get("answer") is not None]
        adversarial = [q for q in qa if q.get("category") == ADVERSARIAL_CATEGORY]
        rng.shuffle(normal)
        rng.shuffle(adversarial)

        picked = normal[: per_conv - per_conv_adv] + adversarial[:per_conv_adv]
        probes = [
            Probe(
                probe_id=f"{cid}:{i:03d}",
                conversation_id=cid,
                question=str(q["question"]).strip(),
                # answers are sometimes numbers in the source file
                gold=str(q.get("answer", "")).strip(),
                category=int(q.get("category", 0)),
                evidence=list(q.get("evidence", []) or []),
            )
            for i, q in enumerate(picked)
        ]
        out.append(
            Conversation(
                conversation_id=cid,
                speakers=(c.get("speaker_a", "A"), c.get("speaker_b", "B")),
                turns=turns,
                probes=probes,
            )
        )
    return out


def summarise(convs: list[Conversation]) -> dict:
    turns = sum(len(c.turns) for c in convs)
    probes = [p for c in convs for p in c.probes]
    by_cat: dict[str, int] = {}
    for p in probes:
        by_cat[p.category_name] = by_cat.get(p.category_name, 0) + 1
    return {
        "conversations": len(convs),
        "sessions": sum(c.sessions for c in convs),
        "turns": turns,
        "probes": len(probes),
        "probes_by_category": by_cat,
    }
