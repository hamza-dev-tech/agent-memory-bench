"""Grading one answer against the gold answer.

Two graders run on every probe and they are meant to disagree sometimes:

  * a normalised string match, which is strict, cheap and never creative;
  * an LLM judge at temperature 0, which catches "7 May 2023" against
    "May 7th, 2023" and "psychology" against "Psychology, counseling".

Where they disagree the probe is written to a review file and a human looks
at it. That is the honest version of "graded by an LLM judge": the judge is
not the last word, it is the thing that flags what to read.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .config import Config
from .llm import Chat

JUDGE_SYSTEM = (
    "You grade short answers against a gold answer from a conversation dataset. "
    "Mark correct if the answer states the same fact as the gold answer, even if the "
    "wording, date format or level of detail differs, and even if it adds a little extra. "
    "Mark incorrect if it states a different fact, contradicts the gold answer, is empty, "
    "or refuses when the gold answer exists. "
    'Reply with JSON only: {"correct": true or false, "why": "at most 15 words"}'
)

_PUNCT = re.compile(r"[^\w\s]")
_SPACE = re.compile(r"\s+")
_FILLER = re.compile(r"\b(the|a|an|on|in|at|of|is|was|to|and)\b")


def normalise(s: str) -> str:
    s = (s or "").strip().lower()
    s = _PUNCT.sub(" ", s)
    s = _FILLER.sub(" ", s)
    return _SPACE.sub(" ", s).strip()


def string_match(answer: str, gold: str) -> bool:
    a, g = normalise(answer), normalise(gold)
    if not a or not g:
        return False
    return a == g or g in a or a in g


@dataclass
class Verdict:
    correct: bool
    by_string: bool
    by_judge: bool | None
    disagreed: bool
    why: str
    abstained: bool


class Judge:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.chat = Chat(cfg, model=cfg.judge_model)

    def grade(self, question: str, gold: str, answer: str, adversarial: bool) -> Verdict:
        abstained = normalise(answer) == normalise(self.cfg.no_answer_token) or not answer.strip()

        # An adversarial probe has no answer in the conversation, so abstaining
        # IS the correct behaviour and anything confident is a hallucination.
        if adversarial:
            return Verdict(
                correct=abstained,
                by_string=abstained,
                by_judge=None,
                disagreed=False,
                why="adversarial probe: abstaining is correct",
                abstained=abstained,
            )

        if abstained:
            return Verdict(False, False, None, False, "abstained on an answerable probe", True)

        by_string = string_match(answer, gold)
        by_judge, why = self._ask(question, gold, answer)
        # The judge decides, the string match is the tripwire.
        return Verdict(
            correct=bool(by_judge),
            by_string=by_string,
            by_judge=by_judge,
            disagreed=by_string != bool(by_judge),
            why=why,
            abstained=False,
        )

    def _ask(self, question: str, gold: str, answer: str) -> tuple[bool, str]:
        user = (
            f"QUESTION: {question}\nGOLD ANSWER: {gold}\nMODEL ANSWER: {answer}\n\nJSON:"
        )
        raw = self.chat.complete(JUDGE_SYSTEM, user, max_tokens=120).text
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            return False, f"judge returned no json: {raw[:60]}"
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return False, f"judge json invalid: {raw[:60]}"
        return bool(data.get("correct")), str(data.get("why", ""))[:120]
