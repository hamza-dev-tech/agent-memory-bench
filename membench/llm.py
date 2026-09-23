"""The one model that answers every probe, for every system.

Free tiers rate limit hard, so the client waits out 429s rather than failing
the run: a benchmark that dies at probe 41 of 60 is worse than one that takes
an hour. Every call is temperature 0 and the usage numbers come back with the
text so the report can state real token counts instead of guesses.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass

import httpx

from .config import Config


@dataclass
class Completion:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    model: str


class Chat:
    def __init__(self, cfg: Config, model: str | None = None):
        self.cfg = cfg
        self.model = model or cfg.llm_model
        self.api_key = cfg.require_llm_key()
        self._client = httpx.Client(base_url=cfg.llm_base_url, timeout=120.0)
        self.calls = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def complete(self, system: str, user: str, max_tokens: int | None = None) -> Completion:
        body = {
            "model": self.model,
            "temperature": self.cfg.llm_temperature,
            "max_tokens": max_tokens or self.cfg.llm_max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            # Groq sits behind Cloudflare, which answers a default Python user
            # agent with a 403 (error 1010). Any real one is fine.
            "User-Agent": "agent-memory-bench/0.1",
            "Accept": "application/json",
        }
        delay = 2.0
        for attempt in range(8):
            start = time.perf_counter()
            try:
                r = self._client.post("/chat/completions", json=body, headers=headers)
            except httpx.RequestError:
                if attempt == 7:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 90)
                continue

            if r.status_code == 200:
                payload = r.json()
                usage = payload.get("usage") or {}
                self.calls += 1
                self.total_prompt_tokens += usage.get("prompt_tokens", 0)
                self.total_completion_tokens += usage.get("completion_tokens", 0)
                return Completion(
                    text=payload["choices"][0]["message"]["content"].strip(),
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    latency_s=time.perf_counter() - start,
                    model=payload.get("model", self.model),
                )

            # 429 means the free tier is doing its job. Wait it out rather than
            # losing an hour of run.
            if r.status_code in (429, 500, 502, 503, 504) and attempt < 7:
                wait = float(r.headers.get("retry-after") or 0) or delay
                time.sleep(min(wait, 90))
                delay = min(delay * 2, 90)
                continue
            raise RuntimeError(f"LLM call failed ({r.status_code}): {r.text[:400]}")
        raise RuntimeError("LLM call failed after retries")


ANSWER_SYSTEM = (
    "You answer questions about two people's past conversations using only the notes provided. "
    "The notes are the only thing you know. Answer with the shortest factual answer: a name, a "
    "date, a place, a short phrase. Do not explain, do not add a sentence around it. "
    "If the notes do not contain the answer, reply with exactly: NO ANSWER"
)


def answer_prompt(context: str, question: str) -> str:
    notes = context.strip() or "(no notes were retrieved)"
    return f"NOTES\n{notes}\n\nQUESTION\n{question}\n\nSHORT ANSWER:"
