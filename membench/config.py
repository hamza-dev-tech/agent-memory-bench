"""Run configuration. Every knob that could change a number lives here, and
the report prints this object verbatim so a reader can reproduce the run.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"


def _env(name: str, default: str | None = None) -> str | None:
    v = os.environ.get(name, default)
    return v.strip() if isinstance(v, str) else v


@dataclass
class Config:
    # ---- shared models: the same for every system in the baseline track ----
    # Groq's free tier serves gpt-oss-120b, an open-weights model, so the whole
    # benchmark costs nothing to run. Temperature 0 everywhere; a benchmark that
    # samples is a benchmark that cannot be reproduced.
    llm_model: str = _env("MEMBENCH_LLM", "openai/gpt-oss-120b")
    llm_base_url: str = _env("MEMBENCH_LLM_BASE_URL", "https://api.groq.com/openai/v1")
    llm_api_key_env: str = "GROQ_API_KEY"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 256

    # nomic-embed-text-v1.5 is what GoodMem's default space runs on Fireworks,
    # so the local systems run the same weights through transformers.
    # The model is asymmetric: documents and queries take different prefixes.
    embed_model: str = _env("MEMBENCH_EMBED", "nomic-ai/nomic-embed-text-v1.5")
    embed_dim: int = 768
    embed_doc_prefix: str = "search_document: "
    embed_query_prefix: str = "search_query: "

    # ---- workload ----
    dataset_path: Path = DATA_DIR / "locomo10.json"
    conversations: int = int(_env("MEMBENCH_CONVERSATIONS", "2"))
    probes: int = int(_env("MEMBENCH_PROBES", "60"))
    # LoCoMo category 5 is adversarial: the answer is not in the conversation.
    # Keeping some is the only way to catch a system that always says something.
    adversarial_probes: int = int(_env("MEMBENCH_ADVERSARIAL", "10"))
    top_k: int = int(_env("MEMBENCH_TOP_K", "10"))
    seed: int = 20260923

    # ---- how much loss makes a number meaningless ----
    # A system that could not store the conversation will still answer probes,
    # abstain on most of them, and produce a recall number that looks like a
    # measurement. It is not one, so the run refuses to grade past this.
    max_ingest_failure_rate: float = 0.02
    # Failing every turn from the first is a misconfiguration, not bad luck,
    # and there is no reason to spend an hour proving it.
    ingest_abort_after: int = 10

    # ---- grading ----
    judge_model: str = _env("MEMBENCH_JUDGE", "openai/gpt-oss-120b")
    no_answer_token: str = "NO ANSWER"

    # ---- systems ----
    systems: tuple[str, ...] = ("goodmem", "mem0", "langmem", "letta", "zep")

    run_id: str = _env("MEMBENCH_RUN_ID", "run")
    results_dir: Path = RESULTS_DIR

    # ---- GoodMem (instance supplied by the vendor, disclosed in the write-up) ----
    goodmem_base_url: str = _env(
        "GOODMEM_BASE_URL", "https://gm-hamzadevtech01-collab-0821dca3.app.goodmem.ai"
    )
    goodmem_default_space: str = _env("GOODMEM_DEFAULT_SPACE", "642d0ac7-2c18-4c35-aed9-bbfde7ad59cd")
    goodmem_optimized_space: str = _env("GOODMEM_OPTIMIZED_SPACE", "dd13a445-8aa5-4294-a561-630561de4e45")
    # post-processor ids for the vendor-recommended track only
    goodmem_llm_id: str = _env("GOODMEM_LLM_ID", "ac3ca82e-3d69-4026-a3dc-a017d5e9902f")
    goodmem_reranker_id: str = _env("GOODMEM_RERANKER_ID", "013495c5-71a3-4de7-af14-a54605630420")

    extra: dict = field(default_factory=dict)

    def require_llm_key(self) -> str:
        key = os.environ.get(self.llm_api_key_env, "")
        if not key:
            raise RuntimeError(
                f"{self.llm_api_key_env} is not set. The benchmark answers every probe with "
                f"{self.llm_model}; without a key nothing can be graded."
            )
        return key

    def as_dict(self) -> dict:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Path):
                d[k] = str(v)
        return d


CONFIG = Config()
