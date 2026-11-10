# agent-memory-bench

An open harness that runs the same long-conversation workload through several
agent memory systems and grades what each one can recall.

The point is not to crown a winner. It is that anyone can rerun this and get
the same numbers, including the vendors being measured.

**Results from the 2026-09-23 run, and the write-up:**
[I Benchmarked 5 AI Agent Memory Systems. The Best Scored 57.6%.](https://hamzashabbir.dev/article/agent-memory-mem0-vs-letta-vs-zep-vs-langmem-benchmark-2026)

| System | Recall | Refused adversarial | Search p50 | Memory tokens | Ingest/turn |
|---|---:|---:|---:|---:|---:|
| GoodMem (vendor config) | 57.6% | 82.1% | 754 ms | 504 | 0.28 s |
| GoodMem (defaults) | 53.3% | 92.9% | 633 ms | 503 | 0.28 s |
| Letta 0.11.7 | 52.2% | 92.9% | 318 ms | 503 | 0.37 s |
| Mem0 2.1.0 | 50.0% | 92.9% | 38 ms | 353 | 1.52 s |
| LangMem 0.0.30 | 45.7% | 85.7% | 68 ms | 884 | 4.15 s |
| Zep / Graphiti 0.30.2 | 37.0% | 92.9% | 163 ms | 212 | 3.65 s |

419 turns, 120 probes, `gpt-4o-mini` at temperature 0 for every system, top 10
retrieved. Published vendor figures for these systems run 80-93%; the
difference is method, and this repo is the method.

If that is useful, a star helps the next person find a number they can check
rather than one they have to take on faith. If your system is not in the table
and you would like it to be, open an issue. The contract is six methods and
[docs/ADDING_A_SYSTEM.md](docs/ADDING_A_SYSTEM.md) walks through one, and the
result gets published either way.

## What it measures

Every system gets identical treatment:

1. Ingest whole multi-session conversations from [LoCoMo](https://github.com/snap-research/locomo), turn by turn.
2. Wait until it says indexing is done.
3. Ask a fixed set of probe questions whose answers live in earlier sessions.
4. Take the text it retrieved, hand it to **one shared model**, and grade that answer.

Step 4 is the part people skip. Some of these products will happily generate
the answer themselves, which turns a memory comparison into a comparison of
whoever writes the nicest sentence. So the harness ignores their generated
replies and uses only the retrieved text.

Recorded per system: recall on answerable questions, refusal rate on
adversarial ones, search latency (p50/p95), median context size, and ingest
time per turn.

## Systems

| System | Licence | How it ran here |
| --- | --- | --- |
| Mem0 | Apache 2.0, open source | self-hosted, local Qdrant |
| Letta | Apache 2.0, open source | self-hosted server in its own venv, see the adapter |
| LangMem | MIT, open source | in process |
| Zep (Graphiti) | Apache 2.0 engine, open source | self-hosted on embedded Kuzu |
| GoodMem | not open source; free to self-host commercially under the vendor's Free Binary License, with perpetual rights to versions obtained under it. Managed hosting is a paid option. | ran on a hosted instance supplied by the vendor |

Two tracks. **Baseline** puts every system on the same models with default
settings. **Recommended** is opt-in: a vendor who provides a configured
instance gets a second run on their own embedder, reranker and LLM, reported
as a separate row.

## Models

- **The published run** used `gpt-4o-mini` on `https://api.openai.com/v1` at
  temperature 0, for answering, for grading, and for the extraction Mem0, LangMem
  and Zep run on every turn. The one-conversation workload cost about $1.70.
  `scripts/reproduce.py` sets exactly this.
- **The code's default** is `openai/gpt-oss-120b` through Groq, temperature 0. It
  is not what the table was measured on. Groq's free tier caps a request at 8,000
  tokens and Mem0's extraction prompt alone is about 8,478, so on the free tier
  every Mem0 write is rejected. Groq's Developer tier serves the same model
  without that cap.
- Embeddings: `nomic-embed-text-v1.5`, run locally through transformers for the
  self-hosted systems and on the vendor's hosting for GoodMem. Same weights,
  different host, noted in the results.

The comparison holds because every system in a run gets the same model. Scores
from runs on different models are not comparable with each other, so a run on
the default will not match the table, and is not meant to.

## Running it

```bash
python -m venv .venv
.venv/Scripts/activate            # source .venv/bin/activate on POSIX
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python scripts/fetch_locomo.py

export OPENAI_API_KEY=...        # the published run used gpt-4o-mini, see Models
export GOODMEM_API_KEY=...       # only if you are running GoodMem

# check the workload matches the published run, without spending anything
python scripts/reproduce.py --check

# start small: a smoke test, the numbers are not comparable to anything
python scripts/run.py --systems mem0 --conversations 1 --probes 10 --max-turns 60

# reproduce the published table: same models, same workload, one command
python scripts/reproduce.py
```

`reproduce.py` is a thin wrapper. By hand, the published workload is
`--conversations 1 --probes 120 --adversarial 28`, and the models come from the
three variables under [Picking a model endpoint](#picking-a-model-endpoint).
Leaving out `--adversarial 28` falls back to 10 and draws a different probe set,
so the numbers will not line up with the table.

Output lands in `results/<run-id>/`:

- `probes.jsonl` - one line per probe, written as it happens. A run that dies
  resumes from here instead of starting over.
- `results.json` - full config, dataset summary and per-system aggregates.
- `RESULTS.md` - the table.
- `review.csv` - probes where the LLM judge and the string match disagreed.
  Read these by hand before publishing anything.

## Picking a model endpoint

Three environment variables, no code change:

```bash
export MEMBENCH_LLM_KEY_ENV=OPENAI_API_KEY
export MEMBENCH_LLM_BASE_URL=https://api.openai.com/v1
export MEMBENCH_LLM=gpt-4o-mini
```

Two things are worth knowing before picking something cheaper.

GoodMem and Letta make no model calls at all while ingesting. Mem0, LangMem and Zep each
run an extraction model on every turn, so a weak model damages those three and leaves the
other two untouched. The comparison stays internally consistent and the absolute numbers
stop meaning anything.

Mem0's extraction prompt is 33,653 characters, about 8,478 tokens, and it goes out on
every turn. On a 788 turn workload that is 10M of the 22.2M input tokens the whole run
costs, so the endpoint has to accept requests of that size: a free tier capping requests
at 8,000 tokens rejects every Mem0 write outright, and no amount of waiting helps. The
prompt is also byte-identical each time, which is worth a lot wherever prompt caching
exists.

## A note on the environment

Install into a virtualenv and leave scikit-learn out of it. transformers only
imports sklearn when it is present, and on a Windows machine with Application
Control enabled, sklearn's compiled `_pairwise_distances_reduction` extension is
blocked, which takes transformers and anything built on it down with it. That is
also why the embedder here talks to transformers directly instead of using
sentence-transformers.

## Grading

Two graders on every probe: a normalised string match, and an LLM judge at
temperature 0 that accepts different date formats and wording. The judge
decides; the string match is a tripwire. Where they disagree the probe goes to
`review.csv` for a human.

Adversarial probes (LoCoMo category 5) have no answer in the conversation.
Refusing is correct. Answering anyway is counted as a hallucination.

## Honest limits

- LoCoMo is synthetic, generated then human-edited. It is not your support inbox.
- Two conversations is a small sample. Treat gaps of a couple of points as noise.
- Self-hosted defaults are not tuned. A vendor who thinks their defaults are
  unfair is welcome to send a configured instance and take the second track.
- Latency is measured from Pakistan, so hosted services carry network distance
  that self-hosted ones do not. The p50/p95 columns are indicative, not a SLA.

## Adding a system

Implement `MemorySystem` in `membench/adapters/base.py`: `reset`, `add`,
`flush`, `search`. Keep the adapter thin. Anything clever added to one adapter
has to be added to all of them or the numbers stop meaning anything.

## Disclosure

Where a vendor has paid for inclusion or supplied a hosted instance, it is
stated in the results and in any article built on them. Payment buys a run and
a fair configuration; it does not buy a number.

## Licence

MIT. The LoCoMo dataset keeps its own licence.
