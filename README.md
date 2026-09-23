# agent-memory-bench

An open harness that runs the same long-conversation workload through several
agent memory systems and grades what each one can recall.

The point is not to crown a winner. It is that anyone can rerun this and get
the same numbers, including the vendors being measured.

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

| System | How it runs |
| --- | --- |
| Mem0 | open source, self-hosted |
| Letta | open source, self-hosted server in its own venv, see the adapter |
| LangMem | open source, in process |
| Zep (Graphiti) | open source engine, self-hosted |
| GoodMem | hosted instance provided by the vendor |

Two tracks. **Baseline** puts every system on the same models with default
settings. **Recommended** is opt-in: a vendor who provides a configured
instance gets a second run on their own embedder, reranker and LLM, reported
as a separate row.

## Models

- Answering and judging: `openai/gpt-oss-120b` through Groq, temperature 0.
- Embeddings: `nomic-embed-text-v1.5`, run locally through transformers for the
  self-hosted systems and on the vendor's hosting for GoodMem. Same weights,
  different host, noted in the results.

An open-weights model extracts memories less well than a frontier model, so every
system's absolute score is lower than it would be on GPT-class models. That is
fine: the comparison holds because every system gets the same model, and the
whole run costs nothing.

## Running it

```bash
python -m venv .venv
.venv/Scripts/activate            # source .venv/bin/activate on POSIX
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python scripts/fetch_locomo.py

export GROQ_API_KEY=...          # free tier is enough
export GOODMEM_API_KEY=...       # only if you are running GoodMem

# start small
python scripts/run.py --systems goodmem --conversations 1 --probes 10

# the real thing
python scripts/run.py --conversations 2 --probes 60
```

Output lands in `results/<run-id>/`:

- `probes.jsonl` - one line per probe, written as it happens. A run that dies
  resumes from here instead of starting over.
- `results.json` - full config, dataset summary and per-system aggregates.
- `RESULTS.md` - the table.
- `review.csv` - probes where the LLM judge and the string match disagreed.
  Read these by hand before publishing anything.

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
