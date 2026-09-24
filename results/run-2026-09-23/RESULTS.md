# Results

Run `run-2026-09-23`, 2026-09-24 08:15 UTC.

| System | Track | Recall | Refused adversarial | Search p50 | Search p95 | Memory tokens | Ingest/turn | Judge disagreements |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Mem0 (open source) | baseline | 50.0% | 92.9% | 38.3 ms | 44.3 ms | 353 | 1.516 s | 25 |
| LangMem | baseline | 45.7% | 85.7% | 68.4 ms | 89.2 ms | 883 | 4.149 s | 24 |
| Zep (Graphiti) | baseline | 37.0% | 92.9% | 163.0 ms | 206.4 ms | 212 | 3.647 s | 14 |
| Letta | baseline | 52.2% | 92.9% | 317.9 ms | 484.8 ms | 503 | 0.37 s | 22 |
| GoodMem | baseline | 53.3% | 92.9% | 633.3 ms | 1122.6 ms | 503 | 0.28 s | 22 |
| GoodMem (recommended config) | recommended | 57.6% | 82.1% | 754.3 ms | 986.0 ms | 504 | 0.276 s | 18 |

## What the columns mean

- **Recall**: share of answerable probes the shared model got right, using only
  what that system retrieved. Graded by an LLM judge at temperature 0, with a
  normalised string match as a second opinion.
- **Refused adversarial**: LoCoMo category 5 questions have no answer in the
  conversation. Refusing is correct; answering is a hallucination.
- **Search p50 / p95**: time in the memory system's own search call. The
  answering model is not included, since it is the same for everyone.
- **Memory tokens**: median size of the injected memory, estimated the same way
  for every system. Cheaper is better at equal recall.
- **Ingest/turn**: wall clock per conversation turn written, including any
  extraction the system does.
- **Judge disagreements**: probes where the judge and the string match differed.
  These are the ones read by hand; see `review.csv`.

## Setup

- Answering model: `gpt-4o-mini` at temperature 0.0, same for every system.
- Grading model: `gpt-4o-mini`, also at temperature 0.
- Endpoint: `https://api.openai.com/v1`. Named because the same model id behaves differently
  across hosts, and a table that only says the model cannot be checked.
- Extraction: GoodMem and Letta make no model calls while ingesting; they store the turn
  and embed it. Mem0, LangMem and Zep each run `gpt-4o-mini` on every turn to decide
  what is worth keeping. So the answering model is shared, and the cost of a weak one is
  not: it falls on those three and not on the other two.
- Embeddings: `nomic-ai/nomic-embed-text-v1.5`, run locally for the self-hosted systems and on
  the vendor's own hosting for GoodMem. Same model, different host.
- Workload: 1 LoCoMo conversation,
  419 turns, 120 probes
  ({'single-hop': 44, 'multi-hop': 17, 'temporal': 22, 'open-domain': 9, 'adversarial': 28}).
- Retrieval depth: top 10 for every system.

125 probes needed a human look.

Raw per-probe records are in `probes.jsonl`; every number above can be recomputed from it.
