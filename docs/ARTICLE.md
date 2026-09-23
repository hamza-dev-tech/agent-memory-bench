# DRAFT — not for publication

Everything here that is not a measured number is finished prose. Every number
comes out of `results/run-2026-09-23/probes.jsonl` once the run completes, and
the placeholders below are marked `<<TK>>` so that nothing unmeasured can reach
a reader by accident. If a `<<TK>>` survives to publication, that is the bug
this whole exercise exists to prevent.

---

## I published a benchmark I never ran. Here is the real one.

Last year I published a comparison of four agent memory systems. It had a
percentage in it, a latency figure, and a line about how the results held up
across reruns. I want to be exact about this: **I did not run that benchmark.**
There was no harness. There were no probes. The numbers were plausible because
they were written to be plausible.

The reason you are reading this is that someone offered to pay me to add a
fifth system to that post, and answering his questions meant admitting there was
nothing underneath it. So I built the thing that should have existed the first
time, ran all five systems on it, and published the code.

This post is both the correction and the benchmark. The correction comes first
because it is the part that matters.

### What was wrong, precisely

- The figures were invented, not measured.
- No model was named, so nothing could have been reproduced even in principle.
- There was no grading method, because there was nothing to grade.
- "Held across reruns" describes reruns that did not happen.

The old post has been replaced with a link to this one. I am not editing the
numbers quietly and leaving the URL working.

---

## Why most memory comparisons do not measure memory

Before the results, the method, because the method is the disagreement.

Nearly every "Mem0 vs Zep vs X" post works like this: give each system a
conversation, ask it a question, let the system answer, score the answer. That
measures the wrong thing. GoodMem and Letta will happily write the answer for
you using their own model. Mem0 and Zep hand back text. If you let each system
answer in its own way, you have compared four answer-writers, and the memory
layer is a rounding error in the result.

So this harness has one rule it does not bend:

> **Retrieved text only.** Every system returns chunks. Those chunks go into one
> shared model, at temperature 0, with the same prompt. The memory system's job
> ends at retrieval.

Four more decisions follow from that.

**Adversarial probes.** LoCoMo's category 5 questions have no answer anywhere in
the conversation. A system that always produces something confident scores well
on a normal benchmark and is useless in production. Here, refusing is the
correct answer and answering is a hallucination. `<<TK: refusal rates>>` is the
column I would look at first.

**Two graders, and a human for the arguments.** A normalised string match and an
LLM judge at temperature 0 both score every probe. Where they disagree, the
probe goes to `review.csv` and I read it. On this run they disagreed on
`<<TK: disagreement count>>` of 120 probes. An LLM judge is not the last word on
whether an LLM was right.

**Same embedder for everyone.** All the self-hosted systems run
`nomic-embed-text-v1.5` through the same weights, including Letta, which embeds
inside its own server and had to be handed a local endpoint to reach them.
Otherwise the table compares two embedders as well as five memory systems.

**Two tracks.** Every system runs at its defaults. GoodMem additionally runs on
the configuration its own team recommends, with their reranker. Defaults are
what most people get; tuned is what the vendor would show you. Both are in the
table, labelled.

---

## The workload

LoCoMo conversation 26: **419 turns across 19 sessions**, a two-person
conversation with real time gaps between sessions, which is what makes it a
memory test rather than a retrieval test.

**120 probes**, drawn from the conversation's own question set:

| category | count | what it asks |
|---|---:|---|
| single-hop | 44 | one fact, stated once |
| temporal | 22 | when, or in what order |
| multi-hop | 17 | two facts from different sessions |
| open-domain | 9 | inference the conversation supports but never states |
| adversarial | 28 | no answer exists; refusing is correct |

Shared answering model `gpt-4o-mini` at temperature 0. Shared embedder
`nomic-embed-text-v1.5`, 768 dimensions. Top 10 retrieved for every system. Seed
fixed, so the same conversation and the same probes come out every time.

---

## The systems

| system | version | how it stores | LLM calls while ingesting |
|---|---|---|---|
| GoodMem | hosted, vendor instance | chunks in a managed space | none |
| Mem0 | 2.1.0 | extracts facts, local Qdrant | one or more per turn |
| LangMem | 0.0.30 | extracts to a LangGraph store | one per turn |
| Zep (Graphiti) | 0.30.2 | temporal knowledge graph, Kuzu | several per turn |
| Letta | 0.11.7 | archival memory, SQLite | none |

That last column decides how you read the ingest numbers, so it is in the table
rather than in a footnote. **GoodMem and Letta do not call a model while
ingesting.** They store the turn and embed it. The other three run an extraction
model on every single turn, which is most of their ingest time and all of their
ingest cost. Comparing "seconds per turn" across those two groups is comparing
two different activities.

---

## Results

`<<TK: the table, generated by membench/report.py>>`

`<<TK: per-category breakdown>>`

`<<TK: two or three paragraphs of reading, written after the numbers exist and
not before>>`

---

## What I found in the systems themselves

This is the part I did not expect to be writing, and it is more useful than the
table.

### Mem0 sends 8,478 tokens of prompt on every turn

Mem0's extraction prompt is 33,653 characters. It goes out with every turn you
write, unchanged. On a 419-turn conversation that is most of the run's entire
token bill before a single question is asked.

I found this because Mem0 would not run at all on a free tier: the requests
exceeded the per-request ceiling outright, so every write came back rejected and
no amount of waiting helped. The prompt is byte-identical each time, so prompt
caching takes most of it back where caching exists — but if you are running Mem0
at volume on an endpoint without caching, that number is your bill.

### Graphiti's Kuzu backend searches indices it never creates

`KuzuDriver.build_indices_and_constraints()` is a no-op. Its comment says the
indices are created in `setup_schema()`. `setup_schema()` creates tables.
Nothing anywhere loads Kuzu's full-text extension. Meanwhile every hybrid search
asks for an index named `edge_name_and_fact`, so every search throws and every
turn is lost.

The subtler half: a Kuzu text index is a **snapshot**. It does not follow later
writes. So creating the indices at startup is not enough — anything ingested
afterwards is invisible to the BM25 half of the hybrid search, and nothing tells
you. It looks like it works. The harness rebuilds the indices after ingest and
before any question is asked.

### Letta's newest server cannot run without Postgres, and its last SQLite release cannot write

Letta 0.16.8's `server/db.py` builds an asyncpg engine unconditionally and never
reads the `SQLITE` value sitting in its own settings. Its ORM is SQLite-ready —
it picks a plain vector column over pgvector when the engine is SQLite — but the
async engine never got the branch. With no Postgres it dies on a refused
connection to localhost:5432, disguised as `generator didn't stop after
athrow()`. Everything from 0.12.1 up is the same, and 0.32.x has no REST server
at all.

So this runs 0.11.7, the last release that works on SQLite, which needed two
fixes of its own: it imports `sqlite_vec` at module scope without declaring it,
and `archives_agents.py` sets a Postgres `server_default="now()"` as a raw
string. SQLite stores those six characters and then cannot read them back as a
date, so the first archival write fails and nothing can ever be stored. That is
one line, it is in `scripts/patch_letta.py`, and it is disclosed here because it
is still a change to a system under test.

### The harness refuses to grade a system that could not store anything

Worth stating because it is the guard that would have caught the original
article. When Mem0's writes were all being rejected, the store stayed empty and
the harness happily went on asking it questions. It refused every answerable
probe and correctly refused the adversarial ones — a full row of entirely
plausible numbers, measured against nothing at all.

That is exactly what a fabricated result looks like from the inside. The runner
now stops a system that loses more than 2% of a conversation, before a single
probe is graded, and says why in the log.

---

## Cost

`<<TK: measured token counts per system>>`

---

## What this does not tell you

- **Letta is a year behind.** 0.11.7 was released in September 2025; everything
  newer needs Postgres. Its numbers are that version's, not today's.
- **GoodMem is hosted; everything else is local.** Its search latency includes a
  network round trip and the others do not. That is the honest comparison for
  anyone using the hosted product and the wrong one for anyone self-hosting it.
- **One conversation.** 419 turns and 120 probes is enough to separate these
  systems, not enough for a confidence interval. The harness takes
  `--conversations 10` and the dataset has ten.
- **Letta and GoodMem are not being measured at what they are for.** Both are
  agent runtimes that would normally curate memory with their own model. Writing
  turns in verbatim and reading chunks back is the only way to compare them to a
  retrieval library, and it is not how you would deploy either.
- **The answering model shapes every number.** A stronger model would lift all
  five, and not by the same amount.

---

## Disclosure

GoodMem's team paid **$250** for their system to be included in this comparison.

- The instance and API key were supplied by them.
- The optimized-config track uses the configuration they recommended.
- They did not see the numbers before publication and did not review this post.
- The harness is public. The instance is theirs, so they can run it themselves,
  which is the main reason a flattering result would have been worthless to
  everybody.

Paying to be measured is not the same as paying for a result, and the way to
tell the difference is whether the code is public. It is.

---

## Run it yourself

```bash
git clone <<TK: repo url>>
pip install -r requirements.txt
python scripts/fetch_locomo.py
python scripts/run.py --systems mem0 langmem zep --conversations 1 --probes 120
```

Every number in the table is recomputed from `probes.jsonl`, one JSON line per
probe: the question, the gold answer, what the system retrieved, what the model
answered, what both graders said, and how long each step took. If you think a
number is wrong, the record that produced it is right there.
