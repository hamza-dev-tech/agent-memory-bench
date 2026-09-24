# DRAFT — not published yet

Two things before this goes live: the disclosure paragraph needs the commercial
terms finalised, and every vendor named here should have had a week to respond.
Everything else is finished. Every number comes from
`results/run-2026-09-23/probes.jsonl` and can be recomputed from it.

---

**Title:** I Benchmarked 5 AI Agent Memory Systems. The Best Scored 57.6%.

**Slug:** `/blog/agent-memory-benchmark-mem0-zep-letta-langmem-goodmem`

**Meta description:** Independent LoCoMo benchmark of Mem0, Zep, Letta, LangMem
and GoodMem. Vendors publish 80-93%. Measured under retrieved-text-only rules,
the best scored 57.6%. Full method, open harness, every number reproducible.

**Published:** 24 September 2026 · **Author:** Malik Hamza Shabbir

---

## TL;DR

I ran five agent memory systems over the same 419-turn conversation with the
same answering model, graded them only on the text they retrieved, and
published the harness.

- **Best score: 57.6%** (GoodMem on its vendor-recommended config). **Worst:
  37.0%** (Zep/Graphiti).
- **Mem0 publishes 92.5% on LoCoMo. I measured 50.0%** on the same dataset. The
  gap is almost entirely method, and that is the point of this article.
- **The two systems that run no LLM while ingesting finished first and third.**
  The three that extract facts on every turn finished second-to-last, fourth
  and last.
- **Tuning buys recall and sells caution.** GoodMem's recommended config gained
  4.3 points of recall and lost 10.8 points on questions that have no answer.
- **Nobody is good at this yet.** Four of six rows sit between 45% and 53%.
  Every system answered fewer than half the multi-hop questions.
- Harness, raw per-probe log and results:
  [github.com/hamza-dev-tech/agent-memory-bench](https://github.com/hamza-dev-tech/agent-memory-bench)

---

## Before anything else: I published a benchmark I never ran

Last year I published a comparison of four agent memory systems. It had a
percentage in it, a latency figure, and a line about how the results held up
across reruns.

I did not run that benchmark. There was no harness. There were no probes. The
numbers were plausible because they were written to be plausible.

I am writing this because a vendor offered to pay to be added to that post, and
answering his technical questions meant admitting there was nothing underneath
it. So I built the thing that should have existed the first time, ran all five
systems on it, and published the code.

You are reading this at the URL that post lived at. I am not moving the
correction somewhere quieter than the mistake.

If that makes you trust this page less, good. Check it. Every number below is
recomputable from a JSON file in the repo, which is more than I can say for
most of the comparisons on this topic, including the ones written by the
vendors.

---

## Why vendors publish 92% and I measured 50%

This is the most important section on the page, so it comes before the results.

Mem0's research page reports **92.5% on LoCoMo**. Published figures for these
systems generally cluster between 80% and 93%. I ran the same benchmark and got
37% to 58%.

Their numbers are not fabricated. Mine are not wrong. **We are measuring
different things**, and almost nobody says which.

First, the difference that is my fault and not theirs: Mem0's figure is over
1,540 questions across all ten LoCoMo conversations. Mine is 120 probes from
one of them. That is a smaller sample and a narrower claim, and you should
weight it accordingly. It is not what produces a forty-point gap.

Three things do.

**1. Who writes the answer.** Most published evaluations let each system answer
in its own way. GoodMem and Letta will happily generate the answer themselves
using their own model. Mem0 and Zep hand back text. If every system answers
however it likes, you have compared four answer-writers and the memory layer is
a rounding error in the result. **This benchmark takes only retrieved text** and
feeds it to one shared model. That single rule costs every system a lot of
points, and it is the only way the comparison is about memory.

**2. Which answering model.** A stronger model recovers answers from worse
retrieval. Mem0's research page does not name the answering model it used. This
one uses `gpt-4o-mini` for every system, named, at temperature 0, chosen because
the entire run had to cost a few dollars. A frontier model would move every row
up, and not by the same amount.

**3. How it is graded.** Published scores rarely say who decided a given answer
was right. Here it is a normalised string match plus an LLM judge at
temperature 0, and every disagreement between the two was read by hand. There
were 125 of them.

The honest summary: **I cannot tell you Mem0 is not 92.5%, because their method
is not published.** I can tell you exactly what mine is, and that under it no
system cleared 58%. Whether those two facts are in tension is a question only
the vendors can answer, and all five are welcome to.

None of that makes 92.5% dishonest. It makes it **incomparable** — to my
numbers, to the other vendors' numbers, and to yours. A benchmark score without
a published method is a marketing asset, not a measurement.

So do not read the table below as "Mem0 is really 50%". Read it as: under one
fixed, published, reproducible method, these systems rank in this order, by
these margins.

---

## Results

Recall appears twice. The reason is in the grading section: the LLM judge has
two repeatable biases, both of which favour every system equally, so the honest
figure is a band rather than a point.

### Overall

| System | Recall (judge) | Recall (strict) | Refused adversarial | Search p50 | Memory tokens | Ingest/turn |
|---|---:|---:|---:|---:|---:|---:|
| **GoodMem** (vendor config) | **57.6%** | **54.3%** | 82.1% | 754 ms | 504 | 0.28 s |
| **GoodMem** (defaults) | 53.3% | 48.9% | 92.9% | 633 ms | 503 | 0.28 s |
| **Letta** 0.11.7 | 52.2% | 47.8% | 92.9% | 318 ms | 503 | 0.37 s |
| **Mem0** 2.1.0 | 50.0% | 46.7% | 92.9% | **38 ms** | 353 | 1.52 s |
| **LangMem** 0.0.30 | 45.7% | 43.5% | 85.7% | 68 ms | 884 | 4.15 s |
| **Zep / Graphiti** 0.30.2 | 37.0% | 32.6% | 92.9% | 163 ms | **212** | 3.65 s |

92 answerable probes and 28 adversarial probes per system, on one 419-turn
conversation. **The order is identical under both readings.** The two columns
differ by 2.2 to 4.3 points and no pair of systems swaps places inside that
band. That is the part I would trust.

### By question type

| System | Single-hop | Temporal | Multi-hop | Open-domain |
|---|---|---|---|---|
| GoodMem (vendor config) | **30/44** | 12/22 | **8/17** | 3/9 |
| GoodMem (defaults) | 26/44 | **13/22** | **8/17** | 2/9 |
| Letta | 26/44 | 12/22 | **8/17** | 2/9 |
| Mem0 | 27/44 | **13/22** | 4/17 | 2/9 |
| LangMem | 28/44 | 7/22 | 5/17 | 2/9 |
| Zep | 20/44 | 8/22 | 4/17 | 2/9 |

---

## What the numbers actually say

### Extracting facts on every turn did not pay for itself

The two systems that call **no LLM at all** while ingesting, GoodMem and Letta,
finished first and third. The three that run an extraction model on every
single turn finished second-to-last, fourth and last.

Letta is the sharpest version of this. It stores the turn verbatim, embeds it,
and beats Mem0 and Zep while ingesting four to eleven times faster. It is also a
year-old release that needed a patch to write to its own database at all.

This is the finding I least expected and the one I would most like someone to
knock down. On this workload the extraction pipelines are paying a large,
continuous cost for something that does not appear in recall. That may be a
property of LoCoMo rather than of extraction: these are conversational facts,
densely stated, in a form that survives being stored verbatim. On a corpus of
long documents the answer could invert.

### Tuning buys recall and sells caution

GoodMem's recommended configuration produced the best score in the table and
the second-worst refusal rate: **+4.3 points of recall, -10.8 points on the
questions that have no answer.** Four more right answers, four more
hallucinations, almost exactly one for one.

A reranker does what a reranker does. Asked something the conversation never
addressed, it still finds the closest-looking chunks and hands them over with
confidence, and the model obliges.

Whether that is a good trade is a question about your product, not about
GoodMem. It is fine behind a search box. It is bad in anything that talks to a
customer unsupervised. **This is the number I would most want from a comparison
and the one nobody publishes**, because you only get it if you deliberately ask
questions that have no answers.

### GoodMem wins on recall and pays for it in latency

Top of the table on both its rows, and the only system that beat Letta. At
defaults it is second on recall with the joint-best refusal rate, which is the
combination I would actually want: it says "I don't know" as reliably as
anything here while finding more than anything except its own tuned
configuration.

Its weakness is the clock. **633 ms at the median and 1,123 ms at p95**, the
slowest search in the table by a factor of two. It is the only hosted system
here, so a good part of that is a network round trip I would not pay
self-hosting it, and none of the others were tested over a network. Read the
latency column as "hosted product, from Pakistan, over the public internet"
rather than as a property of the engine.

It also has the widest strict-versus-lenient gap on its defaults row, 4.3
points, which means more of its correct answers were the arguable kind.

Worth stating plainly because they paid to be here: GoodMem finished first, and
the single largest correction in this whole exercise came from their team
catching a bug of mine that was holding their own score down. That section is
below.

### Zep is losing on retrieval, not on the graph

Last by nine points behind the next worst and twenty behind the leader, and yet
it returns the **smallest context in the table by a wide margin**: 212 tokens against Mem0's 353 and LangMem's 884. It refuses
adversarial probes as well as anyone.

A temporal knowledge graph gives back terse, well-formed facts. There are just
fewer of the right ones. If you are choosing Zep, choose it for what a small
context saves you on every query, not for coverage. Its 8 of 22 on temporal
questions is the genuine surprise, given that time is the thing its graph is
built around.

### LangMem loses on every axis at once

Lowest recall of the four open-source systems, worst refusal rate in the table
at 85.7%, largest context at 884 tokens, slowest ingest at 4.15 s per turn. It
is also the only system whose temporal answers collapse: **7 of 22**, against
12 or 13 for the systems either side of it.

Whatever its extraction is preserving, it is not preserving when things
happened.

### Mem0 is the fastest to search and the cheapest to read

**38 ms at the median**, not quite twice as quick as the next fastest and
seventeen times quicker than GoodMem. It runs on local Qdrant, so that is partly a
hosting comparison rather than a quality one, but if you are self-hosting it is
the number you feel.

It is mid-table on recall and notably weak on multi-hop: **4 of 17**, half what
GoodMem and Letta manage. Facts extracted turn by turn seem to lose the threads
running between sessions.

### Nobody is good at this yet

The best score on the board is 57.6% and it belongs to a tuned commercial
product. Four of six rows sit between 45% and 53%. Every system scored 2 or 3
out of 9 on open-domain questions. **Every system answered fewer than half the
multi-hop questions.**

If you are building on agent memory today, you are building on a component that
gets roughly half the questions right on a long conversation. Plan for that
rather than for the number on the vendor's landing page.

---

## What it costs to run

Three numbers, and the one that matters is not the one people quote.

**Context size is paid on every question, forever.** Zep's 212 tokens against
LangMem's 884 is a four-fold difference in the standing cost of asking
anything, and LangMem is the one with the lower recall. GoodMem and Letta both
sit around 503, Mem0 at 353. If you are serving a lot of queries, this column
outweighs everything else on the page.

**Ingest is where the extraction systems spend.** Over the same 419 turns:
GoodMem 0.28 s per turn, Letta 0.37 s, Mem0 1.52 s, Zep 3.65 s, LangMem 4.15 s.
The first two make no model calls at all. The last three are almost entirely
waiting on one, which is also money.

**And the figure that surprised me most.** Mem0's extraction prompt is 33,653
characters, roughly 8,478 tokens, sent unchanged on every turn. Across this one
conversation that is about ten million input tokens before a single question is
asked, which was most of what the entire benchmark cost. It is byte-identical
each time, so prompt caching reclaims most of it where caching exists. Where it
does not, plan for it.

For scale: the complete run, six configurations over 419 turns and 120 probes
each, cost under three dollars on `gpt-4o-mini`. That is the whole reason a
benchmark like this can be independent. Nobody needs a grant to check this.

---

## Method

### The one rule: retrieved text only

Every system returns chunks. Those chunks go into one shared model, at
temperature 0, with an identical prompt. The memory system's job ends at
retrieval.

Without this rule you are comparing answer-writers. With it, you are comparing
memory. It is the single biggest reason these numbers are lower than published
ones.

### Adversarial probes

LoCoMo's category 5 questions have **no answer anywhere in the conversation**.
A system that always produces something confident scores well on an ordinary
benchmark and is a liability in production. Here, refusing is correct and
answering is a hallucination.

It separates the field more sharply than recall does: four of six rows refuse
exactly 92.9% of them. The two that do not are GoodMem's tuned configuration at
82.1% and LangMem at 85.7%.

### Two graders, and a human for the arguments

A normalised string match and an LLM judge at temperature 0 both score every
probe. Where they disagree, the probe goes to `review.csv` and I read it. **On
this run they disagreed on 125 probes.**

Reading all 125 is what produced the second recall column. 117 were the string
match being naive about paraphrase: "Reading and painting" against a gold of
"Read a book and paint." The rest were the judge being wrong, in two specific
and repeatable ways.

**It accepts wrong dates.** Gold: *"The Friday before 22 October 2023."*
Answer: *"2023-10-13."* The 22nd was a Sunday, so the Friday before it is the
20th. The 13th is a week out. Marked correct.

**It rules both ways on partial answers.** *"A horse painting"* against a gold
of *"Horse, sunset, sunrise"* passed. *"Charlotte's Web"* against a gold naming
two books failed. One of three accepted, one of two rejected.

Both biases hit every system equally, so the ranking survives. Both make a
single figure sound more precise than it is. So recall is reported twice: what
the judge said, and the judge minus every probe it got demonstrably or arguably
wrong. Both adjudication rules are mechanical, with no model in the loop, so
they cannot drift between runs.

**An LLM judge is not the last word on whether an LLM was right.** If a
comparison does not tell you who graded it, that is the most important thing it
is not telling you.

### The workload

LoCoMo conversation 26: **419 turns across 19 sessions**, two people, with real
time gaps between sessions. The gaps are what make it a memory test rather than
a retrieval test.

**120 probes**, drawn from the conversation's own question set:

| Category | Count | What it asks |
|---|---:|---|
| Single-hop | 44 | One fact, stated once |
| Temporal | 22 | When, or in what order |
| Multi-hop | 17 | Two facts from different sessions |
| Open-domain | 9 | Inference the conversation supports but never states |
| Adversarial | 28 | No answer exists; refusing is correct |

Shared answering model `gpt-4o-mini` at temperature 0. Shared embedder
`nomic-embed-text-v1.5` at 768 dimensions, the same weights for every
self-hosted system. Top 10 retrieved for everyone. Fixed seed.

### What each system is

| System | Version | How it stores | LLM calls while ingesting |
|---|---|---|---|
| GoodMem | hosted, vendor instance | chunks in a managed space | **none** |
| Mem0 | 2.1.0 | extracted facts, local Qdrant | one or more per turn |
| LangMem | 0.0.30 | extracted to a LangGraph store | one per turn |
| Zep (Graphiti) | 0.30.2 | temporal knowledge graph, Kuzu | several per turn |
| Letta | 0.11.7 | archival memory, SQLite | **none** |

That last column decides how you read the ingest figures, which is why it is in
the table and not a footnote.

---

## Four bugs I hit that nobody has written up

This is the part I would keep if you skim everything else. All four are
first-hand, all four cost me hours, and none of them are in anyone's docs.

### Mem0 sends 8,478 tokens of prompt on every turn

Mem0's extraction prompt is **33,653 characters**, and it goes out unchanged
with every single turn you write. On a 419-turn conversation that is roughly
ten million input tokens before anyone asks a question, which was most of what
this entire benchmark cost to run.

I found it because Mem0 would not run at all on a free tier: the requests
exceeded the per-request ceiling outright, so every write came back rejected
and no amount of waiting helped. The prompt is byte-identical each time, so
prompt caching reclaims most of it where caching exists. Where it does not,
that number is your bill.

### Graphiti's Kuzu backend searches indices it never creates

`KuzuDriver.build_indices_and_constraints()` is a no-op. Its own comment says
the indices are created in `setup_schema()`. `setup_schema()` creates tables.
Nothing anywhere loads Kuzu's full-text extension. Meanwhile every hybrid
search asks for an index named `edge_name_and_fact`, so every search throws and
every turn is lost.

The subtler half: **a Kuzu text index is a snapshot.** It does not follow later
writes. Creating the indices at startup is not enough, because anything
ingested afterwards is invisible to the BM25 half of the hybrid search and
nothing tells you. It looks like it is working.

### Letta's current server cannot run without Postgres, and its last SQLite release cannot write

Letta 0.16.8's `server/db.py` builds an asyncpg engine unconditionally and
never reads the `SQLITE` value sitting in its own settings. Its ORM is
SQLite-ready, picking a plain vector column over pgvector when the engine is
SQLite, but the async engine never got the branch. With no Postgres it dies on
a refused connection to localhost:5432, disguised as `generator didn't stop
after athrow()`.

Everything from 0.12.1 up is the same, and 0.32.x has no REST server at all. So
this runs **0.11.7**, the last release that works on SQLite, which needed two
more fixes: it imports `sqlite_vec` at module scope without declaring it, and
`archives_agents.py` sets a Postgres `server_default` of `now()` as a raw
string. SQLite stores those six characters and then cannot read them back as a
date, so the first archival write fails and nothing can ever be stored.

### My own harness nearly published a row that never ran

Worth including because it is the subject of this post arriving through the
back door.

GoodMem's two tracks share a system key. Resume was keyed on
`(system, probe_id)`, so the second track found every probe already graded,
skipped the entire conversation and exited cleanly. Then the report, rebuilt
from probe records and keyed on the system alone, took its label from the
summary written last and produced a row reading **GoodMem (recommended config),
50.0%, 92.9%, 652 ms** — a full set of figures, correct to four significant
digits, describing a run that did not happen. The real baseline row had
vanished. **Nothing about it looked wrong.**

That is what a fabricated result looks like from the inside, and it is why the
harness now refuses to grade a system that lost more than 2% of a conversation,
why summaries are keyed on system *and* track, and why a requested system that
produces no rows prints a complaint instead of quietly disappearing.

---

## The vendor who paid found an error that was hurting his own score

Partway through, GoodMem's team checked their spaces and found duplicate
entries left by my repeated indexing runs. They cleaned them and asked me to
rerun retrieval.

It was my bug. My `reset()` deleted an in-memory list of what the current
process had written, so a fresh process had nothing to delete and left the
previous run's data in place. Five smoke runs and one failed attempt had
stacked up. **GoodMem was the only system in the table searching a polluted
index**, since every other adapter drops its whole scope, and duplicates crowd
a top-10 with repeats of one fact and push distinct facts out of it.

Rerunning on clean spaces moved GoodMem up **3.3 points on both tracks**.

So the vendor paying for inclusion caught a defect that was **suppressing his
own result**, and without it the published ranking would have been wrong. I
cannot think of a better argument for publishing the harness.

`reset()` now empties the space and re-lists it to confirm it emptied, and
after ingest it checks the space holds exactly what that run wrote and nothing
else. A delete that silently fails leaves the same state as no delete at all.

---

## Which one should you use?

Not a ranking. It depends on which column you actually pay.

**Choose Mem0 if search latency matters most.** 38 ms median, four times
quicker than anything else, mid-table recall, and the least context to pay for
among the self-hosted options at 353 tokens. Weak on multi-hop.

**Choose Zep if context cost dominates.** 212 tokens per answer, by a distance
the cheapest thing to read. You pay for that with nine points of recall against the next worst system, and
twenty against the best.
Good refusal behaviour.

**Choose Letta if ingest volume is the constraint.** No LLM calls while
writing, 0.37 s a turn, and the joint-best multi-hop score. Be aware you are
choosing 0.11.7 unless you are willing to run Postgres.

**Choose GoodMem if recall is the thing you are buying** and you can tolerate
the refusal trade on its tuned configuration, or run it at defaults, where it
is second on recall with a 92.9% refusal rate. It is hosted, so its 633 ms
includes a network round trip.

**I would not choose LangMem on these numbers.** Worst or near-worst on every
axis measured here.

And for all five: **plan for roughly half your questions being answered
correctly** on a long conversation, not for the number on the landing page.

---

## FAQ

**What is the best AI agent memory system in 2026?**
Under this benchmark's rules, GoodMem on its vendor-recommended configuration
scored highest at 57.6% recall, followed by GoodMem at defaults (53.3%), Letta
(52.2%), Mem0 (50.0%), LangMem (45.7%) and Zep (37.0%). No system exceeded 58%,
and the right choice depends on whether you are optimising for recall, search
latency, context cost or ingest throughput.

**Why is Mem0's published LoCoMo score 92.5% when you measured 50%?**
Different method. This benchmark grades only the text a system retrieves, fed
to one shared model at temperature 0. Most published evaluations let each
system generate its own answer, which measures the answering model as much as
the memory. The answering model, the judge and the grading rules also differ,
and published scores rarely name them.

**Does Zep's temporal knowledge graph help with time-based questions?**
Not in this test. Zep answered 8 of 22 temporal questions, against 13 for Mem0
and for GoodMem at defaults. It was last overall on recall while returning the
smallest context of any system measured.

**Do agent memory systems hallucinate?**
Yes, and it is measurable. On 28 questions with no answer in the conversation,
four of six configurations correctly refused 92.9% of the time. GoodMem's
reranked configuration refused only 82.1% and LangMem 85.7%. Adding a reranker
raised recall by 4.3 points and lowered refusal by 10.8.

**Is it worth running an LLM extraction pass on every turn?**
On this workload, no. The two systems that store turns verbatim and run no
model while ingesting, GoodMem and Letta, placed first and third, ahead of all
three systems that extract facts per turn, while ingesting four to eleven times
faster.

**Can I reproduce these results?**
Yes. The harness, adapters, adjudication script and full per-probe log are at
[github.com/hamza-dev-tech/agent-memory-bench](https://github.com/hamza-dev-tech/agent-memory-bench).
Every number here is recomputable from `probes.jsonl`.

---

## What this does not tell you

- **One conversation.** 419 turns and 120 probes is enough to separate these
  systems, not enough for a confidence interval. The harness takes
  `--conversations 10` and the dataset has ten.
- **Letta is a year behind.** 0.11.7 shipped in September 2025 and everything
  newer needs Postgres. These are that version's numbers.
- **GoodMem is hosted, everything else is local.** Its latency includes a
  network round trip. Right comparison for the hosted product, wrong one for
  self-hosting it.
- **Letta and GoodMem are not measured at what they are for.** Both are agent
  runtimes that would normally curate memory with their own model. Writing
  turns in verbatim and reading chunks back is the only way to compare them
  with a retrieval library, and it is not how you would deploy either.
- **The answering model shapes every number.** `gpt-4o-mini` was chosen because
  the whole run had to fit in a few dollars. A frontier model would lift every
  row, unevenly.
- **LoCoMo is conversation, not documents.** The finding about extraction not
  paying for itself may well invert on long documents.

---

## Disclosure

GoodMem's team are paying for their system to be included in this comparison.

- The instance and API key were supplied by them.
- The vendor-config track uses the configuration they recommended.
- They were shown their own results before publication and invited to respond.
  They did not see the other systems' numbers in advance and did not review
  this post.
- Every other vendor named here is being sent their results with the same
  invitation. Corrections and replies get published on this page as they
  arrive.
- The harness is public and the instance is theirs, so they can run it
  themselves, which is the main reason a flattering result would have been
  worthless to everybody, including them.

Paying to be measured is not the same as paying for a result. The way to tell
the difference is whether the code is public. It is.

---

## Run it yourself

```bash
git clone https://github.com/hamza-dev-tech/agent-memory-bench
python -m venv .venv && .venv/bin/pip install -r requirements.txt
python scripts/fetch_locomo.py
python scripts/run.py --systems mem0 langmem zep --conversations 1 --probes 120
```

`probes.jsonl` carries one JSON line per probe: the question, the gold answer,
what the system retrieved, what the model answered, what both graders said, and
how long every step took. If you think a number here is wrong, the record that
produced it is in the repo.

If the harness is useful to you, **a star on the repo genuinely helps** — it is
how the next person building on agent memory finds a number they can check
instead of one they have to take on faith.

[**github.com/hamza-dev-tech/agent-memory-bench**](https://github.com/hamza-dev-tech/agent-memory-bench)

**Running a memory system I did not test?** Adapters are about 80 lines against
a five-method interface. Open an issue or send me the endpoint and I will run
it on the same workload, same rules, same table, paid or not, and the result
publishes either way. The same offer stands for any of the five above who
thinks a configuration choice here misrepresents them: tell me what to change
and I will rerun it.

---

## Tags

`ai-agent-memory` · `mem0` · `zep` · `graphiti` · `letta` · `memgpt` ·
`langmem` · `goodmem` · `locomo` · `benchmark` · `rag` · `llm-evaluation` ·
`vector-search` · `knowledge-graph` · `agent-infrastructure` ·
`hallucination` · `retrieval` · `open-source`

**Primary keyword:** AI agent memory benchmark

**Secondary:** Mem0 vs Zep vs Letta, LoCoMo benchmark, agent memory comparison
2026, best AI agent memory system, Mem0 alternatives, agent memory
hallucination rate
