# Adding a memory system

An adapter is one file in `membench/adapters/` and one branch in
`scripts/run.py`. The runner only ever talks to the `MemorySystem` contract in
[`membench/adapters/base.py`](../membench/adapters/base.py), so a new system
never needs changes anywhere else.

Before writing one, read the rules below. They are what makes a number in the
table comparable with the others, and an adapter that breaks one produces a
plausible row that measures something else.

## The contract

| Method | Called | Must |
| --- | --- | --- |
| `setup()` | once, before anything | start clients or containers, check the service answers |
| `reset(run_key)` | before each conversation | leave the system with no memory at all |
| `add(user_key, text, meta)` | once per conversation turn | store the turn and return the seconds the call took |
| `flush(user_key)` | after the last turn | block until everything written is searchable |
| `search(user_key, query, k)` | once per probe | return the top `k` chunks the system would give an agent |
| `teardown()` | once, at the end | close what `setup()` opened |

`meta` carries `speaker`, `session`, `timestamp` and `dia_id`. Use them if the
system models time or speakers, ignore them if it does not, and say which in
`config_notes`.

Set these attributes too, because they end up in the results:

- `key`: the name used on the command line, for example `"cognee"`.
- `label`: how the row is shown in the table.
- `track`: `"baseline"` unless this is a vendor-tuned second row (see below).
- `version`: the installed package or server version. Set it in `__init__`,
  because the runner records it before `setup()` runs.
- `config_notes`: one sentence a reader can check, saying exactly what ran.

## A minimal adapter

```python
"""Example memory system. One paragraph on how it stores and retrieves, and
anything a reader would be surprised by."""
from __future__ import annotations

from types import SimpleNamespace

from ..embeddings import LocalEmbedder
from .base import Chunk, MemorySystem, SearchResult, timed


def _imports():
    # every third-party import in one place, so a missing package says what to install
    try:
        import example_memory
    except ImportError as e:
        raise RuntimeError("Run: pip install 'example-memory>=1.0'") from e
    return SimpleNamespace(lib=example_memory)


class ExampleSystem(MemorySystem):
    key = "example"
    label = "Example Memory"
    track = "baseline"

    def __init__(self, cfg):
        self.cfg = cfg
        self.client = None
        try:
            from importlib.metadata import version
            self.version = f"example-memory {version('example-memory')}"
        except Exception:
            self.version = "example-memory unknown"
        self.config_notes = f"in process, defaults, embeddings {cfg.embed_model}, extraction {cfg.llm_model}"

    def setup(self):
        lib = _imports().lib
        self.client = lib.Client(embedder=LocalEmbedder(self.cfg), llm=self.cfg.llm_model)

    def reset(self, run_key):
        self.client.delete_all(namespace=run_key)

    def add(self, user_key, text, meta):
        with timed() as t:
            self.client.add(text, user=user_key, metadata=meta)
        return t[0]

    def search(self, user_key, query, k):
        with timed() as t:
            hits = self.client.search(query, user=user_key, limit=k)
        return SearchResult(
            chunks=[Chunk(text=h.text, score=h.score) for h in hits],
            latency_s=t[0],
        )
```

Then register it in `build_system()` in `scripts/run.py`:

```python
    if name == "example":
        from membench.adapters.example_adapter import ExampleSystem
        return ExampleSystem(cfg)
```

## The rules

1. **Retrieved text only.** `search()` returns what the system retrieved. If the
   system can also write an answer itself, do not use it: the shared model
   answers every probe from the chunks, otherwise the table compares answer
   writers instead of memory layers.
2. **Shared models.** Any extraction or summarisation the system does on write
   runs on `cfg.llm_model`, and embeddings come from `LocalEmbedder(cfg)`, the
   same as every other system in the baseline track.
3. **Defaults.** No tuning the other systems do not get: no extra retries, no
   prompt changes, no caching, no custom chunking. If a default is clearly
   broken, say so in the adapter docstring and in `config_notes` rather than
   quietly fixing it.
4. **Let failures raise.** If a write fails, let the exception through. The
   runner counts failed turns and refuses to grade a system that lost more than
   2% of them, or that failed its first 10 turns in a row. An adapter that
   swallows errors gets graded against an empty store and posts a full row of
   numbers that measure nothing.
5. **`flush()` waits for real.** A system that indexes in the background will
   otherwise look fast on ingest and bad on recall. Poll until a write is
   searchable, with a timeout that raises.
6. **`reset()` empties everything.** Each conversation must start with no
   memory, so no system can score by remembering an earlier one.

## Testing it

Offline, with no key: copy the `FakeSystem` pattern in
[`tests/test_harness.py`](../tests/test_harness.py) if your adapter has logic
worth pinning, such as parsing the system's response into `Chunk`s.

Then a smoke run on a short slice of one conversation. The numbers mean
nothing at this size; you are checking that writes, flush and search all work.

```bash
python scripts/run.py --systems example --conversations 1 --probes 10 --max-turns 60
```

Read the output for `!!` lines. Then run it on the published workload so it can
sit in the same table:

```bash
python scripts/reproduce.py --systems example
```

## The recommended track

A vendor who wants their own configuration measured (their embedder, reranker
or model) gets a second row with `track = "recommended"`, next to the baseline
row, never instead of it. The baseline row is what makes the systems
comparable; the recommended row shows what the vendor's setup adds on top. If a
vendor paid for inclusion, the results say so.

## Getting it into the table

Open an issue or a pull request with the adapter. The result is published
whichever way it comes out.
