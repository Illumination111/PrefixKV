# PrefixKV BioCache adapter

This package extracts a model-independent prefix-selection policy and a
callback-based LLaVA-OV streaming adapter. Existing `prefixkv.py`,
`eval_ppl.py`, and `eval_rouge.py` behavior is unchanged.

## Contract

- `open_stream(stream_id, metadata)` / `reset()`
- `push_frames(timestamped_frames, available_until)`: accepts caller-decoded
  frames only, enforces monotonic timestamps, and rejects future frames.
- `answer(query)` (alias `query`) / `close_stream()`
- `persistent_state()`, `static_state()`, `temporary_state()`, and recursive
  `state_bytes()`

After every video prefill, `PrefixSelectionPolicy.update_after_video_prefill`
updates importance using attention scores keyed by absolute token positions.
`select` then applies a global absolute token-slot and/or actual-byte budget
while protecting configured prefix and recent positions. Query/decode KV is
measured and rolled back to the video-prefill checkpoint after each answer.

A token budget counts retained KV positions across all layers. A byte budget
uses each layer's actual tensor element size and K/V shape. The named `12.5%`,
`25%`, and `50%` tiers map from `full_token_slots` or `full_cache_bytes`;
explicit `token_limit`/`byte_limit` override those derived limits.

## LLaVA-OV callbacks

The adapter deliberately does not import a model implementation. Supply:

- `prefill_fn(frames, cache) -> {"cache": ..., "attentions": ...}` (a cache or
  `(cache, attentions)` is also accepted).
- `query_fn(query, cache) -> {"text": ..., "cache": ...}` (text or
  `(text, cache)` is also accepted).

Tuple/list cache layers must begin with `(key, value)`; sequence dimensions are
configurable. Torch is imported only when real cache selection runs, preserving
legacy environments and allowing dependency-free dry-runs.

## Benchmark schema and outputs

`schema.load_benchmark()` normalizes JSON/JSONL aliases for video paths, query
lists, timestamps, questions, choices, and answers. Run directories contain
`manifest.json`, `results.jsonl`, and `summary.json` with phased timing, state
bytes, selection positions, and pre-rollback query-KV bytes.

## Safe validation

```bash
python -m biocache_adapt.run_dry --run-dir /tmp/prefixkv-biocache-dry
python -m biocache_adapt.run_dry --help
```

These commands never load models, decode videos, download data, or run
inference.
