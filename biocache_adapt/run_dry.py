"""No-model CLI for schema, stream protocol and artifact validation."""

from __future__ import annotations

import argparse
import json
import os

from biocache_adapt.prefixkv_adapter import LlavaOVPrefixKVAdapter, TimestampedFrame
from biocache_adapt.schema import BenchmarkQuery, load_benchmark


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="PrefixKV BioCache adapter dry-run (no model imports)")
    parser.add_argument("--input", help="optional benchmark JSON/JSONL")
    parser.add_argument("--run-dir", default="runs/biocache_dry")
    parser.add_argument("--budget", default="25%", choices=["12.5%", "25%", "50%"])
    parser.add_argument("--full-token-slots", type=int, default=16384)
    parser.add_argument("--token-limit", type=int)
    parser.add_argument("--byte-limit", type=int)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.input:
        samples = load_benchmark(args.input)
    else:
        from biocache_adapt.schema import BenchmarkSample

        samples = [
            BenchmarkSample(
                sample_id="synthetic",
                video_path="",
                duration=2.0,
                queries=[BenchmarkQuery("q0", 1.0, "dry-run question")],
            )
        ]
    outputs = []
    for sample in samples:
        adapter = LlavaOVPrefixKVAdapter(
            dry_run=True,
            budget=args.budget,
            full_token_slots=args.full_token_slots,
            token_limit=args.token_limit,
            byte_limit=args.byte_limit,
            run_dir=os.path.join(args.run_dir, sample.sample_id),
        )
        adapter.open_stream(sample.sample_id, {"video_path": sample.video_path})
        previous = 0.0
        for query in sample.queries:
            timestamps = []
            cursor = previous
            while cursor + 1.0 <= query.timestamp + 1e-9:
                cursor += 1.0
                timestamps.append(cursor)
            adapter.push_frames(
                [TimestampedFrame(timestamp, b"dry-frame") for timestamp in timestamps],
                available_until=query.timestamp,
            )
            adapter.answer(query)
            previous = query.timestamp
        outputs.append(adapter.close_stream())
    print(json.dumps({"dry_run": True, "streams": outputs}, ensure_ascii=False, indent=2))
    return outputs


if __name__ == "__main__":
    main()
