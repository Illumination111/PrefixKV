"""Streaming LLaVA-OV adapter driven by the reusable PrefixKV policy."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Iterable, Mapping, Optional

from .common import JsonlWriter, StageTimer, recursive_nbytes, state_report, write_manifest
from .policy import PrefixSelectionPolicy
from .schema import BenchmarkQuery


@dataclass(frozen=True)
class TimestampedFrame:
    timestamp: float
    frame: Any


def _frame(item: Any) -> TimestampedFrame:
    if isinstance(item, TimestampedFrame):
        return item
    if isinstance(item, Mapping):
        return TimestampedFrame(float(item["timestamp"]), item.get("frame", item.get("image")))
    if isinstance(item, (tuple, list)) and len(item) == 2:
        return TimestampedFrame(float(item[0]), item[1])
    raise TypeError("frame must be TimestampedFrame, mapping, or (timestamp, frame)")


def _prefill_result(result: Any):
    if isinstance(result, Mapping):
        return result.get("cache"), result.get("attentions"), result.get("token_count")
    if isinstance(result, tuple) and len(result) in (2, 3):
        return result[0], result[1], result[2] if len(result) == 3 else None
    return result, None, None


def _query_result(result: Any, fallback_cache: Any):
    if isinstance(result, Mapping):
        return str(result.get("text", result.get("answer", ""))), result.get("cache", fallback_cache)
    if isinstance(result, tuple) and len(result) == 2:
        return str(result[0]), result[1]
    return str(result), fallback_cache


class LlavaOVPrefixKVAdapter:
    """Callback adapter for LLaVA-OV without importing a particular model tree.

    ``prefill_fn(frames, cache)`` returns either a cache, ``(cache, attentions)``,
    or a mapping with ``cache``/``attentions``. ``query_fn(query, cache)`` returns
    text, ``(text, cache)``, or a mapping. This keeps model imports in the
    benchmark integration and preserves the repository's legacy LLaVA-1.5 path.
    """

    def __init__(
        self,
        prefill_fn: Optional[Callable[[Any, Any], Any]] = None,
        query_fn: Optional[Callable[[Any, Any], Any]] = None,
        *,
        dry_run: bool = False,
        budget: Any = "25%",
        full_token_slots: Optional[int] = 16384,
        full_cache_bytes: Optional[int] = None,
        token_limit: Optional[int] = None,
        byte_limit: Optional[int] = None,
        start_size: int = 4,
        recent_size: int = 128,
        k_seq_dim: int = 2,
        v_seq_dim: int = 2,
        run_dir: Optional[str] = None,
    ) -> None:
        if not dry_run and (prefill_fn is None or query_fn is None):
            raise ValueError("prefill_fn and query_fn are required unless dry_run=True")
        from .common import Budget

        limits = Budget.from_value(
            budget,
            full_tokens=full_token_slots,
            full_bytes=full_cache_bytes,
            token_limit=token_limit,
            byte_limit=byte_limit,
        )
        self.policy = PrefixSelectionPolicy(
            budget=limits.ratio,
            token_limit=limits.token_limit,
            byte_limit=limits.byte_limit,
            start_size=start_size,
            recent_size=recent_size,
            k_seq_dim=k_seq_dim,
            v_seq_dim=v_seq_dim,
        )
        self.prefill_fn = prefill_fn
        self.query_fn = query_fn
        self.dry_run = bool(dry_run)
        self.run_dir = run_dir
        self.timer = StageTimer()
        self._writer = None  # type: Optional[JsonlWriter]
        self._opened = False
        self.reset()

    def open_stream(self, stream_id: str, metadata: Optional[Mapping[str, Any]] = None) -> None:
        if self._opened:
            raise RuntimeError("a stream is already open")
        self.reset()
        self._opened = True
        self.stream_id = str(stream_id)
        self.metadata = dict(metadata or {})
        if self.run_dir:
            os.makedirs(self.run_dir, exist_ok=True)
            self._writer = JsonlWriter(os.path.join(self.run_dir, "results.jsonl"))
            repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            write_manifest(
                os.path.join(self.run_dir, "manifest.json"),
                repo_root,
                {
                    "adapter": "prefixkv_llava_ov",
                    "dry_run": self.dry_run,
                    "budget": asdict(self.policy.budget),
                    "start_size": self.policy.start_size,
                    "recent_size": self.policy.recent_size,
                },
                {"stream_id": self.stream_id, "metadata": self.metadata},
            )

    def reset(self) -> None:
        self.stream_id = ""
        self.metadata = {}
        self.video_cache = None
        self.query_cache = None
        self.stream_time = 0.0
        self.last_frame_timestamp = float("-inf")
        self.frames_pushed = 0
        self.queries_answered = 0
        self.last_query_cache_bytes = 0
        self.last_selection = {}
        self.timer = StageTimer()
        if hasattr(self, "policy"):
            self.policy.reset()

    def push_frames(self, frames: Iterable[Any], available_until: float) -> Dict[str, Any]:
        if not self._opened:
            raise RuntimeError("open_stream must be called first")
        watermark = float(available_until)
        if watermark + 1e-9 < self.stream_time:
            raise ValueError("stream watermark cannot move backwards")
        normalized = [_frame(item) for item in frames]
        for item in normalized:
            if item.timestamp > watermark + 1e-9:
                raise ValueError("future frame exceeds available_until watermark")
            if item.timestamp + 1e-9 < self.last_frame_timestamp:
                raise ValueError("frames must be pushed in timestamp order")
            self.last_frame_timestamp = item.timestamp
        self.query_cache = None
        with self.timer.measure("video_prefill", frames=len(normalized), until=watermark):
            if not self.dry_run and normalized:
                result = self.prefill_fn([item.frame for item in normalized], self.video_cache)
                cache, attentions, _ = _prefill_result(result)
                self.policy.update_after_video_prefill(cache, attentions)
                with self.timer.measure("prefix_selection"):
                    selection = self.policy.select(cache)
                self.video_cache = selection.cache
                self.last_selection = {
                    "absolute_positions": selection.absolute_positions,
                    "kept_token_slots": selection.kept_token_slots,
                    "cache_bytes": selection.cache_bytes,
                    "budget_token_limit": selection.budget_token_limit,
                    "budget_byte_limit": selection.budget_byte_limit,
                }
        self.frames_pushed += len(normalized)
        self.stream_time = watermark
        return {"frames": len(normalized), "available_until": watermark, "selection": self.last_selection}

    def answer(self, query: Any) -> str:
        if not self._opened:
            raise RuntimeError("open_stream must be called first")
        if isinstance(query, BenchmarkQuery):
            timestamp, query_id = query.timestamp, query.query_id
        elif isinstance(query, Mapping):
            timestamp = float(query.get("timestamp", query.get("query_time")))
            query_id = str(query.get("query_id", query.get("id", self.queries_answered)))
        else:
            raise TypeError("query must be BenchmarkQuery or mapping")
        if abs(float(timestamp) - self.stream_time) > 1e-9:
            raise ValueError("query timestamp must equal the current stream watermark")
        checkpoint = self.policy.checkpoint_lengths(self.video_cache)
        with self.timer.measure("query", query_id=query_id, timestamp=float(timestamp)):
            if self.dry_run:
                text = "[dry-run]"
            else:
                text, generated_cache = _query_result(
                    self.query_fn(query, self.video_cache), self.video_cache
                )
                self.query_cache = generated_cache
                self.last_query_cache_bytes = recursive_nbytes(self.query_cache)
                self.video_cache = self.policy.rollback_query_kv(generated_cache, checkpoint)
                self.query_cache = None
        self.queries_answered += 1
        record = {
            "stream_id": self.stream_id,
            "query_id": query_id,
            "timestamp": float(timestamp),
            "answer": text,
            "query_cache_bytes_before_rollback": self.last_query_cache_bytes,
            "selection": self.last_selection,
            "state_bytes": self.state_bytes(),
        }
        if self._writer:
            self._writer.write(record)
        return text

    query = answer

    def persistent_state(self) -> Mapping[str, Any]:
        return {"video_kv": self.video_cache, "importance": self.policy.importance}

    def static_state(self) -> Mapping[str, Any]:
        return {"budget": self.policy.budget}

    def temporary_state(self) -> Mapping[str, Any]:
        return {
            "query_kv": self.query_cache,
            "last_query_cache_bytes_before_rollback": self.last_query_cache_bytes,
        }

    def state_bytes(self) -> Dict[str, Any]:
        return state_report(self.persistent_state(), self.static_state(), self.temporary_state())

    def close_stream(self) -> Dict[str, Any]:
        if not self._opened:
            return {}
        summary = {
            "stream_id": self.stream_id,
            "frames_pushed": self.frames_pushed,
            "queries_answered": self.queries_answered,
            "timing": self.timer.summary(),
            "state_bytes": self.state_bytes(),
            "selection": self.last_selection,
            "last_query_cache_bytes_before_rollback": self.last_query_cache_bytes,
        }
        if self.run_dir:
            with open(os.path.join(self.run_dir, "summary.json"), "w", encoding="utf-8") as handle:
                json.dump(summary, handle, ensure_ascii=False, indent=2, default=str)
        self._opened = False
        return summary
