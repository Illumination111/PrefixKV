"""Common JSON/JSONL benchmark loader for timestamped streaming video QA."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional


@dataclass(frozen=True)
class BenchmarkQuery:
    query_id: str
    timestamp: float
    question: str
    options: Optional[List[str]] = None
    answer: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkSample:
    sample_id: str
    video_path: str
    duration: Optional[float]
    queries: List[BenchmarkQuery]
    metadata: Dict[str, Any] = field(default_factory=dict)


def _first(record: Mapping[str, Any], names: Iterable[str], default: Any = None) -> Any:
    for name in names:
        if name in record and record[name] is not None:
            return record[name]
    return default


def _query(record: Mapping[str, Any], index: int, fallback_time: Optional[float]) -> BenchmarkQuery:
    timestamp = _first(record, ("timestamp", "query_timestamp", "query_time", "time"), fallback_time)
    if timestamp is None:
        raise ValueError("query is missing timestamp/query_time")
    known = {
        "id", "query_id", "qid", "timestamp", "query_timestamp", "query_time", "time",
        "question", "query", "prompt", "options", "choices", "answer", "label",
    }
    return BenchmarkQuery(
        query_id=str(_first(record, ("query_id", "qid", "id"), index)),
        timestamp=float(timestamp),
        question=str(_first(record, ("question", "query", "prompt"), "")),
        options=_first(record, ("options", "choices")),
        answer=_first(record, ("answer", "label")),
        metadata={key: value for key, value in record.items() if key not in known},
    )


def normalize_sample(record: Mapping[str, Any], index: int = 0) -> BenchmarkSample:
    raw_queries = _first(record, ("queries", "conversations", "questions"))
    duration = _first(record, ("duration", "video_duration"))
    if raw_queries is None:
        raw_queries = [record]
    if not isinstance(raw_queries, list):
        raise ValueError("queries/conversations must be a list")
    fallback = float(duration) if duration is not None and len(raw_queries) == 1 else None
    queries = sorted(
        [_query(item, query_index, fallback) for query_index, item in enumerate(raw_queries)],
        key=lambda item: item.timestamp,
    )
    known = {
        "id", "sample_id", "video_id", "video", "video_path", "path",
        "duration", "video_duration", "queries", "conversations", "questions",
    }
    return BenchmarkSample(
        sample_id=str(_first(record, ("sample_id", "video_id", "id"), index)),
        video_path=str(_first(record, ("video_path", "video", "path"), "")),
        duration=None if duration is None else float(duration),
        queries=queries,
        metadata={key: value for key, value in record.items() if key not in known},
    )


def load_benchmark(path: str) -> List[BenchmarkSample]:
    with open(path, "r", encoding="utf-8") as handle:
        if path.lower().endswith(".jsonl"):
            records = [json.loads(line) for line in handle if line.strip()]
        else:
            payload = json.load(handle)
            if isinstance(payload, list):
                records = payload
            elif isinstance(payload, dict):
                records = _first(payload, ("samples", "data", "videos"), [payload])
            else:
                raise ValueError("benchmark root must be an object or array")
    if not isinstance(records, list):
        raise ValueError("benchmark samples must be a list")
    return [normalize_sample(record, index) for index, record in enumerate(records)]
