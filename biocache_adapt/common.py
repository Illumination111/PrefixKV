"""Dependency-light accounting, timing, budget and run-output utilities."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Mapping, Optional

BUDGET_TIERS = {"12.5%": 0.125, "25%": 0.25, "50%": 0.5}


def parse_budget(value: Any) -> float:
    if isinstance(value, str):
        text = value.strip()
        if text in BUDGET_TIERS:
            return BUDGET_TIERS[text]
        value = float(text[:-1]) / 100.0 if text.endswith("%") else float(text)
    ratio = float(value)
    ratio = ratio / 100.0 if ratio > 1.0 else ratio
    if not 0.0 < ratio <= 1.0:
        raise ValueError("budget ratio must be in (0, 1]")
    return ratio


@dataclass(frozen=True)
class Budget:
    tier: str
    ratio: float
    token_limit: Optional[int] = None
    byte_limit: Optional[int] = None

    @classmethod
    def from_value(
        cls,
        value: Any,
        full_tokens: Optional[int] = None,
        full_bytes: Optional[int] = None,
        token_limit: Optional[int] = None,
        byte_limit: Optional[int] = None,
    ) -> "Budget":
        ratio = parse_budget(value)
        tier = next((name for name, item in BUDGET_TIERS.items() if item == ratio), str(value))
        return cls(
            tier=tier,
            ratio=ratio,
            token_limit=token_limit if token_limit is not None else (
                None if full_tokens is None else int(full_tokens * ratio)
            ),
            byte_limit=byte_limit if byte_limit is not None else (
                None if full_bytes is None else int(full_bytes * ratio)
            ),
        )


def recursive_nbytes(value: Any, seen: Optional[set] = None) -> int:
    if value is None or isinstance(value, (str, int, float, bool)):
        return 0
    if seen is None:
        seen = set()
    marker = id(value)
    if marker in seen:
        return 0
    seen.add(marker)
    if hasattr(value, "element_size") and hasattr(value, "nelement"):
        return int(value.element_size()) * int(value.nelement())
    if hasattr(value, "nbytes"):
        try:
            return int(value.nbytes)
        except (TypeError, ValueError):
            pass
    if isinstance(value, (bytes, bytearray, memoryview)):
        return len(value)
    if isinstance(value, Mapping):
        return sum(recursive_nbytes(item, seen) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return sum(recursive_nbytes(item, seen) for item in value)
    if hasattr(value, "key_cache") and hasattr(value, "value_cache"):
        return recursive_nbytes(value.key_cache, seen) + recursive_nbytes(value.value_cache, seen)
    layers = getattr(value, "layers", None)
    if layers is not None:
        return recursive_nbytes(
            [(getattr(layer, "keys", None), getattr(layer, "values", None)) for layer in layers],
            seen,
        )
    return 0


class StageTimer:
    def __init__(self) -> None:
        self.records = []
        self.totals = {}  # type: Dict[str, float]

    @contextlib.contextmanager
    def measure(self, stage: str, **metadata: Any) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - started
            self.totals[stage] = self.totals.get(stage, 0.0) + elapsed
            self.records.append({"stage": stage, "seconds": elapsed, **metadata})

    def summary(self) -> Dict[str, Any]:
        return {"totals_seconds": dict(self.totals), "records": list(self.records)}


class JsonlWriter:
    def __init__(self, path: str) -> None:
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def write(self, record: Mapping[str, Any]) -> None:
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(record), ensure_ascii=False, default=str) + "\n")


def _git(repo_root: str, *args: str) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "-C", repo_root, *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def write_manifest(
    path: str,
    repo_root: str,
    config: Mapping[str, Any],
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    package_dir = os.path.dirname(os.path.abspath(__file__))
    hashes = {}
    for filename in sorted(os.listdir(package_dir)):
        if filename.endswith(".py"):
            with open(os.path.join(package_dir, filename), "rb") as handle:
                hashes[filename] = hashlib.sha256(handle.read()).hexdigest()
    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "argv": sys.argv,
        "python": sys.version,
        "config": dict(config),
        "git": {
            "commit": _git(repo_root, "rev-parse", "HEAD"),
            "branch": _git(repo_root, "branch", "--show-current"),
            "dirty": bool(_git(repo_root, "status", "--porcelain")),
        },
        "source_sha256": hashes,
        "extra": dict(extra or {}),
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, default=str)
    return manifest


def state_report(
    persistent: Mapping[str, Any],
    static: Mapping[str, Any],
    temporary: Mapping[str, Any],
) -> Dict[str, Any]:
    groups = {
        "persistent": {name: recursive_nbytes(value) for name, value in persistent.items()},
        "static": {name: recursive_nbytes(value) for name, value in static.items()},
        "temporary": {name: recursive_nbytes(value) for name, value in temporary.items()},
    }
    return {
        name: {"items": items, "bytes": sum(items.values())}
        for name, items in groups.items()
    }
