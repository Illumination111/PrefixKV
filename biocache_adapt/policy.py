"""Reusable PrefixKV selection policy for LLaVA-OV and tuple-style KV caches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .common import Budget


@dataclass
class SelectionResult:
    cache: Any
    absolute_positions: List[List[int]]
    kept_token_slots: int
    cache_bytes: int
    budget_token_limit: Optional[int]
    budget_byte_limit: Optional[int]


class PrefixSelectionPolicy:
    """Global importance selection under absolute token/byte limits.

    A "token" is one retained KV position in one layer (a token slot). Byte
    limits use the actual K+V bytes per position of each layer. Prefix and
    recent positions are mandatory. Imports of torch are delayed until cache
    selection is actually executed.
    """

    def __init__(
        self,
        *,
        budget: Any = "25%",
        token_limit: Optional[int] = None,
        byte_limit: Optional[int] = None,
        start_size: int = 4,
        recent_size: int = 128,
        k_seq_dim: int = 2,
        v_seq_dim: int = 2,
    ) -> None:
        self.budget = Budget.from_value(
            budget, token_limit=token_limit, byte_limit=byte_limit
        )
        if self.budget.token_limit is None and self.budget.byte_limit is None:
            raise ValueError("an absolute token_limit or byte_limit is required")
        self.start_size = int(start_size)
        self.recent_size = int(recent_size)
        self.k_seq_dim = int(k_seq_dim)
        self.v_seq_dim = int(v_seq_dim)
        self.importance = []  # type: List[Dict[int, float]]
        self.absolute_positions = []  # type: List[List[int]]
        self.next_absolute_position = 0
        self.last_selection = None  # type: Optional[SelectionResult]

    def reset(self) -> None:
        self.importance = []
        self.absolute_positions = []
        self.next_absolute_position = 0
        self.last_selection = None

    def _sync_positions(self, cache: Sequence[Any]) -> None:
        for layer_index, layer in enumerate(cache):
            length = int(layer[0].shape[self.k_seq_dim])
            while len(self.absolute_positions) <= layer_index:
                self.absolute_positions.append([])
                self.importance.append({})
            positions = self.absolute_positions[layer_index]
            if len(positions) > length:
                positions = positions[:length]
            elif len(positions) < length:
                count = length - len(positions)
                positions = positions + list(
                    range(self.next_absolute_position, self.next_absolute_position + count)
                )
            self.absolute_positions[layer_index] = positions
        if self.absolute_positions:
            self.next_absolute_position = max(
                (max(items) if items else -1) for items in self.absolute_positions
            ) + 1

    def update_after_video_prefill(self, cache: Any, attentions: Optional[Sequence[Any]]) -> None:
        """Update per-absolute-position importance immediately after video prefill."""

        if cache is None:
            return
        self._sync_positions(cache)
        if attentions is None:
            for layer_index, positions in enumerate(self.absolute_positions):
                for rank, position in enumerate(positions):
                    self.importance[layer_index][position] = max(
                        self.importance[layer_index].get(position, 0.0),
                        float(rank + 1) / max(1, len(positions)),
                    )
            return
        for layer_index, attention in enumerate(attentions):
            if layer_index >= len(self.absolute_positions) or attention is None:
                continue
            # [batch, heads, query, key] -> one score per key.
            score = attention.detach().float()
            while score.ndim > 1:
                score = score.mean(dim=0)
            values = score.cpu().tolist()
            positions = self.absolute_positions[layer_index]
            values = values[-len(positions) :]
            offset = len(positions) - len(values)
            for position, value in zip(positions[offset:], values):
                previous = self.importance[layer_index].get(position, 0.0)
                self.importance[layer_index][position] = previous + float(value)

    def _layer_costs(self, cache: Sequence[Any]) -> List[int]:
        costs = []
        for key, value, *_ in cache:
            length = int(key.shape[self.k_seq_dim])
            key_bytes = int(key.element_size()) * int(key.nelement())
            value_bytes = int(value.element_size()) * int(value.nelement())
            costs.append((key_bytes + value_bytes) // max(1, length))
        return costs

    def select(self, cache: Any) -> SelectionResult:
        if cache is None:
            return SelectionResult(None, [], 0, 0, self.budget.token_limit, self.budget.byte_limit)
        self._sync_positions(cache)
        costs = self._layer_costs(cache)
        mandatory = set()
        candidates = []
        for layer_index, positions in enumerate(self.absolute_positions):
            mandatory_positions = set(positions[: self.start_size])
            if self.recent_size:
                mandatory_positions.update(positions[-self.recent_size :])
            for local_index, absolute in enumerate(positions):
                item = (layer_index, local_index)
                if absolute in mandatory_positions:
                    mandatory.add(item)
                else:
                    score = self.importance[layer_index].get(absolute, 0.0)
                    candidates.append((score, absolute, layer_index, local_index))
        mandatory_tokens = len(mandatory)
        mandatory_bytes = sum(costs[layer] for layer, _ in mandatory)
        if self.budget.token_limit is not None and mandatory_tokens > self.budget.token_limit:
            raise ValueError("absolute token budget is smaller than mandatory prefix/recent slots")
        if self.budget.byte_limit is not None and mandatory_bytes > self.budget.byte_limit:
            raise ValueError("absolute byte budget is smaller than mandatory prefix/recent KV")
        kept = set(mandatory)
        used_tokens = mandatory_tokens
        used_bytes = mandatory_bytes
        for _, _, layer_index, local_index in sorted(candidates, reverse=True):
            next_tokens = used_tokens + 1
            next_bytes = used_bytes + costs[layer_index]
            if self.budget.token_limit is not None and next_tokens > self.budget.token_limit:
                continue
            if self.budget.byte_limit is not None and next_bytes > self.budget.byte_limit:
                continue
            kept.add((layer_index, local_index))
            used_tokens, used_bytes = next_tokens, next_bytes
        import torch

        selected_cache = []
        selected_absolute = []
        for layer_index, layer in enumerate(cache):
            key, value = layer[:2]
            indexes = sorted(local for layer_id, local in kept if layer_id == layer_index)
            index = torch.tensor(indexes, dtype=torch.long, device=key.device)
            selected_key = torch.index_select(key, self.k_seq_dim, index)
            selected_value = torch.index_select(value, self.v_seq_dim, index.to(value.device))
            remainder = tuple(layer[2:])
            selected_cache.append((selected_key, selected_value) + remainder)
            selected_absolute.append([self.absolute_positions[layer_index][item] for item in indexes])
        self.absolute_positions = selected_absolute
        output = tuple(selected_cache) if isinstance(cache, tuple) else selected_cache
        result = SelectionResult(
            cache=output,
            absolute_positions=[list(items) for items in selected_absolute],
            kept_token_slots=used_tokens,
            cache_bytes=used_bytes,
            budget_token_limit=self.budget.token_limit,
            budget_byte_limit=self.budget.byte_limit,
        )
        self.last_selection = result
        return result

    def checkpoint_lengths(self, cache: Any) -> List[int]:
        if cache is None:
            return []
        return [int(layer[0].shape[self.k_seq_dim]) for layer in cache]

    def rollback_query_kv(self, cache: Any, lengths: Sequence[int]) -> Any:
        """Drop query/decode KV and restore each layer to its video checkpoint."""

        if cache is None:
            return None
        rolled = []
        for layer, length in zip(cache, lengths):
            key, value = layer[:2]
            key_slices = [slice(None)] * key.ndim
            value_slices = [slice(None)] * value.ndim
            key_slices[self.k_seq_dim] = slice(0, int(length))
            value_slices[self.v_seq_dim] = slice(0, int(length))
            rolled.append((key[tuple(key_slices)], value[tuple(value_slices)]) + tuple(layer[2:]))
        return tuple(rolled) if isinstance(cache, tuple) else rolled
