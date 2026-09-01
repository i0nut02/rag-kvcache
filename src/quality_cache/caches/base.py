"""Shared cache value objects and namespace identity."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..inference.kv_types import StoredBlock
from ..inference.tensors import release_blocks, slice_stored_blocks
from .article import CacheKey


@dataclass
class StoredKV:
    token_count: int
    blocks: list[StoredBlock] = field(default_factory=list)
    simulated_bytes: int | None = None

    @property
    def stored_bytes(self) -> int:
        if self.simulated_bytes is not None:
            return int(self.simulated_bytes)
        return sum(block.stored_bytes for block in self.blocks)

    @property
    def useful_bytes(self) -> int:
        if self.simulated_bytes is not None:
            return int(self.simulated_bytes)
        return sum(
            int(getattr(block, "useful_bytes", block.stored_bytes))
            for block in self.blocks
        )

    @property
    def stranded_bytes(self) -> int:
        return self.stored_bytes - self.useful_bytes

    def slice(self, start: int, end: int) -> "StoredKV":
        if start < 0 or end < start or end > self.token_count:
            raise ValueError("invalid stored KV slice")
        if self.simulated_bytes is not None:
            left = self.simulated_bytes * start // max(1, self.token_count)
            right = self.simulated_bytes * end // max(1, self.token_count)
            return StoredKV(end - start, simulated_bytes=right - left)
        return StoredKV(
            end - start,
            blocks=slice_stored_blocks(self.blocks, start, end),
        )

    def release(self) -> None:
        if self.simulated_bytes is None:
            release_blocks(self.blocks)


@dataclass
class PrefixLookup:
    matched_tokens: int = 0
    payloads: list[StoredKV] = field(default_factory=list)
    requested_tokens: int = 0

    @property
    def blocks(self) -> list[StoredBlock]:
        return [block for payload in self.payloads for block in payload.blocks]

    @property
    def stored_bytes(self) -> int:
        return sum(payload.stored_bytes for payload in self.payloads)

    @property
    def useful_bytes(self) -> int:
        """Matched payload bytes excluding allocator-internal padding."""
        return sum(payload.useful_bytes for payload in self.payloads)

    @property
    def hit_ratio(self) -> float:
        """Fraction of the requested reusable prefix restored from this cache."""
        if self.requested_tokens <= 0:
            return 0.0
        return self.matched_tokens / self.requested_tokens


class PrefixCache(Protocol):
    """Structural interface shared by every prefix-cache organization."""

    strategy: str
    l0: Any
    max_bytes: int
    current_bytes: int

    def __len__(self) -> int: ...

    def lookup(self, key: CacheKey, tokens: list[int]) -> PrefixLookup: ...

    def insert(
        self,
        key: CacheKey,
        tokens: list[int],
        payload: StoredKV,
        prefill_cost_s: float,
    ) -> bool: ...

    def stats(self) -> dict[str, Any]: ...


def cache_namespace(key: CacheKey) -> str:
    raw = "\x1f".join(
        (
            key.model_revision,
            key.tokenizer_revision,
            key.prompt_version,
            key.dtype,
            key.quantization,
        )
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
