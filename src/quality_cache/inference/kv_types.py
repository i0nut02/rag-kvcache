"""Shared structural types for stored and model-ready KV state."""

from __future__ import annotations

from typing import Any, Protocol


LegacyKV = tuple[tuple[Any, Any], ...]


class StoredBlock(Protocol):
    """Minimal interface shared by tensor-backed and arena-backed blocks."""

    token_count: int

    @property
    def stored_bytes(self) -> int: ...

    @property
    def useful_bytes(self) -> int: ...


class LayeredStoredBlock(StoredBlock, Protocol):
    """Object-backed block whose per-layer tensors can be sliced/restored."""

    layers: tuple[tuple[Any, Any], ...]
