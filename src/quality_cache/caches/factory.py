"""Cache strategy registry and construction."""

from __future__ import annotations

from collections.abc import Callable

from .base import PrefixCache
from .document import DocumentPrefixCache
from .fixed_block import FixedBlockPrefixCache
from .radix import RadixPrefixCache


CacheBuilder = Callable[..., PrefixCache]


def _document_cache(max_bytes: int, **options) -> PrefixCache:
    return DocumentPrefixCache(
        max_bytes,
        policy=options["policy"],
        max_articles=options["max_articles"],
        l0=options["l0"],
        arena=options.get("arena"),
    )


def _fixed_block_cache(max_bytes: int, **options) -> PrefixCache:
    return FixedBlockPrefixCache(
        max_bytes,
        block_tokens=options["block_tokens"],
        policy=options["policy"],
        l0=options["l0"],
    )


def _radix_cache(max_bytes: int, **options) -> PrefixCache:
    return RadixPrefixCache(
        max_bytes,
        policy=options["policy"],
        l0=options["l0"],
    )


CACHE_BUILDERS: dict[str, CacheBuilder] = {
    "document": _document_cache,
    "fixed-block": _fixed_block_cache,
    "radix": _radix_cache,
}
CACHE_STRATEGIES = tuple(CACHE_BUILDERS)


def new_prefix_cache(
    strategy: str,
    max_bytes: int,
    *,
    policy: str = "lru",
    max_articles: int | None = None,
    block_tokens: int = 16,
    l0=None,
    arena=None,
):
    builder = CACHE_BUILDERS.get(strategy)
    if builder is None:
        raise ValueError(f"cache strategy must be one of {CACHE_STRATEGIES}")
    if strategy != "document" and max_articles is not None:
        raise ValueError("--max-articles is only supported by the document strategy")
    return builder(
        max_bytes,
        policy=policy,
        max_articles=max_articles,
        block_tokens=block_tokens,
        l0=l0,
        arena=arena,
    )
