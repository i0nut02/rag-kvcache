"""Cache strategy registry and construction."""

from __future__ import annotations

from .document import DocumentPrefixCache
from .fixed_block import FixedBlockPrefixCache
from .radix import RadixPrefixCache


CACHE_STRATEGIES = ("document", "fixed-block", "radix")


def new_prefix_cache(
    strategy: str,
    max_bytes: int,
    *,
    policy: str = "lru",
    max_articles: int | None = None,
    block_tokens: int = 16,
    l0=None,
):
    if strategy == "document":
        return DocumentPrefixCache(
            max_bytes, policy=policy, max_articles=max_articles, l0=l0
        )
    if max_articles is not None:
        raise ValueError("--max-articles is only supported by the document strategy")
    if strategy == "fixed-block":
        return FixedBlockPrefixCache(
            max_bytes, block_tokens=block_tokens, policy=policy, l0=l0
        )
    if strategy == "radix":
        return RadixPrefixCache(max_bytes, policy=policy, l0=l0)
    raise ValueError(f"cache strategy must be one of {CACHE_STRATEGIES}")
