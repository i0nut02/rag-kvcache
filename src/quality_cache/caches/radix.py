"""Compressed token radix-tree cache strategy."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any

from .article import POLICIES, CacheKey
from .base import PrefixLookup, StoredKV, cache_namespace


@dataclass
class _RadixNode:
    tokens: tuple[int, ...]
    payload: StoredKV
    token_count: int
    parent: "_RadixNode | None" = None
    children: dict[int, "_RadixNode"] = field(default_factory=dict)
    terminal_ids: set[tuple[str, str]] = field(default_factory=set)
    prefill_cost_s: float = 0.0
    frequency: int = 1
    inserted_at: int = 0
    last_access: int = 0
    priority: float = 0.0


class RadixPrefixCache:
    """Compressed token radix tree with policy-based leaf eviction."""

    strategy = "radix"

    def __init__(self, max_bytes: int, *, policy: str = "lru", l0=None):
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        if policy not in POLICIES:
            raise ValueError(f"policy must be one of {POLICIES}")
        self.max_bytes = int(max_bytes)
        self.policy = policy
        self.l0 = l0
        self.roots: dict[str, _RadixNode] = {}
        self.current_bytes = 0
        self.clock = 0
        self.gdsf_clock = 0.0
        self.insertions = 0
        self.evictions = 0
        self._stats_cache: dict[str, Any] | None = None

    def __len__(self):
        return sum(1 for _ in self._nodes())

    def _root(self, key: CacheKey) -> _RadixNode:
        namespace = cache_namespace(key)
        root = self.roots.get(namespace)
        if root is None:
            root = _RadixNode((), StoredKV(0), 0)
            self.roots[namespace] = root
            self._invalidate_stats()
        return root

    def lookup(self, key: CacheKey, tokens: list[int]) -> PrefixLookup:
        self.clock += 1
        node = self._root(key)
        position = 0
        payloads = []
        while position < len(tokens):
            child = node.children.get(tokens[position])
            if child is None:
                break
            common = _common_prefix(child.tokens, tokens, position)
            if common:
                child.frequency += 1
                child.last_access = self.clock
                if self.policy == "gdsf":
                    child.priority = self._gdsf_priority(child)
                payloads.append(child.payload.slice(0, common))
                position += common
            if common != child.token_count:
                break
            node = child
        return PrefixLookup(position, payloads, len(tokens))

    def insert(
        self, key: CacheKey, tokens: list[int], payload: StoredKV, prefill_cost_s: float
    ) -> bool:
        if not tokens:
            return False
        self.clock += 1
        node = self._root(key)
        position = 0
        protected: set[int] = set()
        while position < len(tokens):
            child = node.children.get(tokens[position])
            if child is None:
                suffix = tuple(tokens[position:])
                new = _RadixNode(
                    suffix,
                    payload.slice(position, len(tokens)),
                    len(suffix),
                    parent=node,
                    prefill_cost_s=prefill_cost_s,
                    inserted_at=self.clock,
                    last_access=self.clock,
                )
                new.priority = self._gdsf_priority(new)
                new.terminal_ids.add((key.article_id, key.article_hash))
                node.children[suffix[0]] = new
                protected.add(id(new))
                self.current_bytes += new.payload.stored_bytes
                self.insertions += 1
                node = new
                position = len(tokens)
                break
            common = _common_prefix(child.tokens, tokens, position)
            if common == child.token_count:
                child.last_access = self.clock
                node = child
                position += common
                continue
            common_node = _RadixNode(
                child.tokens[:common],
                child.payload.slice(0, common),
                common,
                parent=node,
                prefill_cost_s=child.prefill_cost_s
                * common
                / max(1, child.token_count),
                frequency=child.frequency,
                inserted_at=child.inserted_at,
                last_access=self.clock,
            )
            common_node.priority = self._gdsf_priority(common_node)
            node.children[common_node.tokens[0]] = common_node
            old_child_tokens = child.token_count
            child.tokens = child.tokens[common:]
            child.payload = child.payload.slice(common, child.payload.token_count)
            child.token_count -= common
            child.prefill_cost_s = (
                child.prefill_cost_s * child.token_count / max(1, old_child_tokens)
            )
            child.priority = self._gdsf_priority(child)
            child.parent = common_node
            common_node.children[child.tokens[0]] = child
            self.insertions += 1
            if position + common == len(tokens):
                common_node.terminal_ids.add((key.article_id, key.article_hash))
                node = common_node
                position = len(tokens)
            else:
                suffix = tuple(tokens[position + common :])
                new = _RadixNode(
                    suffix,
                    payload.slice(position + common, len(tokens)),
                    len(suffix),
                    parent=common_node,
                    prefill_cost_s=prefill_cost_s
                    * len(suffix)
                    / max(1, len(tokens) - position),
                    inserted_at=self.clock,
                    last_access=self.clock,
                )
                new.priority = self._gdsf_priority(new)
                new.terminal_ids.add((key.article_id, key.article_hash))
                common_node.children[suffix[0]] = new
                protected.add(id(new))
                self.insertions += 1
                node = new
                position = len(tokens)
            break
        if position == len(tokens):
            node.terminal_ids.add((key.article_id, key.article_hash))
        self.current_bytes = sum(item.payload.stored_bytes for item in self._nodes())
        while self.current_bytes > self.max_bytes:
            self._evict_leaf(protected)
        self._assert_budget()
        self._invalidate_stats()
        return self.lookup(key, tokens).matched_tokens > 0

    def _evict_leaf(self, protected: set[int] | None = None):
        protected = protected or set()
        leaves = [
            node
            for node in self._nodes()
            if not node.children and id(node) not in protected
        ]
        if not leaves and protected:
            leaves = [node for node in self._nodes() if not node.children]
        if not leaves:
            raise RuntimeError("radix cache has no eviction victim")
        victim = self._victim(leaves)
        parent = victim.parent
        if parent is None:
            raise RuntimeError("cannot evict radix root")
        parent.children.pop(victim.tokens[0])
        self.current_bytes -= victim.payload.stored_bytes
        if self.policy == "gdsf":
            self.gdsf_clock = victim.priority
        victim.payload.blocks.clear()
        self.evictions += 1
        self._invalidate_stats()

    def _victim(self, leaves: list[_RadixNode]) -> _RadixNode:
        if self.policy == "lru":
            return min(leaves, key=lambda node: (node.last_access, node.inserted_at))
        if self.policy == "lfu":
            return min(
                leaves,
                key=lambda node: (node.frequency, node.last_access, node.inserted_at),
            )
        if self.policy == "fifo":
            return min(leaves, key=lambda node: node.inserted_at)
        return min(
            leaves,
            key=lambda node: (node.priority, node.last_access, node.inserted_at),
        )

    def _gdsf_priority(self, node: _RadixNode) -> float:
        return (
            self.gdsf_clock
            + node.frequency
            * max(node.prefill_cost_s, 1e-12)
            / max(node.payload.stored_bytes, 1)
        )

    def _nodes(self):
        stack = [child for root in self.roots.values() for child in root.children.values()]
        while stack:
            node = stack.pop()
            yield node
            stack.extend(node.children.values())

    def stats(self) -> dict[str, Any]:
        if self._stats_cache is not None:
            return dict(self._stats_cache)
        nodes = list(self._nodes())

        terminal_counts: dict[int, int] = {}

        def count_terminals(node: _RadixNode) -> int:
            count = len(node.terminal_ids) + sum(
                count_terminals(child) for child in node.children.values()
            )
            terminal_counts[id(node)] = count
            return count

        for root in self.roots.values():
            for child in root.children.values():
                count_terminals(child)

        shared = sum(
            node.payload.stored_bytes
            for node in nodes
            if terminal_counts[id(node)] > 1
        )
        metadata_bytes = sys.getsizeof(self.roots)
        for namespace, root in self.roots.items():
            metadata_bytes += sys.getsizeof(namespace) + sys.getsizeof(root)
            metadata_bytes += sys.getsizeof(root.children)
        for node in nodes:
            metadata_bytes += sys.getsizeof(node) + sys.getsizeof(node.tokens)
            metadata_bytes += len(node.tokens) * sys.getsizeof(0)
            metadata_bytes += sys.getsizeof(node.children)
            metadata_bytes += sys.getsizeof(node.terminal_ids)
            metadata_bytes += sys.getsizeof(node.payload)
            metadata_bytes += sys.getsizeof(node.payload.blocks)
        stats = {
            "cache_bytes": self.current_bytes,
            "metadata_bytes": metadata_bytes,
            "cache_footprint_bytes": self.current_bytes + metadata_bytes,
            "budget_bytes": self.max_bytes,
            "occupancy": self.current_bytes / self.max_bytes,
            "cached_articles": sum(len(node.terminal_ids) for node in nodes),
            "cached_documents": sum(len(node.terminal_ids) for node in nodes),
            "cached_blocks": 0,
            "radix_nodes": len(nodes),
            "cached_tokens": sum(node.token_count for node in nodes),
            "insertions": self.insertions,
            "evictions": self.evictions,
            "policy": self.policy,
            "cache_strategy": self.strategy,
            "useful_bytes": self.current_bytes,
            "shared_bytes": shared,
            "stranded_bytes": 0,
        }
        self._stats_cache = dict(stats)
        return stats

    def _assert_budget(self):
        assert self.current_bytes <= self.max_bytes

    def _invalidate_stats(self) -> None:
        self._stats_cache = None


def _common_prefix(edge: tuple[int, ...], tokens: list[int], position: int) -> int:
    limit = min(len(edge), len(tokens) - position)
    common = 0
    while common < limit and edge[common] == tokens[position + common]:
        common += 1
    return common
