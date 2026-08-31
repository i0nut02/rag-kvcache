"""Compressed token radix-tree cache strategy."""

from __future__ import annotations

import heapq
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
    node_id: int = 0
    heap_version: int = 0


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
        self._nodes_by_id: dict[int, _RadixNode] = {}
        self._victim_heap: list[tuple[tuple[Any, ...], int, int]] = []
        self._next_node_id = 1
        self._cached_tokens = 0
        self.current_bytes = 0
        self.clock = 0
        self.gdsf_clock = 0.0
        self.insertions = 0
        self.evictions = 0
        self._stats_cache: dict[str, Any] | None = None

    def __len__(self):
        return len(self._nodes_by_id)

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
                self._refresh_leaf(child)
                payloads.append(child.payload.slice(0, common))
                position += common
            if common != child.token_count:
                break
            node = child
        self._maybe_compact_heap()
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
                self._attach_new_node(node, new)
                protected.add(new.node_id)
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
            child_was_leaf = not child.children
            if child_was_leaf:
                self._invalidate_candidate(child)
            old_child_bytes = child.payload.stored_bytes
            old_child_tokens = child.token_count
            common_payload = child.payload.slice(0, common)
            remaining_payload = child.payload.slice(common, child.payload.token_count)
            common_node = _RadixNode(
                child.tokens[:common],
                common_payload,
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
            child.tokens = child.tokens[common:]
            child.payload = remaining_payload
            child.token_count -= common
            child.prefill_cost_s = (
                child.prefill_cost_s * child.token_count / max(1, old_child_tokens)
            )
            child.priority = self._gdsf_priority(child)
            child.parent = common_node
            common_node.children[child.tokens[0]] = child
            self.current_bytes += child.payload.stored_bytes - old_child_bytes
            self._cached_tokens += child.token_count - old_child_tokens
            self._register_node(common_node)
            if child_was_leaf:
                self._schedule_leaf(child)
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
                self._attach_new_node(common_node, new)
                protected.add(new.node_id)
                self.insertions += 1
                node = new
                position = len(tokens)
            break
        if position == len(tokens):
            node.terminal_ids.add((key.article_id, key.article_hash))
        while self.current_bytes > self.max_bytes:
            self._evict_leaf(protected)
        self._assert_budget()
        self._maybe_compact_heap()
        self._invalidate_stats()
        return self.lookup(key, tokens).matched_tokens > 0

    def _evict_leaf(self, protected: set[int] | None = None):
        protected = protected or set()
        victim = self._pop_victim(protected)
        if victim is None and protected:
            victim = self._pop_victim(set())
        if victim is None:
            raise RuntimeError("radix cache has no eviction victim")
        parent = victim.parent
        if parent is None:
            raise RuntimeError("cannot evict radix root")
        removed = parent.children.pop(victim.tokens[0])
        if removed is not victim:
            raise RuntimeError("radix parent no longer owns selected victim")
        if self.policy == "gdsf":
            self.gdsf_clock = victim.priority
        self._unregister_node(victim)
        victim.payload.blocks.clear()
        self.evictions += 1
        if parent.node_id and not parent.children:
            self._schedule_leaf(parent)
        self._invalidate_stats()

    def _victim_priority(self, node: _RadixNode) -> tuple[Any, ...]:
        if self.policy == "lru":
            return (node.last_access, node.inserted_at, node.node_id)
        if self.policy == "lfu":
            return (
                node.frequency,
                node.last_access,
                node.inserted_at,
                node.node_id,
            )
        if self.policy == "fifo":
            return (node.inserted_at, node.node_id)
        return (node.priority, node.last_access, node.inserted_at, node.node_id)

    def _register_node(self, node: _RadixNode) -> None:
        if node.node_id:
            raise RuntimeError("radix node is already registered")
        node.node_id = self._next_node_id
        self._next_node_id += 1
        self._nodes_by_id[node.node_id] = node
        self.current_bytes += node.payload.stored_bytes
        self._cached_tokens += node.token_count

    def _unregister_node(self, node: _RadixNode) -> None:
        node.heap_version += 1
        removed = self._nodes_by_id.pop(node.node_id, None)
        if removed is not node:
            raise RuntimeError("radix victim is not registered")
        self.current_bytes -= node.payload.stored_bytes
        self._cached_tokens -= node.token_count

    def _attach_new_node(self, parent: _RadixNode, node: _RadixNode) -> None:
        if parent.node_id and not parent.children:
            self._invalidate_candidate(parent)
        node.parent = parent
        parent.children[node.tokens[0]] = node
        self._register_node(node)
        self._schedule_leaf(node)

    @staticmethod
    def _invalidate_candidate(node: _RadixNode) -> None:
        node.heap_version += 1

    def _schedule_leaf(self, node: _RadixNode) -> None:
        if node.node_id == 0 or node.children:
            return
        node.heap_version += 1
        heapq.heappush(
            self._victim_heap,
            (self._victim_priority(node), node.heap_version, node.node_id),
        )

    def _refresh_leaf(self, node: _RadixNode) -> None:
        if not node.children:
            self._schedule_leaf(node)

    def _pop_victim(self, protected: set[int]) -> _RadixNode | None:
        deferred: list[tuple[tuple[Any, ...], int, int]] = []
        victim = None
        while self._victim_heap:
            item = heapq.heappop(self._victim_heap)
            _, version, node_id = item
            node = self._nodes_by_id.get(node_id)
            if node is None or node.heap_version != version or node.children:
                continue
            if node_id in protected:
                deferred.append(item)
                continue
            victim = node
            break
        for item in deferred:
            heapq.heappush(self._victim_heap, item)
        return victim

    def _maybe_compact_heap(self) -> None:
        if len(self._victim_heap) <= max(1024, 4 * len(self._nodes_by_id)):
            return
        self._victim_heap.clear()
        for node in self._nodes_by_id.values():
            if not node.children:
                self._schedule_leaf(node)

    def _gdsf_priority(self, node: _RadixNode) -> float:
        return (
            self.gdsf_clock
            + node.frequency
            * max(node.prefill_cost_s, 1e-12)
            / max(node.payload.stored_bytes, 1)
        )

    def _nodes(self):
        yield from self._nodes_by_id.values()

    def stats(self) -> dict[str, Any]:
        if self._stats_cache is not None:
            return self._with_heap_metadata(dict(self._stats_cache))
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
        metadata_bytes += sys.getsizeof(self._nodes_by_id)
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
            "radix_nodes": len(self._nodes_by_id),
            "cached_tokens": self._cached_tokens,
            "insertions": self.insertions,
            "evictions": self.evictions,
            "policy": self.policy,
            "cache_strategy": self.strategy,
            "useful_bytes": self.current_bytes,
            "shared_bytes": shared,
            "stranded_bytes": 0,
        }
        self._stats_cache = dict(stats)
        return self._with_heap_metadata(stats)

    def _with_heap_metadata(self, stats: dict[str, Any]) -> dict[str, Any]:
        heap_bytes = sys.getsizeof(self._victim_heap) + len(
            self._victim_heap
        ) * sys.getsizeof(((), 0, 0))
        stats["metadata_bytes"] += heap_bytes
        stats["cache_footprint_bytes"] = self.current_bytes + stats["metadata_bytes"]
        return stats

    def _assert_budget(self):
        assert self.current_bytes <= self.max_bytes
        assert self.current_bytes >= 0
        assert self._cached_tokens >= 0

    def _invalidate_stats(self) -> None:
        self._stats_cache = None


def _common_prefix(edge: tuple[int, ...], tokens: list[int], position: int) -> int:
    limit = min(len(edge), len(tokens) - position)
    common = 0
    while common < limit and edge[common] == tokens[position + common]:
        common += 1
    return common
