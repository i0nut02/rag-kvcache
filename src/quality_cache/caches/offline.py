"""Offline farthest-next-use cache simulation."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from math import inf


@dataclass(frozen=True)
class FarthestNextUseResult:
    hits: tuple[bool, ...]
    evictions: int
    final_bytes: int
    bytes_after: tuple[int, ...]
    evictions_after: tuple[int, ...]
    insertions_after: tuple[int, ...]
    optimal_for_equal_sizes: bool

    @property
    def hit_rate(self) -> float:
        return sum(self.hits) / max(1, len(self.hits))


def simulate_farthest_next_use(
    trace: list[str],
    sizes: dict[str, int],
    max_bytes: int,
    *,
    max_articles: int | None = None,
) -> FarthestNextUseResult:
    """Offline farthest-next-use policy under byte and optional count limits.

    This is Belady-optimal only when every cached object has the same size. With
    variable-size articles it is a useful clairvoyant heuristic, not an upper
    bound on the achievable hit rate.
    """
    future: dict[str, deque[int]] = defaultdict(deque)
    for index, article_id in enumerate(trace):
        future[article_id].append(index)
    resident: set[str] = set()
    current_bytes = 0
    hits = []
    evictions = 0
    insertions = 0
    bytes_after = []
    evictions_after = []
    insertions_after = []
    for index, article_id in enumerate(trace):
        future[article_id].popleft()
        if article_id in resident:
            hits.append(True)
            bytes_after.append(current_bytes)
            evictions_after.append(evictions)
            insertions_after.append(insertions)
            continue
        hits.append(False)
        size = sizes[article_id]
        if size > max_bytes:
            bytes_after.append(current_bytes)
            evictions_after.append(evictions)
            insertions_after.append(insertions)
            continue
        while current_bytes + size > max_bytes or (
            max_articles is not None and len(resident) + 1 > max_articles
        ):
            victim = max(
                resident,
                key=lambda candidate: future[candidate][0] if future[candidate] else inf,
            )
            resident.remove(victim)
            current_bytes -= sizes[victim]
            evictions += 1
        resident.add(article_id)
        current_bytes += size
        insertions += 1
        assert current_bytes <= max_bytes
        bytes_after.append(current_bytes)
        evictions_after.append(evictions)
        insertions_after.append(insertions)
    return FarthestNextUseResult(
        tuple(hits), evictions, current_bytes,
        tuple(bytes_after), tuple(evictions_after), tuple(insertions_after),
        len(set(sizes.values())) <= 1,
    )
