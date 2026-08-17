from __future__ import annotations

import re
import time
from typing import Iterable

from ..caches import ArticleKVCache, CacheEntry, CacheKey
from ..caches.offline import simulate_farthest_next_use
from ..costs import PrefillCostModel
from ..data import QualityRequest
from ..memory import current_process_rss_bytes, nonnegative_delta
from ..prompt import PROMPT_VERSION, article_tail, l0_text, suffix_text
from ..schema import RESULT_SCHEMA_VERSION


MODEL_BYTES_PER_TOKEN = {
    "Qwen/Qwen2.5-0.5B-Instruct": 12_288,
    "Qwen/Qwen2.5-1.5B-Instruct": 28_672,
    "Qwen/Qwen2.5-3B-Instruct": 36_864,
}
MODEL_KV_SHAPES = {
    "Qwen/Qwen2.5-0.5B-Instruct": (24, 2),
    "Qwen/Qwen2.5-1.5B-Instruct": (28, 2),
    "Qwen/Qwen2.5-3B-Instruct": (36, 2),
}


def approximate_tokens(text: str) -> int:
    return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))


def article_sizes(
    requests: Iterable[QualityRequest], model: str, storage: str, tokenizer=None,
    block_tokens: int = 256,
) -> tuple[dict[str, int], dict[str, int]]:
    bytes_per_token = MODEL_BYTES_PER_TOKEN.get(model, 12_288)
    token_counts = {}
    sizes = {}
    for request in requests:
        if request.article_id not in sizes:
            tokens = (
                len(tokenizer.encode(article_tail(request.article_text), add_special_tokens=False))
                if tokenizer is not None
                else approximate_tokens(article_tail(request.article_text))
            )
            token_counts[request.article_id] = tokens
            if storage == "cpu-int8":
                layers, kv_heads = MODEL_KV_SHAPES.get(model, (24, 2))
                blocks = (tokens + block_tokens - 1) // block_tokens
                scale_bytes = blocks * layers * 2 * kv_heads * 4
                sizes[request.article_id] = max(
                    1, tokens * bytes_per_token // 2 + scale_bytes
                )
            else:
                sizes[request.article_id] = max(1, tokens * bytes_per_token)
    return sizes, token_counts


def working_set_bytes(sizes: dict[str, int]) -> int:
    return sum(sizes.values())


def simulate_trace(
    requests: list[QualityRequest],
    *,
    policy: str,
    budget_bytes: int,
    model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    storage: str = "cpu-fp16",
    max_articles: int | None = None,
    cold_requests: int = 0,
    offline_prefill: bool = False,
    tokenizer=None,
    block_tokens: int = 256,
    article_size_map: dict[str, int] | None = None,
    article_token_counts: dict[str, int] | None = None,
    prefill_cost_model: PrefillCostModel | None = None,
) -> list[dict]:
    if article_size_map is None or article_token_counts is None:
        sizes, tokens = article_sizes(requests, model, storage, tokenizer, block_tokens)
    else:
        sizes, tokens = article_size_map, article_token_counts
    if not sizes:
        return []
    cost_model = prefill_cost_model or PrefillCostModel.linear()
    rss_baseline = current_process_rss_bytes()
    l0_tokens, suffix_tokens = _non_article_token_counts(requests, tokenizer)
    cache = ArticleKVCache(
        budget_bytes,
        max_articles=max_articles,
        policy=policy,
        minimum_article_bytes=min(sizes.values()),
        l0="pinned-simulation-l0",
    )
    popularity = {article_id: 0 for article_id in sizes}
    for request in requests:
        popularity[request.article_id] += 1
    prefill_cost_total = 0.0
    if offline_prefill:
        ranked = sorted(sizes, key=lambda article_id: (-popularity[article_id], article_id))
        for article_id in ranked:
            if cache.current_bytes + sizes[article_id] > budget_bytes:
                continue
            entry = _entry(
                requests, article_id, sizes, tokens, model, storage, cost_model
            )
            cache.put(entry)
            prefill_cost_total += entry.prefill_cost_s

    rows = []
    for index, request in enumerate(requests):
        key = _key(request, model, storage)
        lookup_start = time.perf_counter()
        entry = cache.get(key)
        lookup_s = time.perf_counter() - lookup_start
        hit = entry is not None
        prefill_s = 0.0
        policy_start = time.perf_counter()
        if not hit:
            prefill_s = cost_model.predict(tokens[request.article_id])
            if sizes[request.article_id] <= budget_bytes:
                entry = CacheEntry(
                    key=key,
                    token_count=tokens[request.article_id],
                    prefill_cost_s=prefill_s,
                    simulated_bytes=sizes[request.article_id],
                )
                cache.put(entry)
        policy_s = time.perf_counter() - policy_start
        stats = cache.stats()
        rss = current_process_rss_bytes()
        article_tokens = tokens[request.article_id]
        total_prompt_tokens = l0_tokens + article_tokens + suffix_tokens[request.request_id]
        cached_prompt_tokens = l0_tokens + (article_tokens if hit else 0)
        document_tree_total_tokens = l0_tokens + article_tokens
        document_tree_cached_tokens = cached_prompt_tokens
        document_tree_hit_ratio = (
            document_tree_cached_tokens / max(1, document_tree_total_tokens)
        )
        partial_document_tree_hit = (
            0 < document_tree_cached_tokens < document_tree_total_tokens
        )
        rows.append(
            {
                "result_schema_version": RESULT_SCHEMA_VERSION,
                "request_index": index,
                "request_id": request.request_id,
                "article_id": request.article_id,
                "cache_hit": hit,
                "partial_cache_hit": 0 < cached_prompt_tokens < total_prompt_tokens,
                "partial_article_hit": partial_document_tree_hit,
                "partial_article_text_hit": False,
                "partial_document_tree_hit": partial_document_tree_hit,
                "root_only_hit": partial_document_tree_hit,
                "cache_hit_ratio": cached_prompt_tokens / max(1, total_prompt_tokens),
                "article_cache_hit_ratio": document_tree_hit_ratio,
                "article_text_cache_hit_ratio": float(hit),
                "document_tree_hit_ratio": document_tree_hit_ratio,
                "cached_prompt_tokens": cached_prompt_tokens,
                "total_prompt_tokens": total_prompt_tokens,
                "document_tree_cached_tokens": document_tree_cached_tokens,
                "document_tree_total_tokens": document_tree_total_tokens,
                "uncached_suffix_tokens": suffix_tokens[request.request_id],
                "matched_prefix_tokens": article_tokens if hit else 0,
                "matched_cache_bytes": sizes[request.article_id] if hit else 0,
                "phase": "cold" if index < cold_requests else "steady",
                "cold_requests": cold_requests,
                "article_tokens": tokens[request.article_id],
                "article_bytes": sizes[request.article_id],
                "matched_prefill_tokens": tokens[request.article_id] if hit else 0,
                "avoided_prefill_tokens": tokens[request.article_id] if hit else 0,
                "lookup_s": lookup_s,
                "load_s": 0.0,
                "transfer_s": 0.0,
                "dequant_s": 0.0,
                "policy_s": policy_s,
                "prefill_s": prefill_s,
                "offline_prefill_s": prefill_cost_total,
                "ttft_s": -1.0,
                "prefill_cost_model": cost_model.source,
                "cache_bytes": stats["cache_bytes"],
                "metadata_bytes": stats["metadata_bytes"],
                "cache_footprint_bytes": stats["cache_footprint_bytes"],
                "occupancy": stats["occupancy"],
                "insertions": stats["insertions"],
                "evictions": stats["evictions"],
                "cached_tokens": stats["cached_tokens"],
                "process_rss_bytes": rss,
                "process_rss_delta_bytes": nonnegative_delta(rss, rss_baseline),
                "mps_allocated_bytes": 0,
                "mps_driver_bytes": 0,
                "mps_allocated_delta_bytes": 0,
                "mps_driver_delta_bytes": 0,
                "cuda_allocated_bytes": 0,
                "cuda_reserved_bytes": 0,
                "cuda_allocated_delta_bytes": 0,
                "cuda_reserved_delta_bytes": 0,
                "policy": policy,
                "storage": storage,
                "budget_bytes": budget_bytes,
                "model": model,
                "gold_label": request.question.answer_letter,
                "difficult": request.question.difficult,
                "predicted_label": None,
            }
        )
    return rows


def farthest_next_use_rows(
    requests: list[QualityRequest], budget_bytes: int, model: str, storage: str,
    tokenizer=None, block_tokens: int = 256,
    article_size_map: dict[str, int] | None = None,
    article_token_counts: dict[str, int] | None = None,
    prefill_cost_model: PrefillCostModel | None = None,
) -> list[dict]:
    if article_size_map is None or article_token_counts is None:
        sizes, tokens = article_sizes(requests, model, storage, tokenizer, block_tokens)
    else:
        sizes, tokens = article_size_map, article_token_counts
    l0_tokens, suffix_tokens = _non_article_token_counts(requests, tokenizer)
    cost_model = prefill_cost_model or PrefillCostModel.linear()
    rss_baseline = current_process_rss_bytes()
    result = simulate_farthest_next_use(
        [request.article_id for request in requests], sizes, budget_bytes
    )
    rows = []
    for index, (request, hit) in enumerate(zip(requests, result.hits)):
        prefill_s = 0.0 if hit else cost_model.predict(tokens[request.article_id])
        article_tokens = tokens[request.article_id]
        total_prompt_tokens = l0_tokens + article_tokens + suffix_tokens[request.request_id]
        cached_prompt_tokens = l0_tokens + (article_tokens if hit else 0)
        document_tree_total_tokens = l0_tokens + article_tokens
        document_tree_cached_tokens = cached_prompt_tokens
        document_tree_hit_ratio = (
            document_tree_cached_tokens / max(1, document_tree_total_tokens)
        )
        partial_document_tree_hit = (
            0 < document_tree_cached_tokens < document_tree_total_tokens
        )
        rows.append(
            {
                "result_schema_version": RESULT_SCHEMA_VERSION,
                "request_index": index,
                "request_id": request.request_id,
                "article_id": request.article_id,
                "cache_hit": hit,
                "partial_cache_hit": 0 < cached_prompt_tokens < total_prompt_tokens,
                "partial_article_hit": partial_document_tree_hit,
                "partial_article_text_hit": False,
                "partial_document_tree_hit": partial_document_tree_hit,
                "root_only_hit": partial_document_tree_hit,
                "cache_hit_ratio": cached_prompt_tokens / max(1, total_prompt_tokens),
                "article_cache_hit_ratio": document_tree_hit_ratio,
                "article_text_cache_hit_ratio": float(hit),
                "document_tree_hit_ratio": document_tree_hit_ratio,
                "cached_prompt_tokens": cached_prompt_tokens,
                "total_prompt_tokens": total_prompt_tokens,
                "document_tree_cached_tokens": document_tree_cached_tokens,
                "document_tree_total_tokens": document_tree_total_tokens,
                "uncached_suffix_tokens": suffix_tokens[request.request_id],
                "matched_prefix_tokens": article_tokens if hit else 0,
                "matched_cache_bytes": sizes[request.article_id] if hit else 0,
                "article_tokens": tokens[request.article_id],
                "article_bytes": sizes[request.article_id],
                "matched_prefill_tokens": tokens[request.article_id] if hit else 0,
                "avoided_prefill_tokens": tokens[request.article_id] if hit else 0,
                "ttft_s": -1.0,
                "lookup_s": 0.0,
                "load_s": 0.0,
                "transfer_s": 0.0,
                "dequant_s": 0.0,
                "policy_s": 0.0,
                "prefill_s": prefill_s,
                "policy": "farthest-next-use",
                "offline_policy_optimal": result.optimal_for_equal_sizes,
                "offline_policy_scope": (
                    "equal-size optimal"
                    if result.optimal_for_equal_sizes
                    else "variable-size clairvoyant heuristic"
                ),
                "prefill_cost_model": cost_model.source,
                "storage": storage,
                "budget_bytes": budget_bytes,
                "model": model,
                "cache_bytes": result.bytes_after[index],
                "metadata_bytes": 0,
                "cache_footprint_bytes": result.bytes_after[index],
                "occupancy": result.bytes_after[index] / budget_bytes,
                "insertions": result.insertions_after[index],
                "evictions": result.evictions_after[index],
                "cached_tokens": 0,
                "process_rss_bytes": current_process_rss_bytes(),
                "process_rss_delta_bytes": nonnegative_delta(
                    current_process_rss_bytes(), rss_baseline
                ),
                "mps_allocated_bytes": 0,
                "mps_driver_bytes": 0,
                "mps_allocated_delta_bytes": 0,
                "mps_driver_delta_bytes": 0,
                "cuda_allocated_bytes": 0,
                "cuda_reserved_bytes": 0,
                "cuda_allocated_delta_bytes": 0,
                "cuda_reserved_delta_bytes": 0,
            }
        )
    return rows
def _non_article_token_counts(requests, tokenizer):
    encode = (
        (lambda text: len(tokenizer.encode(text, add_special_tokens=False)))
        if tokenizer is not None
        else approximate_tokens
    )
    l0_tokens = encode(l0_text())
    suffix_tokens = {}
    for request in requests:
        if request.request_id not in suffix_tokens:
            suffix_tokens[request.request_id] = encode(suffix_text(request.question))
    return l0_tokens, suffix_tokens


def _key(request: QualityRequest, model: str, storage: str) -> CacheKey:
    return CacheKey(
        article_id=request.article_id,
        article_hash=request.article_hash,
        model_revision=model,
        tokenizer_revision=model,
        prompt_version=PROMPT_VERSION,
        dtype="float16",
        quantization="int8-per-kv-head" if storage == "cpu-int8" else "none",
    )


def _entry(requests, article_id, sizes, tokens, model, storage, cost_model):
    request = next(item for item in requests if item.article_id == article_id)
    return CacheEntry(
        key=_key(request, model, storage),
        token_count=tokens[article_id],
        prefill_cost_s=cost_model.predict(tokens[article_id]),
        simulated_bytes=sizes[article_id],
    )
