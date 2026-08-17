from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ..caches import CacheKey, StoredKV, new_prefix_cache
from ..costs import load_prefill_cost_model
from ..data import QualityRequest
from ..memory import current_process_rss_bytes, nonnegative_delta
from ..prompt import PROMPT_VERSION, encode_parts, l0_text, suffix_text
from ..schema import RESULT_SCHEMA_VERSION


class NoInferenceRunner:
    """Tokenizer/config-only runner; never constructs a model or KV tensor."""

    model_weights_loaded = False

    def __init__(
        self,
        model_name: str,
        *,
        tokenizer_source: str | None = None,
        device: str = "mps",
        dtype: str = "float16",
        revision: str | None = None,
        prefill_calibration: str | Path | None = None,
    ):
        from transformers import AutoConfig, AutoTokenizer

        source = tokenizer_source or model_name
        local = Path(source).exists()
        self.model_name = model_name
        self.device_name = device
        self.dtype_name = dtype
        self.tokenizer = AutoTokenizer.from_pretrained(
            source,
            revision=None if local else revision,
            local_files_only=local,
        )
        config_source = source if local and (Path(source) / "config.json").exists() else model_name
        config_local = Path(config_source).exists()
        self.config = AutoConfig.from_pretrained(
            config_source,
            revision=None if config_local else revision,
            local_files_only=config_local,
        )
        self.model_revision = str(
            getattr(self.config, "_commit_hash", None) or revision or model_name
        )
        self.tokenizer_revision = str(
            getattr(self.tokenizer, "_commit_hash", None)
            or getattr(self.tokenizer, "init_kwargs", {}).get("_commit_hash")
            or revision
            or source
        )
        self.layers = int(self.config.num_hidden_layers)
        self.kv_heads = int(
            getattr(self.config, "num_key_value_heads", self.config.num_attention_heads)
        )
        self.head_dim = int(
            getattr(
                self.config,
                "head_dim",
                self.config.hidden_size // self.config.num_attention_heads,
            )
        )
        self._article_token_cache: dict[str, list[int]] = {}
        self._l0_token_count = len(
            self.tokenizer.encode(l0_text(), add_special_tokens=False)
        )
        self.prefill_cost = load_prefill_cost_model(prefill_calibration)
        self.prefill_cost_model = self.prefill_cost.source
        self._rss_baseline = current_process_rss_bytes()

    def cache_device(self, storage: str) -> str:
        return self.device_name if storage == "accelerator-fp16" else "cpu"

    def new_cache(
        self,
        budget_bytes: int,
        policy: str,
        max_articles: int | None = None,
        *,
        strategy: str = "document",
        block_tokens: int = 16,
    ):
        return new_prefix_cache(
            strategy,
            budget_bytes,
            policy=policy,
            max_articles=max_articles,
            block_tokens=block_tokens,
            l0="pinned-no-inference-l0",
        )

    def estimate_article_bytes(
        self,
        request: QualityRequest,
        storage: str,
        block_tokens: int = 16,
        cache_strategy: str = "document",
    ) -> int:
        article_ids = self._article_ids(request)
        return self._stored_bytes(
            len(article_ids), storage, block_tokens, cache_strategy
        )

    def serve(
        self,
        request: QualityRequest,
        cache,
        *,
        storage: str,
        block_tokens: int = 16,
        cache_strategy: str = "document",
        validate_agreement: bool = False,
        agreement_atol: float | None = None,
    ) -> dict[str, Any]:
        del agreement_atol
        if validate_agreement:
            raise ValueError("--validate-agreement requires inference")
        article_ids = self._article_ids(request)
        key = self._cache_key(request, storage)
        lookup_start = time.perf_counter()
        match = cache.lookup(key, article_ids)
        lookup_s = time.perf_counter() - lookup_start
        matched = match.matched_tokens
        full_hit = matched == len(article_ids)
        partial_article_text_hit = 0 < matched < len(article_ids)
        l0_tokens = self._l0_tokens()
        suffix_tokens = len(
            self.tokenizer.encode(
                suffix_text(request.question), add_special_tokens=False
            )
        )
        total_prompt_tokens = l0_tokens + len(article_ids) + suffix_tokens
        cached_prompt_tokens = l0_tokens + matched
        cache_hit_ratio = cached_prompt_tokens / max(1, total_prompt_tokens)
        partial_hit = 0 < cached_prompt_tokens < total_prompt_tokens
        document_tree_total_tokens = l0_tokens + len(article_ids)
        document_tree_cached_tokens = l0_tokens + matched
        document_tree_hit_ratio = (
            document_tree_cached_tokens / max(1, document_tree_total_tokens)
        )
        partial_document_tree_hit = (
            0 < document_tree_cached_tokens < document_tree_total_tokens
        )
        root_only_hit = l0_tokens > 0 and matched == 0
        # Keep these older names as strategy-independent aliases for the
        # L0 -> document tree. Article-text reuse has explicit fields below.
        partial_article_hit = partial_document_tree_hit
        article_cache_hit_ratio = document_tree_hit_ratio
        missing = len(article_ids) - matched
        prefill_s = self._prefill_cost_s(len(article_ids), matched)
        article_bytes = self._stored_bytes(
            len(article_ids), storage, block_tokens, cache_strategy
        )
        policy_start = time.perf_counter()
        if missing:
            cache.insert(
                key,
                article_ids,
                StoredKV(len(article_ids), simulated_bytes=article_bytes),
                prefill_s,
            )
        policy_s = time.perf_counter() - policy_start
        stats = cache.stats()
        rss = current_process_rss_bytes()
        return {
            "result_schema_version": RESULT_SCHEMA_VERSION,
            "request_id": request.request_id,
            "article_id": request.article_id,
            "cache_hit": full_hit,
            "partial_cache_hit": partial_hit,
            "partial_article_hit": partial_article_hit,
            "partial_article_text_hit": partial_article_text_hit,
            "partial_document_tree_hit": partial_document_tree_hit,
            "root_only_hit": root_only_hit,
            "cache_hit_ratio": cache_hit_ratio,
            "article_cache_hit_ratio": article_cache_hit_ratio,
            "article_text_cache_hit_ratio": match.hit_ratio,
            "document_tree_hit_ratio": document_tree_hit_ratio,
            "cached_prompt_tokens": cached_prompt_tokens,
            "total_prompt_tokens": total_prompt_tokens,
            "document_tree_cached_tokens": document_tree_cached_tokens,
            "document_tree_total_tokens": document_tree_total_tokens,
            "uncached_suffix_tokens": suffix_tokens,
            "matched_prefix_tokens": matched,
            "matched_cache_bytes": match.stored_bytes,
            "article_tokens": len(article_ids),
            "article_bytes": article_bytes,
            "matched_prefill_tokens": matched,
            "avoided_prefill_tokens": matched,
            "lookup_s": lookup_s,
            "load_s": 0.0,
            "transfer_s": 0.0,
            "dequant_s": 0.0,
            "policy_s": policy_s,
            "prefill_s": prefill_s,
            "ttft_s": -1.0,
            "predicted_label": None,
            "gold_label": request.question.answer_letter,
            "difficult": request.question.difficult,
            "reference_agreement": None,
            "process_rss_bytes": rss,
            "process_rss_delta_bytes": nonnegative_delta(
                rss, getattr(self, "_rss_baseline", rss)
            ),
            "mps_allocated_bytes": 0,
            "mps_driver_bytes": 0,
            "mps_allocated_delta_bytes": 0,
            "mps_driver_delta_bytes": 0,
            "cuda_allocated_bytes": 0,
            "cuda_reserved_bytes": 0,
            "cuda_allocated_delta_bytes": 0,
            "cuda_reserved_delta_bytes": 0,
            "prefill_cost_model": getattr(
                self, "prefill_cost_model", "linear-token-estimate-50000-tokens-per-second"
            ),
            "model_weights_loaded": False,
            **stats,
        }

    def serve_uncached(
        self,
        request: QualityRequest,
        *,
        storage: str = "cpu-fp16",
        block_tokens: int = 16,
        cache_strategy: str = "document",
    ) -> dict[str, Any]:
        article_ids = self._article_ids(request)
        article_bytes = self._stored_bytes(
            len(article_ids), storage, block_tokens, cache_strategy
        )
        l0_tokens = self._l0_tokens()
        suffix_tokens = len(
            self.tokenizer.encode(
                suffix_text(request.question), add_special_tokens=False
            )
        )
        document_tree_total_tokens = l0_tokens + len(article_ids)
        rss = current_process_rss_bytes()
        return {
            "result_schema_version": RESULT_SCHEMA_VERSION,
            "request_id": request.request_id,
            "article_id": request.article_id,
            "cache_hit": False,
            "partial_cache_hit": False,
            "partial_article_hit": False,
            "partial_article_text_hit": False,
            "partial_document_tree_hit": False,
            "root_only_hit": False,
            "cache_hit_ratio": 0.0,
            "article_cache_hit_ratio": 0.0,
            "article_text_cache_hit_ratio": 0.0,
            "document_tree_hit_ratio": 0.0,
            "cached_prompt_tokens": 0,
            "total_prompt_tokens": l0_tokens + len(article_ids) + suffix_tokens,
            "document_tree_cached_tokens": 0,
            "document_tree_total_tokens": document_tree_total_tokens,
            "uncached_suffix_tokens": suffix_tokens,
            "matched_prefix_tokens": 0,
            "matched_cache_bytes": 0,
            "article_tokens": len(article_ids),
            "article_bytes": article_bytes,
            "matched_prefill_tokens": 0,
            "avoided_prefill_tokens": 0,
            "lookup_s": 0.0,
            "load_s": 0.0,
            "transfer_s": 0.0,
            "dequant_s": 0.0,
            "policy_s": 0.0,
            "prefill_s": self._prefill_cost_s(len(article_ids)),
            "ttft_s": -1.0,
            "predicted_label": None,
            "gold_label": request.question.answer_letter,
            "difficult": request.question.difficult,
            "reference_agreement": None,
            "cache_bytes": 0,
            "budget_bytes": 0,
            "occupancy": 0.0,
            "cached_articles": 0,
            "cached_documents": 0,
            "cached_blocks": 0,
            "radix_nodes": 0,
            "root_nodes": 0,
            "document_tree_nodes": 0,
            "cached_tokens": 0,
            "insertions": 0,
            "evictions": 0,
            "useful_bytes": 0,
            "shared_bytes": 0,
            "stranded_bytes": 0,
            "metadata_bytes": 0,
            "cache_footprint_bytes": 0,
            "policy": "none",
            "cache_strategy": "none",
            "process_rss_bytes": rss,
            "process_rss_delta_bytes": nonnegative_delta(
                rss, getattr(self, "_rss_baseline", rss)
            ),
            "mps_allocated_bytes": 0,
            "mps_driver_bytes": 0,
            "mps_allocated_delta_bytes": 0,
            "mps_driver_delta_bytes": 0,
            "cuda_allocated_bytes": 0,
            "cuda_reserved_bytes": 0,
            "cuda_allocated_delta_bytes": 0,
            "cuda_reserved_delta_bytes": 0,
            "prefill_cost_model": getattr(
                self, "prefill_cost_model", "linear-token-estimate-50000-tokens-per-second"
            ),
            "model_weights_loaded": False,
        }

    def _prefill_cost_s(self, total_tokens: int, cached_tokens: int = 0) -> float:
        model = getattr(self, "prefill_cost", None)
        if model is None:
            return max(0, total_tokens - cached_tokens) / 50_000.0
        return model.incremental(total_tokens, cached_tokens)

    def _l0_tokens(self) -> int:
        count = getattr(self, "_l0_token_count", None)
        if count is None:
            count = len(self.tokenizer.encode(l0_text(), add_special_tokens=False))
            self._l0_token_count = count
        return int(count)

    def _stored_bytes(
        self, tokens: int, storage: str, block_tokens: int, strategy: str
    ) -> int:
        elements = tokens * self.layers * 2 * self.kv_heads * self.head_dim
        if storage != "cpu-int8":
            return elements * 2
        blocks = (
            (tokens + block_tokens - 1) // block_tokens
            if strategy == "fixed-block"
            else 1
        )
        scale_bytes = blocks * self.layers * 2 * self.kv_heads * 4
        return elements + scale_bytes

    def _article_ids(self, request: QualityRequest) -> list[int]:
        article_ids = self._article_token_cache.get(request.article_hash)
        if article_ids is None:
            _, article_ids, _ = encode_parts(
                self.tokenizer, request.article_text, request.question
            )
            self._article_token_cache[request.article_hash] = article_ids
        return article_ids

    def _cache_key(self, request: QualityRequest, storage: str) -> CacheKey:
        return CacheKey(
            article_id=request.article_id,
            article_hash=request.article_hash,
            model_revision=self.model_revision,
            tokenizer_revision=self.tokenizer_revision,
            prompt_version=PROMPT_VERSION,
            dtype=self.dtype_name,
            quantization="int8-per-kv-head" if storage == "cpu-int8" else "none",
        )
