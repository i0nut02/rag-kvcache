from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from ..caches import CacheKey, StoredKV, new_prefix_cache
from ..data import QualityRequest
from ..memory import current_process_rss_bytes, nonnegative_delta
from ..prompt import PROMPT_VERSION, article_tail, encode_parts, l0_text, suffix_text
from ..schema import RESULT_SCHEMA_VERSION
from .tensors import (
    concatenate_caches,
    restore_blocks,
    sequence_length,
    slice_cache,
    store_blocks,
    to_legacy,
    to_model_cache,
)


@dataclass
class ScoreResult:
    label: str
    scores: dict[str, float]


def default_reference_logit_atol(dtype) -> float:
    """Absolute agreement tolerance for numerically different inference paths."""
    name = str(dtype).replace("torch.", "")
    return {
        "float16": 0.0625,
        "bfloat16": 0.25,
        "float32": 0.001,
    }.get(name, 0.001)


def transformers_dtype_keyword(version: str) -> str:
    """Transformers 5 renamed the model-loading torch_dtype argument to dtype."""
    try:
        major = int(version.split(".", 1)[0])
    except (TypeError, ValueError):
        major = 4
    return "dtype" if major >= 5 else "torch_dtype"


class QualityModelRunner:
    def __init__(
        self,
        model_name: str,
        *,
        device: str = "mps",
        dtype: str = "float16",
        revision: str | None = None,
    ):
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        if device == "mps" and not torch.backends.mps.is_available():
            built = torch.backends.mps.is_built()
            raise RuntimeError(
                f"requested --device mps, but torch reports MPS unavailable "
                f"(is_built={built}, is_available=False)"
            )
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("requested --device cuda, but torch reports CUDA unavailable")
        self.device = torch.device(device)
        self.dtype = getattr(torch, dtype)
        self.model_name = model_name
        self.requested_revision = revision
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        kwargs = {"revision": revision, "low_cpu_mem_usage": True}
        kwargs[transformers_dtype_keyword(transformers.__version__)] = self.dtype
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs).to(self.device)
        self.model.eval()
        self.model_weights_loaded = True
        self.model_revision = str(getattr(self.model.config, "_commit_hash", None) or revision or model_name)
        self.tokenizer_revision = str(
            getattr(self.tokenizer, "_commit_hash", None)
            or getattr(self.tokenizer, "init_kwargs", {}).get("_commit_hash")
            or revision
            or model_name
        )
        self.l0_ids = self.tokenizer.encode(l0_text(), add_special_tokens=False)
        self.l0_cache = self._prefill(self.l0_ids)
        self.l0_tokens = sequence_length(self.l0_cache)
        self.reference_logit_atol = default_reference_logit_atol(self.dtype)
        self.prefill_cost_model = "measured-online"
        self._memory_baseline = self._raw_memory_stats()

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
            l0=self.l0_cache,
        )

    def estimate_article_bytes(
        self,
        request: QualityRequest,
        storage: str,
        block_tokens: int = 16,
        cache_strategy: str = "document",
    ) -> int:
        _, article_ids, _ = encode_parts(self.tokenizer, request.article_text, request.question)
        config = self.model.config
        layers = int(config.num_hidden_layers)
        kv_heads = int(getattr(config, "num_key_value_heads", config.num_attention_heads))
        head_dim = int(
            getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        )
        elements = len(article_ids) * layers * 2 * kv_heads * head_dim
        if storage != "cpu-int8":
            return elements * 2
        blocks = (
            (len(article_ids) + block_tokens - 1) // block_tokens
            if cache_strategy == "fixed-block"
            else 1
        )
        scale_bytes = blocks * layers * 2 * kv_heads * 4
        return elements + scale_bytes

    def article_token_count(self, request: QualityRequest) -> int:
        _, article_ids, _ = encode_parts(
            self.tokenizer, request.article_text, request.question
        )
        return len(article_ids)

    def measure_article_prefill(self, request: QualityRequest) -> tuple[int, float]:
        """Measure L0-conditioned article prefill without scoring a question."""
        _, article_ids, _ = encode_parts(
            self.tokenizer, request.article_text, request.question
        )
        started = time.perf_counter()
        measured_cache = self._prefill(article_ids, past=self.l0_cache)
        self._synchronize()
        elapsed = time.perf_counter() - started
        del measured_cache
        if self.device.type == "mps":
            self.torch.mps.empty_cache()
        elif self.device.type == "cuda":
            self.torch.cuda.empty_cache()
        return len(article_ids), elapsed

    def prefill_article(
        self,
        request: QualityRequest,
        cache,
        *,
        storage: str,
        block_tokens: int = 16,
    ) -> float:
        _, article_ids, _ = encode_parts(self.tokenizer, request.article_text, request.question)
        key = self._cache_key(request, storage)
        if cache.lookup(key, article_ids).matched_tokens == len(article_ids):
            return 0.0
        started = time.perf_counter()
        full_prefix_cache = self._prefill(article_ids, past=self.l0_cache)
        self._synchronize()
        article_cache = slice_cache(
            full_prefix_cache, self.l0_tokens, self.l0_tokens + len(article_ids)
        )
        physical_block_tokens = (
            len(article_ids) if cache.strategy in {"document", "radix"} else block_tokens
        )
        blocks = store_blocks(
            article_cache,
            storage,
            max(1, physical_block_tokens),
            accelerator_device=self.device,
        )
        measured_prefill = time.perf_counter() - started
        cache.insert(
            key,
            article_ids,
            StoredKV(len(article_ids), blocks=blocks),
            measured_prefill,
        )
        return time.perf_counter() - started

    def _cache_key(self, request: QualityRequest, storage: str) -> CacheKey:
        return CacheKey(
            article_id=request.article_id,
            article_hash=request.article_hash,
            model_revision=self.model_revision,
            tokenizer_revision=self.tokenizer_revision,
            prompt_version=PROMPT_VERSION,
            dtype=str(self.dtype).replace("torch.", ""),
            quantization="int8-per-kv-head" if storage == "cpu-int8" else "none",
        )

    def cache_device(self, storage: str) -> str:
        return self.device.type if storage == "accelerator-fp16" else "cpu"

    def serve_uncached(
        self,
        request: QualityRequest,
        *,
        storage: str = "cpu-fp16",
        block_tokens: int = 16,
        cache_strategy: str = "document",
    ) -> dict[str, Any]:
        l0_ids, article_ids, suffix_ids = encode_parts(
            self.tokenizer, request.article_text, request.question
        )
        total_prompt_tokens = len(l0_ids) + len(article_ids) + len(suffix_ids)
        document_tree_total_tokens = len(l0_ids) + len(article_ids)
        started = time.perf_counter()
        score = self.score_uncached(request)
        self._synchronize()
        ttft_s = time.perf_counter() - started
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
            "total_prompt_tokens": total_prompt_tokens,
            "document_tree_cached_tokens": 0,
            "document_tree_total_tokens": document_tree_total_tokens,
            "uncached_suffix_tokens": len(suffix_ids),
            "matched_prefix_tokens": 0,
            "matched_cache_bytes": 0,
            "article_tokens": len(article_ids),
            "article_bytes": self.estimate_article_bytes(
                request, storage, block_tokens, cache_strategy
            ),
            "matched_prefill_tokens": 0,
            "avoided_prefill_tokens": 0,
            "lookup_s": 0.0,
            "load_s": 0.0,
            "transfer_s": 0.0,
            "dequant_s": 0.0,
            "policy_s": 0.0,
            "prefill_s": ttft_s,
            "ttft_s": ttft_s,
            "predicted_label": score.label,
            "label_scores": score.scores,
            "gold_label": request.question.answer_letter,
            "difficult": request.question.difficult,
            "uncached_label": score.label,
            "fp16_reference_label": score.label,
            "reference_agreement": True,
            "reference_max_label_logit_delta": 0.0,
            "cache_bytes": 0,
            "metadata_bytes": 0,
            "cache_footprint_bytes": 0,
            "budget_bytes": 0,
            "occupancy": 0.0,
            "root_nodes": 0,
            "document_tree_nodes": 0,
            "cached_articles": 0,
            "insertions": 0,
            "evictions": 0,
            "policy": "none",
            "cache_strategy": "none",
            "prefill_cost_model": self.prefill_cost_model,
            "model_weights_loaded": True,
            **self.memory_stats(),
        }

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
        l0_ids, article_ids, suffix_ids = encode_parts(
            self.tokenizer, request.article_text, request.question
        )
        key = self._cache_key(request, storage)
        start = time.perf_counter()
        lookup_start = time.perf_counter()
        match = cache.lookup(key, article_ids)
        lookup_s = time.perf_counter() - lookup_start
        matched = match.matched_tokens
        hit = matched == len(article_ids)
        partial_article_text_hit = 0 < matched < len(article_ids)
        total_prompt_tokens = len(l0_ids) + len(article_ids) + len(suffix_ids)
        cached_prompt_tokens = len(l0_ids) + matched
        cache_hit_ratio = cached_prompt_tokens / max(1, total_prompt_tokens)
        partial_hit = 0 < cached_prompt_tokens < total_prompt_tokens
        document_tree_total_tokens = len(l0_ids) + len(article_ids)
        document_tree_cached_tokens = len(l0_ids) + matched
        document_tree_hit_ratio = (
            document_tree_cached_tokens / max(1, document_tree_total_tokens)
        )
        partial_document_tree_hit = (
            0 < document_tree_cached_tokens < document_tree_total_tokens
        )
        root_only_hit = len(l0_ids) > 0 and matched == 0
        # Keep these older names as strategy-independent aliases for the
        # L0 -> document tree. Article-text reuse has explicit fields below.
        partial_article_hit = partial_document_tree_hit
        article_cache_hit_ratio = document_tree_hit_ratio
        prefill_s = 0.0
        load_s = transfer_s = dequant_s = 0.0
        policy_s = 0.0
        restored = tuple()
        if matched:
            restore_start = time.perf_counter()
            restored = restore_blocks(match.blocks, dtype=self.dtype, device=self.device)
            self._synchronize()
            restore_s = time.perf_counter() - restore_start
            if storage == "cpu-int8":
                dequant_s = restore_s
            elif storage == "cpu-fp16":
                transfer_s = restore_s
            else:
                load_s = restore_s
        prefix = concatenate_caches(self.l0_cache, restored)
        if not hit:
            article_start = time.perf_counter()
            full_prefix_cache = self._prefill(article_ids[matched:], past=prefix)
            self._synchronize()
            prefill_s = time.perf_counter() - article_start
            article_cache = slice_cache(
                full_prefix_cache, self.l0_tokens, self.l0_tokens + len(article_ids)
            )
            physical_block_tokens = (
                len(article_ids)
                if cache.strategy in {"document", "radix"}
                else block_tokens
            )
            blocks = store_blocks(
                article_cache,
                storage,
                max(1, physical_block_tokens),
                accelerator_device=self.device,
            )
            policy_start = time.perf_counter()
            cache.insert(
                key,
                article_ids,
                StoredKV(len(article_ids), blocks=blocks),
                prefill_s,
            )
            policy_s = time.perf_counter() - policy_start
            prefix = full_prefix_cache
        score = self.score_suffix(suffix_ids, prefix, options=request.question.options)
        self._synchronize()
        ttft_s = time.perf_counter() - start

        uncached_label = None
        agreement = None
        resolved_agreement_atol = (
            self.reference_logit_atol if agreement_atol is None else agreement_atol
        )
        if validate_agreement:
            uncached = self.score_uncached(request)
            uncached_label = uncached.label
            agreement = uncached.label == score.label
            logit_delta = max(abs(score.scores[label] - uncached.scores[label]) for label in "ABCD")
            if storage != "cpu-int8" and (
                not agreement or logit_delta > resolved_agreement_atol
            ):
                raise AssertionError(
                    f"cached/uncached mismatch for {self.dtype}: "
                    f"label agreement={agreement}, max label-logit delta={logit_delta:.6g}, "
                    f"atol={resolved_agreement_atol:.6g}"
                )
        else:
            logit_delta = None

        stats = cache.stats()
        memory = self.memory_stats()
        return {
            "result_schema_version": RESULT_SCHEMA_VERSION,
            "request_id": request.request_id,
            "article_id": request.article_id,
            "cache_hit": hit,
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
            "uncached_suffix_tokens": len(suffix_ids),
            "matched_prefix_tokens": matched,
            "matched_cache_bytes": match.stored_bytes,
            "article_tokens": len(article_ids),
            "article_bytes": self.estimate_article_bytes(
                request, storage, block_tokens, cache_strategy
            ),
            "matched_prefill_tokens": matched,
            "avoided_prefill_tokens": matched,
            "lookup_s": lookup_s,
            "load_s": load_s,
            "transfer_s": transfer_s,
            "dequant_s": dequant_s,
            "policy_s": policy_s,
            "prefill_s": prefill_s,
            "ttft_s": ttft_s,
            "predicted_label": score.label,
            "label_scores": score.scores,
            "gold_label": request.question.answer_letter,
            "difficult": request.question.difficult,
            "uncached_label": uncached_label,
            "fp16_reference_label": uncached_label,
            "reference_agreement": agreement,
            "reference_max_label_logit_delta": logit_delta,
            "reference_logit_atol": (
                resolved_agreement_atol if validate_agreement else None
            ),
            "prefill_cost_model": self.prefill_cost_model,
            "model_weights_loaded": True,
            **stats,
            **memory,
        }

    def score_suffix(self, suffix_ids: list[int], prefix, options=None) -> ScoreResult:
        torch = self.torch
        input_ids = torch.tensor([suffix_ids], dtype=torch.long, device=self.device)
        prefix_length = sequence_length(prefix)
        mask = torch.ones((1, prefix_length + len(suffix_ids)), dtype=torch.long, device=self.device)
        with torch.inference_mode():
            output = self.model(
                input_ids=input_ids,
                attention_mask=mask,
                past_key_values=to_model_cache(prefix),
                use_cache=True,
            )
        logits = output.logits[0, -1].float()
        label_ids = {
            label: self.tokenizer.encode(label, add_special_tokens=False) for label in "ABCD"
        }
        if all(len(ids) == 1 for ids in label_ids.values()):
            scores = {label: float(logits[ids[0]].item()) for label, ids in label_ids.items()}
        else:
            sequences = (
                {
                    label: self.tokenizer.encode(option, add_special_tokens=False)
                    for label, option in zip("ABCD", options)
                }
                if options is not None else label_ids
            )
            scores = {
                label: self._sequence_score(suffix_ids, ids, prefix)
                for label, ids in sequences.items()
            }
        return ScoreResult(max(scores, key=scores.get), scores)  # type: ignore[arg-type]

    def score_uncached(self, request: QualityRequest) -> ScoreResult:
        l0_ids, article_ids, suffix_ids = encode_parts(
            self.tokenizer, request.article_text, request.question
        )
        full_ids = l0_ids + article_ids + suffix_ids
        torch = self.torch
        input_ids = torch.tensor([full_ids], dtype=torch.long, device=self.device)
        with torch.inference_mode():
            logits = self.model(input_ids=input_ids, use_cache=False).logits[0, -1].float()
        label_ids = {
            label: self.tokenizer.encode(label, add_special_tokens=False) for label in "ABCD"
        }
        if all(len(ids) == 1 for ids in label_ids.values()):
            scores = {label: float(logits[ids[0]].item()) for label, ids in label_ids.items()}
        else:
            sequences = {
                label: self.tokenizer.encode(option, add_special_tokens=False)
                for label, option in zip("ABCD", request.question.options)
            }
            scores = {
                label: self._full_sequence_score(full_ids, ids)
                for label, ids in sequences.items()
            }
        return ScoreResult(max(scores, key=scores.get), scores)  # type: ignore[arg-type]

    def _full_sequence_score(self, prompt_ids, label_ids) -> float:
        torch = self.torch
        combined = prompt_ids + label_ids
        input_ids = torch.tensor([combined], dtype=torch.long, device=self.device)
        with torch.inference_mode():
            logits = self.model(input_ids=input_ids, use_cache=False).logits[0].log_softmax(-1)
        start = len(prompt_ids) - 1
        return float(
            sum(logits[start + offset, token].item() for offset, token in enumerate(label_ids))
        )

    def _sequence_score(self, suffix_ids, label_ids, prefix) -> float:
        torch = self.torch
        combined = suffix_ids + label_ids
        input_ids = torch.tensor([combined], dtype=torch.long, device=self.device)
        prefix_length = sequence_length(prefix)
        mask = torch.ones((1, prefix_length + len(combined)), dtype=torch.long, device=self.device)
        with torch.inference_mode():
            logits = self.model(
                input_ids=input_ids,
                attention_mask=mask,
                past_key_values=to_model_cache(prefix),
                use_cache=False,
            ).logits[0]
        log_probs = logits.log_softmax(-1)
        start = len(suffix_ids) - 1
        return float(
            sum(log_probs[start + offset, token].item() for offset, token in enumerate(label_ids))
        )

    def _prefill(self, token_ids: list[int], past=None):
        torch = self.torch
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        kwargs = {}
        if past is not None:
            prefix_length = sequence_length(past)
            kwargs["past_key_values"] = to_model_cache(past)
            kwargs["attention_mask"] = torch.ones(
                (1, prefix_length + len(token_ids)), dtype=torch.long, device=self.device
            )
        with torch.inference_mode():
            output = self.model(input_ids=input_ids, use_cache=True, **kwargs)
        return to_legacy(output.past_key_values)

    def _synchronize(self):
        if self.device.type == "mps":
            self.torch.mps.synchronize()
        elif self.device.type == "cuda":
            self.torch.cuda.synchronize()

    def _raw_memory_stats(self) -> dict[str, int]:
        values = {"process_rss_bytes": current_process_rss_bytes()}
        if self.device.type == "mps":
            values["mps_allocated_bytes"] = int(self.torch.mps.current_allocated_memory())
            values["mps_driver_bytes"] = int(self.torch.mps.driver_allocated_memory())
        else:
            values["mps_allocated_bytes"] = 0
            values["mps_driver_bytes"] = 0
        if self.device.type == "cuda":
            values["cuda_allocated_bytes"] = int(self.torch.cuda.memory_allocated(self.device))
            values["cuda_reserved_bytes"] = int(self.torch.cuda.memory_reserved(self.device))
        else:
            values["cuda_allocated_bytes"] = 0
            values["cuda_reserved_bytes"] = 0
        return values

    def memory_stats(self) -> dict[str, int]:
        values = self._raw_memory_stats()
        baseline = getattr(self, "_memory_baseline", values)
        values["process_rss_delta_bytes"] = nonnegative_delta(
            values["process_rss_bytes"], baseline["process_rss_bytes"]
        )
        values["mps_allocated_delta_bytes"] = nonnegative_delta(
            values["mps_allocated_bytes"], baseline["mps_allocated_bytes"]
        )
        values["mps_driver_delta_bytes"] = nonnegative_delta(
            values["mps_driver_bytes"], baseline["mps_driver_bytes"]
        )
        values["cuda_allocated_delta_bytes"] = nonnegative_delta(
            values["cuda_allocated_bytes"], baseline["cuda_allocated_bytes"]
        )
        values["cuda_reserved_delta_bytes"] = nonnegative_delta(
            values["cuda_reserved_bytes"], baseline["cuda_reserved_bytes"]
        )
        return values
