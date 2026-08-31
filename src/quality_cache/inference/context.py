"""Reusable immutable inputs for in-process experiment matrices."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..data import QualityArticle, QualityRequest, load_quality_split
from ..prompt import article_tail, l0_text, suffix_text


@dataclass(frozen=True)
class TokenizedPrompt:
    """Stable token boundaries for one article/question prompt."""

    l0_ids: list[int]
    article_ids: list[int]
    suffix_ids: list[int]

    @property
    def full_ids(self) -> list[int]:
        return self.l0_ids + self.article_ids + self.suffix_ids


class TokenizationContext:
    """Memoize stable prompt pieces for one exact tokenizer revision."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.l0_ids = list(
            tokenizer.encode(l0_text(), add_special_tokens=False)
        )
        self._article_ids: dict[str, list[int]] = {}
        self._suffix_ids: dict[tuple[Any, ...], list[int]] = {}
        self._prompts: dict[tuple[Any, ...], TokenizedPrompt] = {}
        self._label_ids = {
            label: list(tokenizer.encode(label, add_special_tokens=False))
            for label in "ABCD"
        }

    @staticmethod
    def _question_key(request: QualityRequest) -> tuple[Any, ...]:
        question = request.question
        return (
            request.article_hash,
            question.question_id,
            question.text,
            question.options,
        )

    def article_ids(self, request: QualityRequest) -> list[int]:
        ids = self._article_ids.get(request.article_hash)
        if ids is None:
            ids = list(
                self.tokenizer.encode(
                    article_tail(request.article_text), add_special_tokens=False
                )
            )
            self._article_ids[request.article_hash] = ids
        return ids

    def suffix_ids(self, request: QualityRequest) -> list[int]:
        key = self._question_key(request)
        ids = self._suffix_ids.get(key)
        if ids is None:
            ids = list(
                self.tokenizer.encode(
                    suffix_text(request.question), add_special_tokens=False
                )
            )
            self._suffix_ids[key] = ids
        return ids

    def prompt(self, request: QualityRequest) -> TokenizedPrompt:
        key = self._question_key(request)
        prompt = self._prompts.get(key)
        if prompt is not None:
            return prompt
        article_ids = self.article_ids(request)
        suffix_ids = self.suffix_ids(request)
        whole = list(
            self.tokenizer.encode(
                l0_text()
                + article_tail(request.article_text)
                + suffix_text(request.question),
                add_special_tokens=False,
            )
        )
        combined = self.l0_ids + article_ids + suffix_ids
        if combined != whole:
            raise ValueError(
                "tokenizer merges across a prompt boundary; use prefix lengths "
                "from full-prompt encoding"
            )
        prompt = TokenizedPrompt(self.l0_ids, article_ids, suffix_ids)
        self._prompts[key] = prompt
        return prompt

    def label_ids(self) -> dict[str, list[int]]:
        return self._label_ids

    def option_ids(self, request: QualityRequest) -> dict[str, list[int]]:
        return {
            label: list(self.tokenizer.encode(option, add_special_tokens=False))
            for label, option in zip("ABCD", request.question.options)
        }


class ExperimentContext:
    """Reuse datasets, tokenizer/config assets, and token IDs across a matrix.

    Model weights and mutable KV tensors are deliberately not shared: each real
    inference run still starts with an independent model and empty cache.
    """

    def __init__(self):
        self._datasets: dict[tuple[str, str, bool], list[QualityArticle]] = {}
        self._no_inference_assets: dict[tuple[str, str, str | None], tuple[Any, Any]] = {}
        self._tokenization: dict[tuple[Any, ...], TokenizationContext] = {}

    def load_articles(
        self,
        path: str | Path,
        *,
        split: str,
        verify_official_counts: bool,
    ) -> list[QualityArticle]:
        resolved = str(Path(path).resolve())
        key = (resolved, split, bool(verify_official_counts))
        articles = self._datasets.get(key)
        if articles is None:
            articles = load_quality_split(
                path,
                split=split,
                verify_official_counts=verify_official_counts,
            )
            self._datasets[key] = articles
        return articles

    def no_inference_assets(
        self,
        model_name: str,
        *,
        tokenizer_source: str | None,
        revision: str | None,
    ) -> tuple[Any, Any, tuple[Any, ...]]:
        from transformers import AutoConfig, AutoTokenizer

        source = tokenizer_source or model_name
        key = (model_name, source, revision)
        assets = self._no_inference_assets.get(key)
        if assets is None:
            local = Path(source).exists()
            tokenizer = AutoTokenizer.from_pretrained(
                source,
                revision=None if local else revision,
                local_files_only=local,
            )
            config_source = (
                source
                if local and (Path(source) / "config.json").exists()
                else model_name
            )
            config_local = Path(config_source).exists()
            config = AutoConfig.from_pretrained(
                config_source,
                revision=None if config_local else revision,
                local_files_only=config_local,
            )
            assets = (tokenizer, config)
            self._no_inference_assets[key] = assets
        return assets[0], assets[1], ("no-inference",) + key

    def tokenization_context(
        self, identity: tuple[Any, ...], tokenizer
    ) -> TokenizationContext:
        context = self._tokenization.get(identity)
        if context is None:
            context = TokenizationContext(tokenizer)
            self._tokenization[identity] = context
        return context
