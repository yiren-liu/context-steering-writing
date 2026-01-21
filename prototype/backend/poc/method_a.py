from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch

from .bayes_update import GaussianPosterior, adf_update
from .cos_components import EditBundle, build_edit_bundle, weighted_cos_logprob
from .data import EditExample
from .span_mask import build_span_masks


@dataclass
class UserModel:
    posterior: GaussianPosterior
    history: List[torch.Tensor]


class BayesContrastiveModel:
    def __init__(
        self,
        contexts: Sequence[str],
        prior_mean: torch.Tensor,
        prior_cov: torch.Tensor,
        beta: float = 1.0,
        is_chat: bool = True,
        put_context_first: bool = True,
    ) -> None:
        self.contexts = list(contexts)
        self.prior_mean = prior_mean
        self.prior_cov = prior_cov
        self.beta = beta
        self.is_chat = is_chat
        self.put_context_first = put_context_first
        self.user_models: Dict[str, UserModel] = {}

    def _get_user_model(self, user_id: str) -> UserModel:
        if user_id not in self.user_models:
            posterior = GaussianPosterior(
                mean=self.prior_mean.clone(),
                cov=self.prior_cov.clone(),
            )
            self.user_models[user_id] = UserModel(posterior=posterior, history=[])
        return self.user_models[user_id]

    def _build_bundles(
        self,
        model,
        tokenizer,
        example: EditExample,
    ) -> tuple[EditBundle, EditBundle]:
        before_mask, after_mask = build_span_masks(
            example.before, example.after, tokenizer
        )
        bundle_before = build_edit_bundle(
            model=model,
            tokenizer=tokenizer,
            prompt=example.prompt,
            response=example.before,
            contexts=self.contexts,
            response_weights=before_mask,
            is_chat=self.is_chat,
            put_context_first=self.put_context_first,
        )
        bundle_after = build_edit_bundle(
            model=model,
            tokenizer=tokenizer,
            prompt=example.prompt,
            response=example.after,
            contexts=self.contexts,
            response_weights=after_mask,
            is_chat=self.is_chat,
            put_context_first=self.put_context_first,
        )
        return bundle_before, bundle_after

    def update_user(self, model, tokenizer, example: EditExample) -> GaussianPosterior:
        user = self._get_user_model(example.user_id)
        bundle_before, bundle_after = self._build_bundles(model, tokenizer, example)
        posterior = adf_update(
            bundle_pos=bundle_after,
            bundle_neg=bundle_before,
            prior=user.posterior,
            beta=self.beta,
        )
        user.posterior = posterior
        user.history.append(posterior.mean.clone().detach())
        return posterior

    def score_edit(
        self,
        model,
        tokenizer,
        example: EditExample,
        lambdas: torch.Tensor,
    ) -> torch.Tensor:
        bundle_before, bundle_after = self._build_bundles(model, tokenizer, example)
        score_after = weighted_cos_logprob(bundle_after, lambdas)
        score_before = weighted_cos_logprob(bundle_before, lambdas)
        return score_after - score_before

    def get_user_mean(self, user_id: str) -> torch.Tensor:
        return self._get_user_model(user_id).posterior.mean
