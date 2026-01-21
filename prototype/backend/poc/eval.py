from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch

from .data import EditExample
from .method_a import BayesContrastiveModel
from .span_mask import build_span_masks
from .prompt_baseline import StyleProfile, compose_prompt, weighted_logprob_prompt


def _auc(scores: List[float], labels: List[int]) -> float:
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return 0.0
    ranks = np.argsort(np.argsort(scores)) + 1
    rank_sum = sum(r for r, y in zip(ranks, labels) if y == 1)
    n_pos = len(pos)
    n_neg = len(neg)
    return float((rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def evaluate_method_a(
    model,
    tokenizer,
    edits: Iterable[EditExample],
    bayes_model: BayesContrastiveModel,
) -> Dict[str, float]:
    scores = []
    labels = []
    deltas = []
    for ex in edits:
        lambdas = bayes_model.get_user_mean(ex.user_id).to(model.device)
        delta = bayes_model.score_edit(model, tokenizer, ex, lambdas).item()
        deltas.append(delta)
        scores.extend([delta, -delta])
        labels.extend([1, 0])
    accuracy = float(np.mean([d > 0 for d in deltas])) if deltas else 0.0
    return {
        "accuracy": accuracy,
        "auc": _auc(scores, labels),
        "mean_delta": float(np.mean(deltas)) if deltas else 0.0,
    }


def evaluate_prompt_baseline(
    model,
    tokenizer,
    edits: Iterable[EditExample],
    profiles: Dict[str, StyleProfile],
    is_chat: bool = True,
) -> Dict[str, float]:
    scores = []
    labels = []
    deltas = []
    for ex in edits:
        profile = profiles[ex.user_id]
        prompt = compose_prompt(profile.prompt_text, ex.prompt)
        before_mask, after_mask = build_span_masks(ex.before, ex.after, tokenizer)
        before_score = weighted_logprob_prompt(
            model, tokenizer, prompt, ex.before, before_mask, is_chat=is_chat
        ).item()
        after_score = weighted_logprob_prompt(
            model, tokenizer, prompt, ex.after, after_mask, is_chat=is_chat
        ).item()
        delta = after_score - before_score
        deltas.append(delta)
        scores.extend([after_score, before_score])
        labels.extend([1, 0])
    accuracy = float(np.mean([d > 0 for d in deltas])) if deltas else 0.0
    return {
        "accuracy": accuracy,
        "auc": _auc(scores, labels),
        "mean_delta": float(np.mean(deltas)) if deltas else 0.0,
    }


def per_user_accuracy(deltas: List[Tuple[str, float]]) -> Dict[str, float]:
    user_scores: Dict[str, List[float]] = {}
    for user_id, delta in deltas:
        user_scores.setdefault(user_id, []).append(delta)
    return {u: float(np.mean([d > 0 for d in ds])) for u, ds in user_scores.items()}
