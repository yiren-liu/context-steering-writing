from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase

from .data import EditExample
from .cos_components import _tokenize_dialogs, _tokenize_texts, _compute_log_softmax, _append_response


@dataclass
class StyleProfile:
    user_id: str
    prompt_text: str
    stats: Dict[str, float]


def _sentence_stats(text: str) -> Tuple[float, float]:
    sentences = [s for s in text.replace("!", ".").replace("?", ".").split(".") if s.strip()]
    if not sentences:
        return 0.0, 0.0
    lengths = [len(s.split()) for s in sentences]
    return float(np.mean(lengths)), float(np.std(lengths))


def _contraction_rate(text: str) -> float:
    tokens = text.lower().split()
    if not tokens:
        return 0.0
    contractions = sum("'" in t for t in tokens)
    return contractions / len(tokens)


def _punct_rate(text: str) -> float:
    if not text:
        return 0.0
    punct = sum(ch in ",;:!?" for ch in text)
    return punct / max(1, len(text))


def build_style_profile(user_id: str, edits: Iterable[EditExample]) -> StyleProfile:
    before_lens, after_lens = [], []
    sent_means, sent_stds = [], []
    contraction_rates, punct_rates = [], []

    for ex in edits:
        before_lens.append(len(ex.before.split()))
        after_lens.append(len(ex.after.split()))
        mean_len, std_len = _sentence_stats(ex.after)
        sent_means.append(mean_len)
        sent_stds.append(std_len)
        contraction_rates.append(_contraction_rate(ex.after))
        punct_rates.append(_punct_rate(ex.after))

    def avg(vals: List[float]) -> float:
        return float(np.mean(vals)) if vals else 0.0

    stats = {
        "length_ratio": avg(after_lens) / max(1.0, avg(before_lens)),
        "sent_len_mean": avg(sent_means),
        "sent_len_std": avg(sent_stds),
        "contraction_rate": avg(contraction_rates),
        "punct_rate": avg(punct_rates),
    }

    prompt_lines = [
        "Writing style profile inferred from edits:",
        f"- Length ratio (after/before): {stats['length_ratio']:.2f}",
        f"- Avg sentence length: {stats['sent_len_mean']:.1f} words (std {stats['sent_len_std']:.1f})",
        f"- Contraction rate: {stats['contraction_rate']:.2f}",
        f"- Punctuation density: {stats['punct_rate']:.2f}",
        "Apply this style in your response.",
    ]
    prompt_text = "\n".join(prompt_lines)
    return StyleProfile(user_id=user_id, prompt_text=prompt_text, stats=stats)


def compose_prompt(style_prompt: str, user_prompt: str) -> str:
    if not user_prompt:
        return style_prompt
    return f"{style_prompt}\n\n{user_prompt}"


def weighted_logprob_prompt(
    model,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    response: str,
    response_weights: torch.Tensor,
    is_chat: bool = True,
) -> torch.Tensor:
    if is_chat:
        dialog = [{"role": "user", "content": prompt}]
        full = _append_response(dialog, response, is_chat=True)
        toks_prompt = _tokenize_dialogs(tokenizer, dialog).to(model.device)
        toks_full = _tokenize_dialogs(tokenizer, full).to(model.device)
    else:
        full = _append_response(prompt, response, is_chat=False)
        toks_prompt = _tokenize_texts(tokenizer, prompt).to(model.device)
        toks_full = _tokenize_texts(tokenizer, full).to(model.device)

    full_ids = toks_full.input_ids
    full_mask = toks_full.attention_mask
    prompt_mask = toks_prompt.attention_mask

    res_len = (full_mask.sum(dim=1) - prompt_mask.sum(dim=1)).item()
    n_cols = full_ids.size(1)
    len_prompt = int(n_cols - res_len)
    is_prompt = torch.arange(n_cols, device=model.device) < len_prompt
    resp_mask = torch.where(is_prompt[None, :], torch.zeros_like(full_mask), full_mask.bool())
    resp_mask_last = torch.roll(resp_mask, shifts=-1, dims=1)

    logits = _compute_log_softmax(model, full_ids, full_mask)
    response_positions = torch.where(resp_mask[0])[0]
    logit_positions = torch.where(resp_mask_last[0])[0]
    max_len = min(len(response_positions), len(logit_positions))
    response_positions = response_positions[:max_len]
    logit_positions = logit_positions[:max_len]

    token_ids = full_ids[0, response_positions]
    if response_weights.numel() != token_ids.numel():
        response_weights = torch.ones_like(token_ids, dtype=torch.float32, device=token_ids.device)
    else:
        response_weights = response_weights.to(token_ids.device)

    base_lp = logits[0, logit_positions]
    token_lp = base_lp.gather(-1, token_ids[:, None]).squeeze(-1)
    return torch.sum(token_lp * response_weights)
