from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase

from cos.utils import get_multi_context_pair_dialogs, get_multi_context_pair_texts


@dataclass
class EditBundle:
    base_lp: torch.Tensor  # (R, V)
    influences: torch.Tensor  # (K, R, V)
    target_ids: torch.Tensor  # (R,)
    weights: torch.Tensor  # (R,)


def _tokenize_dialogs(
    tokenizer: PreTrainedTokenizerBase,
    dialogs: List[dict],
) -> torch.Tensor:
    tokenized = tokenizer.apply_chat_template(
        dialogs,
        tokenize=True,
        return_tensors="pt",
        padding=False,
        return_dict=True,
    )
    return tokenized


def _tokenize_texts(
    tokenizer: PreTrainedTokenizerBase,
    text: str,
) -> torch.Tensor:
    return tokenizer(text, return_tensors="pt", padding=False)


def _build_prompt_pairs(
    prompt: str,
    contexts: Sequence[str],
    is_chat: bool,
    put_context_first: bool,
):
    all_contexts = [[c] for c in contexts]
    if is_chat:
        dialogs_ctx, dialogs_nc = get_multi_context_pair_dialogs(
            [prompt], all_contexts, put_context_first
        )
        return dialogs_ctx, dialogs_nc
    texts_ctx, texts_nc = get_multi_context_pair_texts(
        [prompt], all_contexts, put_context_first
    )
    return texts_ctx, texts_nc


def _append_response(dialog_or_text, response: str, is_chat: bool):
    if is_chat:
        return dialog_or_text + [{"role": "assistant", "content": response}]
    return f"{dialog_or_text} {response}".strip()


def _compute_log_softmax(model, input_ids, attention_mask):
    with torch.no_grad():
        logits = model(input_ids, attention_mask, use_cache=False).logits
    return F.log_softmax(logits, dim=-1)


def build_edit_bundle(
    model,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    response: str,
    contexts: Sequence[str],
    response_weights: torch.Tensor,
    is_chat: bool = True,
    put_context_first: bool = True,
) -> EditBundle:
    dialogs_ctx, dialogs_nc = _build_prompt_pairs(
        prompt, contexts, is_chat=is_chat, put_context_first=put_context_first
    )

    if is_chat:
        prompt_nc = dialogs_nc[0]
        full_nc = _append_response(prompt_nc, response, is_chat=True)
        full_ctx = [
            _append_response(dialogs_ctx[i][0], response, is_chat=True)
            for i in range(len(contexts))
        ]
        toks_prompt_nc = _tokenize_dialogs(tokenizer, prompt_nc)
        toks_full_nc = _tokenize_dialogs(tokenizer, full_nc)
        toks_full_ctx = [_tokenize_dialogs(tokenizer, fc) for fc in full_ctx]
    else:
        prompt_nc = dialogs_nc[0]
        full_nc = _append_response(prompt_nc, response, is_chat=False)
        full_ctx = [
            _append_response(dialogs_ctx[i][0], response, is_chat=False)
            for i in range(len(contexts))
        ]
        toks_prompt_nc = _tokenize_texts(tokenizer, prompt_nc)
        toks_full_nc = _tokenize_texts(tokenizer, full_nc)
        toks_full_ctx = [_tokenize_texts(tokenizer, fc) for fc in full_ctx]

    toks_prompt_nc = toks_prompt_nc.to(model.device)
    toks_full_nc = toks_full_nc.to(model.device)
    toks_full_ctx = [t.to(model.device) for t in toks_full_ctx]

    full_nc_ids = toks_full_nc.input_ids
    full_nc_mask = toks_full_nc.attention_mask
    prompt_nc_mask = toks_prompt_nc.attention_mask

    res_len = (full_nc_mask.sum(dim=1) - prompt_nc_mask.sum(dim=1)).item()
    n_cols = full_nc_ids.size(1)
    len_prompt = int(n_cols - res_len)
    is_prompt = torch.arange(n_cols, device=model.device) < len_prompt
    resp_mask = torch.where(is_prompt[None, :], torch.zeros_like(full_nc_mask), full_nc_mask.bool())
    resp_mask_last = torch.roll(resp_mask, shifts=-1, dims=1)

    aligned_ctx_ids = []
    aligned_ctx_masks = []
    for toks_ctx in toks_full_ctx:
        ctx_ids = toks_ctx.input_ids
        ctx_masks = toks_ctx.attention_mask
        if ctx_ids.size(1) < n_cols:
            pad_len = n_cols - ctx_ids.size(1)
            ctx_ids = F.pad(ctx_ids, (0, pad_len), value=tokenizer.eos_token_id)
            ctx_masks = F.pad(ctx_masks, (0, pad_len), value=0)
        elif ctx_ids.size(1) > n_cols:
            ctx_ids = ctx_ids[:, -n_cols:]
            ctx_masks = ctx_masks[:, -n_cols:]
        aligned_ctx_ids.append(ctx_ids)
        aligned_ctx_masks.append(ctx_masks)

    logits_nc = _compute_log_softmax(model, full_nc_ids, full_nc_mask)
    all_logits = [
        _compute_log_softmax(model, ctx_ids, ctx_mask)
        for ctx_ids, ctx_mask in zip(aligned_ctx_ids, aligned_ctx_masks)
    ]

    response_positions = torch.where(resp_mask[0])[0]
    logit_positions = torch.where(resp_mask_last[0])[0]
    max_len = min(len(response_positions), len(logit_positions))
    response_positions = response_positions[:max_len]
    logit_positions = logit_positions[:max_len]

    base_lp = logits_nc[0, logit_positions]  # (R, V)
    influences = []
    for logits_ctx in all_logits:
        influences.append(logits_ctx[0, logit_positions] - base_lp)
    influences = torch.stack(influences, dim=0)  # (K, R, V)

    target_ids = full_nc_ids[0, response_positions]
    if response_weights.numel() != target_ids.numel():
        response_weights = torch.ones_like(target_ids, dtype=torch.float32, device=target_ids.device)
    else:
        response_weights = response_weights.to(target_ids.device)

    return EditBundle(
        base_lp=base_lp.detach().float(),
        influences=influences.detach().float(),
        target_ids=target_ids.detach(),
        weights=response_weights.detach().float(),
    )


def weighted_cos_logprob(bundle: EditBundle, lambdas: torch.Tensor) -> torch.Tensor:
    lambdas = lambdas.to(bundle.base_lp.device)
    scaled = (bundle.influences * lambdas[:, None, None]).sum(dim=0)
    cos_lp = F.log_softmax(bundle.base_lp + scaled, dim=-1)
    token_lp = cos_lp.gather(-1, bundle.target_ids[:, None]).squeeze(-1)
    return torch.sum(token_lp * bundle.weights)
