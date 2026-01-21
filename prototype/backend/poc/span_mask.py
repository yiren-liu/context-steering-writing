from __future__ import annotations

import difflib
import re
from typing import Iterable, List, Sequence, Tuple

import torch
from transformers import PreTrainedTokenizerBase


def _word_spans(text: str) -> List[Tuple[int, int]]:
    return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]


def _diff_word_indices(
    before_words: Sequence[str],
    after_words: Sequence[str],
) -> Tuple[List[int], List[int]]:
    matcher = difflib.SequenceMatcher(a=before_words, b=after_words, autojunk=False)
    before_indices: List[int] = []
    after_indices: List[int] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if tag in {"replace", "delete"}:
            before_indices.extend(range(i1, i2))
        if tag in {"replace", "insert"}:
            after_indices.extend(range(j1, j2))
        if tag == "delete" and j1 == j2 and j1 < len(after_words):
            after_indices.append(j1)
    return sorted(set(before_indices)), sorted(set(after_indices))


def _token_mask_from_word_indices(
    text: str,
    tokenizer: PreTrainedTokenizerBase,
    word_indices: Iterable[int],
    window_radius: int = 2,
    decay: float = 0.3,
) -> torch.Tensor:
    if not text.strip():
        return torch.ones(0, dtype=torch.float32)

    word_indices = set(word_indices)
    if not word_indices:
        return torch.ones(len(tokenizer(text).input_ids), dtype=torch.float32)

    word_spans = _word_spans(text)
    tokenized = tokenizer(text, return_offsets_mapping=True)
    offsets = tokenized["offset_mapping"]

    token_to_word: List[int | None] = []
    w_idx = 0
    for start, end in offsets:
        if start == end == 0:
            token_to_word.append(None)
            continue
        while w_idx < len(word_spans) and end > word_spans[w_idx][1]:
            w_idx += 1
        if w_idx < len(word_spans) and not (end <= word_spans[w_idx][0] or start >= word_spans[w_idx][1]):
            token_to_word.append(w_idx)
        else:
            token_to_word.append(None)

    mask = torch.zeros(len(offsets), dtype=torch.float32)
    changed_token_indices = []
    for t_idx, w_idx in enumerate(token_to_word):
        if w_idx is not None and w_idx in word_indices:
            mask[t_idx] = 1.0
            changed_token_indices.append(t_idx)

    if not changed_token_indices:
        return torch.ones(len(offsets), dtype=torch.float32)

    if window_radius > 0:
        for idx in changed_token_indices:
            lo = max(0, idx - window_radius)
            hi = min(len(offsets), idx + window_radius + 1)
            mask[lo:hi] = torch.maximum(mask[lo:hi], torch.tensor(decay, dtype=torch.float32))

    return mask


def build_span_masks(
    before: str,
    after: str,
    tokenizer: PreTrainedTokenizerBase,
    window_radius: int = 2,
    decay: float = 0.3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    before_words = before.split()
    after_words = after.split()
    before_indices, after_indices = _diff_word_indices(before_words, after_words)

    before_mask = _token_mask_from_word_indices(
        before, tokenizer, before_indices, window_radius=window_radius, decay=decay
    )
    after_mask = _token_mask_from_word_indices(
        after, tokenizer, after_indices, window_radius=window_radius, decay=decay
    )
    return before_mask, after_mask
