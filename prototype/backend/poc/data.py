from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import random


@dataclass
class EditExample:
    user_id: str
    before: str
    after: str
    prompt: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)


def _normalize_record(raw: Dict[str, Any]) -> EditExample:
    before = raw.get("before") or raw.get("y_before") or raw.get("input_before")
    after = raw.get("after") or raw.get("y_after") or raw.get("input_after")
    if before is None or after is None:
        raise ValueError("Each JSONL record must have before/after text.")

    user_id = (
        raw.get("user_id")
        or raw.get("user")
        or raw.get("uid")
        or raw.get("author")
        or "unknown"
    )
    prompt = raw.get("prompt") or raw.get("instruction") or raw.get("context") or ""

    meta = {k: v for k, v in raw.items() if k not in {"before", "after", "y_before", "y_after", "input_before", "input_after", "user_id", "user", "uid", "author", "prompt", "instruction", "context"}}

    return EditExample(
        user_id=str(user_id),
        before=str(before),
        after=str(after),
        prompt=str(prompt),
        meta=meta,
    )


def load_jsonl(path: str | Path) -> List[EditExample]:
    path = Path(path)
    examples: List[EditExample] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            examples.append(_normalize_record(raw))
    return examples


def split_by_user(
    examples: Iterable[EditExample],
    val_fraction: float = 0.2,
    seed: int = 13,
) -> Tuple[List[EditExample], List[EditExample]]:
    by_user: Dict[str, List[EditExample]] = {}
    for ex in examples:
        by_user.setdefault(ex.user_id, []).append(ex)

    rng = random.Random(seed)
    train: List[EditExample] = []
    val: List[EditExample] = []
    for user_id, items in by_user.items():
        items = list(items)
        rng.shuffle(items)
        n_val = max(1, int(len(items) * val_fraction)) if len(items) > 1 else 0
        val.extend(items[:n_val])
        train.extend(items[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def group_by_user(examples: Iterable[EditExample]) -> Dict[str, List[EditExample]]:
    by_user: Dict[str, List[EditExample]] = {}
    for ex in examples:
        by_user.setdefault(ex.user_id, []).append(ex)
    return by_user
