from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class StyleContext:
    name: str
    instruction: str


DEFAULT_CONTEXTS: List[StyleContext] = [
    StyleContext(
        name="concise",
        instruction="Write tersely; remove redundancies; keep meaning.",
    ),
    StyleContext(
        name="formal",
        instruction="Use formal, professional tone; avoid slang; precise wording.",
    ),
    StyleContext(
        name="structured",
        instruction="Use clear structure; short paragraphs; explicit transitions.",
    ),
    StyleContext(
        name="empathetic",
        instruction="Write in a warm, empathetic tone; use first-person pronouns; show understanding.",
    ),
    StyleContext(
        name="vivid",
        instruction="Prefer concrete, vivid language; active voice; specific nouns/verbs.",
    ),
    StyleContext(
        name="certainty",
        instruction="Write with confidence; avoid hedging; make clear decisions.",
    ),
]


def contexts_as_strings(contexts: List[StyleContext] | None = None) -> List[str]:
    contexts = contexts or DEFAULT_CONTEXTS
    return [c.instruction for c in contexts]


def slider_to_lambda(scale: int, alpha: float = 1.0) -> float:
    if scale < 1 or scale > 7:
        raise ValueError("Slider scale must be in [1, 7].")
    return alpha * (scale - 4) / 3.0


def lambdas_from_slider(
    slider: int,
    mean_vector: List[float],
    alpha: float = 1.0,
) -> List[float]:
    strength = slider_to_lambda(slider, alpha=alpha)
    return [strength * m for m in mean_vector]
