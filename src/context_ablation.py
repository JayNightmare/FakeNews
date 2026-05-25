"""Helpers for deterministic context-budget ablation experiments."""

from __future__ import annotations

import re


def normalize_context_budget(value: float) -> float:
    """Validate and normalize a context budget.

    Budgets represent the proportion of the available context retained in the
    prompt. Values must be in the interval (0, 1].
    """
    if value <= 0 or value > 1:
        raise ValueError("context budget must be > 0 and <= 1")
    return round(float(value), 4)


def parse_context_budget_levels(raw_value: str | None) -> list[float]:
    """Parse a comma-delimited list of context budgets."""
    if not raw_value:
        return []

    budgets: list[float] = []
    for item in raw_value.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        budgets.append(normalize_context_budget(float(stripped)))

    if not budgets:
        return []
    return sorted(set(budgets), reverse=True)


def truncate_context_text(context_text: str | None, context_budget: float) -> str | None:
    """Keep a deterministic leading slice of context text.

    Sentence boundaries are preferred. If the text does not contain punctuation,
    a word-based fallback is used.
    """
    if context_text is None:
        return None

    budget = normalize_context_budget(context_budget)
    stripped = context_text.strip()
    if not stripped or budget >= 1.0:
        return stripped or None

    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", stripped) if part.strip()]
    if len(sentences) >= 2:
        keep_count = max(1, round(len(sentences) * budget))
        return " ".join(sentences[:keep_count]).strip()

    words = stripped.split()
    if not words:
        return None
    keep_count = max(12, round(len(words) * budget))
    return " ".join(words[:keep_count]).strip()