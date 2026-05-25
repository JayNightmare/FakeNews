"""Prompting system for the misinformation classification pipeline.

Supports three context variants as specified in the experimental design:
- minimal: claim text only
- full: claim + article body + metadata
- misleading: claim + deliberately ambiguous/contradictory context

Each variant generates a prompt for binary classification (real/fake)
plus explanation generation.
"""

from __future__ import annotations

import json
from typing import Any

from src.context_ablation import normalize_context_budget, truncate_context_text
from src.schema import UnifiedRecord

CONTEXT_MODES = ("minimal", "full", "misleading")

_SYSTEM_PROMPT = (
    "You are a misinformation detection system. Analyze the provided content "
    "and determine whether it is real or fake. Return a structured JSON response."
)

_OUTPUT_SCHEMA = """\
Return JSON with this shape:

```json
{
  "classification": "real" or "fake",
  "confidence": 0.0 to 1.0,
  "explanation": "1-3 sentence justification for the classification",
  "reasoning_signals": ["list of key signals used"],
  "requires_external_evidence": true/false
}
```

Rules:
- Use only the information provided.
- If context is missing or insufficient, lower your confidence.
- Do not invent evidence.
- Keep the explanation to 1-3 sentences."""


def build_prompt(
    record: UnifiedRecord,
    context_mode: str = "full",
    *,
    context_budget: float = 1.0,
) -> str:
    """Build a classification prompt for a given record and context mode.

    Args:
        record: A UnifiedRecord instance.
        context_mode: One of 'minimal', 'full', or 'misleading'.

    Returns:
        The formatted prompt string.
    """
    if context_mode not in CONTEXT_MODES:
        raise ValueError(f"Unknown context_mode '{context_mode}', expected one of {CONTEXT_MODES}")
    budget = normalize_context_budget(context_budget)

    sections: list[str] = ["## Task", "Classify the following content as real or fake.\n"]

    if context_mode == "minimal":
        sections.append("## Content")
        sections.append(f"Claim: {record.text}\n")

    elif context_mode == "full":
        sections.append("## Content")
        sections.append(f"Claim: {record.text}\n")
        context_text = truncate_context_text(record.context_text, budget)
        if context_text:
            sections.append("## Additional Context")
            sections.append(f"{context_text}\n")
        meta_lines = _format_metadata(record)
        if meta_lines:
            sections.append("## Metadata")
            sections.append(meta_lines + "\n")

    elif context_mode == "misleading":
        sections.append("## Content")
        sections.append(f"Claim: {record.text}\n")
        sections.append("## Context (note: context may be unreliable)")
        misleading_context = _generate_misleading_context(record)
        sections.append(f"{misleading_context}\n")

    sections.append("## Output Requirements")
    sections.append(_OUTPUT_SCHEMA)

    return "\n".join(sections)


def build_prompt_messages(
    record: UnifiedRecord,
    context_mode: str = "full",
    *,
    context_budget: float = 1.0,
) -> list[dict[str, str]]:
    """Build a chat-format message list for API consumption.

    Returns:
        List of message dicts with 'role' and 'content'.
    """
    user_prompt = build_prompt(record, context_mode, context_budget=context_budget)
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def _format_metadata(record: UnifiedRecord) -> str:
    """Format metadata fields into readable key-value lines."""
    lines: list[str] = []
    meta = record.metadata
    if meta.get("platform"):
        lines.append(f"- Platform: {meta['platform']}")
    if meta.get("publish_date") or meta.get("review_date") or meta.get("timestamp"):
        date = meta.get("publish_date") or meta.get("review_date") or meta.get("timestamp")
        lines.append(f"- Date: {date}")
    if meta.get("source"):
        lines.append(f"- Source: {meta['source']}")
    if meta.get("review_publisher"):
        lines.append(f"- Review publisher: {meta['review_publisher']}")
    if meta.get("language"):
        lines.append(f"- Language: {meta['language']}")
    if record.modality != "text":
        lines.append(f"- Modality: {record.modality}")
    return "\n".join(lines)


def _generate_misleading_context(record: UnifiedRecord) -> str:
    """Generate deliberately ambiguous and contradictory context.

    This constructs context that contains BOTH credibility-boosting and
    credibility-undermining signals, creating genuine ambiguity.

    IMPORTANT: This function is intentionally label-blind. It does NOT
    read `mapped_label` or `mapped_label_name`, ensuring the experimental
    manipulation is methodologically sound.
    """
    # Use a deterministic seed from the record's sample_id so the
    # misleading context is reproducible across runs but varies per record.
    seed = sum(ord(c) for c in record.sample_id) % 3

    # Three fixed misleading-context templates. Each mixes contradictory signals.
    _TEMPLATES = [
        (
            "Multiple reputable sources have confirmed the accuracy of this claim. "
            "However, independent analysts have raised concerns about the methodology "
            "used to verify it, and the original source has faced credibility questions."
        ),
        (
            "Several online communities have flagged this content as potentially "
            "misleading, though a well-known fact-checking organization found it to "
            "be substantially accurate. The discrepancy remains unresolved."
        ),
        (
            "This content originates from a source with a mixed track record. "
            "Some experts endorse the claims made, while others have pointed out "
            "significant factual errors in similar reporting from the same outlet."
        ),
    ]

    parts: list[str] = [_TEMPLATES[seed]]

    # Include a truncated fragment of real context (if available) to make
    # the misleading context more grounded and harder to dismiss.
    if record.context_text:
        words = record.context_text.split()
        if len(words) > 10:
            fragment = " ".join(words[:10]) + "..."
            parts.append(f"Partial related text: {fragment}")

    parts.append(
        "Note: The reliability of the above context has not been independently verified."
    )

    return " ".join(parts)


def build_prompt_payload(
    record: UnifiedRecord,
    context_mode: str = "full",
    *,
    context_budget: float = 1.0,
) -> dict[str, Any]:
    """Build a complete prompt payload for serialization.

    Returns:
        Dict with id, context_mode, prompt text, and messages.
    """
    return {
        "id": record.sample_id,
        "dataset": record.dataset,
        "context_mode": context_mode,
        "context_budget": normalize_context_budget(context_budget),
        "prompt": build_prompt(record, context_mode, context_budget=context_budget),
        "messages": build_prompt_messages(record, context_mode, context_budget=context_budget),
        "ground_truth_label": record.mapped_label,
        "ground_truth_label_name": record.mapped_label_name,
    }
