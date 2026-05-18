"""Optional cost estimation for API-based inference.

Character-based fallback (stdlib, always works).
Optional tiktoken support if installed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Approximate characters per token (conservative estimate)
_CHARS_PER_TOKEN = 4

# Pricing per 1M tokens (USD) as of mid-2026 — update as needed
_PRICING: dict[str, dict[str, float]] = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4": {"input": 30.00, "output": 60.00},
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    "gemini-2.5-pro": {"input": 1.25, "output": 10.00},
    "claude-sonnet-4": {"input": 3.00, "output": 15.00},
    "claude-haiku": {"input": 0.25, "output": 1.25},
}


def estimate_tokens_charcount(text: str) -> int:
    """Estimate token count from character count (no dependencies)."""
    return max(1, len(text) // _CHARS_PER_TOKEN)


def estimate_tokens_tiktoken(text: str, model: str = "gpt-4o-mini") -> int:
    """Estimate token count using tiktoken (if installed)."""
    try:
        import tiktoken  # type: ignore[import-untyped]
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        return estimate_tokens_charcount(text)


def estimate_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """Estimate tokens, using tiktoken if available, char fallback otherwise."""
    try:
        import tiktoken  # type: ignore[import-untyped] # noqa: F401
        return estimate_tokens_tiktoken(text, model)
    except ImportError:
        return estimate_tokens_charcount(text)


def estimate_cost(
    prompts: list[str],
    model: str = "gpt-4o-mini",
    avg_output_tokens: int = 150,
) -> dict[str, Any]:
    """Estimate the cost of running prompts through a model.

    Args:
        prompts: List of prompt strings.
        model: Model identifier.
        avg_output_tokens: Estimated average output tokens per request.

    Returns:
        Cost estimation dictionary.
    """
    input_tokens = sum(estimate_tokens(p, model) for p in prompts)
    output_tokens = len(prompts) * avg_output_tokens
    total_tokens = input_tokens + output_tokens

    pricing = _PRICING.get(model, {"input": 1.0, "output": 3.0})
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]

    return {
        "model": model,
        "prompt_count": len(prompts),
        "estimated_input_tokens": input_tokens,
        "estimated_output_tokens": output_tokens,
        "estimated_total_tokens": total_tokens,
        "estimated_input_cost_usd": round(input_cost, 4),
        "estimated_output_cost_usd": round(output_cost, 4),
        "estimated_total_cost_usd": round(input_cost + output_cost, 4),
        "pricing_per_1m_tokens": pricing,
        "token_estimation_method": _get_estimation_method(),
    }


def estimate_cost_all_models(
    prompts: list[str],
    avg_output_tokens: int = 150,
) -> dict[str, Any]:
    """Estimate costs across all known models."""
    estimates = {}
    for model_name in _PRICING:
        estimates[model_name] = estimate_cost(prompts, model_name, avg_output_tokens)

    return {
        "prompt_count": len(prompts),
        "per_model": estimates,
        "cheapest": min(estimates, key=lambda k: estimates[k]["estimated_total_cost_usd"]),
        "most_expensive": max(estimates, key=lambda k: estimates[k]["estimated_total_cost_usd"]),
    }


def _get_estimation_method() -> str:
    try:
        import tiktoken  # type: ignore[import-untyped] # noqa: F401
        return "tiktoken"
    except ImportError:
        return "character_count"


def write_cost_report(path: Path, report: dict[str, Any]) -> None:
    """Write cost estimation as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
