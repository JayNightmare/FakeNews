"""Utilities for preparing and fine-tuning Hugging Face adapters."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.context_ablation import normalize_context_budget
from src.prompts import build_prompt
from src.schema import UnifiedRecord

_SYSTEM_MESSAGE = "Return JSON only."


@dataclass
class TrainingExample:
    """Supervised chat example for adapter fine-tuning."""

    id: str
    dataset: str
    split: str
    context_mode: str
    context_budget: float
    prompt: str
    response: str
    messages: list[dict[str, str]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_training_response(
    record: UnifiedRecord,
    context_mode: str,
    *,
    context_budget: float = 1.0,
) -> str:
    """Construct a deterministic target JSON object for supervised training."""
    budget = normalize_context_budget(context_budget)
    classification = record.mapped_label_name
    confidence = _default_confidence(record, context_mode, budget)
    requires_external_evidence = not bool(record.context_text) or context_mode == "minimal" or budget < 0.5
    reasoning_signals = _default_reasoning_signals(record, context_mode, budget)
    explanation = _default_explanation(record, classification, context_mode, budget, reasoning_signals)
    payload = {
        "classification": classification,
        "confidence": confidence,
        "explanation": explanation,
        "reasoning_signals": reasoning_signals,
        "requires_external_evidence": requires_external_evidence,
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ": "))


def build_training_example(
    record: UnifiedRecord,
    context_mode: str = "full",
    *,
    context_budget: float = 1.0,
) -> TrainingExample:
    """Build a supervised chat-format example from a unified record."""
    prompt = build_prompt(record, context_mode, context_budget=context_budget)
    response = build_training_response(record, context_mode, context_budget=context_budget)
    budget = normalize_context_budget(context_budget)
    return TrainingExample(
        id=record.sample_id,
        dataset=record.dataset,
        split=record.split or "unspecified",
        context_mode=context_mode,
        context_budget=budget,
        prompt=prompt,
        response=response,
        messages=[
            {"role": "system", "content": _SYSTEM_MESSAGE},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ],
    )


def partition_records_for_training(
    records: list[UnifiedRecord],
    *,
    eval_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[UnifiedRecord], list[UnifiedRecord]]:
    """Split records into train/eval partitions deterministically.

    Uses explicit dataset splits when available; otherwise falls back to a stable,
    seeded shuffle on sample IDs.
    """
    if eval_ratio <= 0 or eval_ratio >= 1:
        raise ValueError("eval_ratio must be > 0 and < 1")

    train_records = [record for record in records if _is_train_split(record.split)]
    eval_records = [record for record in records if _is_eval_split(record.split)]
    remainder = [record for record in records if record not in train_records and record not in eval_records]

    if remainder:
        ranked = sorted(
            remainder,
            key=lambda record: hashlib.sha1(f"{seed}:{record.sample_id}".encode("utf-8")).hexdigest(),
        )
        eval_count = max(1, round(len(ranked) * eval_ratio)) if len(ranked) > 1 else 1
        eval_records.extend(ranked[:eval_count])
        train_records.extend(ranked[eval_count:])

    if not train_records and eval_records:
        train_records, eval_records = eval_records[:-1], eval_records[-1:]
    if not eval_records and train_records:
        eval_records = train_records[-1:]
        train_records = train_records[:-1] or train_records

    return train_records, eval_records


def export_training_corpus(
    records: list[UnifiedRecord],
    output_dir: Path,
    *,
    context_mode: str = "full",
    context_budget: float = 1.0,
    eval_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, Any]:
    """Export deterministic train/eval chat examples for adapter tuning."""
    budget = normalize_context_budget(context_budget)
    train_records, eval_records = partition_records_for_training(records, eval_ratio=eval_ratio, seed=seed)
    train_examples = [build_training_example(record, context_mode, context_budget=budget) for record in train_records]
    eval_examples = [build_training_example(record, context_mode, context_budget=budget) for record in eval_records]

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train_examples.jsonl"
    eval_path = output_dir / "eval_examples.jsonl"
    manifest_path = output_dir / "training_manifest.json"

    _write_examples_jsonl(train_path, train_examples)
    _write_examples_jsonl(eval_path, eval_examples)

    manifest = {
        "context_mode": context_mode,
        "context_budget": budget,
        "eval_ratio": eval_ratio,
        "seed": seed,
        "train_count": len(train_examples),
        "eval_count": len(eval_examples),
        "train_path": str(train_path),
        "eval_path": str(eval_path),
        "datasets": sorted({record.dataset for record in records}),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def load_training_examples(path: Path) -> list[TrainingExample]:
    """Load chat-format training examples from JSONL."""
    examples: list[TrainingExample] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            examples.append(TrainingExample(**payload))
    return examples


def tokenize_training_examples(
    examples: list[TrainingExample],
    tokenizer: Any,
    *,
    max_length: int,
) -> list[dict[str, list[int]]]:
    """Tokenize chat examples and mask prompt tokens from the loss."""
    tokenized: list[dict[str, list[int]]] = []
    for example in examples:
        prompt_messages = example.messages[:-1]
        full_messages = example.messages
        prompt_text = _render_messages(tokenizer, prompt_messages, add_generation_prompt=True)
        full_text = _render_messages(tokenizer, full_messages, add_generation_prompt=False)

        prompt_tokens = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_batch = tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        )
        input_ids = list(full_batch["input_ids"])
        attention_mask = list(full_batch["attention_mask"])
        labels = list(input_ids)
        prompt_length = min(len(prompt_tokens), len(labels))
        labels[:prompt_length] = [-100] * prompt_length
        tokenized.append({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        })
    return tokenized


def _default_confidence(record: UnifiedRecord, context_mode: str, budget: float) -> float:
    confidence = 0.82 if record.context_text else 0.66
    if context_mode == "minimal":
        confidence -= 0.14
    elif context_mode == "misleading":
        confidence -= 0.18
    elif budget < 1.0:
        confidence -= min(0.3, (1.0 - budget) * 0.35)
    if record.has_image:
        confidence += 0.03
    return round(max(0.2, min(confidence, 0.98)), 2)


def _default_reasoning_signals(record: UnifiedRecord, context_mode: str, budget: float) -> list[str]:
    signals = [f"dataset: {record.dataset}"]
    if record.metadata.get("review_publisher"):
        signals.append(f"review publisher: {record.metadata['review_publisher']}")
    if record.metadata.get("source"):
        signals.append(f"source: {record.metadata['source']}")
    if context_mode != "minimal" and record.context_text:
        signals.append("context text available")
    if budget < 1.0:
        signals.append(f"context budget: {budget:.2f}")
    if context_mode == "misleading":
        signals.append("context may be unreliable")
    return signals[:4]


def _default_explanation(
    record: UnifiedRecord,
    classification: str,
    context_mode: str,
    budget: float,
    reasoning_signals: list[str],
) -> str:
    evidence_clause = "available contextual evidence" if record.context_text else "the claim text alone"
    if context_mode == "misleading":
        evidence_clause = "the claim text while discounting unreliable surrounding context"
    elif budget < 1.0:
        evidence_clause = f"the retained {budget:.0%} of the available context"
    signal_clause = ", ".join(reasoning_signals[:2]) if reasoning_signals else "dataset-specific evidence"
    return (
        f"This item is classified as {classification} based on {evidence_clause}. "
        f"The strongest signals come from {signal_clause}."
    )


def _write_examples_jsonl(path: Path, examples: list[TrainingExample]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example.to_dict(), ensure_ascii=False) + "\n")


def _is_train_split(split: str | None) -> bool:
    return (split or "").casefold() in {"train", "training"}


def _is_eval_split(split: str | None) -> bool:
    return (split or "").casefold() in {"val", "valid", "validation", "eval", "test"}


def _render_messages(tokenizer: Any, messages: list[dict[str, str]], *, add_generation_prompt: bool) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    rendered = []
    for message in messages:
        rendered.append(f"{message['role']}: {message['content']}")
    if add_generation_prompt:
        rendered.append("assistant:")
    return "\n\n".join(rendered)