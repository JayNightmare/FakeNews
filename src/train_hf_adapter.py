#!/usr/bin/env python3
"""Prepare and fine-tune a PEFT adapter for misinformation classification."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cleaning import clean_metadata_batch, clean_records
from src.context_ablation import normalize_context_budget
from src.google_factcheck import enrich_records_with_google_factcheck
from src.run_experiment import DATASET_LOADERS
from src.schema import UnifiedRecord, load_records_jsonl
from src.training import export_training_corpus, load_training_examples, tokenize_training_examples


def load_training_records(
    *,
    dataset_name: str | None,
    records_jsonl: Path | None,
    data_dir: Path | None,
    limit: int | None,
) -> list[UnifiedRecord]:
    if records_jsonl is not None:
        return load_records_jsonl(records_jsonl)

    if dataset_name is None:
        raise ValueError("Either --dataset or --records-jsonl must be provided")

    loader_cls = DATASET_LOADERS[dataset_name]
    loader = loader_cls()
    resolved_dir = loader.resolve_data_dir(data_dir)
    raw_records = loader.load(resolved_dir, limit=limit)
    records, _ = clean_records(raw_records)
    return clean_metadata_batch(records)


def train_adapter(args: argparse.Namespace) -> dict[str, Any]:
    training_data_dir = Path(args.output_dir) / "training_data"
    adapter_dir = Path(args.output_dir) / "adapter"

    records = load_training_records(
        dataset_name=args.dataset,
        records_jsonl=Path(args.records_jsonl) if args.records_jsonl else None,
        data_dir=Path(args.data_dir) if args.data_dir else None,
        limit=args.limit,
    )

    grounding_report: dict[str, Any] = {"enabled": False, "records_seen": len(records)}
    if args.ground_with_google:
        cache_path = Path(args.google_factcheck_cache) if args.google_factcheck_cache else Path(args.output_dir) / "google_factcheck_cache.json"
        records, grounding_report = enrich_records_with_google_factcheck(
            records,
            cache_path=cache_path,
            ttl_hours=args.google_factcheck_ttl_hours,
        )

    manifest = export_training_corpus(
        records,
        training_data_dir,
        context_mode=args.context_mode,
        context_budget=args.context_budget,
        eval_ratio=args.eval_ratio,
        seed=args.seed,
    )

    if args.prepare_only:
        prepare_summary = {
            "prepared_only": True,
            "training_manifest": manifest,
            "grounding_report": grounding_report,
            "output_dir": str(Path(args.output_dir)),
        }
        _write_json(Path(args.output_dir) / "prepare_summary.json", prepare_summary)
        return prepare_summary

    try:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    except ImportError as exc:
        raise RuntimeError(
            "transformers, torch, and peft are required for adapter training"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {}
    if args.device_map:
        model_kwargs["device_map"] = args.device_map
    if args.torch_dtype:
        model_kwargs["torch_dtype"] = getattr(torch, args.torch_dtype)

    model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[module.strip() for module in args.lora_target_modules.split(",") if module.strip()],
    )
    model = get_peft_model(model, lora_config)

    train_examples = load_training_examples(training_data_dir / "train_examples.jsonl")
    eval_examples = load_training_examples(training_data_dir / "eval_examples.jsonl")
    train_features = tokenize_training_examples(train_examples, tokenizer, max_length=args.max_length)
    eval_features = tokenize_training_examples(eval_examples, tokenizer, max_length=args.max_length)

    train_dataset = _TokenizedDataset(train_features)
    eval_dataset = _TokenizedDataset(eval_features)
    data_collator = _SupervisedCollator(tokenizer)

    training_args = TrainingArguments(
        output_dir=str(adapter_dir),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        eval_strategy="epoch",
        report_to=[],
        remove_unused_columns=False,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    train_result = trainer.train()
    eval_metrics = trainer.evaluate()

    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    summary = {
        "prepared_only": False,
        "model_id": args.model_id,
        "adapter_dir": str(adapter_dir),
        "training_manifest": manifest,
        "grounding_report": grounding_report,
        "train_metrics": _normalize_metrics(train_result.metrics),
        "eval_metrics": _normalize_metrics(eval_metrics),
    }
    _write_json(Path(args.output_dir) / "training_summary.json", summary)
    return summary


class _TokenizedDataset:
    def __init__(self, rows: list[dict[str, list[int]]]) -> None:
        self._rows = rows

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self._rows[index]


class _SupervisedCollator:
    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        batch = self._tokenizer.pad(
            [{
                "input_ids": feature["input_ids"],
                "attention_mask": feature["attention_mask"],
            } for feature in features],
            padding=True,
            return_tensors="pt",
        )
        max_length = batch["input_ids"].shape[1]
        labels = []
        for feature in features:
            padding = max_length - len(feature["labels"])
            labels.append(feature["labels"] + ([-100] * padding))
        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare and fine-tune a Hugging Face adapter")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--dataset", choices=list(DATASET_LOADERS.keys()), help="Dataset to load and normalize")
    source_group.add_argument("--records-jsonl", help="Path to a normalized_records.jsonl file")
    parser.add_argument("--data-dir", help="Optional dataset directory override")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on source records")
    parser.add_argument("--output-dir", default="artifacts/training/qwen_adapter", help="Training output directory")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-1.5B-Instruct", help="Base Hugging Face model ID")
    parser.add_argument("--context-mode", default="full", choices=["minimal", "full", "misleading"], help="Prompt context mode")
    parser.add_argument("--context-budget", type=float, default=1.0, help="Fraction of context retained in training prompts")
    parser.add_argument("--eval-ratio", type=float, default=0.1, help="Eval split ratio for records without explicit splits")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic partitioning")
    parser.add_argument("--prepare-only", action="store_true", help="Only export the train/eval corpus without training")
    parser.add_argument("--ground-with-google", action="store_true", help="Enrich training records with cached Google fact checks")
    parser.add_argument("--google-factcheck-cache", help="Override path for the Google fact-check cache")
    parser.add_argument("--google-factcheck-ttl-hours", type=int, default=24, help="TTL for Google fact-check cache entries")
    parser.add_argument("--max-length", type=int, default=2048, help="Tokenizer max sequence length")
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--device-map", default="auto", help="Transformers device_map value")
    parser.add_argument("--torch-dtype", default=None, help="Optional torch dtype name, e.g. float16 or bfloat16")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj",
        help="Comma-separated module names to adapt",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    args.context_budget = normalize_context_budget(args.context_budget)
    summary = train_adapter(args)
    print(json.dumps(summary, indent=2))
    return 0


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _normalize_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float, str, bool)) or value is None:
            normalized[key] = value
        else:
            normalized[key] = str(value)
    return normalized


if __name__ == "__main__":
    raise SystemExit(main())