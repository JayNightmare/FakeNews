#!/usr/bin/env python3
"""Prepare and fine-tune a PEFT adapter for misinformation classification."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, TypeVar

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cleaning import clean_metadata_batch, clean_records
from src.context_ablation import normalize_context_budget
from src.evaluation import evaluate_predictions
from src.env import load_project_env
from src.google_factcheck import enrich_records_with_google_factcheck
from src.predictors import create_predictor
from src.prompts import build_prompt
from src.run_experiment import DATASET_LOADERS, MASTER_DATASET_NAME, load_dataset
from src.schema import UnifiedRecord, load_records_jsonl
from src.training import export_training_corpus, load_training_examples, tokenize_training_examples

_FeatureRow = TypeVar("_FeatureRow")

if TYPE_CHECKING:
    from torch.utils.data import Dataset as TorchDataset
else:
    class TorchDataset(Generic[_FeatureRow]):
        def __len__(self) -> int:
            raise NotImplementedError

        def __getitem__(self, index: int) -> _FeatureRow:
            raise NotImplementedError


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

    if dataset_name == MASTER_DATASET_NAME:
        return load_dataset(dataset_name, data_dir, limit)

    loader_cls = DATASET_LOADERS[dataset_name]
    loader = loader_cls()
    resolved_dir = loader.resolve_data_dir(data_dir)
    raw_records = loader.load(resolved_dir, limit=limit)
    records, _ = clean_records(raw_records)
    return clean_metadata_batch(records)


def _resolve_split_records_path(split_dir: Path, split_name: str) -> Path:
    candidates = [
        split_dir / f"{split_name}.jsonl",
        split_dir / "normalized_records.jsonl",
        split_dir / "master_records.jsonl",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"No {split_name} records file found in {split_dir}. Looked for: {searched}"
    )


def _load_split_records(split_dir: Path, split_name: str) -> list[UnifiedRecord]:
    records_path = _resolve_split_records_path(split_dir, split_name)
    records = load_records_jsonl(records_path)
    return [record for record in records if bool(record.text and record.text.strip())]


def _deterministic_take(records: list[UnifiedRecord], limit: int | None, seed: int) -> list[UnifiedRecord]:
    if limit is None or limit <= 0 or len(records) <= limit:
        return records
    ranked = sorted(
        records,
        key=lambda record: hashlib.sha1(f"{seed}:{record.sample_id}".encode("utf-8")).hexdigest(),
    )
    return ranked[:limit]


def _evaluate_records_with_predictor(
    records: list[UnifiedRecord],
    *,
    context_mode: str,
    context_budget: float,
    model_mode: str,
) -> dict[str, Any]:
    predictor = create_predictor(model_mode)
    predictions: list[dict[str, Any]] = []
    for record in records:
        prompt = build_prompt(record, context_mode, context_budget=context_budget)
        predictions.append(predictor.predict(record, prompt, context_mode))
    report = evaluate_predictions(predictions)
    report["prediction_count"] = len(predictions)
    return report


def _gate_release(
    *,
    train_metrics: dict[str, Any],
    eval_metrics: dict[str, Any],
    train_eval_report: dict[str, Any],
    validation_report: dict[str, Any],
    test_report: dict[str, Any],
    min_accuracy: float,
    min_f1: float,
    min_confidence: float,
    max_train_val_f1_gap: float,
    max_val_train_loss_ratio: float,
    max_underfit_loss: float,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add_check(name: str, passed: bool, actual: Any, expected: str) -> None:
        checks.append({
            "name": name,
            "passed": bool(passed),
            "actual": actual,
            "expected": expected,
        })

    val_overall = validation_report.get("overall", {})
    test_overall = test_report.get("overall", {})
    train_overall = train_eval_report.get("overall", {})

    val_accuracy = float(val_overall.get("accuracy", 0.0))
    val_f1 = float(val_overall.get("f1", 0.0))
    test_accuracy = float(test_overall.get("accuracy", 0.0))
    test_f1 = float(test_overall.get("f1", 0.0))
    train_f1 = float(train_overall.get("f1", 0.0))
    test_confidence_mean = float(test_report.get("confidence_stats", {}).get("mean", 0.0))

    add_check("validation_accuracy", val_accuracy >= min_accuracy, val_accuracy, f">= {min_accuracy}")
    add_check("validation_f1", val_f1 >= min_f1, val_f1, f">= {min_f1}")
    add_check("test_accuracy", test_accuracy >= min_accuracy, test_accuracy, f">= {min_accuracy}")
    add_check("test_f1", test_f1 >= min_f1, test_f1, f">= {min_f1}")
    add_check("test_confidence_mean", test_confidence_mean >= min_confidence, test_confidence_mean, f">= {min_confidence}")

    f1_gap = abs(train_f1 - val_f1)
    add_check("train_val_f1_gap", f1_gap <= max_train_val_f1_gap, f1_gap, f"<= {max_train_val_f1_gap}")

    train_loss = train_metrics.get("train_loss")
    val_loss = eval_metrics.get("eval_loss")
    if isinstance(train_loss, (int, float)) and isinstance(val_loss, (int, float)) and float(train_loss) > 0:
        loss_ratio = float(val_loss) / float(train_loss)
        add_check("val_train_loss_ratio", loss_ratio <= max_val_train_loss_ratio, round(loss_ratio, 6), f"<= {max_val_train_loss_ratio}")
        add_check("underfit_loss_ceiling", float(val_loss) <= max_underfit_loss, float(val_loss), f"<= {max_underfit_loss}")
    else:
        add_check("val_train_loss_ratio", False, {"train_loss": train_loss, "eval_loss": val_loss}, "train_loss and eval_loss must be numeric")

    passed = all(check["passed"] for check in checks)
    failures = [check for check in checks if not check["passed"]]
    return {
        "passed": passed,
        "checks": checks,
        "failed_checks": failures,
    }


def _push_model_bundle(model_folder: Path, repo_id: str, private: bool = False) -> dict[str, Any]:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required for --auto-push-to-hub") from exc

    api = HfApi()
    api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
    api.upload_folder(
        folder_path=str(model_folder),
        repo_id=repo_id,
        repo_type="model",
        commit_message="Upload FakeNews model bundle",
    )
    return {
        "repo_id": repo_id,
        "private": private,
        "url": f"https://huggingface.co/{repo_id}",
    }


def train_adapter(args: argparse.Namespace) -> dict[str, Any]:
    training_data_dir = Path(args.output_dir) / "training_data"
    adapter_dir = Path(args.output_dir) / "adapter"

    use_explicit_splits = bool(args.train_records_dir)
    if use_explicit_splits:
        train_records = _load_split_records(Path(args.train_records_dir), "train")
        validation_records = _load_split_records(Path(args.validation_records_dir), "validation")
        test_records = _load_split_records(Path(args.test_records_dir), "test")
        if args.limit is not None:
            train_records = _deterministic_take(train_records, args.limit, args.seed)
            validation_records = _deterministic_take(validation_records, args.limit, args.seed + 1)
            test_records = _deterministic_take(test_records, args.limit, args.seed + 2)
        records = train_records
    else:
        records = load_training_records(
            dataset_name=args.dataset,
            records_jsonl=Path(args.records_jsonl) if args.records_jsonl else None,
            data_dir=Path(args.data_dir) if args.data_dir else None,
            limit=args.limit,
        )
        train_records = records
        validation_records = []
        test_records = []

    grounding_report: dict[str, Any] = {"enabled": False, "records_seen": len(records)}
    if args.ground_with_google:
        cache_path = Path(args.google_factcheck_cache) if args.google_factcheck_cache else Path(args.output_dir) / "google_factcheck_cache.json"
        records, grounding_report = enrich_records_with_google_factcheck(
            records,
            cache_path=cache_path,
            ttl_hours=args.google_factcheck_ttl_hours,
        )
        train_records = records

    manifest = export_training_corpus(
        train_records,
        training_data_dir,
        eval_records=validation_records if use_explicit_splits else None,
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
            "split_inputs": {
                "enabled": use_explicit_splits,
                "train_records_dir": args.train_records_dir,
                "validation_records_dir": args.validation_records_dir,
                "test_records_dir": args.test_records_dir,
                "train_records": len(train_records),
                "validation_records": len(validation_records),
                "test_records": len(test_records),
            },
            "output_dir": str(Path(args.output_dir)),
        }
        _write_json(Path(args.output_dir) / "prepare_summary.json", prepare_summary)
        if args.generate_model_folder:
            model_folder = _generate_model_folder(args, prepare_summary, adapter_dir=None)
            prepare_summary["model_folder"] = str(model_folder)
            _write_json(Path(args.output_dir) / "prepare_summary.json", prepare_summary)
        return prepare_summary

    try:
        import torch
        from peft import LoraConfig, get_peft_model
        import transformers
        from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "transformers, torch, and peft are required for adapter training"
        ) from exc

    torch.manual_seed(args.seed)
    try:
        import random
        random.seed(args.seed)
    except Exception:
        pass

    requested_device_name = (args.device_map or "auto").casefold()
    if requested_device_name not in {"auto", "cuda", "cpu", "mps"}:
        raise RuntimeError(
            "This trainer only supports concrete training devices: auto, cuda, cpu, or mps. "
            "Transformers device_map sharding/offload values such as 'balanced' or 'sequential' "
            "are inference-oriented and are not supported by this manual LoRA training loop."
        )
    if args.offload_folder:
        raise RuntimeError(
            "--offload-folder is not supported for adapter training in this script. "
            "CPU/disk offload through Transformers leaves parameters on the meta device and "
            "breaks backward in this training loop. Use a smaller model, prepare-only mode, "
            "or a distributed setup such as DeepSpeed/FSDP instead."
        )

    model_kwargs: dict[str, Any] = {}
    if args.torch_dtype:
        model_kwargs["dtype"] = getattr(torch, args.torch_dtype)
    elif requested_device_name in {"auto", "cuda"} and torch.cuda.is_available():
        model_kwargs["dtype"] = torch.float16

    config = AutoConfig.from_pretrained(args.model_id)
    config_name = type(config).__name__.casefold()
    is_vision_language = any(token in config_name for token in ("vision", "vl", "multi"))

    tokenizer: Any
    model_loader_name = "AutoModelForCausalLM"
    if is_vision_language:
        model_loader = getattr(transformers, "AutoModelForImageTextToText", None)
        if model_loader is None:
            model_loader = getattr(transformers, "AutoModelForVision2Seq", None)
        if model_loader is None:
            raise RuntimeError(
                "This model appears to be vision-language, but your transformers version "
                "does not expose AutoModelForImageTextToText/AutoModelForVision2Seq. "
                "Upgrade transformers, or use a text-only model such as "
                "Qwen/Qwen2.5-1.5B-Instruct or Qwen/Qwen2.5-7B-Instruct."
            )

        processor = AutoProcessor.from_pretrained(args.model_id)
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError(
                "AutoProcessor did not expose a tokenizer for this vision-language model. "
                "Use a text-only model id for this trainer."
            )
        model = model_loader.from_pretrained(args.model_id, **model_kwargs)
        model_loader_name = type(model_loader).__name__
        print("Detected vision-language model; training text-only adapter using processor tokenizer.")
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_id)
        model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[module.strip() for module in args.lora_target_modules.split(",") if module.strip()],
    )
    model = get_peft_model(model, lora_config)
    if hasattr(model, "config"):
        model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable()
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
        except Exception:
            pass

    training_device = _resolve_training_device(torch, args.device_map)
    model.to(training_device)

    train_examples = load_training_examples(training_data_dir / "train_examples.jsonl")
    eval_examples = load_training_examples(training_data_dir / "eval_examples.jsonl")
    train_features = tokenize_training_examples(train_examples, tokenizer, max_length=args.max_length)
    eval_features = tokenize_training_examples(eval_examples, tokenizer, max_length=args.max_length)

    train_dataset = _TokenizedDataset(train_features)
    eval_dataset = _TokenizedDataset(eval_features)
    data_collator = _SupervisedCollator(tokenizer)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
    )

    train_metrics = _train_model(
        torch,
        model,
        train_dataset,
        data_collator,
        optimizer,
        training_device,
        per_device_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
    )
    eval_metrics = _evaluate_model(
        torch,
        model,
        eval_dataset,
        data_collator,
        training_device,
        per_device_batch_size=args.per_device_eval_batch_size,
    )

    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    summary = {
        "prepared_only": False,
        "model_id": args.model_id,
        "model_loader": model_loader_name,
        "vision_language_model": is_vision_language,
        "adapter_dir": str(adapter_dir),
        "training_manifest": manifest,
        "grounding_report": grounding_report,
        "training_device": str(training_device),
        "train_metrics": _normalize_metrics(train_metrics),
        "eval_metrics": _normalize_metrics(eval_metrics),
        "split_inputs": {
            "enabled": use_explicit_splits,
            "train_records_dir": args.train_records_dir,
            "validation_records_dir": args.validation_records_dir,
            "test_records_dir": args.test_records_dir,
            "train_records": len(train_records),
            "validation_records": len(validation_records),
            "test_records": len(test_records),
        },
    }

    release_gate_report: dict[str, Any] | None = None
    validation_report: dict[str, Any] | None = None
    test_report: dict[str, Any] | None = None
    train_eval_report: dict[str, Any] | None = None
    if use_explicit_splits and not args.skip_release_gates:
        os.environ["HF_MODEL_ID"] = args.model_id
        os.environ["HF_ADAPTER_PATH"] = str(adapter_dir)

        train_gate_records = _deterministic_take(train_records, args.train_gate_max_records, args.seed)
        validation_gate_records = _deterministic_take(validation_records, args.validation_gate_max_records, args.seed + 11)
        test_gate_records = _deterministic_take(test_records, args.test_gate_max_records, args.seed + 17)

        train_eval_report = _evaluate_records_with_predictor(
            train_gate_records,
            context_mode=args.context_mode,
            context_budget=args.context_budget,
            model_mode="huggingface",
        )
        validation_report = _evaluate_records_with_predictor(
            validation_gate_records,
            context_mode=args.context_mode,
            context_budget=args.context_budget,
            model_mode="huggingface",
        )
        test_report = _evaluate_records_with_predictor(
            test_gate_records,
            context_mode=args.context_mode,
            context_budget=args.context_budget,
            model_mode="huggingface",
        )

        _write_json(Path(args.output_dir) / "train_gate_report.json", train_eval_report)
        _write_json(Path(args.output_dir) / "validation_report.json", validation_report)
        _write_json(Path(args.output_dir) / "test_report.json", test_report)

        release_gate_report = _gate_release(
            train_metrics=train_metrics,
            eval_metrics=eval_metrics,
            train_eval_report=train_eval_report,
            validation_report=validation_report,
            test_report=test_report,
            min_accuracy=args.min_accuracy,
            min_f1=args.min_f1,
            min_confidence=args.min_confidence,
            max_train_val_f1_gap=args.max_train_val_f1_gap,
            max_val_train_loss_ratio=args.max_val_train_loss_ratio,
            max_underfit_loss=args.max_underfit_loss,
        )
        _write_json(Path(args.output_dir) / "gate_report.json", release_gate_report)
        summary["release_gate_report"] = release_gate_report
        summary["train_gate_report_path"] = str(Path(args.output_dir) / "train_gate_report.json")
        summary["validation_report_path"] = str(Path(args.output_dir) / "validation_report.json")
        summary["test_report_path"] = str(Path(args.output_dir) / "test_report.json")
    else:
        summary["release_gate_report"] = {
            "passed": True,
            "checks": [],
            "skipped": True,
            "reason": "split-aware gates disabled",
        }

    release_passed = bool(summary["release_gate_report"].get("passed", False))
    summary["release_ready"] = release_passed

    if args.generate_model_folder and release_passed:
        model_folder = _generate_model_folder(args, summary, adapter_dir=adapter_dir)
        summary["model_folder"] = str(model_folder)
    elif args.generate_model_folder:
        summary["model_folder"] = None
        summary["model_folder_blocked"] = True

    if args.auto_push_to_hub and summary.get("model_folder"):
        push_report = _push_model_bundle(
            Path(summary["model_folder"]),
            repo_id=args.hf_repo_id or "JayNightmare/FakeNews",
            private=args.hf_private,
        )
        summary["hub_publish"] = {
            "attempted": True,
            "success": True,
            **push_report,
        }
        _write_json(Path(args.output_dir) / "hub_publish_report.json", summary["hub_publish"])
    elif args.auto_push_to_hub:
        summary["hub_publish"] = {
            "attempted": False,
            "success": False,
            "reason": "model folder generation blocked because release gates failed",
        }

    _write_json(Path(args.output_dir) / "training_summary.json", summary)
    return summary


def _generate_model_folder(
    args: argparse.Namespace,
    summary: dict[str, Any],
    *,
    adapter_dir: Path | None,
) -> Path:
    model_folder = Path(args.model_folder)
    model_folder.mkdir(parents=True, exist_ok=True)

    if adapter_dir is not None and adapter_dir.exists():
        shutil.copytree(adapter_dir, model_folder / "adapter", dirs_exist_ok=True)

    summary_path = Path(args.output_dir) / "training_summary.json"
    if summary_path.exists():
        shutil.copy2(summary_path, model_folder / "training_summary.json")

    _write_json(
        model_folder / "bundle_manifest.json",
        {
            "project": "FakeNews",
            "prepared_only": bool(summary.get("prepared_only", False)),
            "model_id": args.model_id,
            "source_output_dir": str(Path(args.output_dir)),
            "adapter_available": bool(adapter_dir is not None and adapter_dir.exists()),
            "suggested_hf_repo_id": args.hf_repo_id or "your-username/FakeNews-Qwen-Adapter",
        },
    )

    _write_text(
        model_folder / "README.md",
        """# FakeNews Model Bundle

This folder was generated by `src/train_hf_adapter.py` and contains artifacts
needed to publish the fine-tuned FakeNews adapter to Hugging Face.

## Contents

- `adapter/` : LoRA adapter files (generated after training)
- `training_summary.json` : training metrics and run summary
- `bundle_manifest.json` : publish metadata
- `push_to_hub.py` : helper script for uploading this folder

## Publish to Hugging Face

1. `pip install -U huggingface_hub`
2. `huggingface-cli login`
3. `python push_to_hub.py --repo-id <your-username/your-model-repo>`

If `adapter/` is not present, run training without `--prepare-only` first.
""",
    )

    default_repo = args.hf_repo_id or "your-username/FakeNews-Qwen-Adapter"
    _write_text(
        model_folder / "push_to_hub.py",
        f"""#!/usr/bin/env python3
\"\"\"Upload the generated FakeNews model bundle to Hugging Face Hub.\"\"\"

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi


def main() -> int:
    parser = argparse.ArgumentParser(description=\"Upload FakeNews model bundle\")
    parser.add_argument(\"--repo-id\", default=\"{default_repo}\", help=\"Hub repo id, e.g. username/repo\")
    parser.add_argument(\"--folder\", default=str(Path(__file__).resolve().parent), help=\"Bundle folder to upload\")
    parser.add_argument(\"--private\", action=\"store_true\", help=\"Create the repo as private\")
    args = parser.parse_args()

    api = HfApi()
    api.create_repo(repo_id=args.repo_id, private=args.private, exist_ok=True)
    api.upload_folder(
        folder_path=args.folder,
        repo_id=args.repo_id,
        repo_type=\"model\",
        commit_message=\"Upload FakeNews model bundle\",
    )
    print(f\"Uploaded model bundle to https://huggingface.co/{{args.repo_id}}\")
    return 0


if __name__ == \"__main__\":
    raise SystemExit(main())
""",
    )

    _write_text(
        model_folder / "inference_example.py",
        f"""#!/usr/bin/env python3
\"\"\"Load the base model + FakeNews adapter for local inference.\"\"\"

from __future__ import annotations

from peft import PeftModel
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoTokenizer

BASE_MODEL_ID = \"{args.model_id}\"
ADAPTER_PATH = \"adapter\"


def main() -> int:
    config = AutoConfig.from_pretrained(BASE_MODEL_ID)
    config_name = type(config).__name__.casefold()
    is_vl = any(token in config_name for token in (\"vision\", \"vl\", \"multi\"))

    if is_vl:
        model_loader = getattr(transformers, \"AutoModelForImageTextToText\", None)
        if model_loader is None:
            model_loader = getattr(transformers, \"AutoModelForVision2Seq\", None)
        if model_loader is None:
            raise RuntimeError(\"This transformers version does not support VL auto loaders.\")
        processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)
        tokenizer = processor.tokenizer
        model = model_loader.from_pretrained(BASE_MODEL_ID)
    else:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID)

    model = PeftModel.from_pretrained(model, ADAPTER_PATH)

    prompt = \"Classify this claim as real or fake and explain: The moon is made of cheese.\"
    inputs = tokenizer(prompt, return_tensors=\"pt\")
    outputs = model.generate(**inputs, max_new_tokens=128)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    return 0


if __name__ == \"__main__\":
    raise SystemExit(main())
""",
    )

    return model_folder


class _TokenizedDataset(TorchDataset[dict[str, list[int]]]):
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


def _resolve_training_device(torch_module: Any, requested: str | None, model: Any | None = None) -> Any:
    requested_name = (requested or "auto").casefold()
    if requested_name in {"auto", "cuda"} and torch_module.cuda.is_available():
        return torch_module.device("cuda")
    if requested_name in {"auto", "mps"} and hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available():
        return torch_module.device("mps")
    if model is not None:
        try:
            return next(model.parameters()).device
        except StopIteration:
            pass
    return torch_module.device("cpu")


def _train_model(
    torch_module: Any,
    model: Any,
    dataset: _TokenizedDataset,
    data_collator: _SupervisedCollator,
    optimizer: Any,
    device: Any,
    *,
    per_device_batch_size: int,
    gradient_accumulation_steps: int,
    num_train_epochs: float,
    logging_steps: int,
) -> dict[str, Any]:
    from torch.utils.data import DataLoader

    model.train()
    dataloader = DataLoader(
        dataset,
        batch_size=per_device_batch_size,
        shuffle=True,
        collate_fn=data_collator,
    )
    optimizer.zero_grad()

    total_loss = 0.0
    optimization_steps = 0
    forward_steps = 0
    epoch_count = max(1, int(num_train_epochs))

    for epoch_index in range(epoch_count):
        for batch_index, batch in enumerate(dataloader, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += float(loss.detach().cpu().item())
            forward_steps += 1

            scaled_loss = loss / gradient_accumulation_steps
            scaled_loss.backward()

            should_step = batch_index % gradient_accumulation_steps == 0 or batch_index == len(dataloader)
            if should_step:
                optimizer.step()
                optimizer.zero_grad()
                optimization_steps += 1

                if logging_steps > 0 and optimization_steps % logging_steps == 0:
                    average_loss = total_loss / max(1, forward_steps)
                    print(
                        f"epoch={epoch_index + 1} step={optimization_steps} avg_train_loss={average_loss:.4f}"
                    )

    average_train_loss = total_loss / max(1, forward_steps)
    return {
        "train_loss": round(average_train_loss, 6),
        "train_forward_steps": forward_steps,
        "train_optimization_steps": optimization_steps,
        "num_train_epochs": epoch_count,
        "train_examples": len(dataset),
    }


def _evaluate_model(
    torch_module: Any,
    model: Any,
    dataset: _TokenizedDataset,
    data_collator: _SupervisedCollator,
    device: Any,
    *,
    per_device_batch_size: int,
) -> dict[str, Any]:
    from torch.utils.data import DataLoader

    if len(dataset) == 0:
        return {"eval_loss": None, "eval_examples": 0, "eval_batches": 0}

    dataloader = DataLoader(
        dataset,
        batch_size=per_device_batch_size,
        shuffle=False,
        collate_fn=data_collator,
    )
    model.eval()
    total_loss = 0.0
    batch_count = 0
    with torch_module.no_grad():
        for batch in dataloader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            total_loss += float(outputs.loss.detach().cpu().item())
            batch_count += 1
    model.train()

    average_eval_loss = total_loss / max(1, batch_count)
    perplexity: float | None
    try:
        perplexity = round(float(torch_module.exp(torch_module.tensor(average_eval_loss)).item()), 6)
    except OverflowError:
        perplexity = None
    return {
        "eval_loss": round(average_eval_loss, 6),
        "eval_examples": len(dataset),
        "eval_batches": batch_count,
        "perplexity": perplexity,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare and fine-tune a Hugging Face adapter")
    source_group = parser.add_mutually_exclusive_group(required=False)
    source_group.add_argument("--dataset", choices=list(DATASET_LOADERS.keys()) + [MASTER_DATASET_NAME], help="Dataset to load and normalize")
    source_group.add_argument("--records-jsonl", help="Path to a normalized_records.jsonl or master_records.jsonl file")
    source_group.add_argument("--train-records-dir", default=None, help="Directory containing cleaned train split artifacts")
    parser.add_argument("--data-dir", help="Optional dataset directory override")
    parser.add_argument("--validation-records-dir", default="src/datasets/data/cleaned/validation", help="Directory containing cleaned validation split artifacts")
    parser.add_argument("--test-records-dir", default="src/datasets/data/cleaned/test", help="Directory containing cleaned test split artifacts")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on source records")
    parser.add_argument("--output-dir", default="artifacts/training/qwen_adapter", help="Training output directory")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-1.5B-Instruct", help="Base Hugging Face model ID")
    parser.add_argument("--context-mode", default="full", choices=["minimal", "full", "misleading"], help="Prompt context mode")
    parser.add_argument("--context-budget", type=float, default=1.0, help="Fraction of context retained in training prompts")
    parser.add_argument("--eval-ratio", type=float, default=0.1, help="Eval split ratio for records without explicit splits")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic partitioning")
    parser.add_argument("--prepare-only", action="store_true", help="Only export the train/eval corpus without training")
    parser.add_argument("--generate-model-folder", action="store_true", help="Generate artifacts/FakeNews_Model publish bundle")
    parser.add_argument("--model-folder", default="artifacts/FakeNews_Model", help="Path for generated publish bundle")
    parser.add_argument("--hf-repo-id", default="JayNightmare/FakeNews", help="Default Hugging Face repo id for generated push helper and auto-push")
    parser.add_argument("--auto-push-to-hub", action="store_true", help="Automatically push generated model folder to Hugging Face when gates pass")
    parser.add_argument("--hf-private", action="store_true", help="Create/push Hugging Face repo as private")
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
    parser.add_argument("--offload-folder", default=None, help="Optional folder for transformer offload weights")
    parser.add_argument("--torch-dtype", default=None, help="Optional torch dtype name, e.g. float16 or bfloat16")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj",
        help="Comma-separated module names to adapt",
    )
    parser.add_argument("--skip-release-gates", action="store_true", help="Skip split-aware validation/test quality gates")
    parser.add_argument("--min-accuracy", type=float, default=0.8, help="Minimum validation/test accuracy required for release")
    parser.add_argument("--min-f1", type=float, default=0.8, help="Minimum validation/test F1 required for release")
    parser.add_argument("--min-confidence", type=float, default=0.7, help="Minimum mean test confidence required for release")
    parser.add_argument("--max-train-val-f1-gap", type=float, default=0.05, help="Maximum allowed absolute train-vs-validation F1 gap")
    parser.add_argument("--max-val-train-loss-ratio", type=float, default=1.25, help="Maximum allowed validation/train loss ratio")
    parser.add_argument("--max-underfit-loss", type=float, default=1.5, help="Maximum allowed validation loss ceiling")
    parser.add_argument("--train-gate-max-records", type=int, default=500, help="Max train records to score for overfit gate")
    parser.add_argument("--validation-gate-max-records", type=int, default=1000, help="Max validation records to score for gate checks")
    parser.add_argument("--test-gate-max-records", type=int, default=1000, help="Max test records to score for gate checks")
    parser.add_argument("--allow-failed-gates", action="store_true", help="Return success exit code even when release gates fail")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    load_project_env()

    if not args.dataset and not args.records_jsonl and not args.train_records_dir:
        args.train_records_dir = "src/datasets/data/cleaned/train"
    if args.train_records_dir and not args.validation_records_dir:
        args.validation_records_dir = "src/datasets/data/cleaned/validation"
    if args.train_records_dir and not args.test_records_dir:
        args.test_records_dir = "src/datasets/data/cleaned/test"

    # Guard against accidental shell expansion to an empty string, e.g. --model-id "$HF_MODEL_ID".
    requested_model_id = (args.model_id or "").strip()
    if not requested_model_id:
        fallback_model_id = (os.getenv("HF_MODEL_ID") or "").strip()
        if fallback_model_id:
            args.model_id = fallback_model_id
            print(f"Using HF_MODEL_ID from environment: {args.model_id}")
        else:
            parser.error(
                "--model-id is empty. Pass a non-empty model id (for example Qwen/Qwen2.5-1.5B-Instruct) "
                "or set HF_MODEL_ID in your environment/.env before running."
            )

    args.context_budget = normalize_context_budget(args.context_budget)
    summary = train_adapter(args)
    print(json.dumps(summary, indent=2))
    if not summary.get("release_ready", True) and not args.allow_failed_gates:
        return 2
    return 0


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip() + "\n", encoding="utf-8")


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