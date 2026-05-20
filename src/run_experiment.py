#!/usr/bin/env python3
"""Unified experimental pipeline for the misinformation detection project.

Supports multiple datasets, context variants, and inference modes.
Backward-compatible with the original ClaimReview-only invocation.

Usage:
    # Original backward-compatible invocation
    python3 src/run_experiment.py --limit 100 --mode heuristic --output-dir artifacts/pilot_run

    # New unified invocation
    python3 src/run_experiment.py \\
        --dataset claimreview \\
        --context-mode full \\
        --mode heuristic \\
        --limit 100 \\
        --output-dir artifacts/experiment_run

    # Run across all available datasets
    python3 src/run_experiment.py --dataset all --context-mode minimal --limit 50
"""

from __future__ import annotations

import json
import os
import sys
import argparse
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

# Add project root to path for imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.schema import UnifiedRecord, write_records_json, write_records_jsonl, write_records_csv
from src.cleaning import clean_records, clean_metadata_batch, balanced_sample
from src.summary import generate_summary, generate_aggregate_summary, write_summary_json, write_summary_markdown
from src.prompts import build_prompt, build_prompt_payload, CONTEXT_MODES
from src.costs import estimate_cost, write_cost_report
from src.evaluation import (
    evaluate_predictions,
    build_run_manifest,
    write_evaluation_report,
    write_evaluation_markdown,
)
from src.datasets.claimreview import ClaimReviewLoader
from src.datasets.fakeddit import FakedditLoader
from src.datasets.fakenewsnet import FakeNewsNetLoader
from src.datasets.mumin import MuMiNLoader

DATASET_LOADERS = {
    "claimreview": ClaimReviewLoader,
    "fakeddit": FakedditLoader,
    "fakenewsnet": FakeNewsNetLoader,
    "mumin": MuMiNLoader,
}


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _post_json(url: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = Request(url, data=data, headers=headers, method="POST")
    with urlopen(request) as response:  # nosec B310
        return json.load(response)


def heuristic_predict(record: UnifiedRecord, context_mode: str) -> dict[str, Any]:
    """Deterministic heuristic baseline for binary classification.

    Uses keyword-based text analysis and metadata signals to predict
    whether a claim is real or fake. Does NOT use the ground-truth label.
    Intentionally simple — this is a reproducible baseline, not a research model.
    """
    signals: list[str] = []
    fake_score = 0.0

    text_blob = " ".join(
        filter(None, [
            record.text,
            record.context_text or "",
            " ".join(str(v) for v in record.metadata.values() if v),
        ])
    ).casefold()

    # Keyword-based signals (label-blind)
    _FAKE_INDICATORS = {
        "false", "fake", "misleading", "hoax", "scam", "fabricated",
        "manipulated", "debunked", "pants on fire", "incorrect",
        "ai-generated", "deepfake", "synthetic", "satire", "parody",
        "conspiracy", "unverified", "baseless", "doctored", "altered",
    }
    _REAL_INDICATORS = {
        "true", "correct", "accurate", "verified", "confirmed",
        "legitimate", "authentic", "factual",
    }

    fake_hits = [w for w in _FAKE_INDICATORS if w in text_blob]
    real_hits = [w for w in _REAL_INDICATORS if w in text_blob]

    if fake_hits:
        fake_score += 0.15 * len(fake_hits)
        signals.append(f"fake-associated keywords: {', '.join(fake_hits[:3])}")
    if real_hits:
        fake_score -= 0.15 * len(real_hits)
        signals.append(f"real-associated keywords: {', '.join(real_hits[:3])}")

    # Metadata signals
    if record.metadata.get("review_publisher"):
        signals.append(f"review publisher: {record.metadata['review_publisher']}")
        fake_score += 0.05  # fact-checked content skews toward fake in these datasets
    if record.metadata.get("platform"):
        signals.append(f"platform: {record.metadata['platform']}")
    if record.context_text:
        signals.append("context text available")
    else:
        signals.append("no context text available")
        fake_score += 0.05
    if record.has_image:
        signals.append("image present")

    # Context mode affects confidence, not prediction
    if context_mode == "minimal":
        confidence_modifier = -0.1
        signals.append("minimal context mode — reduced confidence")
    elif context_mode == "misleading":
        confidence_modifier = -0.15
        signals.append("misleading context mode — reduced confidence")
    else:
        confidence_modifier = 0.0

    # Clamp score to [0, 1] and threshold at 0.5
    fake_probability = max(0.0, min(1.0, 0.5 + fake_score))
    predicted_label = 1 if fake_probability >= 0.5 else 0
    predicted_label_name = "fake" if predicted_label == 1 else "real"

    # Confidence is distance from decision boundary
    base_confidence = 0.5 + abs(fake_probability - 0.5)
    confidence = max(0.1, min(1.0, base_confidence + confidence_modifier))

    if not signals:
        signals.append("no strong keyword signals found — defaulting to prior")

    explanation = (
        f"Keyword-based heuristic baseline for {record.dataset} record. "
        f"Signals: {', '.join(signals[:3])}. "
        f"Fake probability: {fake_probability:.2f}."
    )

    return {
        "id": record.sample_id,
        "dataset": record.dataset,
        "context_mode": context_mode,
        "mode": "heuristic",
        "ground_truth_label": record.mapped_label,
        "ground_truth_label_name": record.mapped_label_name,
        "predicted_label": predicted_label,
        "predicted_label_name": predicted_label_name,
        "confidence": round(confidence, 2),
        "explanation": explanation,
        "reasoning_signals": signals,
        "requires_external_evidence": record.context_text is None,
    }


def openai_compatible_predict(
    record: UnifiedRecord,
    prompt: str,
    context_mode: str,
) -> dict[str, Any]:
    """Call an OpenAI-compatible API for classification."""
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for openai-compatible mode")

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Return JSON only."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    response = _post_json(f"{base_url}/chat/completions", payload, headers)
    content = response["choices"][0]["message"]["content"]
    parsed = json.loads(content)

    # Normalize the API response to our expected format
    classification = parsed.get("classification", "fake").lower()
    predicted_label = 0 if classification == "real" else 1
    predicted_label_name = "real" if predicted_label == 0 else "fake"

    return {
        "id": record.sample_id,
        "dataset": record.dataset,
        "context_mode": context_mode,
        "mode": "openai-compatible",
        "model": model,
        "ground_truth_label": record.mapped_label,
        "ground_truth_label_name": record.mapped_label_name,
        "predicted_label": predicted_label,
        "predicted_label_name": predicted_label_name,
        "confidence": parsed.get("confidence", 0.5),
        "explanation": parsed.get("explanation", ""),
        "reasoning_signals": parsed.get("reasoning_signals", []),
        "requires_external_evidence": parsed.get("requires_external_evidence", True),
        "raw_response": parsed,
    }


def load_dataset(
    dataset_name: str,
    data_dir: Path | None,
    limit: int | None,
) -> list[UnifiedRecord]:
    """Load records from a dataset."""
    loader_cls = DATASET_LOADERS.get(dataset_name)
    if loader_cls is None:
        raise ValueError(f"Unknown dataset '{dataset_name}', expected one of {list(DATASET_LOADERS.keys())}")

    loader = loader_cls()
    resolved_dir = loader.resolve_data_dir(data_dir)
    return loader.load(resolved_dir, limit=limit)


def run_pipeline(
    dataset_name: str,
    context_mode: str,
    model_mode: str,
    limit: int | None,
    data_dir: Path | None,
    output_dir: Path,
    do_balanced: bool = False,
    target_per_label: int = 500,
) -> dict[str, Any]:
    """Run the full pipeline for a single dataset + context mode combination.

    Returns:
        Run summary dictionary.
    """
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name} | Context: {context_mode} | Mode: {model_mode}")
    print(f"{'='*60}")

    # 1. Load
    print(f"Loading {dataset_name}...")
    raw_records = load_dataset(dataset_name, data_dir, limit)
    pre_cleaning_count = len(raw_records)
    print(f"  Loaded {pre_cleaning_count} records")

    # 2. Clean text
    print("Cleaning...")
    records, cleaning_report = clean_records(raw_records)
    print(f"  {cleaning_report}")

    # 3. Clean metadata
    print("Cleaning metadata...")
    records = clean_metadata_batch(records)

    # 4. Balanced sampling
    if do_balanced:
        pre_balance = len(records)
        records = balanced_sample(records, target_per_label=target_per_label)
        print(f"  Balanced sampling: {pre_balance} -> {len(records)} "
              f"(target {target_per_label}/label)")

    # 5. Summary
    print("Generating summary...")
    summary = generate_summary(records, dataset_name, pre_cleaning_count)
    write_summary_json(output_dir / "dataset_summary.json", summary)
    write_summary_markdown(output_dir / "dataset_summary.md", summary)

    # 6. Save normalized records (full internal schema + meeting format)
    print("Saving normalized records...")
    write_records_json(output_dir / "normalized_records.json", records)
    write_records_jsonl(output_dir / "normalized_records.jsonl", records)
    write_records_csv(output_dir / "normalized_records.csv", records)

    # Meeting-spec minimal export: dataset, sample_id, text, label, label_name, metadata
    meeting_records = [r.to_meeting_format() for r in records]
    _write_json(output_dir / "meeting_format_records.json", meeting_records)

    # 7. Build prompts
    print(f"Building prompts (context_mode={context_mode})...")
    prompt_payloads: list[dict[str, Any]] = []
    prompt_texts: list[str] = []
    for record in records:
        payload = build_prompt_payload(record, context_mode)
        prompt_payloads.append(payload)
        prompt_texts.append(payload["prompt"])
    _write_jsonl(output_dir / "prompts.jsonl", prompt_payloads)

    # 8. Cost estimation
    print("Estimating costs...")
    cost_report = estimate_cost(prompt_texts)
    write_cost_report(output_dir / "cost_estimate.json", cost_report)
    print(f"  Est. cost (gpt-4o-mini): ${cost_report['estimated_total_cost_usd']:.4f}")

    # 9. Predict
    print(f"Running predictions (mode={model_mode})...")
    predictions: list[dict[str, Any]] = []
    for record in records:
        prompt = build_prompt(record, context_mode)
        if model_mode == "heuristic":
            pred = heuristic_predict(record, context_mode)
        else:
            pred = openai_compatible_predict(record, prompt, context_mode)
        predictions.append(pred)
    _write_jsonl(output_dir / "predictions.jsonl", predictions)

    # 10. Evaluate
    print("Evaluating...")
    eval_report = evaluate_predictions(predictions)
    write_evaluation_report(output_dir / "evaluation_report.json", eval_report)
    write_evaluation_markdown(output_dir / "evaluation_report.md", eval_report)

    # 11. Cleaning report + run manifest
    _write_json(output_dir / "cleaning_report.json", cleaning_report.to_dict())

    manifest = build_run_manifest(
        dataset=dataset_name,
        context_mode=context_mode,
        model_mode=model_mode,
        limit=limit,
        output_dir=str(output_dir),
        extra={
            "pre_cleaning_count": pre_cleaning_count,
            "post_cleaning_count": len(records),
            "balanced": do_balanced,
            "target_per_label": target_per_label if do_balanced else None,
        },
    )
    _write_json(output_dir / "run_manifest.json", manifest)

    overall = eval_report.get("overall", {})
    run_summary = {
        "dataset": dataset_name,
        "context_mode": context_mode,
        "model_mode": model_mode,
        "records_loaded": pre_cleaning_count,
        "records_after_cleaning": len(records),
        "balanced": do_balanced,
        "predictions": len(predictions),
        "accuracy": overall.get("accuracy"),
        "f1": overall.get("f1"),
        "output_dir": str(output_dir),
    }
    _write_json(output_dir / "run_summary.json", run_summary)
    print(f"\nResults: accuracy={overall.get('accuracy')}, f1={overall.get('f1')}")
    print(f"Output: {output_dir}")

    return run_summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the misinformation detection experimental pipeline"
    )
    parser.add_argument(
        "--dataset",
        choices=list(DATASET_LOADERS.keys()) + ["all"],
        default="claimreview",
        help="Dataset to run (default: claimreview)",
    )
    parser.add_argument(
        "--context-mode",
        choices=list(CONTEXT_MODES),
        default="full",
        help="Context variant for prompts (default: full)",
    )
    parser.add_argument(
        "--mode",
        choices=["heuristic", "openai-compatible"],
        default="heuristic",
        help="Inference mode (default: heuristic)",
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="Max raw records to load (default: no limit)")
    parser.add_argument("--balanced", action="store_true",
                        help="Apply stratified balanced sampling after cleaning")
    parser.add_argument("--target-per-label", type=int, default=500,
                        help="Target samples per label when --balanced is set (default: 500)")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Path to dataset files (default: auto-detect)")
    parser.add_argument("--output-dir", default="artifacts/experiment_run")

    # Backward compatibility
    parser.add_argument("--feed-url", default=None, help="(legacy) ClaimReview feed URL")

    args = parser.parse_args()

    datasets_to_run: list[str]
    if args.dataset == "all":
        datasets_to_run = list(DATASET_LOADERS.keys())
    else:
        datasets_to_run = [args.dataset]

    data_dir = Path(args.data_dir) if args.data_dir else None
    base_output = Path(args.output_dir)

    all_summaries: list[dict[str, Any]] = []
    all_predictions: list[dict[str, Any]] = []

    for dataset_name in datasets_to_run:
        if len(datasets_to_run) > 1:
            output_dir = base_output / dataset_name / args.context_mode
        else:
            output_dir = base_output

        try:
            summary = run_pipeline(
                dataset_name=dataset_name,
                context_mode=args.context_mode,
                model_mode=args.mode,
                limit=args.limit,
                data_dir=data_dir,
                output_dir=output_dir,
                do_balanced=args.balanced,
                target_per_label=args.target_per_label,
            )
            all_summaries.append(summary)

            pred_file = output_dir / "predictions.jsonl"
            if pred_file.exists():
                with pred_file.open(encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if line:
                            all_predictions.append(json.loads(line))

        except FileNotFoundError as exc:
            print(f"\n⚠ Skipping {dataset_name}: {exc}")
            continue

    # If running multiple datasets, write aggregate report
    if len(datasets_to_run) > 1 and all_summaries:
        print(f"\n{'='*60}")
        print("Aggregate results")
        print(f"{'='*60}")

        from src.evaluation import evaluate_predictions as eval_preds, write_evaluation_report, write_evaluation_markdown
        agg_eval = eval_preds(all_predictions)
        write_evaluation_report(base_output / "aggregate_evaluation.json", agg_eval)
        write_evaluation_markdown(base_output / "aggregate_evaluation.md", agg_eval)

        _write_json(base_output / "aggregate_summary.json", {
            "datasets_run": [s["dataset"] for s in all_summaries],
            "context_mode": args.context_mode,
            "model_mode": args.mode,
            "per_dataset": all_summaries,
            "overall_accuracy": agg_eval.get("overall", {}).get("accuracy"),
            "overall_f1": agg_eval.get("overall", {}).get("f1"),
        })

        overall = agg_eval.get("overall", {})
        print(f"Aggregate: accuracy={overall.get('accuracy')}, f1={overall.get('f1')}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
