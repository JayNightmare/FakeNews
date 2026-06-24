#!/usr/bin/env python3
"""Unified experimental pipeline for the misinformation detection project.

Usage:
    # Run master dataset
    python3 src/run_experiment.py --dataset master_dataset --context-mode minimal --limit 1000 --output-dir artifacts/master_run
    python3 src/run_experiment.py --dataset master_dataset --context-mode misleading --limit 1000 --output-dir artifacts/master_run
    python3 src/run_experiment.py --dataset master_dataset --context-mode full --limit 1000 --output-dir artifacts/master_run
"""

from __future__ import annotations

import json
import os
import sys
import random
import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

# Add project root to path for imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.schema import UnifiedRecord, load_records_jsonl, write_records_csv, write_records_json, write_records_jsonl
from src.cleaning import clean_records, clean_metadata_batch, balanced_sample
from src.context_ablation import normalize_context_budget, parse_context_budget_levels
from src.summary import generate_summary, generate_aggregate_summary, write_summary_json, write_summary_markdown
from src.prompts import build_prompt, build_prompt_payload, CONTEXT_MODES
from src.costs import estimate_cost_all_models, write_cost_report
from src.evaluation import (
    evaluate_predictions,
    build_run_manifest,
    write_evaluation_report,
    write_evaluation_markdown,
)
from src.google_factcheck import enrich_records_with_google_factcheck
from src.predictors import create_predictor
from src.datasets.claimreview import ClaimReviewLoader
from src.datasets.fakeddit import FakedditLoader
from src.datasets.fakenewsnet import FakeNewsNetLoader
from src.datasets.mumin import MuMiNLoader
from src.visualization import generate_aggregate_visualizations, generate_run_visualizations
from src.datasets.split import split_dataset

DATASET_LOADERS = {
    "claimreview": ClaimReviewLoader,
    "fakeddit": FakedditLoader,
    "fakenewsnet": FakeNewsNetLoader,
    "mumin": MuMiNLoader,
}

MASTER_DATASET_NAME = "master_dataset"
MASTER_DATASET_FILENAMES = ("normalized_records.jsonl", "master_records.jsonl")
MASTER_DATASET_SPLIT_NAMES = ("train", "validation", "test")


def _step(label: str, t0: float | None = None) -> float:
    """Print a pipeline step header with elapsed time since last step."""
    ts = time.strftime("%H:%M:%S")
    if t0 is not None:
        elapsed = time.time() - t0
        print(f"[{ts}] {label}  ({elapsed:.1f}s)", flush=True)
    else:
        print(f"[{ts}] {label}", flush=True)
    return time.time()


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _resolve_master_dataset_records_path(data_dir: Path | None) -> Path:
    if data_dir is not None and data_dir.is_file():
        return data_dir

    candidate_dirs: list[Path] = []
    if data_dir is not None:
        candidate_dirs.append(data_dir)
        if data_dir.name != MASTER_DATASET_NAME:
            candidate_dirs.append(data_dir / MASTER_DATASET_NAME)
    else:
        candidate_dirs.extend([
            ROOT / "artifacts" / MASTER_DATASET_NAME,
            ROOT / "src" / "datasets" / "data" / MASTER_DATASET_NAME,
        ])

    for candidate_dir in candidate_dirs:
        for filename in MASTER_DATASET_FILENAMES:
            candidate_path = candidate_dir / filename
            if candidate_path.exists():
                return candidate_path

    searched = ", ".join(str(path) for path in candidate_dirs) or "(none)"
    expected_files = ", ".join(MASTER_DATASET_FILENAMES)
    raise FileNotFoundError(
        f"No master dataset export found. Looked in: {searched}. "
        f"Expected one of: {expected_files}."
    )


def _resolve_split_dataset_input(args: argparse.Namespace) -> tuple[str, Path | None]:
    known_datasets = set(DATASET_LOADERS) | {MASTER_DATASET_NAME}
    if args.dataset in known_datasets:
        return args.dataset, Path(args.data_dir) if args.data_dir else None

    dataset_path = Path(args.dataset)
    dataset_name = MASTER_DATASET_NAME if dataset_path.exists() or dataset_path.suffix or "/" in args.dataset or "\\" in args.dataset else args.dataset
    return dataset_name, dataset_path


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

def load_dataset(
    dataset_name: str,
    data_dir: Path | None,
    limit: int | None,
) -> list[UnifiedRecord]:
    """Load records from a dataset."""
    if dataset_name == MASTER_DATASET_NAME:
        records = load_records_jsonl(_resolve_master_dataset_records_path(data_dir))
        if limit is not None:
            return records[:limit]
        return records

    loader_cls = DATASET_LOADERS.get(dataset_name)
    if loader_cls is None:
        expected = list(DATASET_LOADERS.keys()) + [MASTER_DATASET_NAME]
        raise ValueError(f"Unknown dataset '{dataset_name}', expected one of {expected}")

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
    context_budget: float = 1.0,
    context_ablation_levels: list[float] | None = None,
    ground_with_google: bool = False,
    google_factcheck_cache: Path | None = None,
    google_factcheck_ttl_hours: int = 24,
    max_workers: int = 4,
) -> dict[str, Any]:
    """Run the full pipeline for a single dataset + context mode combination.

    Returns:
        Run summary dictionary.
    """
    t_pipeline = time.time()
    print(f"\n{'='*60}", flush=True)
    print(f"Dataset: {dataset_name} | Context: {context_mode} | Mode: {model_mode}", flush=True)
    print(f"output_dir: {output_dir}", flush=True)
    print(f"limit: {limit}  balanced: {do_balanced}  context_budget: {context_budget}", flush=True)
    print(f"{'='*60}", flush=True)

    # 1. Load and randomize records
    t = _step(f"[1/12] Loading {dataset_name}...")
    raw_records = load_dataset(dataset_name, data_dir, limit=None)
    random.shuffle(raw_records)
    if limit is not None:
        raw_records = raw_records[:limit]
    pre_cleaning_count = len(raw_records)
    t = _step(f"  -> {pre_cleaning_count} records loaded", t)

    # 2. Clean text
    t = _step("[2/12] Cleaning text...")
    records, cleaning_report = clean_records(raw_records)
    t = _step(f"  -> {cleaning_report}", t)

    # 3. Clean metadata
    t = _step("[3/12] Cleaning metadata...")
    records = clean_metadata_batch(records)
    t = _step(f"  -> {len(records)} records after metadata cleaning", t)

    grounding_report: dict[str, Any] = {
        "enabled": False,
        "records_seen": len(records),
    }
    if ground_with_google:
        t = _step("[3b/12] Grounding with Google Fact Check...")
        cache_path = google_factcheck_cache or (output_dir / "google_factcheck_cache.json")
        records, grounding_report = enrich_records_with_google_factcheck(
            records,
            cache_path=cache_path,
            ttl_hours=google_factcheck_ttl_hours,
        )
        t = _step(
            f"  -> cache hits={grounding_report['cache_hits']}, "
            f"live fetches={grounding_report['live_fetches']}, "
            f"matches={grounding_report['matches_found']}",
            t,
        )

    # 4. Balanced sampling
    if do_balanced:
        t = _step("[4/12] Balanced sampling...")
        pre_balance = len(records)
        records = balanced_sample(records, target_per_label=target_per_label)
        t = _step(f"  -> {pre_balance} -> {len(records)} (target {target_per_label}/label)", t)

    # 5. Summary
    t = _step("[5/12] Generating dataset summary...")
    summary = generate_summary(records, dataset_name, pre_cleaning_count)
    write_summary_json(output_dir / "dataset_summary.json", summary)
    write_summary_markdown(output_dir / "dataset_summary.md", summary)
    t = _step("  -> summary written", t)

    # 6. Save normalized records (full internal schema + meeting format)
    t = _step("[6/12] Saving normalized records...")
    write_records_json(output_dir / "normalized_records.json", records)
    write_records_jsonl(output_dir / "normalized_records.jsonl", records)
    write_records_csv(output_dir / "normalized_records.csv", records)
    meeting_records = [r.to_meeting_format() for r in records]
    _write_json(output_dir / "meeting_format_records.json", meeting_records)
    t = _step(f"  -> {len(records)} records written to {output_dir}", t)

    # 7. Build prompts
    t = _step(f"[7/12] Building prompts (context_mode={context_mode})...")
    prompt_payloads: list[dict[str, Any]] = []
    prompt_texts: list[str] = []
    prediction_inputs: list[tuple[UnifiedRecord, str, float]] = []
    budgets = context_ablation_levels or [normalize_context_budget(context_budget)]
    for record in records:
        for budget in budgets:
            payload = build_prompt_payload(record, context_mode, context_budget=budget)
            prompt_payloads.append(payload)
            prompt_texts.append(payload["prompt"])
            prediction_inputs.append((record, payload["prompt"], budget))
    _write_jsonl(output_dir / "prompts.jsonl", prompt_payloads)
    t = _step(f"  -> {len(prompt_payloads)} prompts built (budgets={budgets})", t)

    # 8. Cost estimation
    t = _step("[8/12] Estimating costs...")
    cost_report = estimate_cost_all_models(prompt_texts, avg_output_tokens=150)
    write_cost_report(output_dir / "cost_estimate.json", cost_report)
    t = _step(f"  -> cheapest={cost_report['cheapest']}  most_expensive={cost_report['most_expensive']}", t)

    # 9. Predict
    t = _step(f"[9/12] Running predictions (mode={model_mode}, records={len(prediction_inputs)})...")
    predictions: list[dict[str, Any]] = []
    predictor = create_predictor(model_mode, heuristic_predictor=heuristic_predict)

    # Respect the backend's concurrency capability; HuggingFace serialises
    # internally via a lock, but there's no throughput benefit from threads.
    effective_workers = 1 if not predictor.supports_concurrency else max(1, max_workers)
    total = len(prediction_inputs)
    print(f"  effective_workers={effective_workers}  total_predictions={total}", flush=True)

    def _predict_one(args: tuple[UnifiedRecord, str, float]) -> dict[str, Any]:
        record, prompt, budget = args
        pred = predictor.predict(record, prompt, context_mode)
        pred["context_budget"] = budget
        return pred

    if effective_workers == 1:
        t_first = None
        for i, item in enumerate(prediction_inputs, 1):
            t_item = time.time()
            predictions.append(_predict_one(item))
            elapsed_item = time.time() - t_item
            if t_first is None:
                t_first = time.time()
                eta_s = elapsed_item * total
                print(f"  first prediction: {elapsed_item:.1f}s  estimated total: {eta_s/60:.1f} min", flush=True)
            if i % 10 == 0 or i == total:
                elapsed_total = time.time() - t
                rate = i / elapsed_total if elapsed_total > 0 else 0
                remaining = (total - i) / rate if rate > 0 else float("inf")
                print(
                    f"  [{time.strftime('%H:%M:%S')}] {i}/{total} predictions  "
                    f"{rate:.2f}/s  ETA {remaining/60:.1f} min",
                    flush=True,
                )
    else:
        print(f"  Using {effective_workers} concurrent workers", flush=True)
        futures = {}
        with ThreadPoolExecutor(max_workers=effective_workers) as pool:
            for idx, item in enumerate(prediction_inputs):
                futures[pool.submit(_predict_one, item)] = idx
            completed = 0
            results: dict[int, dict[str, Any]] = {}
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()
                completed += 1
                if completed % 10 == 0 or completed == total:
                    elapsed_total = time.time() - t
                    rate = completed / elapsed_total if elapsed_total > 0 else 0
                    remaining = (total - completed) / rate if rate > 0 else float("inf")
                    print(
                        f"  [{time.strftime('%H:%M:%S')}] {completed}/{total} predictions  "
                        f"{rate:.2f}/s  ETA {remaining/60:.1f} min",
                        flush=True,
                    )
        predictions = [results[i] for i in range(len(prediction_inputs))]

    t = _step(f"  -> {len(predictions)} predictions done", t)
    _write_jsonl(output_dir / "predictions.jsonl", predictions)

    # 10. Evaluate
    t = _step("[10/12] Evaluating predictions...")
    eval_report = evaluate_predictions(predictions)
    write_evaluation_report(output_dir / "evaluation_report.json", eval_report)
    write_evaluation_markdown(output_dir / "evaluation_report.md", eval_report)
    t = _step("  -> evaluation written", t)

    # 11. Visualization artifacts
    t = _step("[11/12] Generating visualizations...")
    generate_run_visualizations(records, summary, eval_report, predictions, output_dir)
    t = _step("  -> visualizations done", t)

    # 12. Cleaning report + run manifest
    t = _step("[12/12] Writing manifests and final artifacts...")
    _write_json(output_dir / "cleaning_report.json", cleaning_report.to_dict())
    _write_json(output_dir / "grounding_report.json", grounding_report)

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
            "context_budget": normalize_context_budget(context_budget),
            "context_ablation_levels": budgets,
            "ground_with_google": ground_with_google,
            "google_factcheck_cache": str(google_factcheck_cache) if google_factcheck_cache else None,
            "google_factcheck_ttl_hours": google_factcheck_ttl_hours,
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
        "context_budget": normalize_context_budget(context_budget),
        "context_ablation_levels": budgets,
        "ground_with_google": ground_with_google,
        "output_dir": str(output_dir),
    }
    _write_json(output_dir / "run_summary.json", run_summary)
    total_elapsed = time.time() - t_pipeline
    print(f"\nPipeline complete in {total_elapsed/60:.1f} min — accuracy={overall.get('accuracy')}, f1={overall.get('f1')}", flush=True)
    print(f"Output: {output_dir}", flush=True)
    return run_summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the misinformation detection experimental pipeline"
    )
    parser.add_argument(
        "--dataset",
        # choices=list(DATASET_LOADERS.keys()) + [MASTER_DATASET_NAME, "all"],
        default=MASTER_DATASET_NAME,
        help="Dataset to run (default: master_dataset)",
    )
    parser.add_argument(
        "--context-mode",
        choices=list(CONTEXT_MODES),
        default="full",
        help="Context variant for prompts (default: full)",
    )
    parser.add_argument(
        "--mode",
        choices=["openai-compatible", "huggingface"],
        default="huggingface",
        help="Inference mode (default: huggingface)",
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="Max raw records to load (default: no limit)")
    parser.add_argument("--balanced", action="store_true",
                        help="Apply stratified balanced sampling after cleaning")
    parser.add_argument("--target-per-label", type=int, default=500,
                        help="Target samples per label when --balanced is set (default: 500)")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Path to dataset files (default: auto-detect)")
    parser.add_argument("--output-dir", "--output", dest="output_dir", default="artifacts/experiment_run")
    parser.add_argument("--context-budget", type=float, default=1.0,
                        help="Fraction of available context to retain in prompts (default: 1.0)")
    parser.add_argument("--context-ablation-levels", type=str, default=None,
                        help="Comma-separated context budgets to evaluate in one run, e.g. 1.0,0.75,0.5,0.25")
    parser.add_argument("--ground-with-google", action="store_true",
                        help="Enrich records with cached Google Fact Check search results")
    parser.add_argument("--google-factcheck-cache", type=str, default=None,
                        help="Path to the Google fact-check cache file (default: output-dir/google_factcheck_cache.json)")
    parser.add_argument("--google-factcheck-ttl-hours", type=int, default=24,
                        help="TTL for Google fact-check cache entries in hours (default: 24)")
    parser.add_argument("--max-workers", type=int, default=4,
                        help="Concurrent workers for API prediction calls (default: 4; ignored for HuggingFace)")
    parser.add_argument("--split-dataset", action="store_true",
                        help="Split the master dataset into train/validation/test sets and exit")
    parser.add_argument("--split-type", choices=["train", "validation", "test"], default=None,
                        help="Specify which split to create (default: all splits)")

    args = parser.parse_args()

    if args.split_dataset:
        split_dataset_name, split_data_path = _resolve_split_dataset_input(args)
        splits_to_create = [args.split_type] if args.split_type else MASTER_DATASET_SPLIT_NAMES
        split_output_root = Path(args.output_dir)
        for split in splits_to_create:
            output_path = split_output_root if args.split_type else split_output_root / split
            print(f"Splitting dataset into {split} set at {output_path}...")
            print(split_dataset(split_dataset_name, split, output_path, args.limit, data_path=split_data_path))
        return 0

    datasets_to_run: list[str]
    if args.dataset == "all":
        datasets_to_run = list(DATASET_LOADERS.keys())
    else:
        datasets_to_run = [args.dataset]

    data_dir = Path(args.data_dir) if args.data_dir else None
    base_output = Path(args.output_dir)
    context_budget = normalize_context_budget(args.context_budget)
    context_ablation_levels = parse_context_budget_levels(args.context_ablation_levels)
    google_factcheck_cache = Path(args.google_factcheck_cache) if args.google_factcheck_cache else None

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
                context_budget=context_budget,
                context_ablation_levels=context_ablation_levels,
                ground_with_google=args.ground_with_google,
                google_factcheck_cache=google_factcheck_cache,
                google_factcheck_ttl_hours=args.google_factcheck_ttl_hours,
                max_workers=args.max_workers,
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
        generate_aggregate_visualizations(all_summaries, agg_eval, base_output)

        overall = agg_eval.get("overall", {})
        print(f"Aggregate: accuracy={overall.get('accuracy')}, f1={overall.get('f1')}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
