"""Evaluation and metrics module.

Computes classification metrics and comparison reports across
datasets and context variants:
- Accuracy, precision, recall, F1
- Confusion matrix
- Per-dataset and per-context-variant breakdown
- Explanation capture
- Run manifest
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def confusion_matrix(
    ground_truth: list[int],
    predictions: list[int],
) -> dict[str, int]:
    """Compute binary confusion matrix counts.

    Args:
        ground_truth: List of true labels (0=real, 1=fake).
        predictions: List of predicted labels (0=real, 1=fake).

    Returns:
        Dict with tp, tn, fp, fn counts.
    """
    tp = tn = fp = fn = 0
    for gt, pred in zip(ground_truth, predictions):
        if gt == 1 and pred == 1:
            tp += 1
        elif gt == 0 and pred == 0:
            tn += 1
        elif gt == 0 and pred == 1:
            fp += 1
        elif gt == 1 and pred == 0:
            fn += 1
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def compute_metrics(cm: dict[str, int]) -> dict[str, float]:
    """Compute classification metrics from a confusion matrix.

    Args:
        cm: Dict with tp, tn, fp, fn counts.

    Returns:
        Dict with accuracy, precision, recall, f1.
    """
    tp, tn, fp, fn = cm["tp"], cm["tn"], cm["fp"], cm["fn"]
    total = tp + tn + fp + fn

    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "support": total,
    }


def evaluate_predictions(
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate a list of predictions against ground truth.

    Each prediction dict must have:
    - ground_truth_label: int (0 or 1)
    - predicted_label: int (0 or 1)
    - dataset: str
    - context_mode: str

    Returns:
        Evaluation report with overall and per-slice metrics.
    """
    if not predictions:
        return {"error": "no predictions to evaluate"}

    gt_all: list[int] = []
    pred_all: list[int] = []
    by_dataset: dict[str, dict[str, list[int]]] = defaultdict(lambda: {"gt": [], "pred": []})
    by_context: dict[str, dict[str, list[int]]] = defaultdict(lambda: {"gt": [], "pred": []})
    by_combo: dict[str, dict[str, list[int]]] = defaultdict(lambda: {"gt": [], "pred": []})

    confidence_values: list[float] = []
    explanations: list[dict[str, Any]] = []

    for pred in predictions:
        gt = pred.get("ground_truth_label")
        predicted = pred.get("predicted_label")
        if gt is None or predicted is None:
            continue

        gt = int(gt)
        predicted = int(predicted)
        dataset = pred.get("dataset", "unknown")
        context_mode = pred.get("context_mode", "unknown")

        gt_all.append(gt)
        pred_all.append(predicted)
        by_dataset[dataset]["gt"].append(gt)
        by_dataset[dataset]["pred"].append(predicted)
        by_context[context_mode]["gt"].append(gt)
        by_context[context_mode]["pred"].append(predicted)

        combo_key = f"{dataset}_{context_mode}"
        by_combo[combo_key]["gt"].append(gt)
        by_combo[combo_key]["pred"].append(predicted)

        if pred.get("confidence") is not None:
            confidence_values.append(float(pred["confidence"]))

        if pred.get("explanation"):
            explanations.append({
                "id": pred.get("id"),
                "dataset": dataset,
                "context_mode": context_mode,
                "ground_truth": gt,
                "predicted": predicted,
                "correct": gt == predicted,
                "explanation": pred["explanation"],
            })

    overall_cm = confusion_matrix(gt_all, pred_all)
    overall_metrics = compute_metrics(overall_cm)

    dataset_metrics = {}
    for ds_name, data in by_dataset.items():
        cm = confusion_matrix(data["gt"], data["pred"])
        dataset_metrics[ds_name] = {
            "confusion_matrix": cm,
            **compute_metrics(cm),
        }

    context_metrics = {}
    for ctx_name, data in by_context.items():
        cm = confusion_matrix(data["gt"], data["pred"])
        context_metrics[ctx_name] = {
            "confusion_matrix": cm,
            **compute_metrics(cm),
        }

    combo_metrics = {}
    for combo_key, data in by_combo.items():
        cm = confusion_matrix(data["gt"], data["pred"])
        combo_metrics[combo_key] = {
            "confusion_matrix": cm,
            **compute_metrics(cm),
        }

    report: dict[str, Any] = {
        "overall": {
            "confusion_matrix": overall_cm,
            **overall_metrics,
        },
        "by_dataset": dataset_metrics,
        "by_context_mode": context_metrics,
        "by_dataset_context": combo_metrics,
        "prediction_count": len(gt_all),
        "label_distribution": {
            "ground_truth": dict(Counter(gt_all)),
            "predicted": dict(Counter(pred_all)),
        },
    }

    if confidence_values:
        report["confidence_stats"] = {
            "mean": round(sum(confidence_values) / len(confidence_values), 4),
            "min": round(min(confidence_values), 4),
            "max": round(max(confidence_values), 4),
        }

    if explanations:
        report["explanation_count"] = len(explanations)
        report["explanations_sample"] = explanations[:10]

    return report


def build_run_manifest(
    dataset: str,
    context_mode: str,
    model_mode: str,
    limit: int | None,
    output_dir: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a manifest documenting run parameters.

    Args:
        dataset: Dataset name or 'all'.
        context_mode: Context variant used.
        model_mode: Inference mode (heuristic, openai-compatible).
        limit: Record limit.
        output_dir: Output directory path.
        extra: Additional parameters to include.

    Returns:
        Run manifest dictionary.
    """
    import datetime

    manifest: dict[str, Any] = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "dataset": dataset,
        "context_mode": context_mode,
        "model_mode": model_mode,
        "limit": limit,
        "output_dir": output_dir,
    }
    if extra:
        manifest.update(extra)
    return manifest


def write_evaluation_report(path: Path, report: dict[str, Any]) -> None:
    """Write evaluation report as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_evaluation_markdown(path: Path, report: dict[str, Any]) -> None:
    """Write evaluation report as readable markdown."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = ["# Evaluation Report\n"]

    overall = report.get("overall", {})
    lines.append("## Overall Metrics\n")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    for key in ["accuracy", "precision", "recall", "f1", "support"]:
        if key in overall:
            lines.append(f"| {key} | {overall[key]} |")
    lines.append("")

    cm = overall.get("confusion_matrix", {})
    if cm:
        lines.append("## Confusion Matrix\n")
        lines.append("| | Predicted Real | Predicted Fake |")
        lines.append("|---|---|---|")
        lines.append(f"| **Actual Real** | {cm.get('tn', 0)} | {cm.get('fp', 0)} |")
        lines.append(f"| **Actual Fake** | {cm.get('fn', 0)} | {cm.get('tp', 0)} |")
        lines.append("")

    by_dataset = report.get("by_dataset", {})
    if by_dataset:
        lines.append("## By Dataset\n")
        lines.append("| Dataset | Accuracy | Precision | Recall | F1 | Support |")
        lines.append("|---|---|---|---|---|---|")
        for ds_name, metrics in by_dataset.items():
            lines.append(
                f"| {ds_name} | {metrics.get('accuracy', '-')} | "
                f"{metrics.get('precision', '-')} | {metrics.get('recall', '-')} | "
                f"{metrics.get('f1', '-')} | {metrics.get('support', '-')} |"
            )
        lines.append("")

    by_context = report.get("by_context_mode", {})
    if by_context:
        lines.append("## By Context Mode\n")
        lines.append("| Context | Accuracy | Precision | Recall | F1 | Support |")
        lines.append("|---|---|---|---|---|---|")
        for ctx_name, metrics in by_context.items():
            lines.append(
                f"| {ctx_name} | {metrics.get('accuracy', '-')} | "
                f"{metrics.get('precision', '-')} | {metrics.get('recall', '-')} | "
                f"{metrics.get('f1', '-')} | {metrics.get('support', '-')} |"
            )
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
