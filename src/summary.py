"""Dataset summary sheet generator.

Produces the per-dataset summary required by the meeting document:
- Number of samples (before/after cleaning)
- Available text fields
- Available metadata fields
- Label types and distribution
- Missing data statistics

Output as both JSON and markdown.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from src.schema import UnifiedRecord


def generate_summary(
    records: list[UnifiedRecord],
    dataset_name: str,
    pre_cleaning_count: int | None = None,
) -> dict[str, Any]:
    """Generate a summary sheet for a list of unified records.

    Args:
        records: Cleaned unified records.
        dataset_name: Name of the dataset.
        pre_cleaning_count: Number of records before cleaning (if known).

    Returns:
        Summary dictionary.
    """
    if not records:
        return {
            "dataset": dataset_name,
            "sample_count": 0,
            "note": "no records available",
        }

    label_dist = Counter()
    original_label_dist = Counter()
    modality_dist = Counter()
    split_dist = Counter()
    missing: Counter[str] = Counter()
    text_lengths: list[int] = []
    has_context = 0
    has_image_count = 0

    check_fields = [
        "text", "context_text", "original_label", "mapped_label",
        "split", "sample_id",
    ]
    metadata_keys: Counter[str] = Counter()

    for record in records:
        label_dist[record.mapped_label_name] += 1
        original_label_dist[str(record.original_label_name)] += 1
        modality_dist[record.modality] += 1
        split_dist[record.split or "(unassigned)"] += 1
        text_lengths.append(len(record.text))

        if record.context_text:
            has_context += 1
        if record.has_image:
            has_image_count += 1

        rd = record.to_dict()
        for field in check_fields:
            val = rd.get(field)
            if val is None or val == "" or val == []:
                missing[field] += 1

        for key in record.metadata:
            metadata_keys[key] += 1

    n = len(records)
    source_fields_seen = set()
    for record in records:
        source_fields_seen.update(record.source_fields_used)

    summary: dict[str, Any] = {
        "dataset": dataset_name,
        "sample_count": n,
        "pre_cleaning_count": pre_cleaning_count,
        "post_cleaning_count": n,
        "retention_rate": round(n / pre_cleaning_count, 4) if pre_cleaning_count else None,
        "text_fields_used": sorted(source_fields_seen),
        "metadata_fields_available": dict(metadata_keys.most_common()),
        "label_types": {
            "mapped_labels": dict(label_dist.most_common()),
            "original_labels": dict(original_label_dist.most_common(20)),
        },
        "modality_distribution": dict(modality_dist.most_common()),
        "split_distribution": dict(split_dist.most_common()),
        "missing_data": dict(missing.most_common()),
        "text_length_stats": {
            "min": min(text_lengths),
            "max": max(text_lengths),
            "mean": round(sum(text_lengths) / n, 1),
            "median": sorted(text_lengths)[n // 2],
        },
        "records_with_context": has_context,
        "records_with_image": has_image_count,
        "balance_ratio": round(
            min(label_dist.values()) / max(label_dist.values()), 3
        ) if label_dist and max(label_dist.values()) > 0 else None,
    }

    return summary


def generate_aggregate_summary(
    per_dataset: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate summaries across all datasets."""
    total = sum(s.get("sample_count", 0) for s in per_dataset)
    all_labels: Counter[str] = Counter()
    all_modalities: Counter[str] = Counter()

    for s in per_dataset:
        for label, count in s.get("label_types", {}).get("mapped_labels", {}).items():
            all_labels[label] += count
        for mod, count in s.get("modality_distribution", {}).items():
            all_modalities[mod] += count

    return {
        "total_samples": total,
        "datasets": [s.get("dataset") for s in per_dataset],
        "per_dataset_counts": {s["dataset"]: s["sample_count"] for s in per_dataset},
        "aggregate_label_distribution": dict(all_labels.most_common()),
        "aggregate_modality_distribution": dict(all_modalities.most_common()),
    }


def write_summary_json(path: Path, summary: dict[str, Any]) -> None:
    """Write summary as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_summary_markdown(path: Path, summary: dict[str, Any]) -> None:
    """Write summary as a readable markdown document."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    ds = summary.get("dataset", "Unknown")
    lines.append(f"# Dataset Summary: {ds}\n")

    n = summary.get("sample_count", 0)
    pre = summary.get("pre_cleaning_count")
    lines.append(f"**Samples:** {n}")
    if pre is not None:
        lines.append(f"  (before cleaning: {pre}, retention: {summary.get('retention_rate', '?')})")
    lines.append("")

    labels = summary.get("label_types", {}).get("mapped_labels", {})
    if labels:
        lines.append("## Label Distribution (mapped)\n")
        lines.append("| Label | Count |")
        lines.append("|---|---|")
        for label, count in labels.items():
            lines.append(f"| {label} | {count} |")
        lines.append("")

    orig_labels = summary.get("label_types", {}).get("original_labels", {})
    if orig_labels:
        lines.append("## Original Label Distribution\n")
        lines.append("| Label | Count |")
        lines.append("|---|---|")
        for label, count in orig_labels.items():
            lines.append(f"| {label} | {count} |")
        lines.append("")

    text_fields = summary.get("text_fields_used", [])
    if text_fields:
        lines.append(f"## Text Fields Used\n")
        for f in text_fields:
            lines.append(f"- `{f}`")
        lines.append("")

    meta_fields = summary.get("metadata_fields_available", {})
    if meta_fields:
        lines.append("## Metadata Fields\n")
        lines.append("| Field | Records with field |")
        lines.append("|---|---|")
        for f, count in meta_fields.items():
            lines.append(f"| `{f}` | {count} |")
        lines.append("")

    missing = summary.get("missing_data", {})
    if missing:
        lines.append("## Missing Data\n")
        lines.append("| Field | Missing count |")
        lines.append("|---|---|")
        for f, count in missing.items():
            lines.append(f"| `{f}` | {count} |")
        lines.append("")

    text_stats = summary.get("text_length_stats", {})
    if text_stats:
        lines.append("## Text Length Stats\n")
        for k, v in text_stats.items():
            lines.append(f"- **{k}:** {v}")
        lines.append("")

    lines.append(f"**Records with context:** {summary.get('records_with_context', 0)}")
    lines.append(f"**Records with image:** {summary.get('records_with_image', 0)}")
    balance = summary.get("balance_ratio")
    if balance is not None:
        lines.append(f"**Balance ratio (min/max label):** {balance}")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
