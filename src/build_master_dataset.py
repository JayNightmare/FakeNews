#!/usr/bin/env python3
"""Build a merged master dataset from all supported local dataset mirrors.

The builder reuses the existing dataset loaders and cleaning pipeline,
then exports two merged views:

- a canonical master dataset with provenance-rich fields
- a flat six-column dataset for downstream analysis

The current implementation prefers the fullest local mirror available for
each dataset directory and records whether that local source looks like a
verified full mirror, an unverified local mirror, or an obvious sample
fallback.
"""

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
from src.datasets.base import DATASETS_ROOT, DatasetLoader
from src.datasets.claimreview import ClaimReviewLoader
from src.datasets.fakeddit import FakedditLoader
from src.datasets.fakenewsnet import FakeNewsNetLoader
from src.datasets.mumin import MuMiNLoader
from src.schema import (
    MasterRecord,
    UnifiedRecord,
    write_master_flat_csv,
    write_master_records_csv,
    write_master_records_json,
    write_master_records_jsonl,
)

DATASET_LOADERS: dict[str, type[DatasetLoader]] = {
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


def _resolve_dataset_dir(loader: DatasetLoader, data_root: Path | None) -> Path:
    if data_root is None:
        return loader.resolve_data_dir(None)
    return data_root / loader.default_data_dir


def _iter_local_files(data_dir: Path) -> list[Path]:
    if not data_dir.exists():
        return []
    return sorted(path for path in data_dir.iterdir() if path.is_file())


def _coverage_stats(values: list[Any]) -> dict[str, Any]:
    total = len(values)
    non_null = 0
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        non_null += 1
    null_count = total - non_null
    return {
        "total": total,
        "non_null": non_null,
        "null": null_count,
        "null_rate": round(null_count / total, 4) if total else 0,
    }


def _fingerprint(record: MasterRecord) -> str:
    title = (record.title or "").strip().casefold()
    primary_text = (record.primary_text or "").strip().casefold()
    source_name = (record.source_name or "").strip().casefold()
    return "|".join([title, primary_text, source_name])


def _duplicate_summary(records: list[MasterRecord]) -> dict[str, Any]:
    seen_ids: set[str] = set()
    duplicate_ids: list[str] = []
    fingerprint_counts: dict[str, int] = {}

    for record in records:
        if record.id in seen_ids:
            duplicate_ids.append(record.id)
        seen_ids.add(record.id)

        fingerprint = _fingerprint(record)
        if fingerprint.strip("|"):
            fingerprint_counts[fingerprint] = fingerprint_counts.get(fingerprint, 0) + 1

    return {
        "duplicate_id_count": len(duplicate_ids),
        "duplicate_ids": duplicate_ids[:50],
        "duplicate_fingerprint_count": sum(1 for count in fingerprint_counts.values() if count > 1),
    }


def _infer_acquisition_status(dataset_name: str, data_dir: Path, raw_count: int) -> dict[str, Any]:
    files = [path.name for path in _iter_local_files(data_dir)]
    lower_names = [name.casefold() for name in files]
    fallback_used = False
    status = "local_existing"
    notes: list[str] = []

    if dataset_name == "fakeddit":
        if any("sample" in name for name in lower_names):
            fallback_used = True
            status = "sample_fallback"
            notes.append("sample-named local mirror detected")
        elif any(
            name in {"train.tsv", "test.tsv", "validate.tsv", "all_train.tsv", "all_test_public.tsv", "all_validate.tsv"}
            for name in lower_names
        ):
            status = "full_or_split_local_mirror"
    elif dataset_name in {"fakenewsnet", "mumin"}:
        if any(name.endswith(".csv") for name in lower_names) and raw_count <= 5000:
            fallback_used = True
            status = "csv_export_fallback"
            notes.append("single CSV export is consistent with the current sampled/local mirror path")
        elif any(name.endswith(".csv") for name in lower_names):
            status = "csv_export_local_mirror"
    elif dataset_name == "claimreview" and "data.json" in lower_names:
        status = "claimreview_feed_local_json"

    return {
        "attempted_full_corpus": True,
        "selected_input_status": status,
        "fallback_used": fallback_used,
        "data_dir": str(data_dir),
        "files": files,
        "notes": notes,
    }


def _load_and_convert_records(
    dataset_name: str,
    data_root: Path | None,
    limit: int | None,
) -> tuple[list[UnifiedRecord], dict[str, Any]]:
    loader_cls = DATASET_LOADERS[dataset_name]
    loader = loader_cls()
    resolved_dir = _resolve_dataset_dir(loader, data_root)

    raw_items = loader.load_raw(resolved_dir)
    raw_rows_available = len(raw_items)
    if limit is not None:
        raw_items = raw_items[:limit]

    converted: list[UnifiedRecord] = []
    excluded_count = 0
    for raw_item in raw_items:
        record = loader.to_unified(raw_item)
        if record is None:
            excluded_count += 1
            continue
        converted.append(record)

    manifest = {
        "dataset": dataset_name,
        "loader": loader.__class__.__name__,
        "raw_rows_available": raw_rows_available,
        "raw_rows_used": len(raw_items),
        "excluded_during_mapping": excluded_count,
        "converted_rows": len(converted),
        "acquisition": _infer_acquisition_status(dataset_name, resolved_dir, raw_rows_available),
    }
    return converted, manifest


def _build_dataset_slice(
    dataset_name: str,
    data_root: Path | None,
    limit: int | None,
) -> tuple[list[MasterRecord], list[dict[str, Any]], dict[str, Any]]:
    converted, manifest = _load_and_convert_records(dataset_name, data_root, limit)

    cleaned, cleaning_report = clean_records(converted)
    cleaned = clean_metadata_batch(cleaned)
    master_records = [record.to_master_record() for record in cleaned]
    flat_rows = [record.to_master_flat_format() for record in cleaned]

    manifest.update(
        {
            "rows_after_cleaning": len(cleaned),
            "cleaning_report": cleaning_report.to_dict(),
            "coverage": {
                "Title": _coverage_stats([row["Title"] for row in flat_rows]),
                "Claim": _coverage_stats([row["Claim"] for row in flat_rows]),
                "news source": _coverage_stats([row["news source"] for row in flat_rows]),
                "tweet_num": _coverage_stats([row["tweet_num"] for row in flat_rows]),
            },
        }
    )
    return master_records, flat_rows, manifest


def build_master_dataset(
    datasets: list[str],
    data_root: Path | None,
    limit: int | None,
    output_dir: Path,
) -> dict[str, Any]:
    all_master_records: list[MasterRecord] = []
    all_flat_rows: list[dict[str, Any]] = []
    per_dataset: list[dict[str, Any]] = []

    for dataset_name in datasets:
        print(f"Loading {dataset_name}...")
        master_records, flat_rows, manifest = _build_dataset_slice(dataset_name, data_root, limit)
        all_master_records.extend(master_records)
        all_flat_rows.extend(flat_rows)
        per_dataset.append(manifest)
        print(
            f"  raw={manifest['raw_rows_used']} converted={manifest['converted_rows']} "
            f"cleaned={manifest['rows_after_cleaning']}"
        )

    manifest = {
        "datasets": datasets,
        "data_root": str(data_root or DATASETS_ROOT),
        "total_rows": len(all_master_records),
        "per_dataset": per_dataset,
        "duplicates": _duplicate_summary(all_master_records),
    }
    coverage = {
        "total_rows": len(all_flat_rows),
        "aggregate": {
            "Title": _coverage_stats([row["Title"] for row in all_flat_rows]),
            "Claim": _coverage_stats([row["Claim"] for row in all_flat_rows]),
            "news source": _coverage_stats([row["news source"] for row in all_flat_rows]),
            "tweet_num": _coverage_stats([row["tweet_num"] for row in all_flat_rows]),
        },
        "per_dataset": [
            {
                "dataset": item["dataset"],
                "acquisition": item["acquisition"],
                "coverage": item["coverage"],
            }
            for item in per_dataset
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_master_records_json(output_dir / "master_records.json", all_master_records)
    write_master_records_jsonl(output_dir / "master_records.jsonl", all_master_records)
    write_master_records_csv(output_dir / "master_records.csv", all_master_records)
    _write_json(output_dir / "master_flat_records.json", all_flat_rows)
    _write_jsonl(output_dir / "master_flat_records.jsonl", all_flat_rows)
    write_master_flat_csv(output_dir / "master_flat_records.csv", all_flat_rows)
    _write_json(output_dir / "master_dataset_manifest.json", manifest)
    _write_json(output_dir / "master_dataset_coverage.json", coverage)

    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a canonical master dataset from all supported sources")
    parser.add_argument(
        "--dataset",
        choices=list(DATASET_LOADERS.keys()) + ["all"],
        default="all",
        help="Dataset to include (default: all)",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Optional root directory containing per-dataset subdirectories",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on raw rows per dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/master_dataset",
        help="Directory for canonical and flat master exports",
    )
    args = parser.parse_args()

    datasets = list(DATASET_LOADERS.keys()) if args.dataset == "all" else [args.dataset]
    data_root = Path(args.data_root) if args.data_root else None
    output_dir = Path(args.output_dir)

    manifest = build_master_dataset(datasets, data_root, args.limit, output_dir)
    print(f"Wrote {manifest['total_rows']} master rows to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())