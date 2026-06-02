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
    write_records_csv,
    write_records_json,
    write_records_jsonl,
)

DATASET_LOADERS: dict[str, type[DatasetLoader]] = {
    "claimreview": ClaimReviewLoader,
    "fakeddit": FakedditLoader,
    "fakenewsnet": FakeNewsNetLoader,
    "mumin": MuMiNLoader,
}

DATASET_PRIORITY: dict[str, int] = {
    "claimreview": 0,
    "fakenewsnet": 1,
    "fakeddit": 2,
    "mumin": 3,
}

_COVERAGE_COLUMNS = ["Title", "Claim", "news source", "Real", "tweet_num"]


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


def _coverage_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        column: _coverage_stats([row.get(column) for row in rows])
        for column in _COVERAGE_COLUMNS
    }


def _fingerprint(record: MasterRecord) -> str:
    title = (record.title or "").strip().casefold()
    primary_text = (record.primary_text or "").strip().casefold()
    source_name = (record.source_name or "").strip().casefold()
    return "|".join([title, primary_text, source_name])


def _dataset_priority(record: MasterRecord) -> int:
    return DATASET_PRIORITY.get(record.dataset.casefold(), len(DATASET_PRIORITY))


def _deduplicate_master_records(
    records: list[MasterRecord],
) -> tuple[list[MasterRecord], dict[str, Any], list[dict[str, Any]]]:
    deduplication_notes: list[dict[str, Any]] = []
    seen_ids: dict[str, MasterRecord] = {}
    duplicate_ids: list[str] = []
    id_deduped_records: list[MasterRecord] = []

    for record in records:
        kept_record = seen_ids.get(record.id)
        if kept_record is not None:
            duplicate_ids.append(record.id)
            deduplication_notes.append(
                {
                    "reason": "duplicate_id",
                    "removed_record_id": record.id,
                    "removed_source_sample_id": record.source_sample_id,
                    "removed_dataset": record.dataset,
                    "kept_record_id": kept_record.id,
                    "kept_source_sample_id": kept_record.source_sample_id,
                    "kept_dataset": kept_record.dataset,
                }
            )
            continue
        seen_ids[record.id] = record
        id_deduped_records.append(record)

    fingerprint_groups: dict[str, list[tuple[int, MasterRecord]]] = {}
    for index, record in enumerate(id_deduped_records):
        fingerprint = _fingerprint(record)
        if fingerprint.strip("|"):
            fingerprint_groups.setdefault(fingerprint, []).append((index, record))

    keep_indexes = set(range(len(id_deduped_records)))
    duplicate_fingerprint_group_count = 0
    duplicate_fingerprint_count = 0

    for fingerprint, grouped_records in fingerprint_groups.items():
        if len(grouped_records) == 1:
            continue

        duplicate_fingerprint_group_count += 1
        kept_index, kept_record = min(
            grouped_records,
            key=lambda item: (_dataset_priority(item[1]), item[0]),
        )

        for record_index, record in grouped_records:
            if record_index == kept_index:
                continue
            keep_indexes.discard(record_index)
            duplicate_fingerprint_count += 1
            deduplication_notes.append(
                {
                    "reason": "duplicate_fingerprint",
                    "fingerprint": fingerprint,
                    "removed_record_id": record.id,
                    "removed_source_sample_id": record.source_sample_id,
                    "removed_dataset": record.dataset,
                    "kept_record_id": kept_record.id,
                    "kept_source_sample_id": kept_record.source_sample_id,
                    "kept_dataset": kept_record.dataset,
                }
            )

    deduped_records = [
        record for index, record in enumerate(id_deduped_records)
        if index in keep_indexes
    ]
    duplicate_id_count = len(duplicate_ids)
    duplicates_detected = duplicate_id_count + duplicate_fingerprint_count

    return (
        deduped_records,
        {
            "duplicate_id_count": duplicate_id_count,
            "duplicate_ids": duplicate_ids[:50],
            "duplicate_fingerprint_group_count": duplicate_fingerprint_group_count,
            "duplicate_fingerprint_count": duplicate_fingerprint_count,
            "duplicates_detected": duplicates_detected,
            "duplicates_removed": duplicates_detected,
        },
        deduplication_notes,
    )


def _deduplicate_record_pairs(
    record_pairs: list[tuple[UnifiedRecord, MasterRecord]],
) -> tuple[list[UnifiedRecord], list[MasterRecord], dict[str, Any], list[dict[str, Any]]]:
    deduplication_notes: list[dict[str, Any]] = []
    seen_ids: dict[str, tuple[UnifiedRecord, MasterRecord]] = {}
    duplicate_ids: list[str] = []
    id_deduped_pairs: list[tuple[UnifiedRecord, MasterRecord]] = []

    for unified_record, master_record in record_pairs:
        kept_pair = seen_ids.get(master_record.id)
        if kept_pair is not None:
            _, kept_master_record = kept_pair
            duplicate_ids.append(master_record.id)
            deduplication_notes.append(
                {
                    "reason": "duplicate_id",
                    "removed_record_id": master_record.id,
                    "removed_source_sample_id": master_record.source_sample_id,
                    "removed_dataset": master_record.dataset,
                    "kept_record_id": kept_master_record.id,
                    "kept_source_sample_id": kept_master_record.source_sample_id,
                    "kept_dataset": kept_master_record.dataset,
                }
            )
            continue
        seen_ids[master_record.id] = (unified_record, master_record)
        id_deduped_pairs.append((unified_record, master_record))

    fingerprint_groups: dict[str, list[tuple[int, tuple[UnifiedRecord, MasterRecord]]]] = {}
    for index, pair in enumerate(id_deduped_pairs):
        _, master_record = pair
        fingerprint = _fingerprint(master_record)
        if fingerprint.strip("|"):
            fingerprint_groups.setdefault(fingerprint, []).append((index, pair))

    keep_indexes = set(range(len(id_deduped_pairs)))
    duplicate_fingerprint_group_count = 0
    duplicate_fingerprint_count = 0

    for fingerprint, grouped_pairs in fingerprint_groups.items():
        if len(grouped_pairs) == 1:
            continue

        duplicate_fingerprint_group_count += 1
        kept_index, kept_pair = min(
            grouped_pairs,
            key=lambda item: (_dataset_priority(item[1][1]), item[0]),
        )
        _, kept_master_record = kept_pair

        for pair_index, (_, master_record) in grouped_pairs:
            if pair_index == kept_index:
                continue
            keep_indexes.discard(pair_index)
            duplicate_fingerprint_count += 1
            deduplication_notes.append(
                {
                    "reason": "duplicate_fingerprint",
                    "fingerprint": fingerprint,
                    "removed_record_id": master_record.id,
                    "removed_source_sample_id": master_record.source_sample_id,
                    "removed_dataset": master_record.dataset,
                    "kept_record_id": kept_master_record.id,
                    "kept_source_sample_id": kept_master_record.source_sample_id,
                    "kept_dataset": kept_master_record.dataset,
                }
            )

    deduped_pairs = [
        pair for index, pair in enumerate(id_deduped_pairs)
        if index in keep_indexes
    ]
    duplicate_id_count = len(duplicate_ids)
    duplicates_detected = duplicate_id_count + duplicate_fingerprint_count

    return (
        [unified_record for unified_record, _ in deduped_pairs],
        [master_record for _, master_record in deduped_pairs],
        {
            "duplicate_id_count": duplicate_id_count,
            "duplicate_ids": duplicate_ids[:50],
            "duplicate_fingerprint_group_count": duplicate_fingerprint_group_count,
            "duplicate_fingerprint_count": duplicate_fingerprint_count,
            "duplicates_detected": duplicates_detected,
            "duplicates_removed": duplicates_detected,
        },
        deduplication_notes,
    )


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
) -> tuple[list[UnifiedRecord], list[MasterRecord], list[dict[str, Any]], dict[str, Any]]:
    converted, manifest = _load_and_convert_records(dataset_name, data_root, limit)

    cleaned, cleaning_report = clean_records(converted)
    cleaned = clean_metadata_batch(cleaned)
    master_records = [record.to_master_record() for record in cleaned]
    flat_rows = [record.to_master_flat_format() for record in cleaned]

    manifest.update(
        {
            "rows_after_cleaning": len(cleaned),
            "cleaning_report": cleaning_report.to_dict(),
            "coverage": _coverage_summary(flat_rows),
        }
    )
    return cleaned, master_records, flat_rows, manifest


def build_master_dataset(
    datasets: list[str],
    data_root: Path | None,
    limit: int | None,
    output_dir: Path,
) -> dict[str, Any]:
    all_unified_records: list[UnifiedRecord] = []
    all_master_records: list[MasterRecord] = []
    per_dataset: list[dict[str, Any]] = []

    for dataset_name in datasets:
        print(f"Loading {dataset_name}...")
        unified_records, master_records, flat_rows, manifest = _build_dataset_slice(dataset_name, data_root, limit)
        all_unified_records.extend(unified_records)
        all_master_records.extend(master_records)
        per_dataset.append(manifest)
        print(
            f"  raw={manifest['raw_rows_used']} converted={manifest['converted_rows']} "
            f"cleaned={manifest['rows_after_cleaning']}"
        )

    deduped_unified_records, deduped_master_records, duplicate_summary, deduplication_notes = _deduplicate_record_pairs(
        list(zip(all_unified_records, all_master_records))
    )
    deduped_flat_rows = [record.to_flat_dict() for record in deduped_master_records]

    manifest = {
        "datasets": datasets,
        "data_root": str(data_root or DATASETS_ROOT),
        "total_rows_before_deduplication": len(all_master_records),
        "total_rows": len(deduped_master_records),
        "per_dataset": per_dataset,
        "duplicates": duplicate_summary,
        "deduplication_notes": deduplication_notes,
    }
    coverage = {
        "total_rows": len(deduped_flat_rows),
        "aggregate": _coverage_summary(deduped_flat_rows),
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
    write_records_json(output_dir / "normalized_records.json", deduped_unified_records)
    write_records_jsonl(output_dir / "normalized_records.jsonl", deduped_unified_records)
    write_records_csv(output_dir / "normalized_records.csv", deduped_unified_records)
    write_master_records_json(output_dir / "master_records.json", deduped_master_records)
    write_master_records_jsonl(output_dir / "master_records.jsonl", deduped_master_records)
    write_master_records_csv(output_dir / "master_records.csv", deduped_master_records)
    _write_json(output_dir / "master_flat_records.json", deduped_flat_rows)
    _write_jsonl(output_dir / "master_flat_records.jsonl", deduped_flat_rows)
    write_master_flat_csv(output_dir / "master_flat_records.csv", deduped_flat_rows)
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