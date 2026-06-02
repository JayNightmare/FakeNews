"""Canonical unified record schema for the misinformation detection pipeline.

Richer than the meeting's minimal export format (dataset, sample_id, text,
label, label_name, metadata) to preserve provenance, modality, and cleaning
history across all four data sources.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class UnifiedRecord:
    """Internal canonical representation of a single misinformation sample."""

    dataset: str
    sample_id: str
    text: str

    original_label: str | int
    original_label_name: str
    mapped_label: int  # 0 = real, 1 = fake
    mapped_label_name: str  # "real" | "fake"

    split: str | None = None
    source_fields_used: list[str] = field(default_factory=list)
    context_text: str | None = None
    modality: str = "text"
    has_image: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    cleaning_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Full internal representation."""
        return asdict(self)

    def to_meeting_format(self) -> dict[str, Any]:
        """Export in the meeting's minimal required format."""
        return {
            "dataset": self.dataset,
            "sample_id": self.sample_id,
            "text": self.text,
            "label": self.mapped_label,
            "label_name": self.mapped_label_name,
            "metadata": self.metadata,
        }

    def to_master_record(self) -> "MasterRecord":
        """Export a canonical master-dataset record."""
        return master_record_from_unified(self)

    def to_master_flat_format(self) -> dict[str, Any]:
        """Export the user-requested flat master-dataset row."""
        return flat_master_export_from_unified(self)


@dataclass
class MasterRecord:
    """Canonical merged export for the multi-source master dataset."""

    id: str
    dataset: str
    source_sample_id: str
    title: str | None
    primary_text: str
    source_name: str | None
    is_real: int  # 1 = real, 0 = fake
    tweet_num: int | None
    split: str | None
    original_label: str | int
    original_label_name: str
    mapped_label: int
    mapped_label_name: str
    source_fields_used: list[str] = field(default_factory=list)
    modality: str = "text"
    has_image: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    cleaning_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_flat_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "Title": self.title,
            "Claim": self.primary_text,
            "news source": self.source_name,
            "Real": self.is_real,
            "tweet_num": self.tweet_num,
        }


@dataclass
class LabelMapping:
    """Documents how a dataset's native labels map to binary real/fake."""

    dataset: str
    original_labels: list[str | int]
    mapped_labels: dict[str, str]  # original -> "real" | "fake"
    excluded_labels: list[str | int]
    justification: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _slugify(value: str) -> str:
    lowered = re.sub(r"[^a-z0-9]+", "_", value.lower())
    lowered = re.sub(r"_+", "_", lowered).strip("_")
    return lowered or "record"


def _best_non_empty(*values: Any) -> str | None:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str):
            stripped = value.strip()
            if stripped:
                return stripped
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _extract_title_and_primary_text(text: str, context_text: str | None) -> tuple[str | None, str]:
    cleaned_text = text.strip()
    cleaned_context = context_text.strip() if context_text else None
    if not cleaned_context:
        return cleaned_text or None, cleaned_text
    if cleaned_text == cleaned_context:
        return None, cleaned_context
    if cleaned_text.endswith(cleaned_context):
        prefix = cleaned_text[:-len(cleaned_context)].strip()
        return prefix or None, cleaned_context
    return None, cleaned_context


def _canonical_master_id(record: UnifiedRecord) -> str:
    return f"{_slugify(record.dataset)}__{record.sample_id}"


def _claimreview_master_record(record: UnifiedRecord) -> MasterRecord:
    title = _best_non_empty(record.metadata.get("review_title"), record.context_text)
    source_name = _best_non_empty(record.metadata.get("review_publisher"))
    return MasterRecord(
        id=_canonical_master_id(record),
        dataset=record.dataset,
        source_sample_id=record.sample_id,
        title=title,
        primary_text=record.text,
        source_name=source_name,
        is_real=1 if record.mapped_label == 0 else 0,
        tweet_num=None,
        split=record.split,
        original_label=record.original_label,
        original_label_name=record.original_label_name,
        mapped_label=record.mapped_label,
        mapped_label_name=record.mapped_label_name,
        source_fields_used=list(record.source_fields_used),
        modality=record.modality,
        has_image=record.has_image,
        metadata=dict(record.metadata),
        cleaning_notes=list(record.cleaning_notes),
    )


def _fakeddit_master_record(record: UnifiedRecord) -> MasterRecord:
    title, primary_text = _extract_title_and_primary_text(record.text, record.context_text)
    source_name = _best_non_empty(
        record.metadata.get("domain"),
        record.metadata.get("subreddit"),
        record.metadata.get("platform"),
    )
    return MasterRecord(
        id=_canonical_master_id(record),
        dataset=record.dataset,
        source_sample_id=record.sample_id,
        title=title,
        primary_text=primary_text,
        source_name=source_name,
        is_real=1 if record.mapped_label == 0 else 0,
        tweet_num=None,
        split=record.split,
        original_label=record.original_label,
        original_label_name=record.original_label_name,
        mapped_label=record.mapped_label,
        mapped_label_name=record.mapped_label_name,
        source_fields_used=list(record.source_fields_used),
        modality=record.modality,
        has_image=record.has_image,
        metadata=dict(record.metadata),
        cleaning_notes=list(record.cleaning_notes),
    )


def _fakenewsnet_master_record(record: UnifiedRecord) -> MasterRecord:
    title, primary_text = _extract_title_and_primary_text(record.text, record.context_text)
    source_name = _best_non_empty(
        record.metadata.get("source"),
        record.metadata.get("source_domain"),
        record.metadata.get("platform"),
    )
    return MasterRecord(
        id=_canonical_master_id(record),
        dataset=record.dataset,
        source_sample_id=record.sample_id,
        title=title,
        primary_text=primary_text,
        source_name=source_name,
        is_real=1 if record.mapped_label == 0 else 0,
        tweet_num=_coerce_int(record.metadata.get("tweet_num")),
        split=record.split,
        original_label=record.original_label,
        original_label_name=record.original_label_name,
        mapped_label=record.mapped_label,
        mapped_label_name=record.mapped_label_name,
        source_fields_used=list(record.source_fields_used),
        modality=record.modality,
        has_image=record.has_image,
        metadata=dict(record.metadata),
        cleaning_notes=list(record.cleaning_notes),
    )


def _mumin_master_record(record: UnifiedRecord) -> MasterRecord:
    source_name = _best_non_empty(
        record.metadata.get("claim_origin"),
        record.metadata.get("source"),
        record.metadata.get("platform"),
    )
    return MasterRecord(
        id=_canonical_master_id(record),
        dataset=record.dataset,
        source_sample_id=record.sample_id,
        title=None,
        primary_text=record.text,
        source_name=source_name,
        is_real=1 if record.mapped_label == 0 else 0,
        tweet_num=_coerce_int(record.metadata.get("tweet_num")),
        split=record.split,
        original_label=record.original_label,
        original_label_name=record.original_label_name,
        mapped_label=record.mapped_label,
        mapped_label_name=record.mapped_label_name,
        source_fields_used=list(record.source_fields_used),
        modality=record.modality,
        has_image=record.has_image,
        metadata=dict(record.metadata),
        cleaning_notes=list(record.cleaning_notes),
    )


def master_record_from_unified(record: UnifiedRecord) -> MasterRecord:
    """Convert a UnifiedRecord into the canonical master export."""
    dataset_key = record.dataset.casefold()
    if dataset_key == "claimreview":
        return _claimreview_master_record(record)
    if dataset_key == "fakeddit":
        return _fakeddit_master_record(record)
    if dataset_key == "fakenewsnet":
        return _fakenewsnet_master_record(record)
    if dataset_key == "mumin":
        return _mumin_master_record(record)

    title, primary_text = _extract_title_and_primary_text(record.text, record.context_text)
    source_name = _best_non_empty(
        record.metadata.get("source_name"),
        record.metadata.get("source"),
        record.metadata.get("source_domain"),
        record.metadata.get("platform"),
    )
    return MasterRecord(
        id=_canonical_master_id(record),
        dataset=record.dataset,
        source_sample_id=record.sample_id,
        title=title,
        primary_text=primary_text,
        source_name=source_name,
        is_real=1 if record.mapped_label == 0 else 0,
        tweet_num=_coerce_int(record.metadata.get("tweet_num")),
        split=record.split,
        original_label=record.original_label,
        original_label_name=record.original_label_name,
        mapped_label=record.mapped_label,
        mapped_label_name=record.mapped_label_name,
        source_fields_used=list(record.source_fields_used),
        modality=record.modality,
        has_image=record.has_image,
        metadata=dict(record.metadata),
        cleaning_notes=list(record.cleaning_notes),
    )


def flat_master_export_from_unified(record: UnifiedRecord) -> dict[str, Any]:
    """Convert a UnifiedRecord into the flat master export row."""
    return master_record_from_unified(record).to_flat_dict()


def _reconstruct_text_and_context_from_master(record: MasterRecord) -> tuple[str, str | None]:
    title = (record.title or "").strip()
    primary_text = record.primary_text.strip()
    dataset_key = record.dataset.casefold()

    if dataset_key == "claimreview":
        return primary_text, title or None

    if dataset_key in {"fakeddit", "fakenewsnet"}:
        if title and primary_text and title != primary_text:
            return f"{title}\n\n{primary_text}", primary_text
        return primary_text or title, None

    return primary_text or title, None


def unified_record_from_master(record: MasterRecord) -> UnifiedRecord:
    """Convert a MasterRecord back into the canonical UnifiedRecord shape."""
    text, context_text = _reconstruct_text_and_context_from_master(record)
    return UnifiedRecord(
        dataset=record.dataset,
        sample_id=record.source_sample_id,
        text=text,
        original_label=record.original_label,
        original_label_name=record.original_label_name,
        mapped_label=record.mapped_label,
        mapped_label_name=record.mapped_label_name,
        split=record.split,
        source_fields_used=list(record.source_fields_used),
        context_text=context_text,
        modality=record.modality,
        has_image=record.has_image,
        metadata=dict(record.metadata),
        cleaning_notes=list(record.cleaning_notes),
    )


def write_master_records_json(path: Path, records: list[MasterRecord]) -> None:
    """Write canonical master records as a JSON array."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [record.to_dict() for record in records]
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_master_records_jsonl(path: Path, records: list[MasterRecord]) -> None:
    """Write canonical master records as newline-delimited JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")


def write_master_records_csv(path: Path, records: list[MasterRecord]) -> None:
    """Write canonical master records as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        return

    rows: list[dict[str, Any]] = []
    for record in records:
        flat = record.to_dict()
        flat["source_fields_used"] = " | ".join(flat.get("source_fields_used") or [])
        flat["cleaning_notes"] = " | ".join(flat.get("cleaning_notes") or [])
        flat["metadata"] = json.dumps(flat.get("metadata") or {}, ensure_ascii=False)
        rows.append(flat)

    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_master_flat_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write the requested flat master export as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_records_json(path: Path, records: list[UnifiedRecord]) -> None:
    """Write unified records as a JSON array."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [r.to_dict() for r in records]
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_records_jsonl(path: Path, records: list[UnifiedRecord]) -> None:
    """Write unified records as newline-delimited JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")


def write_records_csv(path: Path, records: list[UnifiedRecord]) -> None:
    """Write unified records as CSV (flattening list/dict fields)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        return

    rows: list[dict[str, Any]] = []
    for record in records:
        flat = record.to_dict()
        flat["source_fields_used"] = " | ".join(flat.get("source_fields_used") or [])
        flat["cleaning_notes"] = " | ".join(flat.get("cleaning_notes") or [])
        flat["metadata"] = json.dumps(flat.get("metadata") or {}, ensure_ascii=False)
        rows.append(flat)

    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_records_jsonl(path: Path) -> list[UnifiedRecord]:
    """Read unified records from a UnifiedRecord or MasterRecord JSONL file."""
    records: list[UnifiedRecord] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if "sample_id" in data:
                records.append(UnifiedRecord(**data))
                continue
            if "source_sample_id" in data:
                records.append(unified_record_from_master(MasterRecord(**data)))
                continue
            raise ValueError(f"Unsupported record schema in {path}: missing sample_id/source_sample_id")
    return records
