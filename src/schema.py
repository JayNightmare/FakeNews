"""Canonical unified record schema for the misinformation detection pipeline.

Richer than the meeting's minimal export format (dataset, sample_id, text,
label, label_name, metadata) to preserve provenance, modality, and cleaning
history across all four data sources.
"""

from __future__ import annotations

import csv
import json
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
    """Read unified records from a JSONL file."""
    records: list[UnifiedRecord] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            records.append(UnifiedRecord(**data))
    return records
