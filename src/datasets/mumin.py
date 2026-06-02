"""MuMiN dataset loader (stub).

MuMiN depends on the `mumin` Python package and Twitter/X credentials
for full hydration. This loader is intentionally minimal — it can load
from pre-exported JSONL/JSON files but does not handle hydration.

Full implementation deferred as the highest-risk dependency.
See: https://github.com/MuMiN-dataset/mumin
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from src.datasets.base import DatasetLoader
from src.schema import LabelMapping, UnifiedRecord


def _first_non_empty(raw: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = raw.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            stripped = value.strip()
            if stripped:
                return stripped
            continue
        return value
    return None


class MuMiNLoader(DatasetLoader):
    """Loads misinformation records from MuMiN (pre-exported format)."""

    @property
    def name(self) -> str:
        return "MuMiN"

    @property
    def default_data_dir(self) -> str:
        return "mumin"

    def load_raw(self, data_dir: Path) -> list[dict[str, Any]]:
        """Load from pre-exported JSONL or JSON files.

        MuMiN's native access path uses the `mumin` Python package,
        which requires Twitter API credentials. This loader expects
        pre-exported data in JSONL or JSON format.
        """
        records: list[dict[str, Any]] = []

        for jsonl_file in sorted(data_dir.glob("*.jsonl")):
            with jsonl_file.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))

        for json_file in sorted(data_dir.glob("*.json")):
            try:
                data = json.loads(json_file.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    records.extend(data)
                elif isinstance(data, dict) and "records" in data:
                    records.extend(data["records"])
                elif isinstance(data, dict):
                    records.append(data)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue

        for csv_file in sorted(data_dir.glob("*.csv")):
            records.extend(self._read_csv(csv_file))

        if not records:
            raise FileNotFoundError(
                f"No MuMiN data found in {data_dir}. "
                f"This loader expects pre-exported CSV/JSONL/JSON files. "
                f"For native access, install the mumin package and export first: "
                f"https://github.com/MuMiN-dataset/mumin"
            )

        return records

    @staticmethod
    def _read_csv(filepath: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with filepath.open(encoding="utf-8", errors="replace") as fh:
            reader = csv.DictReader(fh)
            for index, row in enumerate(reader, start=1):
                row["_file"] = filepath.name
                row["_row_number"] = str(index)
                rows.append(row)
        return rows

    def to_unified(self, raw: dict[str, Any]) -> UnifiedRecord:
        claim_text = (
            raw.get("claim_en") or raw.get("claim") or raw.get("text") or raw.get("tweet_text") or ""
        ).strip()
        thread_text = raw.get("thread_text") or raw.get("context") or ""

        label_raw = raw.get("label") or raw.get("verdict") or "unknown"
        label_str = str(label_raw).lower().strip()

        if label_str == "unknown":
            return None

        if label_str in ("factual", "true", "real", "0"):
            mapped_label, mapped_label_name = 0, "real"
        elif label_str in ("misinformation", "false", "fake", "1"):
            mapped_label, mapped_label_name = 1, "fake"
        else:
            return None

        cleaning_notes = [
            f"Label mapped: {label_raw} ({label_str}) -> {mapped_label_name}"
        ]

        sample_id = f"mn_{raw.get('claim_id', raw.get('id', raw.get('_row_number', 'unknown')))}"
        has_image = bool(raw.get("image_url") or raw.get("has_image"))

        split = raw.get("split")
        if isinstance(split, str) and split.strip().casefold() == "validate":
            split = "validation"
        if not split:
            if str(raw.get("train_mask", "")).lower() == "true":
                split = "train"
            elif str(raw.get("val_mask", "")).lower() == "true":
                split = "validation"
            elif str(raw.get("test_mask", "")).lower() == "true":
                split = "test"

        claim_origin = _first_non_empty(raw, "claim_source", "claim_origin", "source")
        tweet_num = raw.get("num_replies")

        metadata: dict[str, Any] = {
            "claim_id": raw.get("claim_id"),
            "thread_id": raw.get("thread_id"),
            "timestamp": raw.get("timestamp") or raw.get("created_at"),
            "platform": raw.get("platform", "Twitter"),
            "language": raw.get("lang") or raw.get("language"),
            "num_replies": raw.get("num_replies"),
            "tweet_num": tweet_num,
        }
        if claim_origin is not None:
            metadata["claim_origin"] = claim_origin
        if raw.get("claim") and raw.get("claim_en") and metadata.get("language") is None:
            metadata["language"] = "pt"
        metadata = {k: v for k, v in metadata.items() if v is not None or k == "tweet_num"}

        if raw.get("claim_en"):
            source_fields = ["claim_en"]
        elif raw.get("claim"):
            source_fields = ["claim"]
        else:
            source_fields = ["text"]

        return UnifiedRecord(
            dataset="MuMiN",
            sample_id=sample_id,
            text=claim_text,
            original_label=label_raw,
            original_label_name=label_str,
            mapped_label=mapped_label,
            mapped_label_name=mapped_label_name,
            split=split,
            source_fields_used=source_fields,
            context_text=thread_text if thread_text else None,
            modality="text+image" if has_image else "text",
            has_image=has_image,
            metadata=metadata,
            cleaning_notes=cleaning_notes,
        )

    def label_mapping(self) -> LabelMapping:
        return LabelMapping(
            dataset="MuMiN",
            original_labels=["misinformation", "factual", "unknown"],
            mapped_labels={
                "misinformation": "fake",
                "factual": "real",
            },
            excluded_labels=["unknown"],
            justification=(
                "MuMiN's claim-level labels are binary (misinformation vs factual). "
                "Claims labelled 'unknown' lack ground truth and are excluded from "
                "training/evaluation to avoid noise."
            ),
        )
