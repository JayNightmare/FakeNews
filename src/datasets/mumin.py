"""MuMiN dataset loader (stub).

MuMiN depends on the `mumin` Python package and Twitter/X credentials
for full hydration. This loader is intentionally minimal — it can load
from pre-exported JSONL/JSON files but does not handle hydration.

Full implementation deferred as the highest-risk dependency.
See: https://github.com/MuMiN-dataset/mumin
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.datasets.base import DatasetLoader
from src.schema import LabelMapping, UnifiedRecord


class MuMiNLoader(DatasetLoader):
    """Loads misinformation records from MuMiN (pre-exported format)."""

    @property
    def name(self) -> str:
        return "MuMiN"

    @property
    def default_data_dir(self) -> str:
        return "data/mumin"

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

        if not records:
            raise FileNotFoundError(
                f"No MuMiN data found in {data_dir}. "
                f"This loader expects pre-exported JSONL/JSON files. "
                f"For native access, install the mumin package and export first: "
                f"https://github.com/MuMiN-dataset/mumin"
            )

        return records

    def to_unified(self, raw: dict[str, Any]) -> UnifiedRecord:
        claim_text = (raw.get("claim") or raw.get("text") or raw.get("tweet_text") or "").strip()
        thread_text = raw.get("thread_text") or raw.get("context") or ""

        label_raw = raw.get("label") or raw.get("verdict") or "unknown"
        label_str = str(label_raw).lower().strip()

        if label_str in ("factual", "true", "real", "0"):
            mapped_label, mapped_label_name = 0, "real"
        elif label_str in ("misinformation", "false", "fake", "1"):
            mapped_label, mapped_label_name = 1, "fake"
        else:
            mapped_label, mapped_label_name = 1, "fake"

        cleaning_notes: list[str] = []
        if label_str in ("unknown",):
            cleaning_notes.append(f"label '{label_str}' is ambiguous, mapped to fake by default")

        sample_id = f"mn_{raw.get('claim_id', raw.get('id', 'unknown'))}"
        has_image = bool(raw.get("image_url") or raw.get("has_image"))

        metadata: dict[str, Any] = {
            "claim_id": raw.get("claim_id"),
            "thread_id": raw.get("thread_id"),
            "timestamp": raw.get("timestamp") or raw.get("created_at"),
            "platform": raw.get("platform", "Twitter"),
            "language": raw.get("lang") or raw.get("language"),
            "num_replies": raw.get("num_replies"),
        }
        metadata = {k: v for k, v in metadata.items() if v is not None}

        source_fields = ["claim" if raw.get("claim") else "text"]

        return UnifiedRecord(
            dataset="MuMiN",
            sample_id=sample_id,
            text=claim_text,
            original_label=label_raw,
            original_label_name=label_str,
            mapped_label=mapped_label,
            mapped_label_name=mapped_label_name,
            split=raw.get("split"),
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
