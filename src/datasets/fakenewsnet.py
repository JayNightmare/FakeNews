"""FakeNewsNet dataset loader.

Expects local files from the FakeNewsNet repository
(https://github.com/KaiDMML/FakeNewsNet).

The article-level data does not require X/Twitter hydration.
Typical structure:
    data/fakenewsnet/
        politifact/
            real/ or fake/
                news_article_*.json  (or a single CSV/JSONL)
        gossipcop/
            real/ or fake/
                ...

Also supports a pre-processed CSV/JSONL format where each row has
at minimum: title, text/body, label, source.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from src.datasets.base import DatasetLoader
from src.schema import LabelMapping, UnifiedRecord


class FakeNewsNetLoader(DatasetLoader):
    """Loads misinformation records from FakeNewsNet."""

    @property
    def name(self) -> str:
        return "FakeNewsNet"

    @property
    def default_data_dir(self) -> str:
        return "fakenewsnet"

    def load_raw(self, data_dir: Path) -> list[dict[str, Any]]:
        """Load from local files. Supports multiple directory layouts."""
        records: list[dict[str, Any]] = []

        # Try structured directory layout first (politifact/gossipcop with real/fake subdirs)
        for source in ["politifact", "gossipcop"]:
            source_dir = data_dir / source
            if not source_dir.is_dir():
                continue
            for label_dir in ["real", "fake"]:
                label_path = source_dir / label_dir
                if not label_path.is_dir():
                    continue
                for json_file in sorted(label_path.glob("*.json")):
                    try:
                        data = json.loads(json_file.read_text(encoding="utf-8"))
                        data["_source"] = source
                        data["_label"] = label_dir
                        data["_file"] = json_file.name
                        records.append(data)
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        continue

        # Fallback: look for a pre-processed JSONL or CSV
        if not records:
            for jsonl_file in sorted(data_dir.glob("*.jsonl")):
                with jsonl_file.open(encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if line:
                            records.append(json.loads(line))

        if not records:
            for csv_file in sorted(data_dir.glob("*.csv")):
                records.extend(self._read_csv(csv_file))

        if not records:
            for json_file in sorted(data_dir.glob("*.json")):
                try:
                    data = json.loads(json_file.read_text(encoding="utf-8"))
                    if isinstance(data, list):
                        records.extend(data)
                    elif isinstance(data, dict):
                        records.append(data)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue

        if not records:
            raise FileNotFoundError(
                f"No FakeNewsNet data found in {data_dir}. "
                f"Expected either politifact/gossipcop subdirectories with real/fake folders, "
                f"or pre-processed CSV/JSONL/JSON files. "
                f"Download from https://github.com/KaiDMML/FakeNewsNet"
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
        title = (raw.get("title") or raw.get("headline") or "").strip()
        body = (raw.get("text") or raw.get("body") or raw.get("content") or "").strip()
        source = raw.get("_source") or raw.get("source") or raw.get("source_domain") or "unknown"

        text = title
        source_fields = ["title"]
        if body:
            text = f"{title}\n\n{body}" if title else body
            source_fields.append("text" if raw.get("text") else "body")

        label_raw = raw.get("_label") or raw.get("label")
        if label_raw in (None, ""):
            real_flag = raw.get("real")
            if real_flag not in (None, ""):
                real_flag_str = str(real_flag).strip().lower()
                if real_flag_str in ("1", "true", "yes", "real"):
                    label_raw = "real"
                elif real_flag_str in ("0", "false", "no", "fake"):
                    label_raw = "fake"
                else:
                    label_raw = real_flag
        if label_raw in (None, ""):
            label_raw = "unknown"
        label_str = str(label_raw).lower().strip()

        if label_str in ("real", "true", "0"):
            mapped_label, mapped_label_name = 0, "real"
        elif label_str in ("fake", "false", "1"):
            mapped_label, mapped_label_name = 1, "fake"
        else:
            mapped_label, mapped_label_name = 1, "fake"

        sample_token = raw.get("id") or raw.get("_row_number") or raw.get("_file") or "unknown"
        sample_id = f"fn_{source}_{sample_token}"
        sample_id = sample_id.replace(".json", "")
        sample_id = sample_id.replace(".csv", "")

        has_image = bool(raw.get("images") or raw.get("top_img") or raw.get("image_url"))

        metadata: dict[str, Any] = {
            "source": source,
            "url": raw.get("url") or raw.get("news_url"),
            "publish_date": raw.get("publish_date") or raw.get("date"),
            "authors": raw.get("authors"),
            "source_domain": raw.get("source_domain"),
            "tweet_num": raw.get("tweet_num"),
            "platform": "news",
        }
        metadata = {k: v for k, v in metadata.items() if v is not None}

        return UnifiedRecord(
            dataset="FakeNewsNet",
            sample_id=sample_id,
            text=text,
            original_label=label_raw,
            original_label_name=label_str,
            mapped_label=mapped_label,
            mapped_label_name=mapped_label_name,
            split=raw.get("split"),
            source_fields_used=source_fields,
            context_text=body if body else None,
            modality="text+image" if has_image else "text",
            has_image=has_image,
            metadata=metadata,
            cleaning_notes=[],
        )

    def label_mapping(self) -> LabelMapping:
        return LabelMapping(
            dataset="FakeNewsNet",
            original_labels=["real", "fake"],
            mapped_labels={"real": "real", "fake": "fake"},
            excluded_labels=[],
            justification=(
                "FakeNewsNet uses a clean binary label natively (real/fake). "
                "No remapping needed."
            ),
        )
