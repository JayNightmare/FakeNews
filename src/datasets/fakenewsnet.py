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
                f"or pre-processed JSONL/JSON files. "
                f"Download from https://github.com/KaiDMML/FakeNewsNet"
            )

        return records

    def to_unified(self, raw: dict[str, Any]) -> UnifiedRecord:
        title = (raw.get("title") or raw.get("headline") or "").strip()
        body = (raw.get("text") or raw.get("body") or raw.get("content") or "").strip()
        source = raw.get("_source") or raw.get("source") or "unknown"

        text = title
        source_fields = ["title"]
        if body:
            text = f"{title}\n\n{body}" if title else body
            source_fields.append("text" if raw.get("text") else "body")

        label_raw = raw.get("_label") or raw.get("label") or "unknown"
        label_str = str(label_raw).lower().strip()

        if label_str in ("real", "true", "0"):
            mapped_label, mapped_label_name = 0, "real"
        elif label_str in ("fake", "false", "1"):
            mapped_label, mapped_label_name = 1, "fake"
        else:
            mapped_label, mapped_label_name = 1, "fake"

        sample_id = f"fn_{source}_{raw.get('_file', raw.get('id', 'unknown'))}"
        sample_id = sample_id.replace(".json", "")

        has_image = bool(raw.get("images") or raw.get("top_img") or raw.get("image_url"))

        metadata: dict[str, Any] = {
            "source": source,
            "url": raw.get("url") or raw.get("news_url"),
            "publish_date": raw.get("publish_date") or raw.get("date"),
            "authors": raw.get("authors"),
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
