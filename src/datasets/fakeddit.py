"""Fakeddit dataset loader.

Expects local TSV/CSV files from the Fakeddit dataset
(https://github.com/entitize/Fakeddit).

Fakeddit uses multi-way labels (2-way, 3-way, 6-way). We use the 6-way
column as original_label to preserve maximum granularity, then collapse
to binary for the mapped target.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from src.datasets.base import DatasetLoader
from src.schema import LabelMapping, UnifiedRecord

_LABEL_NAMES_6WAY: dict[int, str] = {
    0: "true",
    1: "satire/parody",
    2: "misleading content",
    3: "imposter content",
    4: "false connection",
    5: "manipulated content",
}

_LABEL_TO_BINARY: dict[int, tuple[int, str]] = {
    0: (0, "real"),
    1: (1, "fake"),
    2: (1, "fake"),
    3: (1, "fake"),
    4: (1, "fake"),
    5: (1, "fake"),
}


class FakedditLoader(DatasetLoader):
    """Loads misinformation records from the Fakeddit dataset."""

    @property
    def name(self) -> str:
        return "Fakeddit"

    @property
    def default_data_dir(self) -> str:
        return "data/fakeddit"

    def load_raw(self, data_dir: Path) -> list[dict[str, Any]]:
        """Load from local TSV files.

        Looks for files matching common Fakeddit naming:
        - train.tsv / test.tsv / validate.tsv
        - or any .tsv / .csv in the directory
        """
        records: list[dict[str, Any]] = []
        split_files = {
            "train": ["train.tsv", "train.csv", "all_train.tsv"],
            "test": ["test.tsv", "test.csv", "all_test_public.tsv"],
            "validate": ["validate.tsv", "validate.csv", "all_validate.tsv"],
        }

        found_any = False
        for split_name, filenames in split_files.items():
            for filename in filenames:
                filepath = data_dir / filename
                if filepath.exists():
                    found_any = True
                    records.extend(self._read_tsv(filepath, split_name))
                    break

        if not found_any:
            for filepath in sorted(data_dir.glob("*.tsv")) + sorted(data_dir.glob("*.csv")):
                records.extend(self._read_tsv(filepath, filepath.stem))

        if not records:
            raise FileNotFoundError(
                f"No Fakeddit data files found in {data_dir}. "
                f"Expected TSV/CSV files (e.g. train.tsv, test.tsv). "
                f"Download from https://github.com/entitize/Fakeddit"
            )

        return records

    @staticmethod
    def _read_tsv(filepath: Path, split: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        delimiter = "\t" if filepath.suffix == ".tsv" else ","
        with filepath.open(encoding="utf-8", errors="replace") as fh:
            reader = csv.DictReader(fh, delimiter=delimiter)
            for row in reader:
                row["_split"] = split
                row["_source_file"] = filepath.name
                rows.append(row)
        return rows

    def to_unified(self, raw: dict[str, Any]) -> UnifiedRecord | None:
        title = (raw.get("clean_title") or raw.get("title") or "").strip()
        selftext = (raw.get("selftext") or raw.get("body") or "").strip()
        sample_id = f"fd_{raw.get('id', 'unknown')}"

        label_6way = int(raw.get("6_way_label", raw.get("2_way_label", 0)))
        original_label_name = _LABEL_NAMES_6WAY.get(label_6way, f"unknown_{label_6way}")
        mapped_label, mapped_label_name = _LABEL_TO_BINARY.get(label_6way, (1, "fake"))

        has_image = bool(raw.get("hasImage") in ("True", "true", "1", True, 1)
                         or raw.get("image_url"))

        # Merge title + selftext (meeting requirement: merge relevant text fields)
        if selftext and selftext not in ("[removed]", "[deleted]"):
            text = f"{title}\n\n{selftext}" if title else selftext
            source_fields = ["clean_title" if raw.get("clean_title") else "title", "selftext"]
            context_text = selftext
        else:
            text = title
            source_fields = ["clean_title" if raw.get("clean_title") else "title"]
            context_text = None

        metadata: dict[str, Any] = {
            "subreddit": raw.get("subreddit"),
            "author": raw.get("author"),
            "score": raw.get("score"),
            "num_comments": raw.get("num_comments"),
            "created_utc": raw.get("created_utc"),
            "permalink": raw.get("permalink"),
            "domain": raw.get("domain"),
            "image_url": raw.get("image_url"),
            "platform": "Reddit",
        }
        metadata = {k: v for k, v in metadata.items() if v is not None}

        return UnifiedRecord(
            dataset="Fakeddit",
            sample_id=sample_id,
            text=text,
            original_label=label_6way,
            original_label_name=original_label_name,
            mapped_label=mapped_label,
            mapped_label_name=mapped_label_name,
            split=raw.get("_split"),
            source_fields_used=source_fields,
            context_text=context_text,
            modality="text+image" if has_image else "text",
            has_image=has_image,
            metadata=metadata,
            cleaning_notes=[],
        )

    def label_mapping(self) -> LabelMapping:
        return LabelMapping(
            dataset="Fakeddit",
            original_labels=[0, 1, 2, 3, 4, 5],
            mapped_labels={
                "0 (true)": "real",
                "1 (satire/parody)": "fake",
                "2 (misleading content)": "fake",
                "3 (imposter content)": "fake",
                "4 (false connection)": "fake",
                "5 (manipulated content)": "fake",
            },
            excluded_labels=[],
            justification=(
                "Fakeddit label 0 (true) maps to real. All other categories represent "
                "some form of misinformation and map to fake. The 6-way original label "
                "is preserved for fine-grained analysis."
            ),
        )
