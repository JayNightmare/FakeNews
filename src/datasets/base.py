"""Abstract base class for dataset loaders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from src.schema import LabelMapping, UnifiedRecord


DATASETS_ROOT = Path(__file__).resolve().parent / "data"


class DatasetLoader(ABC):
    """Interface that every dataset loader must implement."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable dataset name (e.g. 'FakeNewsNet')."""

    @property
    @abstractmethod
    def default_data_dir(self) -> str:
        """Default dataset subdirectory name under src/datasets/data/."""

    def resolve_data_dir(self, data_dir: Path | None = None) -> Path:
        """Resolve the dataset directory, honoring an explicit override."""
        if data_dir is not None:
            return data_dir
        return DATASETS_ROOT / self.default_data_dir

    @abstractmethod
    def load_raw(self, data_dir: Path) -> list[dict[str, Any]]:
        """Load raw records from local files.

        Args:
            data_dir: Path to the directory containing dataset files.

        Returns:
            List of raw dictionaries, one per sample.
        """

    @abstractmethod
    def to_unified(self, raw: dict[str, Any]) -> UnifiedRecord | None:
        """Convert a single raw record to the unified schema.

        Args:
            raw: A single raw dictionary from load_raw().

        Returns:
            A UnifiedRecord instance, or None if the record should be
            excluded (e.g. unmappable labels).
        """

    @abstractmethod
    def label_mapping(self) -> LabelMapping:
        """Return the label mapping documentation for this dataset."""

    def load(self, data_dir: Path, limit: int | None = None) -> list[UnifiedRecord]:
        """Load, convert, filter excluded records, and optionally limit.

        Args:
            data_dir: Path to dataset files.
            limit: Maximum number of records to return.

        Returns:
            List of UnifiedRecord instances (excludes None results).
        """
        raw_items = self.load_raw(data_dir)
        if limit is not None:
            raw_items = raw_items[:limit]
        records: list[UnifiedRecord] = []
        for item in raw_items:
            result = self.to_unified(item)
            if result is not None:
                records.append(result)
        return records

