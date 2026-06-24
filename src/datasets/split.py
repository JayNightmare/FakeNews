"""Dataset split helpers for the unified master dataset.

Split %:
- Training: 70%
- Validation: 15%
- Test: 15%
"""

from __future__ import annotations

import random
from dataclasses import replace
from pathlib import Path

from src.schema import load_records_jsonl, write_records_csv, write_records_json, write_records_jsonl

VALID_SPLITS = ("train", "validation", "test")
DEFAULT_SPLIT_FILENAMES = ("normalized_records.jsonl", "master_records.jsonl")


def _resolve_split_source_path(dataset_name: str, data_path: Path | None) -> Path:
    candidates: list[Path] = []

    if data_path is not None:
        if data_path.suffix:
            candidates.append(data_path)
        else:
            candidates.extend(data_path.with_suffix(ext) for ext in (".jsonl", ".json"))
            candidates.append(data_path)
    else:
        source_root = Path(__file__).resolve().parent / "data"
        dataset_dir = source_root / dataset_name
        for filename in DEFAULT_SPLIT_FILENAMES:
            candidates.append(dataset_dir / filename)

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate

    searched = ", ".join(str(candidate) for candidate in candidates) or "(none)"
    raise FileNotFoundError(
        f"Unable to resolve dataset records for splitting. Looked in: {searched}."
    )


def _build_splits(total_records: int) -> dict[str, tuple[int, int]]:
    train_end = int(0.7 * total_records)
    validation_end = int(0.85 * total_records)
    return {
        "train": (0, train_end),
        "validation": (train_end, validation_end),
        "test": (validation_end, total_records),
    }

def split_dataset(
    dataset_name: str,
    split: str,
    output_path: Path,
    max_samples: int | None = None,
    data_path: Path | None = None,
    seed: int = 42,
) -> str:
    """
    Splits the dataset into training, validation, and test datasets.

    Args:
        dataset_name (str): The name of the dataset to split.
        split (str): The split to use (e.g., "train", "validation", "test").
        output_path (Path): The path to save the split dataset.
        max_samples (int | None): The maximum number of samples to include in the split. If None, all samples are included.

    Returns:
        Successful split message (str): A message indicating the success of the split operation.
    
    Logic:
        1. Load the dataset using the provided dataset_name and split.
        2. If max_samples is specified and the dataset size exceeds max_samples, randomly sample max_samples from the dataset.
        3. Save the split dataset to the specified output_path.
    """
    if split not in VALID_SPLITS:
        raise ValueError(f"Invalid split '{split}'. Must be one of 'train', 'validation', or 'test'.")

    source_path = _resolve_split_source_path(dataset_name, data_path)
    print(f"Splitting dataset '{dataset_name}' from {source_path} into {split} and saving to {output_path}...")

    records = load_records_jsonl(source_path)
    random.Random(seed).shuffle(records)

    if max_samples is not None:
        records = records[:max_samples]

    split_bounds = _build_splits(len(records))
    start, end = split_bounds[split]
    split_records = [replace(record, split=split) for record in records[start:end]]

    output_path.mkdir(parents=True, exist_ok=True)
    write_records_jsonl(output_path / f"{split}.jsonl", split_records)
    write_records_json(output_path / f"{split}.json", split_records)
    write_records_csv(output_path / f"{split}.csv", split_records)

    return (
        f"Dataset {dataset_name} split into {split} with {len(split_records)} records "
        f"and saved to {output_path}."
    )