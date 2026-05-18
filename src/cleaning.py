"""Data cleaning pipeline.

Implements all cleaning rules specified in the 17/04 meeting document:
- Remove entries with empty or null text
- Remove entries with fewer than 5 words
- Remove corrupted or unreadable characters
- Strip unnecessary whitespace
- Drop rows with missing essential fields

Every modification is logged in the record's cleaning_notes field.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import replace
from typing import Any

from src.schema import UnifiedRecord

_CORRUPTION_PATTERN = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]"
    r"|[\ufffd\ufffe\uffff]"
    r"|[\ud800-\udfff]"
)

ESSENTIAL_FIELDS = ("text", "mapped_label", "dataset", "sample_id")
MIN_WORD_COUNT = 5


def _strip_whitespace(text: str) -> str:
    """Collapse all runs of whitespace into single spaces, strip ends."""
    return re.sub(r"\s+", " ", text).strip()


def _remove_corrupted_chars(text: str) -> tuple[str, int]:
    """Remove control characters and replacement characters.

    Returns:
        Tuple of (cleaned text, number of characters removed).
    """
    cleaned = _CORRUPTION_PATTERN.sub("", text)
    cleaned = unicodedata.normalize("NFC", cleaned)
    removed_count = len(text) - len(cleaned)
    return cleaned, removed_count


def _word_count(text: str) -> int:
    """Count words (whitespace-separated tokens)."""
    return len(text.split())


def clean_record(record: UnifiedRecord) -> UnifiedRecord | None:
    """Apply all cleaning rules to a single record.

    Args:
        record: The record to clean.

    Returns:
        Cleaned record, or None if the record should be dropped.
    """
    notes = list(record.cleaning_notes)

    # 1. Check essential fields
    if not record.text:
        return None
    if not record.dataset or not record.sample_id:
        return None

    text = record.text

    # 2. Remove corrupted characters
    text, removed_count = _remove_corrupted_chars(text)
    if removed_count > 0:
        notes.append(f"removed {removed_count} corrupted/control character(s)")

    # 3. Strip whitespace
    original_len = len(text)
    text = _strip_whitespace(text)
    if len(text) != original_len:
        notes.append("stripped excess whitespace")

    # 4. Check for empty after cleaning
    if not text:
        return None

    # 5. Minimum word count
    wc = _word_count(text)
    if wc < MIN_WORD_COUNT:
        return None

    # 6. Clean context_text if present
    context = record.context_text
    if context:
        context, ctx_removed = _remove_corrupted_chars(context)
        context = _strip_whitespace(context)
        if ctx_removed > 0:
            notes.append(f"removed {ctx_removed} corrupted char(s) from context_text")
        if not context:
            context = None
            notes.append("context_text became empty after cleaning")

    return replace(record, text=text, context_text=context, cleaning_notes=notes)


def clean_records(records: list[UnifiedRecord]) -> tuple[list[UnifiedRecord], CleaningReport]:
    """Apply cleaning to a batch of records.

    Args:
        records: List of records to clean.

    Returns:
        Tuple of (cleaned records, cleaning report).
    """
    cleaned: list[UnifiedRecord] = []
    dropped_empty = 0
    dropped_short = 0
    dropped_missing = 0
    total_corrupted_chars = 0

    for record in records:
        if not record.text:
            dropped_empty += 1
            continue
        if not record.dataset or not record.sample_id:
            dropped_missing += 1
            continue

        result = clean_record(record)
        if result is None:
            wc = _word_count(_strip_whitespace(record.text))
            if wc < MIN_WORD_COUNT:
                dropped_short += 1
            else:
                dropped_empty += 1
            continue

        for note in result.cleaning_notes:
            if "corrupted" in note and "context" not in note:
                try:
                    count = int(note.split()[1])
                    total_corrupted_chars += count
                except (ValueError, IndexError):
                    pass

        cleaned.append(result)

    report = CleaningReport(
        total_input=len(records),
        total_output=len(cleaned),
        dropped_empty_text=dropped_empty,
        dropped_short_text=dropped_short,
        dropped_missing_fields=dropped_missing,
        corrupted_characters_removed=total_corrupted_chars,
    )
    return cleaned, report


class CleaningReport:
    """Summary of what the cleaning pipeline did."""

    def __init__(
        self,
        total_input: int,
        total_output: int,
        dropped_empty_text: int,
        dropped_short_text: int,
        dropped_missing_fields: int,
        corrupted_characters_removed: int,
    ) -> None:
        self.total_input = total_input
        self.total_output = total_output
        self.dropped_empty_text = dropped_empty_text
        self.dropped_short_text = dropped_short_text
        self.dropped_missing_fields = dropped_missing_fields
        self.corrupted_characters_removed = corrupted_characters_removed

    @property
    def total_dropped(self) -> int:
        return self.total_input - self.total_output

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_input": self.total_input,
            "total_output": self.total_output,
            "total_dropped": self.total_dropped,
            "dropped_empty_text": self.dropped_empty_text,
            "dropped_short_text": self.dropped_short_text,
            "dropped_missing_fields": self.dropped_missing_fields,
            "corrupted_characters_removed": self.corrupted_characters_removed,
            "retention_rate": round(self.total_output / self.total_input, 4) if self.total_input else 0,
        }

    def __repr__(self) -> str:
        return (
            f"CleaningReport(input={self.total_input}, output={self.total_output}, "
            f"dropped={self.total_dropped})"
        )


# ---------------------------------------------------------------------------
# Metadata cleaning
# ---------------------------------------------------------------------------

_DATE_FIELDS = {"created_utc", "timestamp", "publish_date", "review_date", "claim_date"}
_DROP_VALUES = {None, "", "None", "none", "null", "N/A", "n/a", "nan", "NaN"}


def _normalise_date(value: Any) -> str | None:
    """Best-effort date normalisation.

    Handles unix timestamps (int/float/str-of-digits), ISO strings,
    and common date formats.  Returns ISO-8601 string or None.
    """
    if value is None:
        return None

    # Unix timestamp (seconds)
    if isinstance(value, (int, float)):
        try:
            from datetime import datetime, timezone
            return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()
        except (OSError, ValueError):
            return str(value)

    s = str(value).strip()
    if not s or s in _DROP_VALUES:
        return None

    # String that looks like a unix timestamp
    if s.replace(".", "").isdigit():
        try:
            ts = float(s)
            from datetime import datetime, timezone
            return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        except (OSError, ValueError):
            return s

    # Already an ISO-ish string — return as-is
    return s


def clean_metadata(record: UnifiedRecord) -> UnifiedRecord:
    """Normalise and clean metadata fields.

    - Removes keys with null/empty/sentinel values
    - Normalises date-like fields to ISO-8601
    - Strips whitespace from string values
    """
    notes = list(record.cleaning_notes)
    cleaned_meta: dict[str, Any] = {}

    for key, value in record.metadata.items():
        # Drop empty/null sentinel values
        try:
            if value in _DROP_VALUES:
                continue
        except TypeError:
            pass  # unhashable type (list, dict) — not a sentinel
        if isinstance(value, list) and not value:
            continue

        # Normalise dates
        if key in _DATE_FIELDS:
            normalised = _normalise_date(value)
            if normalised is None:
                continue
            cleaned_meta[key] = normalised
            if str(value) != normalised:
                notes.append(f"normalised metadata.{key}")
            continue

        # Strip string whitespace
        if isinstance(value, str):
            value = value.strip()
            if not value or value in _DROP_VALUES:
                continue

        cleaned_meta[key] = value

    return replace(record, metadata=cleaned_meta, cleaning_notes=notes)


def clean_metadata_batch(records: list[UnifiedRecord]) -> list[UnifiedRecord]:
    """Apply metadata cleaning to a batch of records."""
    return [clean_metadata(r) for r in records]


# ---------------------------------------------------------------------------
# Balanced / stratified sampling
# ---------------------------------------------------------------------------

def balanced_sample(
    records: list[UnifiedRecord],
    target_per_label: int = 500,
    seed: int = 42,
) -> list[UnifiedRecord]:
    """Draw a balanced subset with equal representation per mapped_label.

    Args:
        records: Full list of cleaned records.
        target_per_label: Max samples per label value.
        seed: Random seed for reproducibility.

    Returns:
        Balanced subset. If a label has fewer than target_per_label
        records, all are included (no oversampling).
    """
    import random

    by_label: dict[int, list[UnifiedRecord]] = {}
    for record in records:
        by_label.setdefault(record.mapped_label, []).append(record)

    rng = random.Random(seed)
    sampled: list[UnifiedRecord] = []

    for label in sorted(by_label.keys()):
        pool = by_label[label]
        if len(pool) <= target_per_label:
            sampled.extend(pool)
        else:
            sampled.extend(rng.sample(pool, target_per_label))

    # Shuffle so labels aren't grouped together
    rng.shuffle(sampled)
    return sampled

