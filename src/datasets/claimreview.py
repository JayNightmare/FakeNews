"""ClaimReview dataset loader.

Refactored from the original run_experiment.py logic to conform to the
DatasetLoader interface. Uses the pinned local ClaimReview snapshot
from MartinoMensio/claimreview-data (2026_06_12).
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from src.datasets.base import DatasetLoader
from src.schema import LabelMapping, UnifiedRecord

CLAIMREVIEW_SNAPSHOT_VERSION = "2026_06_12"
CLAIMREVIEW_SNAPSHOT_FILE = "claim_reviews.json"


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _first_dict(items: list[Any]) -> dict[str, Any]:
    for item in items:
        if isinstance(item, dict):
            return item
    return {}


def _first_non_empty_string(items: list[Any]) -> str | None:
    for item in items:
        if isinstance(item, str):
            value = item.strip()
            if value:
                return value
    return None


def _to_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        first_text = _first_non_empty_string(value)
        if first_text:
            return first_text
    return ""


def _slugify(value: str) -> str:
    lowered = re.sub(r"[^a-z0-9]+", "_", value.lower())
    lowered = re.sub(r"_+", "_", lowered).strip("_")
    return lowered or "record"


def _stable_id(claim_text: str | None, review_url: str | None,
               publisher: str | None, claim_date: str | None) -> str:
    base = "|".join([review_url or "", claim_text or "", publisher or "", claim_date or ""])
    digest = hashlib.sha1(base.encode("utf-8")).hexdigest()[:12]
    title = publisher or claim_text or "record"
    return f"cr_{_slugify(title)[:24]}_{digest}"


def _normalize_rating(raw_rating: str | None) -> str:
    if not raw_rating:
        return "other"

    text = raw_rating.casefold()
    if text in {"credible", "mostly_credible"}:
        return "true"
    if text == "not_credible":
        return "false_or_misleading"
    if text in {"uncertain", "not_verifiable"}:
        return "mixed_or_unverified"
    if any(t in text for t in ["ai-generated", "ai generated", "synthetic", "deepfake"]):
        return "ai_generated_or_synthetic"
    if any(t in text for t in ["satire", "joke", "parody", "skit", "originated as satire"]):
        return "satire_or_joke"
    if any(t in text for t in ["false", "fake", "misleading", "incorrect", "mostly false",
                                "pants on fire", "hoax", "scam"]):
        return "false_or_misleading"
    if any(t in text for t in ["exaggerated", "partly false", "partly true", "half true",
                                "mixed", "unverified", "needs context", "inaccurate"]):
        return "mixed_or_unverified"
    if any(t in text for t in ["true", "correct", "accurate"]):
        return "true"
    if any(t in text for t in ["誤り", "不正確", "نادرست", "مضلل"]):
        return "false_or_misleading"
    return "other"


_RATING_TO_BINARY: dict[str, tuple[int, str]] = {
    "true": (0, "real"),
    "false_or_misleading": (1, "fake"),
    "mixed_or_unverified": (1, "fake"),
    "satire_or_joke": (1, "fake"),
    "ai_generated_or_synthetic": (1, "fake"),
}


class ClaimReviewLoader(DatasetLoader):
    """Loads misinformation records from the pinned ClaimReview snapshot."""

    def __init__(self, snapshot_version: str = CLAIMREVIEW_SNAPSHOT_VERSION) -> None:
        self._snapshot_version = snapshot_version

    @property
    def name(self) -> str:
        return "ClaimReview"

    @property
    def default_data_dir(self) -> str:
        return "claimreview"

    def _snapshot_candidates(self, data_dir: Path) -> list[Path]:
        snapshot_dir = data_dir / self._snapshot_version
        return [
            snapshot_dir / CLAIMREVIEW_SNAPSHOT_FILE,
            data_dir / CLAIMREVIEW_SNAPSHOT_FILE,
        ]

    def load_raw(self, data_dir: Path) -> list[dict[str, Any]]:
        """Load from local pinned snapshot and fail fast if missing.

        Supports release format and legacy payloads for compatibility:
        1. Snapshot release format (list of dicts)
        2. Data Commons feed format (dict with 'dataFeedElement')
        3. Legacy flat JSON/JSONL records
        """
        for candidate in self._snapshot_candidates(data_dir):
            if not candidate.exists():
                continue
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            records = self._parse_payload(payload)
            if records:
                return records

        expected_paths = " or ".join(str(path) for path in self._snapshot_candidates(data_dir))
        raise FileNotFoundError(
            "ClaimReview snapshot not found. Expected "
            f"{expected_paths}. Download release {self._snapshot_version} from "
            "https://github.com/MartinoMensio/claimreview-data/releases and place "
            f"{CLAIMREVIEW_SNAPSHOT_FILE} under the snapshot directory."
        )

    @staticmethod
    def _parse_payload(payload: Any) -> list[dict[str, Any]]:
        """Parse a JSON payload into a list of raw records."""
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if isinstance(payload, dict):
            # Data Commons feed format
            feed_items = _as_list(payload.get("dataFeedElement"))
            if feed_items:
                return [item for item in feed_items if isinstance(item, dict)]
            # Single record
            return [payload]
        return []

    def _is_explorer_format(self, raw: dict[str, Any]) -> bool:
        """Detect if a record is from flat release/explorer formats."""
        return "claim_text" in raw and "item" not in raw

    def to_unified(self, raw: dict[str, Any]) -> UnifiedRecord | None:
        if self._is_explorer_format(raw):
            return self._explorer_to_unified(raw)
        return self._feed_to_unified(raw)

    def _explorer_to_unified(self, raw: dict[str, Any]) -> UnifiedRecord | None:
        """Convert a flat release/explorer-format record."""
        claim_text = _to_text(raw.get("claim_text"))
        fact_checker = raw.get("fact_checker") if isinstance(raw.get("fact_checker"), dict) else {}
        reviews = _as_list(raw.get("reviews"))
        first_review = _first_dict(reviews)

        publisher = raw.get("review_publisher") or fact_checker.get("name")
        review_url = raw.get("review_url")
        raw_rating = (
            raw.get("label")
            or raw.get("review_rating")
            or first_review.get("label")
            or first_review.get("review_rating")
            or first_review.get("original_label")
        )
        normalized_bucket = _normalize_rating(raw_rating)

        mapped = _RATING_TO_BINARY.get(normalized_bucket)
        if mapped is None:
            return None  # Excluded per label-mapping.md
        mapped_label, mapped_label_name = mapped
        cleaning_notes = [
            f"Label mapped: {raw_rating or '(missing)'} ({normalized_bucket}) -> {mapped_label_name}"
        ]

        review_title = _to_text(
            raw.get("review_title")
            or first_review.get("title")
            or first_review.get("name")
        )
        claim_date = raw.get("claim_date") or first_review.get("date_published")

        return UnifiedRecord(
            dataset="ClaimReview",
            sample_id=_stable_id(claim_text, review_url, publisher, claim_date),
            text=claim_text,
            original_label=raw_rating or "(missing)",
            original_label_name=normalized_bucket,
            mapped_label=mapped_label,
            mapped_label_name=mapped_label_name,
            split=None,
            source_fields_used=["claim_text"],
            context_text=review_title if review_title else None,
            modality="text",
            has_image=False,
            metadata={
                "review_url": review_url,
                "review_publisher": publisher,
                "review_title": review_title,
                "claim_date": claim_date,
                "claimant": raw.get("claimant") or first_review.get("claimant"),
                "tags": raw.get("tags"),
                "source": raw.get("source") or first_review.get("retrieved_by"),
                "language_filter": raw.get("language_filter"),
                "fact_checker_country": fact_checker.get("country"),
                "fact_checker_language": fact_checker.get("language"),
                "snapshot_version": self._snapshot_version,
                "rating_bucket": normalized_bucket,
            },
            cleaning_notes=cleaning_notes,
        )

    def _feed_to_unified(self, raw: dict[str, Any]) -> UnifiedRecord | None:
        """Convert a Data Commons feed-format record."""
        claim_review = _first_dict(_as_list(raw.get("item")))
        item_reviewed_raw = claim_review.get("itemReviewed")
        item_reviewed = item_reviewed_raw if isinstance(item_reviewed_raw, dict) else {}

        review_author_raw = claim_review.get("author")
        review_author = review_author_raw if isinstance(review_author_raw, dict) else {}
        review_rating_raw = claim_review.get("reviewRating")
        review_rating = review_rating_raw if isinstance(review_rating_raw, dict) else {}
        claim_author_raw = item_reviewed.get("author")
        claim_author = claim_author_raw if isinstance(claim_author_raw, dict) else {}

        claim_text = claim_review.get("claimReviewed") or ""
        publisher = review_author.get("name")
        review_url = claim_review.get("url")
        claim_date = item_reviewed.get("datePublished")
        raw_rating = review_rating.get("alternateName") or review_rating.get("ratingValue")
        normalized_bucket = _normalize_rating(raw_rating)

        mapped = _RATING_TO_BINARY.get(normalized_bucket)
        if mapped is None:
            return None  # Excluded per label-mapping.md
        mapped_label, mapped_label_name = mapped
        cleaning_notes = [
            f"Label mapped: {raw_rating or '(missing)'} ({normalized_bucket}) -> {mapped_label_name}"
        ]

        review_title = (claim_review.get("name") or claim_review.get("headline")
                        or claim_review.get("title") or "")

        return UnifiedRecord(
            dataset="ClaimReview",
            sample_id=_stable_id(claim_text, review_url, publisher, claim_date),
            text=claim_text,
            original_label=raw_rating or "(missing)",
            original_label_name=normalized_bucket,
            mapped_label=mapped_label,
            mapped_label_name=mapped_label_name,
            split=None,
            source_fields_used=["claimReviewed"],
            context_text=review_title if review_title else None,
            modality="text",
            has_image=False,
            metadata={
                "review_url": review_url,
                "review_publisher": publisher,
                "review_date": claim_review.get("reviewDate") or claim_review.get("datePublished"),
                "claim_date": claim_date,
                "claimant": claim_author.get("name"),
                "language": claim_review.get("inLanguage") or claim_review.get("languageCode"),
                "rating_bucket": normalized_bucket,
            },
            cleaning_notes=cleaning_notes,
        )

    def label_mapping(self) -> LabelMapping:
        return LabelMapping(
            dataset="ClaimReview",
            original_labels=["true", "false_or_misleading", "mixed_or_unverified",
                             "satire_or_joke", "ai_generated_or_synthetic", "other"],
            mapped_labels={
                "true": "real",
                "false_or_misleading": "fake",
                "mixed_or_unverified": "fake",
                "satire_or_joke": "fake",
                "ai_generated_or_synthetic": "fake",
            },
            excluded_labels=["other"],
            justification=(
                "ClaimReview ratings are free-text from publishers, normalized into six buckets. "
                "'true' maps to real; all verifiable misinformation categories map to fake; "
                "'other' is excluded as too ambiguous for reliable binary assignment."
            ),
        )
