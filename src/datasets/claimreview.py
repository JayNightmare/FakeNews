"""ClaimReview dataset loader.

Refactored from the original run_experiment.py logic to conform to the
DatasetLoader interface. Downloads from the public Data Commons daily
ClaimReview feed or loads from a local JSON file.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from src.datasets.base import DatasetLoader
from src.schema import LabelMapping, UnifiedRecord

DEFAULT_FEED_URL = "https://storage.googleapis.com/datacommons-feeds/claimreview/latest/data.json"


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
    """Loads misinformation records from the Google/Data Commons ClaimReview feed."""

    def __init__(self, feed_url: str = DEFAULT_FEED_URL) -> None:
        self._feed_url = feed_url

    @property
    def name(self) -> str:
        return "ClaimReview"

    @property
    def default_data_dir(self) -> str:
        return "claimreview"

    def load_raw(self, data_dir: Path) -> list[dict[str, Any]]:
        """Load from local files, falling back to remote feed.

        Supports three formats:
        1. Data Commons feed format (dict with 'dataFeedElement')
        2. Explorer sample format (flat list of dicts with 'claim_text')
        3. Any JSON/JSONL files in the data directory
        """
        records: list[dict[str, Any]] = []

        # Try data.json first (canonical file)
        local_file = data_dir / "data.json"
        if local_file.exists():
            payload = json.loads(local_file.read_text(encoding="utf-8"))
            records = self._parse_payload(payload)

        # Scan for other JSON/JSONL files if data.json didn't yield results
        if not records and data_dir.is_dir():
            for json_file in sorted(data_dir.glob("*.json")):
                if json_file.name == "data.json":
                    continue
                try:
                    payload = json.loads(json_file.read_text(encoding="utf-8"))
                    records.extend(self._parse_payload(payload))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
            for jsonl_file in sorted(data_dir.glob("*.jsonl")):
                try:
                    with jsonl_file.open(encoding="utf-8") as fh:
                        for line in fh:
                            line = line.strip()
                            if line:
                                records.append(json.loads(line))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue

        # Fallback: fetch from remote feed
        if not records:
            try:
                request = Request(self._feed_url,
                                  headers={"User-Agent": "openclaw-google-factcheck-pipeline/1.0"})
                with urlopen(request) as response:  # nosec B310
                    payload = json.load(response)
                data_dir.mkdir(parents=True, exist_ok=True)
                local_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                                      encoding="utf-8")
                records = self._parse_payload(payload)
            except Exception as exc:
                raise FileNotFoundError(
                    f"No ClaimReview data found in {data_dir} and remote fetch failed: {exc}"
                ) from exc

        return records

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
        """Detect if a record is from the Explorer UI (flat format)."""
        return "claim_text" in raw and "item" not in raw

    def to_unified(self, raw: dict[str, Any]) -> UnifiedRecord | None:
        if self._is_explorer_format(raw):
            return self._explorer_to_unified(raw)
        return self._feed_to_unified(raw)

    def _explorer_to_unified(self, raw: dict[str, Any]) -> UnifiedRecord | None:
        """Convert a flat Explorer-format record."""
        claim_text = raw.get("claim_text") or ""
        publisher = raw.get("review_publisher")
        review_url = raw.get("review_url")
        raw_rating = raw.get("review_rating")
        normalized_bucket = _normalize_rating(raw_rating)

        mapped = _RATING_TO_BINARY.get(normalized_bucket)
        if mapped is None:
            return None  # Excluded per label-mapping.md
        mapped_label, mapped_label_name = mapped
        cleaning_notes: list[str] = []

        review_title = raw.get("review_title") or ""

        return UnifiedRecord(
            dataset="ClaimReview",
            sample_id=_stable_id(claim_text, review_url, publisher, None),
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
                "claimant": raw.get("claimant"),
                "tags": raw.get("tags"),
                "source": raw.get("source"),
                "language_filter": raw.get("language_filter"),
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
        cleaning_notes: list[str] = []

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
