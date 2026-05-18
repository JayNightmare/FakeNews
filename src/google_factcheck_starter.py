#!/usr/bin/env python3
"""Simple starter downloader/normalizer for Google's public ClaimReview feed.

No third-party dependencies.

Usage:
    python3 src/google_factcheck_starter.py --limit 100
    python3 src/google_factcheck_starter.py --limit 200 --output-dir output
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any
from urllib.request import urlopen

DEFAULT_FEED_URL = "https://storage.googleapis.com/datacommons-feeds/claimreview/latest/data.json"


def fetch_json(url: str) -> dict[str, Any]:
    with urlopen(url) as response:  # nosec B310 - trusted public Google endpoint for this starter
        return json.load(response)


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def first_dict(items: list[Any]) -> dict[str, Any]:
    for item in items:
        if isinstance(item, dict):
            return item
    return {}


def normalize_feed_item(feed_item: dict[str, Any]) -> dict[str, Any]:
    claim_review = first_dict(as_list(feed_item.get("item")))
    item_reviewed = claim_review.get("itemReviewed") if isinstance(claim_review.get("itemReviewed"), dict) else {}
    item_reviewed = item_reviewed or {}

    author = claim_review.get("author") if isinstance(claim_review.get("author"), dict) else {}
    review_rating = claim_review.get("reviewRating") if isinstance(claim_review.get("reviewRating"), dict) else {}
    claim_author = item_reviewed.get("author") if isinstance(item_reviewed.get("author"), dict) else {}
    first_appearance = item_reviewed.get("firstAppearance") if isinstance(item_reviewed.get("firstAppearance"), dict) else {}

    appearance_urls = []
    for appearance in as_list(item_reviewed.get("appearance")):
        if isinstance(appearance, dict) and appearance.get("url"):
            appearance_urls.append(appearance["url"])

    if first_appearance.get("url") and first_appearance["url"] not in appearance_urls:
        appearance_urls.insert(0, first_appearance["url"])

    normalized = {
        "feed_item_url": feed_item.get("url"),
        "feed_item_date_created": feed_item.get("dateCreated"),
        "claim_text": claim_review.get("claimReviewed"),
        "claimant": claim_author.get("name"),
        "claim_date": item_reviewed.get("datePublished"),
        "claim_first_appearance_url": first_appearance.get("url"),
        "claim_appearance_urls": appearance_urls,
        "review_url": claim_review.get("url"),
        "review_title": claim_review.get("name") or claim_review.get("headline") or claim_review.get("title"),
        "review_date": claim_review.get("reviewDate") or claim_review.get("datePublished"),
        "review_rating": review_rating.get("alternateName") or review_rating.get("ratingValue"),
        "review_publisher": author.get("name"),
        "review_publisher_url": author.get("url"),
        "language": claim_review.get("inLanguage") or claim_review.get("languageCode"),
        "sd_publisher": (claim_review.get("sdPublisher") or {}).get("name") if isinstance(claim_review.get("sdPublisher"), dict) else None,
        "raw": claim_review,
    }
    return normalized


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flat_rows = []
    for row in rows:
        flat_row = dict(row)
        flat_row["claim_appearance_urls"] = " | ".join(row.get("claim_appearance_urls") or [])
        flat_row.pop("raw", None)
        flat_rows.append(flat_row)

    if not flat_rows:
        return

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0].keys()))
        writer.writeheader()
        writer.writerows(flat_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and normalize Google's public ClaimReview feed")
    parser.add_argument("--url", default=DEFAULT_FEED_URL, help="Feed URL to fetch")
    parser.add_argument("--limit", type=int, default=100, help="Number of items to keep")
    parser.add_argument("--output-dir", default="output", help="Where to write results")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    payload = fetch_json(args.url)
    feed_items = as_list(payload.get("dataFeedElement"))[: args.limit]
    normalized = [normalize_feed_item(item) for item in feed_items if isinstance(item, dict)]

    write_json(output_dir / "claimreview_raw_sample.json", {"source_url": args.url, "items": feed_items})
    write_json(output_dir / "claimreview_normalized_sample.json", normalized)
    write_jsonl(output_dir / "claimreview_normalized_sample.jsonl", normalized)
    write_csv(output_dir / "claimreview_normalized_sample.csv", normalized)

    summary = {
        "source_url": args.url,
        "requested_limit": args.limit,
        "saved_items": len(normalized),
        "output_dir": str(output_dir),
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
