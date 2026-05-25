"""Google Fact Check Tools integration with a local JSON cache."""

from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from src.schema import UnifiedRecord

DEFAULT_FACTCHECK_SEARCH_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _cache_key(claim_text: str) -> str:
    return " ".join(claim_text.strip().casefold().split())


def _default_fetch_json(url: str) -> dict[str, Any]:
    request = Request(url, headers={"User-Agent": "google-factcheck-starter/1.0"})
    with urlopen(request) as response:  # nosec B310
        return json.load(response)


class GoogleFactCheckCache:
    """Simple JSON cache keyed by normalized claim text."""

    def __init__(self, path: Path, ttl_hours: int = 24) -> None:
        self._path = path
        self._ttl = dt.timedelta(hours=ttl_hours)
        self._entries: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        payload = json.loads(self._path.read_text(encoding="utf-8"))
        entries = payload.get("entries", {}) if isinstance(payload, dict) else {}
        if isinstance(entries, dict):
            self._entries = entries

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "saved_at": _utc_now().isoformat(),
            "entries": self._entries,
        }
        self._path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def get(self, claim_text: str, now: dt.datetime | None = None) -> dict[str, Any] | None:
        now = now or _utc_now()
        entry = self._entries.get(_cache_key(claim_text))
        if not entry:
            return None
        fetched_at = entry.get("fetched_at")
        if not fetched_at:
            return None
        age = now - dt.datetime.fromisoformat(fetched_at)
        if age > self._ttl:
            return None
        return entry.get("response")

    def set(self, claim_text: str, response: dict[str, Any], now: dt.datetime | None = None) -> None:
        now = now or _utc_now()
        self._entries[_cache_key(claim_text)] = {
            "claim_text": claim_text,
            "fetched_at": now.isoformat(),
            "response": response,
        }


class GoogleFactCheckClient:
    """Thin wrapper around the Google Fact Check Tools claims search API."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        endpoint: str = DEFAULT_FACTCHECK_SEARCH_URL,
        fetch_json: Callable[[str], dict[str, Any]] | None = None,
    ) -> None:
        self._api_key = api_key or os.getenv("GOOGLE_FACTCHECK_API_KEY")
        self._endpoint = endpoint
        self._fetch_json = fetch_json or _default_fetch_json

    @property
    def is_configured(self) -> bool:
        return bool(self._api_key)

    def search_claim(self, claim_text: str, *, page_size: int = 5, language_code: str | None = None) -> dict[str, Any]:
        if not self._api_key:
            raise RuntimeError("GOOGLE_FACTCHECK_API_KEY is required for live Google fact-check lookups")
        query = {
            "query": claim_text,
            "pageSize": page_size,
            "key": self._api_key,
        }
        if language_code:
            query["languageCode"] = language_code
        url = f"{self._endpoint}?{urlencode(query)}"
        return self._fetch_json(url)


def summarize_claim_search(response: dict[str, Any]) -> dict[str, Any]:
    """Extract the top-grounding candidate from a Google API response."""
    claims = response.get("claims", []) if isinstance(response, dict) else []
    if not claims:
        return {
            "status": "no_match",
            "match_count": 0,
            "grounding_summary": None,
        }

    top_claim = claims[0] if isinstance(claims[0], dict) else {}
    reviews = top_claim.get("claimReview", []) if isinstance(top_claim.get("claimReview"), list) else []
    top_review = reviews[0] if reviews and isinstance(reviews[0], dict) else {}
    publisher = top_review.get("publisher") if isinstance(top_review.get("publisher"), dict) else {}

    review_publisher = publisher.get("name")
    review_title = top_review.get("title")
    textual_rating = top_review.get("textualRating")
    review_url = top_review.get("url")
    claim_text = top_claim.get("text")
    claimant = top_claim.get("claimant")

    summary_parts = [
        part for part in [
            f"Matched claim: {claim_text}" if claim_text else None,
            f"Claimant: {claimant}" if claimant else None,
            f"Review: {review_title}" if review_title else None,
            f"Publisher: {review_publisher}" if review_publisher else None,
            f"Rating: {textual_rating}" if textual_rating else None,
            f"Source URL: {review_url}" if review_url else None,
        ] if part
    ]

    return {
        "status": "match_found",
        "match_count": len(claims),
        "matched_claim_text": claim_text,
        "claimant": claimant,
        "review_title": review_title,
        "review_publisher": review_publisher,
        "textual_rating": textual_rating,
        "review_url": review_url,
        "grounding_summary": " | ".join(summary_parts) if summary_parts else None,
    }


def enrich_records_with_google_factcheck(
    records: list[UnifiedRecord],
    *,
    cache_path: Path,
    ttl_hours: int = 24,
    client: GoogleFactCheckClient | None = None,
) -> tuple[list[UnifiedRecord], dict[str, Any]]:
    """Enrich records with cached Google fact-check metadata."""
    client = client or GoogleFactCheckClient()
    cache = GoogleFactCheckCache(cache_path, ttl_hours=ttl_hours)

    report = {
        "enabled": True,
        "cache_path": str(cache_path),
        "ttl_hours": ttl_hours,
        "records_seen": len(records),
        "cache_hits": 0,
        "cache_misses": 0,
        "live_fetches": 0,
        "missing_api_key": 0,
        "matches_found": 0,
        "no_match": 0,
    }

    for record in records:
        cached = cache.get(record.text)
        response: dict[str, Any] | None = None
        cache_status: str

        if cached is not None:
            report["cache_hits"] += 1
            response = cached
            cache_status = "hit"
        elif client.is_configured:
            report["cache_misses"] += 1
            response = client.search_claim(record.text, language_code=record.metadata.get("language"))
            cache.set(record.text, response)
            report["live_fetches"] += 1
            cache_status = "fetched"
        else:
            report["cache_misses"] += 1
            report["missing_api_key"] += 1
            cache_status = "missing_api_key"

        summary = summarize_claim_search(response or {})
        if summary["status"] == "match_found":
            report["matches_found"] += 1
        else:
            report["no_match"] += 1

        record.metadata["google_fact_check"] = {
            **summary,
            "cache_status": cache_status,
        }

        if summary.get("grounding_summary"):
            if record.context_text:
                record.context_text = (
                    f"{record.context_text}\n\nGoogle fact-check evidence: {summary['grounding_summary']}"
                )
            else:
                record.context_text = f"Google fact-check evidence: {summary['grounding_summary']}"

    cache.save()
    return records, report