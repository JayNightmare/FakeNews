# Normalized schema proposal

This is a practical starter schema for the misinformation project.

## Goal

Turn each fact-check instance into a stable JSON record that can be:

- inspected manually
- exported to CSV/JSONL
- converted into prompts for LLM evaluation
- extended later with fetched article text or image metadata

## Core fields

```json
{
	"id": "string",
	"source_collection": "google_claimreview_feed",
	"claim_text": "string | null",
	"claimant": "string | null",
	"claim_date": "string | null",
	"claim_first_appearance_url": "string | null",
	"claim_appearance_urls": ["string"],
	"review_url": "string | null",
	"review_title": "string | null",
	"review_date": "string | null",
	"review_publisher": "string | null",
	"review_publisher_url": "string | null",
	"review_rating": "string | null",
	"language": "string | null",
	"topic_tags": ["string"],
	"article_text": null,
	"notes": null
}
```

## Notes on design

- `id`: deterministic local id, for example a hash of `review_url` or `claim_text + review_url`
- `source_collection`: keeps later multi-dataset merges sane
- `claim_appearance_urls`: preserve all discovered source/appearance links
- `topic_tags`: optional; can be empty if unavailable in the API/feed
- `article_text`: keep `null` until a second-stage fetch enriches records
- `notes`: free space for manual annotations, failure reasons, or parsing comments

## Minimal required fields for first pilot

If you want the leanest possible version, keep at least:

- `claim_text`
- `review_url`
- `review_publisher`
- `review_rating`
- `language`

## Nice-to-have quality checks

For each saved record, later validate:

- claim text present?
- review URL reachable?
- rating present?
- language present?
- duplicate review URLs?
- malformed appearance URLs?

## Suggested rating normalization

Because publishers use different labels, keep both:

```json
{
	"review_rating": "Mostly False",
	"review_rating_normalized": "false_or_misleading"
}
```

Starter normalized buckets:

- `true`
- `false_or_misleading`
- `mixed_or_unverified`
- `satire_or_joke`
- `ai_generated_or_synthetic`
- `other`

Do **not** overwrite the original rating string.
