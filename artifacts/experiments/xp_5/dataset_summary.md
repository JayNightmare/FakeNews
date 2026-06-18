# Dataset Summary: master_dataset

**Samples:** 1000
  (before cleaning: 1000, retention: 1.0)

## Label Distribution (mapped)

| Label | Count |
|---|---|
| fake | 788 |
| real | 212 |

## Original Label Distribution

| Label | Count |
|---|---|
| false_or_misleading | 482 |
| true | 171 |
| mixed_or_unverified | 170 |
| manipulated content | 50 |
| real | 41 |
| imposter content | 35 |
| misleading content | 29 |
| fake | 19 |
| misinformation | 3 |

## Text Fields Used

- `claim_en`
- `claim_text`
- `clean_title`
- `title`

## Metadata Fields

| Field | Records with field |
|---|---|
| `review_url` | 704 |
| `review_publisher` | 704 |
| `fact_checker_country` | 704 |
| `fact_checker_language` | 704 |
| `snapshot_version` | 704 |
| `rating_bucket` | 704 |
| `claim_date` | 639 |
| `platform` | 296 |
| `subreddit` | 233 |
| `score` | 233 |
| `num_comments` | 233 |
| `created_utc` | 233 |
| `domain` | 233 |
| `image_url` | 232 |
| `author` | 225 |
| `source` | 60 |
| `tweet_num` | 60 |
| `url` | 58 |
| `source_domain` | 58 |
| `language` | 3 |

## Missing Data

| Field | Missing count |
|---|---|
| `context_text` | 1000 |
| `split` | 764 |

## Text Length Stats

- **min:** 21
- **max:** 11528
- **mean:** 103.1
- **median:** 74

**Records with context:** 0
**Records with image:** 233
**Balance ratio (min/max label):** 0.269
