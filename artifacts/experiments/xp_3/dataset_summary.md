# Dataset Summary: master_dataset

**Samples:** 1000
  (before cleaning: 1000, retention: 1.0)

## Label Distribution (mapped)

| Label | Count |
|---|---|
| fake | 823 |
| real | 177 |

## Original Label Distribution

| Label | Count |
|---|---|
| false_or_misleading | 505 |
| mixed_or_unverified | 153 |
| true | 142 |
| manipulated content | 70 |
| misleading content | 39 |
| imposter content | 37 |
| real | 35 |
| fake | 15 |
| misinformation | 4 |

## Text Fields Used

- `claim_en`
- `claim_text`
- `clean_title`
- `title`

## Metadata Fields

| Field | Records with field |
|---|---|
| `review_url` | 697 |
| `review_publisher` | 697 |
| `fact_checker_country` | 697 |
| `fact_checker_language` | 697 |
| `snapshot_version` | 697 |
| `rating_bucket` | 697 |
| `claim_date` | 628 |
| `platform` | 303 |
| `subreddit` | 249 |
| `score` | 249 |
| `num_comments` | 249 |
| `created_utc` | 249 |
| `domain` | 249 |
| `image_url` | 248 |
| `author` | 235 |
| `source` | 50 |
| `tweet_num` | 50 |
| `url` | 49 |
| `source_domain` | 49 |
| `language` | 4 |

## Missing Data

| Field | Missing count |
|---|---|
| `context_text` | 1000 |
| `split` | 747 |

## Text Length Stats

- **min:** 22
- **max:** 7034
- **mean:** 99.4
- **median:** 72

**Records with context:** 0
**Records with image:** 249
**Balance ratio (min/max label):** 0.215
