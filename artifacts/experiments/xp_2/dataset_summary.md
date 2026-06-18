# Dataset Summary: master_dataset

**Samples:** 100
  (before cleaning: 100, retention: 1.0)

## Label Distribution (mapped)

| Label | Count |
|---|---|
| fake | 80 |
| real | 20 |

## Original Label Distribution

| Label | Count |
|---|---|
| false_or_misleading | 39 |
| mixed_or_unverified | 20 |
| true | 17 |
| manipulated content | 10 |
| fake | 5 |
| misleading content | 4 |
| real | 3 |
| imposter content | 2 |

## Text Fields Used

- `claim_text`
- `clean_title`
- `title`

## Metadata Fields

| Field | Records with field |
|---|---|
| `review_url` | 65 |
| `review_publisher` | 65 |
| `fact_checker_country` | 65 |
| `fact_checker_language` | 65 |
| `snapshot_version` | 65 |
| `rating_bucket` | 65 |
| `claim_date` | 59 |
| `platform` | 35 |
| `subreddit` | 27 |
| `score` | 27 |
| `num_comments` | 27 |
| `created_utc` | 27 |
| `domain` | 27 |
| `image_url` | 27 |
| `author` | 26 |
| `source` | 8 |
| `url` | 8 |
| `source_domain` | 8 |
| `tweet_num` | 8 |

## Missing Data

| Field | Missing count |
|---|---|
| `context_text` | 100 |
| `split` | 73 |

## Text Length Stats

- **min:** 25
- **max:** 407
- **mean:** 90.1
- **median:** 73

**Records with context:** 0
**Records with image:** 27
**Balance ratio (min/max label):** 0.25
