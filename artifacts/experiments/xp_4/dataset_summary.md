# Dataset Summary: master_dataset

**Samples:** 1000
  (before cleaning: 1000, retention: 1.0)

## Label Distribution (mapped)

| Label | Count |
|---|---|
| fake | 803 |
| real | 197 |

## Original Label Distribution

| Label | Count |
|---|---|
| false_or_misleading | 484 |
| mixed_or_unverified | 190 |
| true | 144 |
| manipulated content | 64 |
| real | 53 |
| imposter content | 29 |
| misleading content | 26 |
| misinformation | 5 |
| fake | 5 |

## Text Fields Used

- `claim_en`
- `claim_text`
- `clean_title`
- `title`

## Metadata Fields

| Field | Records with field |
|---|---|
| `review_url` | 718 |
| `review_publisher` | 718 |
| `fact_checker_country` | 718 |
| `fact_checker_language` | 718 |
| `snapshot_version` | 718 |
| `rating_bucket` | 718 |
| `claim_date` | 644 |
| `platform` | 282 |
| `subreddit` | 219 |
| `score` | 219 |
| `num_comments` | 219 |
| `created_utc` | 219 |
| `domain` | 219 |
| `image_url` | 217 |
| `author` | 205 |
| `source` | 58 |
| `tweet_num` | 58 |
| `url` | 57 |
| `source_domain` | 57 |
| `language` | 5 |

## Missing Data

| Field | Missing count |
|---|---|
| `context_text` | 1000 |
| `split` | 776 |

## Text Length Stats

- **min:** 17
- **max:** 4096
- **mean:** 96.8
- **median:** 74

**Records with context:** 0
**Records with image:** 219
**Balance ratio (min/max label):** 0.245
