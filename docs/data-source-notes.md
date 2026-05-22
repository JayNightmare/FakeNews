# Google Fact Check data source notes

## 1) Google Fact Check Tools API

Official docs: <https://developers.google.com/fact-check/tools/api/>

The Claim Search API is the official way to query the same fact-check result pool exposed by the Explorer UI.

According to the docs, useful fields include:

- claim text
- claimant
- claim date
- claim reviews
- review publisher
- review URL
- review title
- review date
- textual rating
- language code

## 2) Data Commons research dataset / daily feed

Research dataset page: <https://datacommons.org/factcheck/download>
Daily feed: <https://storage.googleapis.com/datacommons-feeds/claimreview/latest/data.json>

Important caveat from Google/Data Commons:

- the release contains structured `ClaimReview` markup
- the original fact-check article content is **not** included
- the `url` field points back to the publisher article
- the daily feed is refreshed regularly

## 3) What we saw in the live feed

The live JSON feed includes fields like:

- `claimReviewed`
- `itemReviewed.author.name`
- `itemReviewed.datePublished`
- `itemReviewed.firstAppearance.url` or `itemReviewed.appearance[].url`
- `reviewRating.alternateName`
- `author.name` / `author.url`
- `url`
- `datePublished` on the review object in some entries
- `sdPublisher`

This is enough to build an initial metadata-driven misinformation pipeline.

## 4) What we saw in the Explorer UI

The recent-results page had varied examples spanning:

- politics / elections
- health / medical misinformation
- AI-generated or manipulated images
- consumer / brand claims
- transport / public-policy claims

The Explorer UI also exposes useful tag/topic buttons, for example people, places, topics, and publishers.

## 5) Recommendation for the project

Best low-friction pilot:

1. Start with the daily feed.
2. Normalize 100-200 entries into JSONL/CSV.
3. Measure data quality.
4. Only add article-body scraping if your downstream prompts truly need full article text.

This keeps the project moving without paying for X hydration.

## 6) CSV-backed dataset mirrors used in the pipeline

The current `src/datasets/fetch_datasets.py` workflow does not only fetch ClaimReview.
It also writes CSV-backed local mirrors for FakeNewsNet and MuMiN under `src/datasets/data/`.

### FakeNewsNet

- The upstream FakeNewsNet repository distributes separate `politifact_real.csv`, `politifact_fake.csv`, `gossipcop_real.csv`, and `gossipcop_fake.csv` files.
- The Hugging Face mirror used by this project exposes a merged CSV with a `real` column.
- The pipeline interprets that field as `1 => real` and `0 => fake`, which is consistent with the upstream split naming.

### MuMiN

- The fetched MuMiN mirror currently arrives as CSV rather than JSONL.
- The pipeline reads `claim_en` when available, keeps `verdict` as the native label, and derives split assignment from `train_mask`, `val_mask`, and `test_mask`.
- This is a practical export-driven path, distinct from the full native MuMiN hydration workflow.
