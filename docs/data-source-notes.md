# Google Fact Check data source notes

## 1) Google Fact Check Tools API

Official docs: https://developers.google.com/fact-check/tools/api/

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

Research dataset page: https://datacommons.org/factcheck/download
Daily feed: https://storage.googleapis.com/datacommons-feeds/claimreview/latest/data.json

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
