# Talk track

## 1) Opening

"The project goal was to establish a fully functional and reproducible experimental pipeline for misinformation data."

## 2) Why the plan changed

"The original X/Twitter-style path was not a good fit because it introduces cost and reproducibility issues. So the project pivots to Google's fact-check ecosystem instead."

## 3) Why Google / Data Commons

"This path gives structured ClaimReview data, a public daily feed, and a cleaner baseline for repeatable runs."

## 4) What the pipeline actually does

"The pipeline downloads data, normalizes it, builds prompt-ready records, runs a deterministic baseline or optional API-backed model, and saves every stage as explicit artifacts."

## 5) What was verified

"I verified the available metadata, the feed shape, the Explorer topic spread, and built a local project structure around that."

## 6) Biggest caveat

"The base Google/Data Commons path gives structured metadata, not the full fact-check article body. So richer evidence gathering would be a later enrichment step."

## 7) Closing line

"So the outcome is not just a dataset choice, but a reproducible pipeline that can be rerun, extended, and compared across future experiments."
