# Presentation brief

## Best file to open live

Open:

- `slides/presentation.html`

## Demo backup files

If asked for concrete evidence, show:

- `artifacts/manual_pilot/normalized_records.sample.json`
- `artifacts/manual_pilot/predictions.sample.jsonl`
- `artifacts/manual_pilot/quality_report.sample.json`
- `src/run_experiment.py`

## One-sentence pitch

"We replaced an expensive and fragile X-dependent workflow with a reproducible multi-dataset pipeline (ClaimReview, Fakeddit, FakeNewsNet, MuMiN) that standardizes ingestion, token-cost tracking, balanced sampling, and saves inspectable artifacts."

## If asked about limitations

Say:

- Datasets like Fakeddit and FakeNewsNet must be acquired separately (e.g., via our HuggingFace fetch script).
- The heuristic baseline is purely for pipeline plumbing validation, not a research result.
- We deliberately built the scaffold to support future multimodal (image) extensions and stronger local model backends.
