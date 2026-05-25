# Presentation brief

## Best file to open live

Open:

- `slides/presentation.html`

## Demo backup files

If asked for concrete evidence, show:

- `artifacts/manual_pilot/normalized_records.sample.json`
- `artifacts/manual_pilot/predictions.sample.jsonl`
- `artifacts/manual_pilot/quality_report.sample.json` (legacy sample report; current live runs write `evaluation_report.json`)
- `src/run_experiment.py`

## One-sentence pitch

"We built a reproducible multi-dataset misinformation pipeline that now runs open-source Hugging Face models, tests how much context can be removed before predictions flip, and can ground runs against Google fact-check results while still saving fully inspectable artifacts."

## If asked about limitations

Say:

- Datasets like Fakeddit and FakeNewsNet must be acquired separately (e.g., via our HuggingFace fetch script).
- Google/Data Commons gives structured fact-check metadata, not the full article body.
- Adapter fine-tuning supervision is currently deterministic and reproducible, but the explanations are not yet human-quality rationales.
- We deliberately built the scaffold to support future multimodal (image) extensions and stronger calibration/evaluation loops.
