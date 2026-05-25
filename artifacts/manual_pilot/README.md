# Manual pilot artifacts

These files are presentation-ready sample artifacts prepared from the accessible Google Fact Check Explorer sample.

## Included

- `normalized_records.sample.json`
- `predictions.sample.jsonl`
- `quality_report.sample.json` (legacy sample artifact analogous to `evaluation_report.json`)

## Why these exist

The session could not execute the local Python CLI because `exec` remained approval-blocked here, so these files show the expected artifact shape for a small pilot while the code path is still ready to run locally.

## How to upgrade this into a live run

```bash
export HF_MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct

python3 src/run_experiment.py \
  --limit 100 \
  --mode huggingface \
  --output-dir artifacts/pilot_run
```
