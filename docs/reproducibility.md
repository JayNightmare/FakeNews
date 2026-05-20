# Reproducibility guide

## Baseline run

The default run path does not require API keys.

```bash
# 1. Fetch datasets
python src/datasets/fetch_datasets.py

# 2. Run deterministic baseline
python3 src/run_experiment.py \
  --dataset all \
  --balanced \
  --limit 100 \
  --mode heuristic \
  --output-dir artifacts/pilot_run
```

## What this guarantees

- same source endpoint
- same normalization logic
- same prompt template
- same deterministic heuristic baseline
- same artifact structure

This makes the base run easy to rerun, compare, and inspect.

Fetched datasets are stored under `src/datasets/data/` unless you override `--data-dir`.

## Saved artifacts

Each run writes:

- `normalized_records.json`
- `normalized_records.jsonl`
- `normalized_records.csv`
- `prompts.jsonl`
- `predictions.jsonl`
- `evaluation_report.json`
- `evaluation_report.md`
- `dataset_summary.json`
- `dataset_summary.md`
- `cleaning_report.json`
- `run_manifest.json`
- `run_summary.json`

## Optional remote-model run

If you later want a model-backed experiment:

```bash
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=https://api.openai.com/v1
export OPENAI_MODEL=gpt-4o-mini

python3 src/run_experiment.py \
  --limit 100 \
  --mode openai-compatible \
  --output-dir artifacts/pilot_run_llm
```

## Recommended presentation demo flow

1. Show the repository structure.
2. Show the command above.
3. Explain that the heuristic mode is deterministic and works without keys.
4. Open one normalized record, one prompt, one prediction, and the evaluation report.
5. Explain that the API-backed mode is an extension, not a prerequisite.
