# Reproducibility guide

## Baseline run

The default run path does not require API keys.

```bash
# 1. Fetch datasets
python fetch_datasets.py

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

## Saved artifacts

Each run writes:

- `raw_feed_sample.json`
- `normalized_records.json`
- `normalized_records.jsonl`
- `normalized_records.csv`
- `prompts.jsonl`
- `predictions.jsonl`
- `quality_report.json`
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
4. Open one normalized record, one prompt, one prediction, and the quality report.
5. Explain that the API-backed mode is an extension, not a prerequisite.

## Current session note

The code and docs were created successfully in this session and read back for verification.
Local execution from this session remained blocked by the environment's exec approval state, so runtime verification should be done by running the command above in a terminal when convenient.
