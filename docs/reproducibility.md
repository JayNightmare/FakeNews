# Reproducibility guide

## Baseline run

The default run path now uses a local Hugging Face model instead of the removed heuristic baseline.

```bash
# 1. Fetch datasets
python src/datasets/fetch_datasets.py

# 2. Run an open-source pilot
export HF_MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct

python3 src/run_experiment.py \
  --dataset all \
  --balanced \
  --limit 100 \
  --mode huggingface \
  --output-dir artifacts/pilot_run
```

## What this guarantees

- same source endpoint
- same normalization logic
- same prompt template
- same model identifier and prompt contract
- same artifact structure

This makes the base run easy to rerun, compare, and inspect as long as the same model weights are available locally.

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
- `visualizations/dashboard.html`
- `visualizations/visualization_report.md`
- `cleaning_report.json`
- `run_manifest.json`
- `run_summary.json`

## Deterministic training corpus export

You can export a stable train/eval corpus for adapter tuning without starting training:

```bash
export HF_MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct

python3 src/train_hf_adapter.py \
  --dataset claimreview \
  --model-id "$HF_MODEL_ID" \
  --prepare-only \
  --output-dir artifacts/training/qwen_adapter
```

This writes deterministic chat-format train/eval JSONL files and a manifest that captures the split ratio, context mode, and context budget.

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
3. Explain that the primary path is now open-source and reproducible through explicit model IDs, manifests, and cached grounding.
4. Open one normalized record, one prompt, one prediction, and the visualization report.
5. Explain that adapter training and API-backed mode are extensions, not prerequisites.
