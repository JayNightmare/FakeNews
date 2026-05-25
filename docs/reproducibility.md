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

## Reproducible context-ablation run

To measure how predictions change as context is removed, keep the model ID fixed and run all context budgets in one command:

```bash
export HF_MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct

python3 src/run_experiment.py \
  --dataset claimreview \
  --context-mode full \
  --context-ablation-levels 1.0,0.75,0.5,0.25 \
  --mode huggingface \
  --output-dir artifacts/context_ablation
```

This writes the usual run artifacts plus `by_context_budget` and `context_ablation` sections in `evaluation_report.json`.

## Reproducible grounded run

If you need Google Fact Check grounding, fix both the model ID and the cache TTL, and reuse the same cache path when comparing runs:

```bash
export HF_MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct
export GOOGLE_FACTCHECK_API_KEY=your-api-key

python3 src/run_experiment.py \
  --dataset claimreview \
  --context-mode full \
  --mode huggingface \
  --ground-with-google \
  --google-factcheck-cache artifacts/grounded_run/google_factcheck_cache.json \
  --google-factcheck-ttl-hours 24 \
  --output-dir artifacts/grounded_run
```

This adds `grounding_report.json` and a reusable cache file to the run output.

## Reproducible adapter training

Prepare-only export is the cheapest reproducibility checkpoint because it avoids a full model load. Once the exported corpus looks correct, the matching training command is:

```bash
export HF_MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct

python3 src/train_hf_adapter.py \
  --dataset claimreview \
  --model-id "$HF_MODEL_ID" \
  --output-dir artifacts/training/qwen_adapter
```

For training comparisons, keep `--model-id`, `--context-mode`, `--context-budget`, `--eval-ratio`, and `--seed` fixed, and compare the emitted `training_manifest.json` files before comparing adapter metrics.

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

Grounded runs additionally write:

- `grounding_report.json`
- `google_factcheck_cache.json` or the file given via `--google-factcheck-cache`

Training runs additionally write:

- `training_data/train_examples.jsonl`
- `training_data/eval_examples.jsonl`
- `training_data/training_manifest.json`
- `training_summary.json`

## Optional remote-model run

If you later want a remote API-backed comparison run:

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
