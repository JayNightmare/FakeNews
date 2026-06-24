# Claude Development Guide

This repository is a research pipeline for testing whether open-source models can classify misinformation from different amounts and qualities of context. Optimize for reproducible experiments and clear artifacts, not production-style precision gates.

## Quick Start

```bash
pip install -r requirements.txt
python src/datasets/fetch_datasets.py
python src/build_master_dataset.py --dataset all --output-dir artifacts/dataset/master_dataset
```

The cleaned split datasets used by adapter training live in:

- `src/datasets/data/cleaned/train`
- `src/datasets/data/cleaned/validation`
- `src/datasets/data/cleaned/test`

Prepare a corpus without loading a model:

```bash
python3 src/train_hf_adapter.py \
  --train-records-dir src/datasets/data/cleaned/train \
  --validation-records-dir src/datasets/data/cleaned/validation \
  --test-records-dir src/datasets/data/cleaned/test \
  --model-id "$HF_MODEL_ID" \
  --prepare-only \
  --output-dir artifacts/training/qwen_split_prepare
```

Run a tiny smoke training job before full split training:

```bash
python3 src/train_hf_adapter.py \
  --train-records-dir src/datasets/data/cleaned/train \
  --validation-records-dir src/datasets/data/cleaned/validation \
  --test-records-dir src/datasets/data/cleaned/test \
  --model-id Qwen/Qwen2.5-0.5B-Instruct \
  --output-dir artifacts/training/qwen_tiny_smoke \
  --limit 4 \
  --num-train-epochs 1 \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps 1 \
  --logging-steps 1 \
  --max-length 256 \
  --lora-rank 4 \
  --lora-alpha 8 \
  --gate-policy skip
```

## Architecture

```text
fetch datasets -> build master dataset -> split cleaned data
      -> export chat training corpus -> fine-tune LoRA adapter
      -> score train/validation/test -> write reports and bundle artifacts
```

Key modules:

- `src/schema.py`: canonical `UnifiedRecord` and master export schemas.
- `src/datasets/`: source-specific loaders and cleaned split export.
- `src/prompts.py`: context-mode prompts for `minimal`, `full`, and `misleading` runs.
- `src/training.py`: deterministic chat corpus export and tokenization helpers.
- `src/train_hf_adapter.py`: PEFT/LoRA trainer, evaluation reports, and model bundle generation.
- `src/predictors.py`: inference backends, including Hugging Face text and image-text models.
- `src/evaluation.py`: accuracy/F1/confidence/context-ablation reporting.

## Research Workflow Decisions

- Train one adapter per base model. LoRA adapters are not generally portable across arbitrary model families or text-only versus image-text-to-text architectures.
- Validation/test gates are diagnostics for research unless explicitly configured as blocking. Failed thresholds should be recorded, not treated as evidence the experiment is useless.
- Current vision-language training is text-only: the trainer uses the processor tokenizer, but does not feed image tensors during fine-tuning. Hugging Face inference can use images when `HF_ENABLE_IMAGES=1` and records contain image URLs.
- Compare context effects by holding `model-id`, split, seed, and training settings fixed while varying `context-mode` and `context-budget`.

## Artifact Conventions

Use output paths that encode the model and context condition, for example:

```text
artifacts/training/qwen2_5_1_5b/full/budget_1_0
artifacts/training/qwen2_5_1_5b/minimal/budget_1_0
artifacts/training/qwen2_vl_7b/full/budget_0_5
```

Training outputs include `training_summary.json`, `training_data/training_manifest.json`, optional train/validation/test reports, `gate_report.json`, and model bundles under `artifacts/FakeNews_Model` or a custom `--model-folder`.

## Troubleshooting

- Long first epoch: the master split is large. With batch size 1 and gradient accumulation 8, each optimization step represents about 8 examples. Use `--limit` for smoke runs.
- `avg_train_loss=nan`: treat this as a trainer bug or numeric instability. Training should stop immediately with a finite-loss error rather than continue logging `nan`.
- Missing cleaned splits: rebuild the master dataset and split outputs before running split-aware training.
- Empty `--model-id "$HF_MODEL_ID"`: set `HF_MODEL_ID` in the shell or repository `.env`; the trainer can fall back to the environment when shell expansion produces an empty value.
- Hugging Face rate limits: set `HF_TOKEN` for authenticated downloads and pushes.
- GPU memory pressure: use a smaller model, lower `--max-length`, lower LoRA rank, use `--prepare-only`, or move to a proper distributed setup. This manual trainer does not support sharded/offloaded backprop.

## Validation Commands

```bash
python -m pytest tests/test_smoke.py -v

python3 src/train_hf_adapter.py \
  --train-records-dir src/datasets/data/cleaned/train \
  --validation-records-dir src/datasets/data/cleaned/validation \
  --test-records-dir src/datasets/data/cleaned/test \
  --model-id "$HF_MODEL_ID" \
  --prepare-only \
  --limit 8 \
  --output-dir artifacts/training/prepare_smoke
```

Keep generated artifacts out of source changes unless the user explicitly asks to commit them.
