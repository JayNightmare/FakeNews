# Misinformation Detection Experimental Pipeline

Experimental pipeline for studying how contextual information affects open-source LLM performance in misinformation detection.

## Research Aim

Investigate whether increasing context consistently improves model performance, or whether excessive or misleading context degrades model reliability.

**Task:** Binary classification (real/fake) + explanation generation.

## Datasets

| Dataset         | Domain              | Modality      | Status                        |
| --------------- | ------------------- | ------------- | ----------------------------- |
| **ClaimReview** | fact-check metadata | text          | ✅ working (auto-downloads)   |
| **Fakeddit**    | Reddit posts        | text + images | ✅ working (via fetch script) |
| **FakeNewsNet** | news articles       | text + images | ✅ working (via fetch script) |
| **MuMiN**       | social media claims | text + images | ✅ working (via fetch script) |

## Project Structure

```text
src/
├── run_experiment.py          # main unified pipeline runner
├── predictors.py              # inference backend abstraction + HF/OpenAI providers
├── google_factcheck.py        # cached Google Fact Check grounding layer
├── context_ablation.py        # context-budget parsing and truncation helpers
├── schema.py                  # canonical UnifiedRecord schema
├── cleaning.py                # data cleaning pipeline
├── prompts.py                 # context-variant prompt generation
├── evaluation.py              # metrics, confusion matrix, comparison
├── training.py                # deterministic corpus export for adapter tuning
├── train_hf_adapter.py        # LoRA/PEFT training entrypoint
├── summary.py                 # dataset summary sheets
├── costs.py                   # optional cost estimation
├── google_factcheck_starter.py # original starter downloader (legacy)
└── datasets/
    ├── base.py                # abstract dataset loader
    ├── claimreview.py         # Google/Data Commons ClaimReview feed
    ├── fakeddit.py            # Fakeddit (Reddit)
    ├── fakenewsnet.py         # FakeNewsNet (PolitiFact + GossipCop)
    ├── mumin.py               # MuMiN (stub)
    └── data/                  # local dataset files (gitignored)

docs/
├── experimental-design.md     # experiment framing + context variants
├── label-mapping.md           # per-dataset label mapping + justification
├── meeting-task-checklist.md  # RA1/RA2 task status
├── normalized-schema.md       # schema design notes
├── data-source-notes.md       # data source investigation
├── reproducibility.md         # how to rerun consistently
├── visualisation.md           # embedded charts and narrative interpretation
├── presentation-brief.md      # presentation guide
└── sources.md                 # references

templates/                     # prompt templates
config/                        # configuration examples
slides/                        # presentation materials
CHANGELOG.md                   # recent history + unreleased changes
artifacts/                     # pipeline output artifacts
```

## Quick Start

### 0) Fetch Datasets

The fetcher downloads ClaimReview plus manageable sample subsets for Fakeddit, FakeNewsNet, and MuMiN into `src/datasets/data/`:

- `claimreview` is stored as the raw feed JSON payload
- `fakeddit` is stored as TSV
- `fakenewsnet` is stored as a fetched CSV export and loaded directly by the pipeline
- `mumin` is stored as a fetched CSV export and loaded directly by the pipeline

```bash
# Install required dependencies
pip install -r requirements.txt

# Run the fetcher (default: attempt full corpora, then fall back to a sampled mirror when needed)
python src/datasets/fetch_datasets.py

# Force a sampled acquisition path for a quick local refresh
python src/datasets/fetch_datasets.py --max-samples 5000
```

If the local dataset mirror already exists, the fetcher leaves it in place. For the Hugging Face-backed sources, omitting `--max-samples` attempts a full split first and only retries as a sample when that full export fails.

### 0.1) Configure environment variables

The repository reads provider credentials and model settings from process environment variables. A repository-local `.env` file is loaded automatically when present.

Create `.env` from `.env.example` and set the values you actually need for your workflow.

Common variables:

- `HF_MODEL_ID` — Hugging Face model used for local inference and adapter training
- `HF_TOKEN` — optional but recommended for higher HF Hub rate limits and faster authenticated downloads
- `HF_DEVICE_MAP` / `HF_TORCH_DTYPE` — optional local loading controls for large models
- `HF_MAX_NEW_TOKENS`, `HF_TEMPERATURE`, `HF_TOP_P` — local generation controls
- `GOOGLE_FACTCHECK_API_KEY` — required for `--ground-with-google`
- `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL` — required for `--mode openai-compatible`

If `HF_TOKEN` is unset, Hugging Face downloads still work but the Hub treats them as unauthenticated requests.

### 0.5) Build the master dataset

Use the dedicated builder to merge the currently available local mirrors into one canonical dataset plus one flat analysis export:

```bash
python3 src/build_master_dataset.py \
  --dataset all \
  --output-dir artifacts/master_dataset
```

The builder reuses the existing loaders and cleaning pipeline, then writes:

- `normalized_records.{json,jsonl,csv}` — deduplicated canonical `UnifiedRecord` corpus for experiments and training
- `master_records.{json,jsonl,csv}` — canonical merged dataset with provenance fields
- `master_flat_records.{json,jsonl,csv}` — flat export with `id`, `Title`, `Claim`, `news source`, `Real`, `tweet_num`
- `master_dataset_manifest.json` — per-dataset acquisition and row-count summary
- `master_dataset_coverage.json` — null-rate coverage for the flat export fields

Important label note:

- Internal pipeline records still use `mapped_label: 0 = real, 1 = fake`
- The flat master export uses `Real: 1 = real, 0 = fake`

### 1) Run a Hugging Face pilot

The default inference path now targets the combined master dataset instead of an individual source dataset. Qwen is the current first-wave target:

```bash
export HF_MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct

python3 src/run_experiment.py \
  --dataset master_dataset \
  --limit 100 \
  --balanced \
  --mode huggingface \
  --output-dir artifacts/master_run
```

`--dataset master_dataset` resolves the canonical builder outputs automatically. By default it prefers `artifacts/master_dataset/` and falls back to `src/datasets/data/master_dataset/` if needed.

### 2) Run with context variants

```bash
# Minimal context (claim text only)
python3 src/run_experiment.py \
  --dataset master_dataset \
  --context-mode minimal \
  --limit 100 \
  --mode huggingface \
  --output-dir artifacts/context_minimal

# Full context (claim + article + metadata)
python3 src/run_experiment.py \
  --dataset master_dataset \
  --context-mode full \
  --limit 100 \
  --mode huggingface \
  --output-dir artifacts/context_full

# Misleading context (adversarial)
python3 src/run_experiment.py \
  --dataset master_dataset \
  --context-mode misleading \
  --limit 100 \
  --mode huggingface \
  --output-dir artifacts/context_misleading
```

### 3) Run the context-ablation suite

This evaluates the same records across multiple context budgets so you can measure when predictions flip as context is removed:

```bash
export HF_MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct

python3 src/run_experiment.py \
  --dataset master_dataset \
  --context-mode full \
  --context-ablation-levels 1.0,0.75,0.5,0.25 \
  --limit 100 \
  --mode huggingface \
  --output-dir artifacts/context_ablation
```

The evaluation report now includes `by_context_budget` metrics and `context_ablation` threshold summaries.

### 4) Ground runs with Google Fact Check Tools

You can enrich records with cached Google fact-check lookups before prompt generation:

```bash
export HF_MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct
export GOOGLE_FACTCHECK_API_KEY=your-api-key

python3 src/run_experiment.py \
  --dataset master_dataset \
  --context-mode full \
  --mode huggingface \
  --ground-with-google \
  --google-factcheck-ttl-hours 24 \
  --output-dir artifacts/grounded_run
```

Each grounded run writes `grounding_report.json`, a reusable Google cache file, and enriched prompts/records when matches are found.

### 5) Prepare and fine-tune a Qwen adapter

The adapter trainer reuses the same prompt contract as inference and exports deterministic train/eval chat examples before training.

```bash
export HF_MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct

# Export train/eval corpora only
python3 src/train_hf_adapter.py \
  --dataset master_dataset \
  --model-id "$HF_MODEL_ID" \
  --prepare-only \
  --output-dir artifacts/training/qwen_adapter

# Fine-tune a LoRA adapter
python3 src/train_hf_adapter.py \
  --dataset master_dataset \
  --model-id "$HF_MODEL_ID" \
  --output-dir artifacts/training/qwen_adapter
```

For local Apple Silicon validation, prefer `--device-map mps`. CPU-only training remains available, but even tiny Qwen adapter runs can be impractically slow without MPS acceleration.

```bash
# Validated tiny local training run on Apple Silicon
python3 src/train_hf_adapter.py \
  --dataset master_dataset \
  --limit 4 \
  --context-mode minimal \
  --model-id Qwen/Qwen2.5-0.5B-Instruct \
  --output-dir artifacts/training/qwen_adapter_tiny_run_mps \
  --num-train-epochs 1 \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps 1 \
  --logging-steps 1 \
  --device-map mps \
  --max-length 256 \
  --lora-rank 4 \
  --lora-alpha 8
```

Training outputs include:

- `training_data/train_examples.jsonl`
- `training_data/eval_examples.jsonl`
- `training_data/training_manifest.json`
- `adapter/` with the saved LoRA weights and tokenizer files
- `training_summary.json`

If you want grounded training examples, add `--ground-with-google` and set `GOOGLE_FACTCHECK_API_KEY`.

### 6) Run other datasets

Per-dataset runs remain available as an explicit fallback when you want source-specific comparisons.

```bash
# Fakeddit (uses fetched data in src/datasets/data/fakeddit/)
python3 src/run_experiment.py --dataset fakeddit --limit 100 --mode huggingface

# FakeNewsNet (uses fetched data in src/datasets/data/fakenewsnet/)
python3 src/run_experiment.py --dataset fakenewsnet --limit 100 --mode huggingface

# All individual datasets as separate runs
python3 src/run_experiment.py --dataset all --limit 50 --mode huggingface
```

### 7) Optional OpenAI-compatible model run

This remains available for comparison, but it is not the primary research path.

```bash
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=https://api.openai.com/v1
export OPENAI_MODEL=gpt-4o-mini

python3 src/run_experiment.py \
  --dataset claimreview \
  --context-mode full \
  --mode openai-compatible \
  --output-dir artifacts/llm_run
```

### 8) Expected outputs per run

- `normalized_records.{json,jsonl,csv}` — cleaned unified records
- `prompts.jsonl` — generated prompts with context
- `predictions.jsonl` — model predictions
- `evaluation_report.{json,md}` — accuracy, F1, confusion matrix
- `dataset_summary.{json,md}` — sample counts, label distribution, missing data
- `visualizations/dashboard.html` — browser-friendly chart dashboard for the run
- `visualizations/visualization_report.md` — explains each chart, its source artifact, and the main trend
- `cleaning_report.json` — what was removed during cleaning
- `grounding_report.json` — Google fact-check cache and match statistics when grounding is enabled
- `run_manifest.json` — full run parameters

Training runs additionally write `training_data/` manifests and adapter checkpoints under the chosen training output directory.

For a concise summary of recent committed and unreleased changes, see `CHANGELOG.md`.

See [docs/visualisation.md](docs/visualisation.md) for a docs-native page that embeds the latest aggregate visuals and explains them.

## Context Conditions

The pipeline supports three context variants to study context sensitivity:

| Mode         | What's included                 | Tests                                           |
| ------------ | ------------------------------- | ----------------------------------------------- |
| `minimal`    | claim text only                 | can the model classify from surface-level text? |
| `full`       | claim + article body + metadata | does relevant context improve classification?   |
| `misleading` | claim + adversarial context     | does irrelevant context degrade reliability?    |

The full-context path also supports `--context-budget` and `--context-ablation-levels` so runs can quantify how much context can be removed before predictions flip.

## Reproducibility

- CLI-first, no hidden notebook state
- explicit CLI arguments and saved run manifests
- explicit model selection and context-budget settings saved in run manifests
- stable record IDs across reruns
- artifacts saved at every pipeline stage
- offline-capable token cost tracking and balanced sampling
- automated `src/datasets/fetch_datasets.py` for standardizing data ingestion under `src/datasets/data/`
- cached Google Fact Check lookups with TTL-based reuse
- deterministic train/eval corpus export for adapter fine-tuning

## Model Selection

**Phase 1 (open-source):** Qwen, Gemma, Granite via Hugging Face
**Phase 2 (optional closed-source APIs):** GPT-4/4o, Gemini, Claude

## Important Caveats

- Google/Data Commons gives structured ClaimReview metadata, **not the full article body**.
- Dataset loaders read from `src/datasets/data/` by default; use `python src/datasets/fetch_datasets.py` to populate that directory.
- FakeNewsNet's Hugging Face mirror exposes a `real` column in CSV form; the pipeline interprets `1 => real` and `0 => fake`, which is consistent with the upstream FakeNewsNet repository's separate `*_real.csv` and `*_fake.csv` splits.
- MuMiN is currently loaded from the fetched CSV export under `src/datasets/data/mumin/mumin.csv`; the loader derives the split from the `train_mask`, `val_mask`, and `test_mask` columns.
- The primary workflow is now open-source-model-first; the OpenAI-compatible path remains optional and secondary.
- Adapter fine-tuning supervision is deterministic and derived from the current binary labels plus prompt context, which makes it reproducible but not yet explanation-rich.

## Reference

- TechRxiv survey: _From Fact Verification to Understanding Misleadingness: A Survey and Roadmap on Reader-Centric Multimodal Misinformation Detection_
