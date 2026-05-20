# Misinformation Detection Experimental Pipeline

Experimental pipeline for studying how contextual information affects LLM performance in misinformation detection.

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
├── schema.py                  # canonical UnifiedRecord schema
├── cleaning.py                # data cleaning pipeline
├── prompts.py                 # context-variant prompt generation
├── evaluation.py              # metrics, confusion matrix, comparison
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
artifacts/                     # pipeline output artifacts
```

## Quick Start

### 0) Fetch Datasets

The fetcher downloads ClaimReview plus manageable sample subsets for Fakeddit, FakeNewsNet, and MuMiN into `src/datasets/data/`:

```bash
# Install required dependencies
pip install -r requirements.txt

# Run the fetcher
python src/datasets/fetch_datasets.py
```

### 1) Run a heuristic baseline (no API key required)

You can run a deterministic smoke-test using the `--balanced` flag to ensure stratified sampling:

```bash
python3 src/run_experiment.py \
  --dataset claimreview \
  --limit 100 \
  --mode heuristic \
  --output-dir artifacts/pilot_run
```

### 2) Run with context variants

```bash
# Minimal context (claim text only)
python3 src/run_experiment.py \
  --dataset claimreview \
  --context-mode minimal \
  --limit 100 \
  --output-dir artifacts/context_minimal

# Full context (claim + article + metadata)
python3 src/run_experiment.py \
  --dataset claimreview \
  --context-mode full \
  --limit 100 \
  --output-dir artifacts/context_full

# Misleading context (adversarial)
python3 src/run_experiment.py \
  --dataset claimreview \
  --context-mode misleading \
  --limit 100 \
  --output-dir artifacts/context_misleading
```

### 3) Run other datasets

```bash
# Fakeddit (uses fetched data in src/datasets/data/fakeddit/)
python3 src/run_experiment.py --dataset fakeddit --limit 100

# FakeNewsNet (uses fetched data in src/datasets/data/fakenewsnet/)
python3 src/run_experiment.py --dataset fakenewsnet --limit 100

# All available datasets
python3 src/run_experiment.py --dataset all --limit 50
```

### 4) Optional OpenAI-compatible model run

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

### 5) Expected outputs per run

- `normalized_records.{json,jsonl,csv}` — cleaned unified records
- `prompts.jsonl` — generated prompts with context
- `predictions.jsonl` — model/heuristic predictions
- `evaluation_report.{json,md}` — accuracy, F1, confusion matrix
- `dataset_summary.{json,md}` — sample counts, label distribution, missing data
- `visualizations/dashboard.html` — browser-friendly chart dashboard for the run
- `visualizations/visualization_report.md` — explains each chart, its source artifact, and the main trend
- `cleaning_report.json` — what was removed during cleaning
- `run_manifest.json` — full run parameters

See [docs/visualisation.md](docs/visualisation.md) for a docs-native page that embeds the latest aggregate visuals and explains them.

## Context Conditions

The pipeline supports three context variants to study context sensitivity:

| Mode         | What's included                 | Tests                                           |
| ------------ | ------------------------------- | ----------------------------------------------- |
| `minimal`    | claim text only                 | can the model classify from surface-level text? |
| `full`       | claim + article body + metadata | does relevant context improve classification?   |
| `misleading` | claim + adversarial context     | does irrelevant context degrade reliability?    |

## Reproducibility

- CLI-first, no hidden notebook state
- deterministic heuristic baseline (no API needed)
- explicit CLI arguments and saved run manifests
- stable record IDs across reruns
- artifacts saved at every pipeline stage
- offline-capable token cost tracking and balanced sampling
- automated `src/datasets/fetch_datasets.py` for standardizing data ingestion under `src/datasets/data/`

## Model Selection

**Phase 1 (open-source):** LLaMA 3, Qwen2.5
**Phase 2 (closed-source APIs):** GPT-4/4o, Gemini, Claude

## Important Caveats

- Google/Data Commons gives structured ClaimReview metadata, **not the full article body**.
- Dataset loaders read from `src/datasets/data/` by default; use `python src/datasets/fetch_datasets.py` to populate that directory.
- The heuristic baseline is for pipeline plumbing validation, not a research result.

## Reference

- TechRxiv survey: _From Fact Verification to Understanding Misleadingness: A Survey and Roadmap on Reader-Centric Multimodal Misinformation Detection_
