# Google Fact Check Experimental Pipeline

## Slide 1 — Title

- Multi-Dataset Experimental Scaffold for Misinformation Detection
- Kingston misinformation project
- Goal: Establish a fully functional and reproducible experimental pipeline (RA1/RA2 alignment)

## Slide 2 — Problem

- Original route depended on X/Twitter-style hydration workflows
- API pricing and instability makes that route poor for a reproducible student research pipeline
- We needed a robust, offline-capable experimental scaffold that handles multiple data formats

## Slide 3 — Unified Architecture

- Created a unified `UnifiedRecord` schema
- Supported Datasets:
    1. **ClaimReview** (Google/Data Commons fact-check metadata feed)
    2. **Fakeddit** (Reddit multimodal fake news)
    3. **FakeNewsNet** (PolitiFact + GossipCop news articles)
    4. **MuMiN** (Social media claims)
- Automated ingestion via Hugging Face mirrors for reproducible setup

## Slide 4 — Pipeline Stages

1. **Collection**: Raw data loaders + `fetch_datasets.py` for offline-capable reproducible syncing
2. **Normalization**: Canonical schema mapping
3. **Cleaning & Sampling**: Metadata cleaning, deterministic balanced/stratified sampling
4. **Prompting**: 3 context variants (minimal, full, misleading)
5. **Cost Estimation**: Token-level offline/online cost tracking for budget management
6. **Inference & Evaluation**: Heuristic baseline + LLM integration ready

## Slide 5 — Context Variants

To study how context affects model reliability:
- **Minimal**: Claim text only
- **Full**: Claim + article body + metadata
- **Misleading**: Claim + deliberately adversarial/conflicting context

## Slide 6 — Cost Tracking & Sampling

- **Cost Tracking**: Automated token estimation (via `tiktoken` or character heuristics) before inference to prevent API bill shocks
- **Balanced Sampling**: Deterministic stratified sampling (e.g., 500 fake / 500 real) to ensure statistically sound evaluation across imbalanced datasets

## Slide 7 — Reproducibility Choices

- CLI-first pipeline (`run_experiment.py`)
- Deterministic default baseline for testing plumbing without API keys
- Explicit output directories with saved JSON/CSV artifacts at each stage
- Fully containerizable, offline-capable core

## Slide 8 — Deliverables Completed

- Reproducible Python pipeline
- Automated dataset fetchers
- Unified schema and data cleaners
- Automated cost and evaluation reporting
- Comprehensive documentation

## Slide 9 — Next Steps (Research Phase)

- Execute full 1,000-sample runs across Fakeddit, FakeNewsNet, and MuMiN
- Compare heuristic baselines with API-backed models (LLaMA 3, GPT-4o, Claude)
- Analyze performance delta between `minimal`, `full`, and `misleading` context modes
