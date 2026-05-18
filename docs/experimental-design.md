# Experimental design

## Research aim

Investigate how contextual information affects LLM performance in misinformation detection. Specifically, whether increasing context consistently improves model performance, or whether excessive or misleading context degrades reliability.

## Task definition

1. **Classification**: determine whether content is real or fake (binary)
2. **Explanation generation**: provide a justification for the prediction

## Datasets

| Dataset     | Domain              | Modality                 | Source                      |
| ----------- | ------------------- | ------------------------ | --------------------------- |
| FakeNewsNet | news articles       | text (+ optional images) | PolitiFact, GossipCop       |
| MuMiN       | social media claims | text + images            | Twitter/X via claim threads |
| Fakeddit    | Reddit posts        | text + images            | Reddit                      |
| ClaimReview | fact-check metadata | text                     | Google/Data Commons feed    |

Target: ~1000 samples per dataset (balanced where possible), ~3000–4000 total.

## Context conditions

Each sample is evaluated under three context variants to measure context sensitivity:

### 1. Minimal context

- Claim text only (headline or short claim)
- No article body, metadata, or source information
- Tests: can the model classify from surface-level text alone?

### 2. Full context

- Claim text + article body / thread text
- Publisher metadata, dates, source URLs
- Tests: does additional relevant context improve classification?

### 3. Misleading context

- Claim text + deliberately ambiguous or contradictory context
- Conflicting metadata, unrelated article fragments
- Tests: does irrelevant or adversarial context degrade model reliability?

## Model selection

### Phase 1 — Open-source models

- LLaMA 3 (primary)
- Qwen2.5

### Phase 2 — Closed-source APIs

- OpenAI GPT-4 / GPT-4o
- Google Gemini
- Anthropic Claude

## Pipeline stages

### Stage 1: Collection

- Load raw data from local files (per-dataset loaders)
- Use `fetch_datasets.py` to automate downloading balanced sample subsets from Hugging Face mirrors for Fakeddit, FakeNewsNet, and MuMiN.

### Stage 2: Normalization

- Convert each dataset's native format into `UnifiedRecord` schema
- Preserve original labels alongside binary mapped labels
- Track modality, source fields used, and split assignment

### Stage 3: Cleaning & Sampling

- Apply meeting-specified text and metadata cleaning rules (remove nulls, normalize ISO-8601 dates).
- Perform deterministic stratified balanced sampling to ensure equal class distribution (e.g. 500 real / 500 fake).
- Log all modifications in `cleaning_notes`.

### Stage 4: Prompt construction

- Generate prompts for each context variant (minimal, full, misleading)
- Use `templates/claim_prompt_template.md` as base

### Stage 5: Inference

- Deterministic heuristic baseline (no API needed)
- OpenAI-compatible API mode
- Future: local open-source model inference

### Stage 6: Evaluation

- Per-dataset metrics: accuracy, precision, recall, F1
- Per-context-variant breakdown
- Confusion matrices
- Explanation capture and quality assessment
- Cross-variant comparison report

### Stage 7: Quality reporting

- Missing-field counts
- Label distribution
- Language distribution
- Duplicate detection
- Appearance URL stats

## Reproducibility principles

- CLI-first, not notebook-first
- deterministic default baseline
- explicit output directory per run
- stable record IDs
- artifacts saved at each stage
- no hidden UI state required
- run manifest with full parameters saved per experiment

## Extension path

If richer experiments are needed later:

1. Article-body enrichment from publisher URLs
2. Image download / multimodal evidence capture
3. Stronger open-source model baselines (local inference)
4. Evaluation splits / stratified sampling
5. Rating-label harmonization across publishers and languages
6. Human evaluation of generated explanations

## Reference

- TechRxiv survey: _From Fact Verification to Understanding Misleadingness: A Survey and Roadmap on Reader-Centric Multimodal Misinformation Detection_
