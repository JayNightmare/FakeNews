# Meeting task checklist (17/04/2026)

Mapped against the RA1 and RA2 task list from `docs/Project_meeting_1704.docx`.

---

## RA1 — Data Preparation

### 1) Dataset Preparation (FakeNewsNet, MuMiN, Fakeddit)

**Status:** done

- [x] Google ClaimReview feed: working loader and normalizer
- [x] FakeNewsNet: loader implemented, dataset fetcher created
- [x] Fakeddit: loader implemented, dataset fetcher created
- [x] MuMiN: stub loader, dataset fetcher created

### 2) Data Cleaning and Standardisation

**Status:** in progress

- [x] Cleaning pipeline implemented (`src/cleaning.py`)
     - Remove empty/null text
     - Remove entries with fewer than 5 words
     - Remove corrupted characters
     - Strip whitespace
     - Drop rows with missing essential fields
- [x] Cleaning notes logged per record
- [ ] Merge relevant text fields (title + body) — handled per-loader

### 3) Metadata Cleaning

**Status:** done

- [x] Metadata extracted and structured per dataset loader
- [x] Cross-dataset consistency validation (date normalization, null handling via `cleaning.py`)

### 4) Dataset Summary Sheet

**Status:** done

- [x] Summary generator implemented (`src/summary.py`)
     - Number of samples (before/after cleaning)
     - Available text fields
     - Available metadata fields
     - Label types and distribution
     - Missing data statistics
- [x] Generated summaries for all four datasets during inference runs

### 5) Label Mapping Document

**Status:** done

- [x] `docs/label-mapping.md` — covers all four datasets
     - Original labels
     - Mapped labels (real/fake)
     - Excluded categories
     - Justification for mapping decisions

### 6) Unified Dataset Format

**Status:** done

- [x] `src/schema.py` — `UnifiedRecord` dataclass
     - `dataset`, `sample_id`, `text`, `label`, `label_name`, `metadata`
     - Plus extended fields: `original_label`, `mapped_label`, `split`, `modality`, `context_text`, `cleaning_notes`
- [x] JSON, JSONL, CSV export

---

## RA2 — Engineering Focus

### 1) Prompting Pipeline

**Status:** done

- [x] Prompt template system (`src/prompts.py`)
     - Takes dataset instance as input
     - Formats into prompt with context variants (minimal, full, misleading)
     - Supports classification + explanation output
- [x] Context-budget ablation support for full-context prompts
- [x] Cached Google Fact Check grounding support
- [x] OpenAI-compatible API mode
- [x] Open-source model integration (Qwen-first Hugging Face workflow)
- [x] Adapter training/export pipeline for open-source fine-tuning

### 2) Cost Count

**Status:** done

- [x] Character-based cost estimation (`src/costs.py`)
- [x] Token-level estimation (`tiktoken` integration with offline fallback)
- [x] Per-model cost projections (gpt-4o, claude-3, gemini, etc.)

---

## Additional — Evaluation (inferred from meeting objectives)

### Metrics and Comparison

**Status:** in progress

- [x] Evaluation module (`src/evaluation.py`)
     - Accuracy, precision, recall, F1
     - Confusion matrix
     - Per-dataset and per-context-variant breakdown
     - Run manifest
- [x] Cross-variant and context-budget comparison report
- [ ] Explanation quality assessment

---

## Previously completed (pre-meeting pivot work)

- [x] Google ClaimReview data source investigation
- [x] Data Commons feed integration
- [x] Presentation materials (slides, talk track, lecturer reply)
- [x] Reproducibility documentation
