# Project Memory

## Current State

- The project is a CLI-first experimental pipeline for misinformation detection across ClaimReview, Fakeddit, FakeNewsNet, and MuMiN.
- The canonical local workflow is `fetch_datasets.py` -> `build_master_dataset.py` -> `run_experiment.py` or `train_hf_adapter.py`.
- Loaders now normalize labels and split values at the source so all emitted `UnifiedRecord` instances use a consistent internal contract.
- The master merge pipeline now deduplicates merged records before writing `master_records.*`, `master_flat_records.*`, `master_dataset_manifest.json`, and `master_dataset_coverage.json`.

## Completed Work

- Implemented the core schema-cleaning-dataset-merge subsystem around `UnifiedRecord`, `MasterRecord`, and `LabelMapping`.
- Implemented dataset fetching and local mirror support through `src/datasets/fetch_datasets.py`; this file exists and is part of the normal setup flow.
- Implemented prompt construction, evaluation, visualization, and experiment orchestration in `src/run_experiment.py` and neighboring modules.
- Added label-mapping audit notes to kept normalized records and exclusion-by-`None` for ambiguous labels where required.
- Added post-merge deduplication with manifest audit notes and aggregate coverage tracking for the master dataset.
- Drafted presentation material under `drafts/slides/`.

## Active Focus

- Use the master dataset outputs as the canonical merged corpus for inspection, reproducibility, and downstream model runs.
- Keep documentation aligned with the implemented loader contracts, merge rules, and label semantics.

## Architectural Decisions

- CLI-first execution keeps the pipeline reproducible and avoids hidden notebook state.
- Ambiguous labels are excluded at the loader boundary instead of being forced into the binary target.
- Internal binary labels use `mapped_label: 0 = real, 1 = fake`; master exports expose `is_real` / `Real` as `1 = real, 0 = fake`.
- Cross-dataset deduplication keeps the highest-priority record in this order: ClaimReview > FakeNewsNet > Fakeddit > MuMiN.
