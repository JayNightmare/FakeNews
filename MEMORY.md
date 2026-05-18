# Project Memory

## Current State
- The project is an Experimental Pipeline for Misinformation Detection.
- It evaluates LLM performance based on contextual information (minimal, full, misleading).
- Supports datasets: ClaimReview, Fakeddit, FakeNewsNet, MuMiN.
- A functional CLI pipeline exists in `src/run_experiment.py`.

## Active Tasks
- **Completed**:
  - Implementation of the core data processing pipeline and prompt generation.
  - Setup of fetching scripts for datasets (`fetch_datasets.py`).
  - Drafted presentation slides (`drafts/slides/presentation.html`).
- **Next Steps (Feature Implementation)**:
  - Create a GitHub Pages website to host the slides with interactive navigation.
  - Integrate visual graphs and diagrams into the slides.
  - Setup deployment strategy for the presentation website.

## Architectural Decisions
- CLI-first pipeline to ensure reproducibility without hidden notebook state.
- Supports offline deterministic baseline testing.
- Uses `docs/` for project documentation and now as the source for GitHub Pages.
