# Changelog

This file summarizes notable changes from recent git history and the current unreleased worktree.

## Unreleased

Current worktree changes not yet committed:

- Added a deterministic adapter-training workflow for open-source models:
     - `src/training.py` exports train/eval chat corpora from normalized records.
     - `src/train_hf_adapter.py` prepares data or fine-tunes a LoRA adapter for Hugging Face models such as Qwen.
- Extended smoke coverage for training-example export, deterministic split handling, and prepare-only corpus generation.
- Refreshed the documentation and command surface so README, reproducibility guidance, sources, experimental design, and presentation notes all reflect the Hugging Face-first workflow, Google fact-check grounding, and context-ablation support.

## 2026-05-25

### `cee6c84` feat: Enhance context budget handling and integrate Google Fact Check

- Added context-budget support for prompt generation and evaluation.
- Added context-ablation reporting with prediction-flip and threshold summaries.
- Added cached Google Fact Check grounding before prompt construction.
- Switched the main experiment runner toward a Hugging Face-first inference path.

## 2026-05-22

### `b870cce` docs: update dataset handling and trends in documentation

- Updated project documentation to better reflect dataset handling and analysis trends.

## 2026-05-20

### Reporting and documentation viewer improvements

Included commits: `a905ae4`, `57a559b`, `d5da986`, `4cfc48f`, `2af29f6`, `06083b2`

- Added fallback handling for `visualisation.md` in the documentation viewer.
- Added static visualizations and reports for experiment artifacts.
- Added run manifest and run-summary artifacts for pilot outputs.
- Refactored the code structure for maintainability.
- Improved mobile responsiveness and markdown file handling in the docs viewer.
- Added meeting notes for contextual-information and dataset-analysis discussion.

## 2026-05-18

### Project initialization and presentation/docs foundation

Included commits: `d0b8946`, `cee4097`, `37e7c67`, `9729fde`

- Initialized the Google Fact Check starter project with the core evaluation pipeline, datasets, and experiment artifacts.
- Added the documentation viewer and responsive documentation layout.
- Added project memory and presentation slide materials.
