# Visualisation Guide

This page brings the generated experiment visuals into the project documentation so they can be reviewed from the docs folder rather than only from an artifact directory.

These visuals come from the aggregate heuristic run written to `artifacts/pilot_run/visualizations/`.

## Source Run

The current visuals were generated from:

```bash
python3 src/run_experiment.py \
  --dataset all \
  --balanced \
  --limit 100 \
  --mode heuristic \
  --output-dir artifacts/pilot_run
```

Open the full dashboard here:

- [Aggregate dashboard](../artifacts/pilot_run/visualizations/dashboard.html)

## Records After Cleaning

![Records After Cleaning](../artifacts/pilot_run/visualizations/dataset_sizes_bar.svg)

**What this visual is:** A bar chart comparing how many records remained in each dataset after cleaning and optional balancing.

**Where it came from:** Derived from `artifacts/pilot_run/aggregate_summary.json -> per_dataset[].records_after_cleaning`.

**Trend seen:** MuMiN and FakeNewsNet contribute the largest retained slices in the current aggregate run, while ClaimReview remains much smaller because the local feed sample is tiny by comparison.

## F1 by Dataset

![F1 by Dataset](../artifacts/pilot_run/visualizations/dataset_f1_bar.svg)

**What this visual is:** A bar chart comparing the final F1 score achieved on each dataset that ran successfully.

**Where it came from:** Derived from `artifacts/pilot_run/aggregate_summary.json -> per_dataset[].f1`.

**Trend seen:** ClaimReview and MuMiN sit at the top of the current F1 ranking, while FakeNewsNet is the hardest dataset for the heuristic baseline and Fakeddit lands in the middle.

## Accuracy by Dataset

![Accuracy by Dataset](../artifacts/pilot_run/visualizations/dataset_accuracy_bar.svg)

**What this visual is:** A bar chart comparing overall accuracy across datasets.

**Where it came from:** Derived from `artifacts/pilot_run/aggregate_summary.json -> per_dataset[].accuracy`.

**Trend seen:** Accuracy follows the same pattern as F1: ClaimReview and MuMiN are strongest in this run, Fakeddit is moderate, and FakeNewsNet is the weakest slice for the heuristic baseline.

## Aggregate Ground-Truth Labels

![Aggregate Ground-Truth Labels](../artifacts/pilot_run/visualizations/aggregate_labels_pie.svg)

**What this visual is:** A pie chart showing the combined label mix across all datasets included in the aggregate run.

**Where it came from:** Derived from `artifacts/pilot_run/aggregate_evaluation.json -> label_distribution.ground_truth`.

**Trend seen:** The combined label pool is still skewed toward the `1` label in the current run, which helps explain why the heuristic baseline over-predicts fake and achieves very high recall but weak true-negative performance.

## How To Refresh

1. Re-run the aggregate pipeline command above.
2. Re-open this page.
3. If you want a different experiment represented here, update the image links to point at that run's `visualizations/` folder.
