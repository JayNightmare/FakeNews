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

**Trend seen:** Fakeddit dominates the retained sample count because ClaimReview only has a very small locally available sample in the current run, while FakeNewsNet and MuMiN were skipped due to missing local exports.

## F1 by Dataset

![F1 by Dataset](../artifacts/pilot_run/visualizations/dataset_f1_bar.svg)

**What this visual is:** A bar chart comparing the final F1 score achieved on each dataset that ran successfully.

**Where it came from:** Derived from `artifacts/pilot_run/aggregate_summary.json -> per_dataset[].f1`.

**Trend seen:** ClaimReview is effectively saturated on the tiny available sample, while Fakeddit is notably harder and produces a lower but still usable F1 score. The gap is more about data regime and task difficulty than a meaningful benchmark comparison.

## Accuracy by Dataset

![Accuracy by Dataset](../artifacts/pilot_run/visualizations/dataset_accuracy_bar.svg)

**What this visual is:** A bar chart comparing overall accuracy across datasets.

**Where it came from:** Derived from `artifacts/pilot_run/aggregate_summary.json -> per_dataset[].accuracy`.

**Trend seen:** Accuracy follows the same pattern as F1: ClaimReview appears perfect on the tiny sample, whereas Fakeddit is materially noisier and sits much closer to the decision boundary.

## Aggregate Ground-Truth Labels

![Aggregate Ground-Truth Labels](../artifacts/pilot_run/visualizations/aggregate_labels_pie.svg)

**What this visual is:** A pie chart showing the combined label mix across all datasets included in the aggregate run.

**Where it came from:** Derived from `artifacts/pilot_run/aggregate_evaluation.json -> label_distribution.ground_truth`.

**Trend seen:** The combined label pool is skewed toward the `1` label because both the small ClaimReview slice and the sampled Fakeddit slice lean fake-heavy in this run, even after balancing is requested.

## How To Refresh

1. Re-run the aggregate pipeline command above.
2. Re-open this page.
3. If you want a different experiment represented here, update the image links to point at that run's `visualizations/` folder.
