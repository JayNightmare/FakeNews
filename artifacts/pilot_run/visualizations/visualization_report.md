# Aggregate Visualization Report

This file explains the cross-dataset visualization artifacts created for a multi-dataset run.

## Records After Cleaning

![Records After Cleaning](dataset_sizes_bar.svg)

**What this visual is:** A bar chart comparing how many records remained in each dataset after cleaning and optional balancing.

**Where it came from:** Derived from `aggregate_summary.json -> per_dataset[].records_after_cleaning`.

**Trend seen:** The dataset size mix is relatively balanced, with `mumin` only slightly ahead of `fakenewsnet`.

## F1 by Dataset

![F1 by Dataset](dataset_f1_bar.svg)

**What this visual is:** A bar chart comparing the final F1 score achieved on each dataset.

**Where it came from:** Derived from `aggregate_summary.json -> per_dataset[].f1`.

**Trend seen:** The F1 score mix is relatively balanced, with `claimreview` only slightly ahead of `mumin`.

## Accuracy by Dataset

![Accuracy by Dataset](dataset_accuracy_bar.svg)

**What this visual is:** A bar chart comparing overall accuracy across datasets.

**Where it came from:** Derived from `aggregate_summary.json -> per_dataset[].accuracy`.

**Trend seen:** The accuracy score mix is relatively balanced, with `claimreview` only slightly ahead of `mumin`.

## Aggregate Ground-Truth Labels

![Aggregate Ground-Truth Labels](aggregate_labels_pie.svg)

**What this visual is:** A pie chart showing the combined label mix across all datasets included in the run.

**Where it came from:** Derived from `aggregate_evaluation.json -> label_distribution.ground_truth`.

**Trend seen:** `1` is the largest ground-truth label segment, ahead of `0` by 69.0.
