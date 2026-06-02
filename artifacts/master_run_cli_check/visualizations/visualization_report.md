# master_dataset Visualization Report

This file explains the generated visualization artifacts for the current run and summarizes the main pattern visible in each chart.

## Mapped Label Distribution

![Mapped Label Distribution](label_distribution_pie.svg)

**What this visual is:** A pie chart showing how many cleaned samples map to each binary target label.

**Where it came from:** Derived from `dataset_summary.json -> label_types.mapped_labels` after cleaning and any optional balanced sampling.

**Trend seen:** No label distribution was available for this run.

## Modality Distribution

![Modality Distribution](modality_distribution_bar.svg)

**What this visual is:** A bar chart showing the count of text-only versus multimodal records that survived cleaning.

**Where it came from:** Derived from `dataset_summary.json -> modality_distribution`, using the normalized records written by the pipeline.

**Trend seen:** No modality distribution was available for this run.

## Context Coverage

![Context Coverage](context_coverage_pie.svg)

**What this visual is:** A pie chart comparing records that include context text with records that do not.

**Where it came from:** Derived from `dataset_summary.json -> records_with_context` and `sample_count`.

**Trend seen:** The context coverage mix is relatively balanced, with `with context` only slightly ahead of `without context`.

## Overall Metrics

![Overall Metrics](overall_metrics_bar.svg)

**What this visual is:** A bar chart comparing the top-line evaluation metrics for the current run.

**Where it came from:** Derived from `evaluation_report.json -> overall` after prediction and scoring.

**Trend seen:** `accuracy` is the strongest metric at 0.000, while `f1` is the main constraint at 0.000.
