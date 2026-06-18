# Evaluation Report

## Overall Metrics

| Metric | Value |
|---|---|
| accuracy | 0.8 |
| precision | 0.8 |
| recall | 1.0 |
| f1 | 0.8889 |
| support | 100 |

## Confusion Matrix

| | Predicted Real | Predicted Fake |
|---|---|---|
| **Actual Real** | 0 | 20 |
| **Actual Fake** | 0 | 80 |

## By Dataset

| Dataset | Accuracy | Precision | Recall | F1 | Support |
|---|---|---|---|---|---|
| ClaimReview | 0.9077 | 0.9077 | 1.0 | 0.9516 | 65 |
| Fakeddit | 0.5926 | 0.5926 | 1.0 | 0.7442 | 27 |
| FakeNewsNet | 0.625 | 0.625 | 1.0 | 0.7692 | 8 |

## By Context Mode

| Context | Accuracy | Precision | Recall | F1 | Support |
|---|---|---|---|---|---|
| full | 0.8 | 0.8 | 1.0 | 0.8889 | 100 |

## By Context Budget

| Budget | Accuracy | Precision | Recall | F1 | Support |
|---|---|---|---|---|---|
| 1.00 | 0.8 | 0.8 | 1.0 | 0.8889 | 100 |
