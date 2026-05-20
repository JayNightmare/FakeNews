# Evaluation Report

## Overall Metrics

| Metric | Value |
|---|---|
| accuracy | 0.6137 |
| precision | 0.6204 |
| recall | 0.9827 |
| f1 | 0.7606 |
| support | 277 |

## Confusion Matrix

| | Predicted Real | Predicted Fake |
|---|---|---|
| **Actual Real** | 0 | 104 |
| **Actual Fake** | 3 | 170 |

## By Dataset

| Dataset | Accuracy | Precision | Recall | F1 | Support |
|---|---|---|---|---|---|
| ClaimReview | 1.0 | 1.0 | 1.0 | 1.0 | 4 |
| Fakeddit | 0.5465 | 0.5465 | 1.0 | 0.7068 | 86 |
| FakeNewsNet | 0.2778 | 0.2778 | 1.0 | 0.4348 | 90 |
| MuMiN | 0.9691 | 1.0 | 0.9691 | 0.9843 | 97 |

## By Context Mode

| Context | Accuracy | Precision | Recall | F1 | Support |
|---|---|---|---|---|---|
| full | 0.6137 | 0.6204 | 0.9827 | 0.7606 | 277 |
