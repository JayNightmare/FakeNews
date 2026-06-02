# Label mapping

This document defines how each dataset loader maps native labels to the binary misinformation target used in the pipeline.

## Binary target

| Field               | Value           | Meaning                     |
| ------------------- | --------------- | --------------------------- |
| `mapped_label`      | `0`             | real                        |
| `mapped_label`      | `1`             | fake                        |
| `mapped_label_name` | `real` / `fake` | Human-readable binary label |

The master export also derives `is_real` and the flat `Real` column, where the direction is reversed for readability:

| Field              | Value | Meaning |
| ------------------ | ----- | ------- |
| `is_real` / `Real` | `1`   | real    |
| `is_real` / `Real` | `0`   | fake    |

## General handling rules

- The original dataset label is preserved in `original_label` and `original_label_name`.
- Ambiguous labels are excluded by returning `None` in the loader instead of forcing a binary class.
- Kept records append a label audit note to `cleaning_notes` using the format `Label mapped: {original_label} ({original_label_name}) -> {mapped_label_name}`.

## ClaimReview

ClaimReview ratings are publisher-specific free text. The loader first normalizes them into six buckets and then maps those buckets to the binary target.

| Normalized bucket           | mapped_label | mapped_label_name |
| --------------------------- | ------------ | ----------------- |
| `true`                      | `0`          | `real`            |
| `false_or_misleading`       | `1`          | `fake`            |
| `mixed_or_unverified`       | `1`          | `fake`            |
| `satire_or_joke`            | `1`          | `fake`            |
| `ai_generated_or_synthetic` | `1`          | `fake`            |
| `other`                     | excluded     | excluded          |

**Excluded labels:** `other`

**Justification:** `true` is the only reliable positive signal for factual content. All other explicit misinformation buckets represent false, misleading, synthetic, or satire-like claims and therefore map to `fake`. `other` is excluded because it is too ambiguous or unparseable to support a trustworthy binary assignment.

## Fakeddit

Fakeddit exposes multi-class misinformation labels. The loader preserves the six-way label as the original label and collapses it to the binary target.

| original_label | original_label_name   | mapped_label | mapped_label_name |
| -------------- | --------------------- | ------------ | ----------------- |
| `0`            | `true`                | `0`          | `real`            |
| `1`            | `satire/parody`       | `1`          | `fake`            |
| `2`            | `misleading content`  | `1`          | `fake`            |
| `3`            | `imposter content`    | `1`          | `fake`            |
| `4`            | `false connection`    | `1`          | `fake`            |
| `5`            | `manipulated content` | `1`          | `fake`            |

**Excluded labels:** none

**Justification:** label `0` is the only class representing genuine content. Every other six-way label describes a misinformation subtype, so the binary collapse is lossless with respect to the real/fake decision while still preserving the original fine-grained label for analysis.

## FakeNewsNet

FakeNewsNet is treated as a native binary source. The loader accepts textual labels and equivalent boolean or numeric aliases coming from fetched CSV exports.

| Accepted source value                                                                       | mapped_label | mapped_label_name |
| ------------------------------------------------------------------------------------------- | ------------ | ----------------- |
| `real`, `true`, `1` in the `real` flag path, or `0` in the label path meaning real content  | `0`          | `real`            |
| `fake`, `false`, `0` in the `real` flag path, or `1` in the label path meaning fake content | `1`          | `fake`            |
| `unknown` or any other unparseable label                                                    | excluded     | excluded          |

**Excluded labels:** `unknown` and other unparseable values

**Justification:** FakeNewsNet is intended as a binary dataset, but local mirrors can arrive with different label encodings. The loader normalizes supported aliases to `real` or `fake` and excludes anything else rather than biasing the dataset by silently mapping ambiguous values to `fake`.

## MuMiN

MuMiN is also treated as a binary claim-level dataset after normalization.

| Accepted source value                  | mapped_label | mapped_label_name |
| -------------------------------------- | ------------ | ----------------- |
| `factual`, `true`, `real`, `0`         | `0`          | `real`            |
| `misinformation`, `false`, `fake`, `1` | `1`          | `fake`            |
| `unknown`                              | excluded     | excluded          |

**Excluded labels:** `unknown`

**Justification:** MuMiN's usable claim-level supervision is factual versus misinformation. Claims with `unknown` verdicts do not provide reliable ground truth and are excluded before they reach the normalized corpus.

## Ambiguous and unknown labels

These labels are excluded by returning `None` from the loader:

| Dataset     | Excluded values                              | Why                                                      |
| ----------- | -------------------------------------------- | -------------------------------------------------------- |
| ClaimReview | `other`                                      | Non-mappable or ambiguous rating bucket                  |
| FakeNewsNet | `unknown` and other unparseable label values | Avoid silently biasing the binary target                 |
| MuMiN       | `unknown`                                    | Missing or unreliable ground truth                       |
| Fakeddit    | none                                         | All six-way labels collapse cleanly to the binary target |

Because excluded rows never become `UnifiedRecord` instances, they do not carry `cleaning_notes`. Only kept records receive the label audit note.
