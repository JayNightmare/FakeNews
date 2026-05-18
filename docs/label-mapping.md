# Label mapping

This document maps each dataset's native labels to the binary classification target (real/fake) used in the experimental pipeline.

## Binary target

| mapped_label | mapped_label_name | meaning                                           |
| ------------ | ----------------- | ------------------------------------------------- |
| 0            | real              | content is factually accurate or genuine          |
| 1            | fake              | content is fabricated, misleading, or manipulated |

---

## FakeNewsNet

**Source splits:** PolitiFact, GossipCop

| original_label | mapped_label | mapped_label_name |
| -------------- | ------------ | ----------------- |
| real           | 0            | real              |
| fake           | 1            | fake              |

**Excluded:** none

**Justification:** FakeNewsNet uses a clean binary label natively. No remapping needed.

---

## Fakeddit

**Source labels (6-way):**

| original_label | original_label_name | mapped_label | mapped_label_name |
| -------------- | ------------------- | ------------ | ----------------- |
| 0              | true                | 0            | real              |
| 1              | satire/parody       | 1            | fake              |
| 2              | misleading content  | 1            | fake              |
| 3              | imposter content    | 1            | fake              |
| 4              | false connection    | 1            | fake              |
| 5              | manipulated content | 1            | fake              |

**Excluded:** none (all categories map to one of the two targets)

**Justification:** The meeting requires binary real/fake classification. Fakeddit's label 0 (true) maps to real; all other categories represent some form of misinformation and map to fake. The original fine-grained label is preserved in `original_label` and `original_label_name` for downstream analysis.

Fakeddit also offers 2-way and 3-way label columns. We use the 6-way column as the canonical `original_label` because it retains the most information, then collapse to binary for the mapped target.

---

## MuMiN

**Source labels:**

| original_label | mapped_label | mapped_label_name |
| -------------- | ------------ | ----------------- |
| misinformation | 1            | fake              |
| factual        | 0            | real              |

**Excluded:**

- `unknown` / unlabelled claims — excluded from training/evaluation sets

**Justification:** MuMiN's core claim-level labels are binary (misinformation vs factual). Claims labelled `unknown` lack ground truth and are excluded to avoid noise. The original label is preserved verbatim.

---

## ClaimReview (Google/Data Commons)

**Source labels:** free-text `reviewRating.alternateName` from publishers (e.g. "False", "Mostly True", "Pants on Fire", "Satire")

**Normalized buckets → binary mapping:**

| review_rating_normalized  | mapped_label | mapped_label_name |
| ------------------------- | ------------ | ----------------- |
| true                      | 0            | real              |
| false_or_misleading       | 1            | fake              |
| mixed_or_unverified       | 1            | fake              |
| satire_or_joke            | 1            | fake              |
| ai_generated_or_synthetic | 1            | fake              |
| other                     | —            | excluded          |

**Excluded:** `other` — too ambiguous to assign a reliable binary label

**Justification:** ClaimReview ratings are publisher-specific free text. The pipeline first normalizes them into six buckets (already implemented in `run_experiment.py`), then maps to binary. `true` → real; all verifiable misinformation categories → fake; `other` is excluded because it typically represents non-English or unparseable ratings where the mapping would be unreliable.

---

## Design notes

- The **original label** is always preserved in `original_label` / `original_label_name`.
- The **mapped label** is the binary target used for classification and metrics.
- Exclusions are logged in `cleaning_notes` on each affected record.
- If a dataset is extended with new label categories, update this document before re-running the pipeline.
