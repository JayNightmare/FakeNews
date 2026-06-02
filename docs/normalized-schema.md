# Normalized schema

This repository uses three related dataclasses during ingestion and merge:

- `UnifiedRecord` is the canonical internal record emitted by each dataset loader.
- `MasterRecord` is the provenance-rich merged export written by `build_master_dataset.py`.
- `LabelMapping` documents how each loader turns source labels into the binary task target.

The pipeline is: loader `to_unified()` -> cleaning -> `UnifiedRecord` -> `MasterRecord` / flat export -> deduplicated master dataset.

## UnifiedRecord

`UnifiedRecord` is the shared internal schema used across ClaimReview, Fakeddit, FakeNewsNet, and MuMiN before merging.

| Field                 | Type             | Purpose                                                                                 |
| --------------------- | ---------------- | --------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| `dataset`             | `str`            | Human-readable dataset name such as `ClaimReview` or `MuMiN`.                           |
| `sample_id`           | `str`            | Stable dataset-local identifier used for prompts, predictions, and traceability.        |
| `text`                | `str`            | Primary text used for classification. This is the field passed into prompt generation.  |
| `original_label`      | `str             | int`                                                                                    | Native label value from the source dataset before binary normalization.                  |
| `original_label_name` | `str`            | Human-readable normalized name for the original label.                                  |
| `mapped_label`        | `int`            | Internal binary target: `0 = real`, `1 = fake`.                                         |
| `mapped_label_name`   | `str`            | Binary label name: `real` or `fake`.                                                    |
| `split`               | `str             | None`                                                                                   | Dataset split when known. Canonical values are `train`, `validation`, `test`, or `None`. |
| `source_fields_used`  | `list[str]`      | Raw source fields that were actually combined to create `text`.                         |
| `context_text`        | `str             | None`                                                                                   | Secondary context used in prompts or master export transformations.                      |
| `modality`            | `str`            | Modality hint such as `text` or `text+image`.                                           |
| `has_image`           | `bool`           | Whether the source item has image evidence or image metadata.                           |
| `metadata`            | `dict[str, Any]` | Dataset-specific provenance and auxiliary fields preserved after normalization.         |
| `cleaning_notes`      | `list[str]`      | Audit trail for transformations applied to kept records, including label-mapping notes. |

### UnifiedRecord usage notes

- Ambiguous records are excluded before a `UnifiedRecord` is created. Loaders return `None` instead of forcing uncertain labels into the binary target.
- Kept records now append a label audit note in the format `Label mapped: {original_label} ({original_label_name}) -> {mapped_label_name}`.
- `metadata` is intentionally heterogeneous, but common keys such as `tweet_num` and `claim_origin` are normalized where possible so downstream exports can rely on them.

## MasterRecord

`MasterRecord` is the export schema written by `build_master_dataset.py` after cleaning and cross-dataset merge.

| Field                 | Type             | Purpose                                                                        |
| --------------------- | ---------------- | ------------------------------------------------------------------------------ | ----------------------------------------------------------------------- |
| `id`                  | `str`            | Canonical merged identifier, derived from dataset name and `source_sample_id`. |
| `dataset`             | `str`            | Source dataset name.                                                           |
| `source_sample_id`    | `str`            | Original `UnifiedRecord.sample_id` retained for provenance.                    |
| `title`               | `str             | None`                                                                          | Title-like text extracted from the source when the dataset provides it. |
| `primary_text`        | `str`            | Main claim/article/body text used for the master export.                       |
| `source_name`         | `str             | None`                                                                          | Preferred source or publisher field chosen per dataset.                 |
| `is_real`             | `int`            | Export-facing binary label: `1 = real`, `0 = fake`.                            |
| `tweet_num`           | `int             | None`                                                                          | Interaction count field used in the flat export when available.         |
| `split`               | `str             | None`                                                                          | Split propagated from the normalized record.                            |
| `original_label`      | `str             | int`                                                                           | Native label copied from `UnifiedRecord`.                               |
| `original_label_name` | `str`            | Human-readable original label name copied from `UnifiedRecord`.                |
| `mapped_label`        | `int`            | Internal binary label copied from `UnifiedRecord` (`0 = real`, `1 = fake`).    |
| `mapped_label_name`   | `str`            | Internal binary label name copied from `UnifiedRecord`.                        |
| `source_fields_used`  | `list[str]`      | Provenance for the text fields used to build the record.                       |
| `modality`            | `str`            | Modality hint carried over from `UnifiedRecord`.                               |
| `has_image`           | `bool`           | Image-presence flag carried over from `UnifiedRecord`.                         |
| `metadata`            | `dict[str, Any]` | Dataset-specific metadata preserved after export.                              |
| `cleaning_notes`      | `list[str]`      | Audit notes preserved from the normalized record.                              |

### MasterRecord usage notes

- `UnifiedRecord.to_master_record()` performs dataset-specific field selection.
- `_mumin_master_record()` prefers `metadata["claim_origin"]` before falling back to `platform`.
- The master builder deduplicates `MasterRecord` rows before writing output files.

## Flat master export

The flat export is a six-column analysis-friendly view derived from `MasterRecord.to_flat_dict()`.

| Column        | Source                      | Meaning                                           |
| ------------- | --------------------------- | ------------------------------------------------- |
| `id`          | `MasterRecord.id`           | Canonical merged identifier.                      |
| `Title`       | `MasterRecord.title`        | Title-like field when available.                  |
| `Claim`       | `MasterRecord.primary_text` | Main textual content.                             |
| `news source` | `MasterRecord.source_name`  | Preferred publisher, outlet, or source string.    |
| `Real`        | `MasterRecord.is_real`      | Export-facing label where `1 = real`, `0 = fake`. |
| `tweet_num`   | `MasterRecord.tweet_num`    | Interaction-count proxy where available.          |

## LabelMapping

`LabelMapping` is the small documentation dataclass returned by each loader's `label_mapping()` method.

| Field             | Type             | Purpose                                                   |
| ----------------- | ---------------- | --------------------------------------------------------- | ------------------------------------------------------------------------- |
| `dataset`         | `str`            | Dataset name.                                             |
| `original_labels` | `list[str        | int]`                                                     | Source label values or normalized label buckets recognized by the loader. |
| `mapped_labels`   | `dict[str, str]` | Documented mapping from source label to `real` or `fake`. |
| `excluded_labels` | `list[str        | int]`                                                     | Ambiguous labels that are excluded by returning `None`.                   |
| `justification`   | `str`            | Rationale for the mapping and any exclusions.             |

## Relationship between the three dataclasses

1. Each dataset loader reads raw rows and emits a `UnifiedRecord` or `None`.
2. Cleaning runs on `UnifiedRecord` objects and preserves their schema.
3. `UnifiedRecord.to_master_record()` converts each kept record into a `MasterRecord`.
4. `MasterRecord.to_flat_dict()` produces the six-column export used for lightweight analysis.
5. `LabelMapping` documents the label policy used in step 1.

## Deduplication strategy

The master builder now removes duplicates before writing `master_records.*` and `master_flat_records.*`.

### Pass 1: exact ID duplicates

- Duplicates are detected by exact `MasterRecord.id` match.
- The first occurrence is kept.
- Later occurrences are removed with manifest notes marked `reason = duplicate_id`.

### Pass 2: content fingerprint duplicates

- Fingerprints are computed from `title`, `primary_text`, and `source_name` after `strip()` and `casefold()`.
- The fingerprint format is `title|primary_text|source_name`.
- Blank fingerprints are ignored.

When multiple records share the same fingerprint, the builder keeps the highest-priority dataset in this order:

1. ClaimReview
2. FakeNewsNet
3. Fakeddit
4. MuMiN

Within the same dataset priority, the earliest record in merge order is kept.

## Interpreting deduplication outputs

`master_dataset_manifest.json` now includes:

- `total_rows_before_deduplication`
- `total_rows`
- `duplicates.duplicate_id_count`
- `duplicates.duplicate_fingerprint_group_count`
- `duplicates.duplicate_fingerprint_count`
- `duplicates.duplicates_detected`
- `duplicates.duplicates_removed`
- `deduplication_notes`

Each `deduplication_notes` entry records the removed record, the kept record, and whether removal happened because of an exact ID collision or a fingerprint collision.

`master_dataset_coverage.json` now tracks `Title`, `Claim`, `news source`, `Real`, and `tweet_num` for the deduplicated flat export.
