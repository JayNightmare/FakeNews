import argparse
import csv
import json
import os
from pathlib import Path
import sys
from typing import Any

# Add parent directory to sys.path to allow running this script directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))


CLAIMREVIEW_SNAPSHOT_VERSION = "2026_06_12"
CLAIMREVIEW_SNAPSHOT_FILE = "claim_reviews.json"
CLAIMREVIEW_RELEASE_URL = (
    "https://github.com/MartinoMensio/claimreview-data/releases/tag/2026_06_12"
)

def _serialise_cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return value


def _write_hf_split(ds: Any, output_path: Path) -> int:
    delimiter = "\t" if output_path.suffix == ".tsv" else ","
    fieldnames = list(getattr(ds, "column_names", []) or [])
    if not fieldnames:
        raise ValueError(f"Dataset split for {output_path} has no columns")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter=delimiter)
        writer.writeheader()
        for row in ds:
            writer.writerow({name: _serialise_cell(row.get(name)) for name in fieldnames})
            count += 1
    return count


def _download_hf_dataset_once(
    dataset_name: str,
    split: str,
    output_path: Path,
    max_samples: int | None,
) -> int:
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split=split, token=os.getenv("HF_TOKEN"))
    original_size = len(ds)
    selected = ds
    if max_samples is not None and original_size > max_samples:
        print(f"Dataset is large ({original_size} rows). Taking a balanced/random sample of {max_samples}...")
        selected = ds.shuffle(seed=42).select(range(max_samples))

    saved = _write_hf_split(selected, output_path)
    return saved


def download_hf_dataset(
    dataset_name: str,
    split: str,
    output_path: Path,
    max_samples: int | None = None,
    fallback_samples: int | None = None,
):
    """Attempt to download a Hugging Face dataset, preferring the full split first.

    When a full-corpus attempt fails, retry with `fallback_samples` when provided.
    """
    try:
        attempt_label = "full corpus" if max_samples is None else f"sample ({max_samples})"
        print(f"Downloading {dataset_name} ({split}) from Hugging Face [{attempt_label}]...")
        saved = _download_hf_dataset_once(dataset_name, split, output_path, max_samples)
        print(f"✅ Saved {saved} records to {output_path}")
        return True
    except Exception as e:
        import traceback

        print(f"❌ Failed to download {dataset_name}: {type(e).__name__}: {e}")
        traceback.print_exc()
        if max_samples is None and fallback_samples:
            print(f"Retrying {dataset_name} with sampled fallback ({fallback_samples})...")
            try:
                saved = _download_hf_dataset_once(dataset_name, split, output_path, fallback_samples)
                print(f"✅ Saved fallback sample with {saved} records to {output_path}")
                return True
            except Exception as retry_error:
                print(
                    f"❌ Fallback download also failed for {dataset_name}: "
                    f"{type(retry_error).__name__}: {retry_error}"
                )
                traceback.print_exc()
        return False


def validate_claimreview_snapshot(claimreview_dir: Path) -> bool:
    """Verify that the pinned ClaimReview snapshot exists locally."""
    snapshot_file = claimreview_dir / CLAIMREVIEW_SNAPSHOT_VERSION / CLAIMREVIEW_SNAPSHOT_FILE
    if snapshot_file.exists():
        print(
            "✅ ClaimReview snapshot found at "
            f"{snapshot_file}"
        )
        return True

    print(
        "❌ Missing pinned ClaimReview snapshot. Expected file: "
        f"{snapshot_file}"
    )
    print(f"   Download release assets from: {CLAIMREVIEW_RELEASE_URL}")
    return False

def main():
    parser = argparse.ArgumentParser(description="Fetch datasets for fact-checking pipeline.")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max rows per dataset to keep. Omit to attempt a full corpus.",
    )
    parser.add_argument(
        "--fallback-samples",
        type=int,
        default=5000,
        help="If a full Hugging Face download fails, retry with this sample size. Use 0 to disable.",
    )
    args = parser.parse_args()

    # Define absolute path to data directory
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    data_dir.mkdir(exist_ok=True)

    print(f"=== Dataset Acquisition Pipeline (Saving to {data_dir}) ===\n")

    # 1. ClaimReview (Priority 1)
    claimreview_dir = data_dir / "claimreview"
    claimreview_dir.mkdir(exist_ok=True)
    if not validate_claimreview_snapshot(claimreview_dir):
        print("Stopping because ClaimReview is now pinned to the 2026_06_12 snapshot.")
        raise SystemExit(1)

    print("\n-----------------------------------\n")

    # 2. Fakeddit (Priority 2)
    fakeddit_dir = data_dir / "fakeddit"
    fakeddit_dir.mkdir(exist_ok=True)
    fakeddit_out = fakeddit_dir / "fakeddit_sample.tsv"
    
    if not fakeddit_out.exists():
        download_hf_dataset(
            "rtfarchitect/fakeddit_sample",
            "train",
            fakeddit_out,
            max_samples=args.max_samples,
            fallback_samples=args.fallback_samples or None,
        )
    else:
        print(f"✅ Fakeddit already exists at {fakeddit_out}")

    print("\n-----------------------------------\n")

    # 3. FakeNewsNet (Priority 3)
    fnn_dir = data_dir / "fakenewsnet"
    fnn_dir.mkdir(exist_ok=True)
    fnn_out = fnn_dir / "fakenewsnet.csv"
    
    if not fnn_out.exists():
        download_hf_dataset(
            "rickstello/FakeNewsNet",
            "train",
            fnn_out,
            max_samples=args.max_samples,
            fallback_samples=args.fallback_samples or None,
        )
    else:
        print(f"✅ FakeNewsNet already exists at {fnn_out}")

    print("\n-----------------------------------\n")

    # 4. MuMiN (Priority 4)
    mumin_dir = data_dir / "mumin"
    mumin_dir.mkdir(exist_ok=True)
    mumin_out = mumin_dir / "mumin.csv"
    
    if not mumin_out.exists():
        download_hf_dataset(
            "ju-resplande/MuMiN-PT",
            "train",
            mumin_out,
            max_samples=args.max_samples,
            fallback_samples=args.fallback_samples or None,
        )
    else:
        print(f"✅ MuMiN already exists at {mumin_out}")

    print("\n=== Acquisition Complete ===")
    print("If you had missing libraries, please install them and re-run this script:")
    print("pip install datasets pyarrow")

if __name__ == "__main__":
    main()
