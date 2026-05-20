import argparse
import json
from pathlib import Path
import sys
import os
from typing import Any
from urllib.request import Request, urlopen

# Add parent directory to sys.path to allow running this script directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))


DEFAULT_CLAIMREVIEW_FEED_URL = "https://storage.googleapis.com/datacommons-feeds/claimreview/latest/data.json"

def download_hf_dataset(dataset_name: str, split: str, output_path: Path, max_samples: int = 5000):
    """Attempt to download a HuggingFace dataset using the datasets library."""
    try:
        from datasets import load_dataset
        import pandas as pd

        print(f"Downloading {dataset_name} ({split}) from Hugging Face...")
        ds = load_dataset(dataset_name, split=split)
        
        # Take a subset if it's huge
        if len(ds) > max_samples:
            print(f"Dataset is large ({len(ds)} rows). Taking a balanced/random sample of {max_samples}...")
            # We shuffle with a fixed seed to get a mix of labels
            ds = ds.shuffle(seed=42).select(range(max_samples))
            
        df_or_iter: Any = ds.to_pandas()
        if hasattr(df_or_iter, "to_csv"):
            df = df_or_iter
        else:
            df = pd.concat(list(df_or_iter), ignore_index=True)
        
        # Determine format by extension
        if output_path.suffix == ".tsv":
            df.to_csv(output_path, sep="\t", index=False)
        else:
            df.to_csv(output_path, index=False)
            
        print(f"✅ Saved {len(df)} records to {output_path}")
        return True
    except Exception as e:
        import traceback
        print(f"❌ Failed to download {dataset_name}: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False


def download_claimreview_dataset(
    output_path: Path,
    max_samples: int = 5000,
    feed_url: str = DEFAULT_CLAIMREVIEW_FEED_URL,
):
    """Download the public ClaimReview feed and save it in canonical loader format."""
    try:
        print(f"Downloading ClaimReview feed from {feed_url}...")
        request = Request(feed_url, headers={"User-Agent": "google-factcheck-starter/1.0"})
        with urlopen(request) as response:  # nosec B310
            payload = json.load(response)

        feed_items = payload.get("dataFeedElement")
        if isinstance(feed_items, list) and len(feed_items) > max_samples:
            print(f"ClaimReview feed is large ({len(feed_items)} rows). Keeping the first {max_samples}...")
            payload["dataFeedElement"] = feed_items[:max_samples]

        output_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        saved_count = len(payload.get("dataFeedElement", [])) if isinstance(payload, dict) else 0
        print(f"✅ Saved {saved_count} ClaimReview records to {output_path}")
        return True
    except Exception as e:
        import traceback
        print(f"❌ Failed to download ClaimReview: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description="Fetch datasets for fact-checking pipeline.")
    parser.add_argument("--max-samples", type=int, default=5000, help="Max rows per dataset to keep")
    args = parser.parse_args()

    # Define absolute path to data directory
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    data_dir.mkdir(exist_ok=True)

    print(f"=== Dataset Acquisition Pipeline (Saving to {data_dir}) ===\n")

    # 1. ClaimReview (Priority 1)
    claimreview_dir = data_dir / "claimreview"
    claimreview_dir.mkdir(exist_ok=True)
    claimreview_out = claimreview_dir / "data.json"

    if not claimreview_out.exists():
        download_claimreview_dataset(claimreview_out, args.max_samples)
    else:
        print(f"✅ ClaimReview already exists at {claimreview_out}")

    print("\n-----------------------------------\n")

    # 2. Fakeddit (Priority 2)
    fakeddit_dir = data_dir / "fakeddit"
    fakeddit_dir.mkdir(exist_ok=True)
    fakeddit_out = fakeddit_dir / "fakeddit_sample.tsv"
    
    if not fakeddit_out.exists():
        download_hf_dataset("rtfarchitect/fakeddit_sample", "train", fakeddit_out, args.max_samples)
    else:
        print(f"✅ Fakeddit already exists at {fakeddit_out}")

    print("\n-----------------------------------\n")

    # 3. FakeNewsNet (Priority 3)
    fnn_dir = data_dir / "fakenewsnet"
    fnn_dir.mkdir(exist_ok=True)
    fnn_out = fnn_dir / "fakenewsnet.csv"
    
    if not fnn_out.exists():
        download_hf_dataset("rickstello/FakeNewsNet", "train", fnn_out, args.max_samples)
    else:
        print(f"✅ FakeNewsNet already exists at {fnn_out}")

    print("\n-----------------------------------\n")

    # 4. MuMiN (Priority 4)
    mumin_dir = data_dir / "mumin"
    mumin_dir.mkdir(exist_ok=True)
    mumin_out = mumin_dir / "mumin.csv"
    
    if not mumin_out.exists():
        download_hf_dataset("ju-resplande/MuMiN-PT", "train", mumin_out, args.max_samples)
    else:
        print(f"✅ MuMiN already exists at {mumin_out}")

    print("\n=== Acquisition Complete ===")
    print("If you had missing libraries, please install them and re-run this script:")
    print("pip install datasets pandas pyarrow")

if __name__ == "__main__":
    main()
