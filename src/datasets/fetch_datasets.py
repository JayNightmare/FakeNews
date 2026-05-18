import os
import argparse
from pathlib import Path

import sys
import os

# Add parent directory to sys.path to allow running this script directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def download_hf_dataset(dataset_name: str, split: str, output_path: Path, max_samples: int = 5000):
    """Attempt to download a HuggingFace dataset using the datasets library."""
    try:
        from datasets import load_dataset
        print(f"Downloading {dataset_name} ({split}) from Hugging Face...")
        ds = load_dataset(dataset_name, split=split)
        
        # Take a subset if it's huge
        if len(ds) > max_samples:
            print(f"Dataset is large ({len(ds)} rows). Taking a balanced/random sample of {max_samples}...")
            # We shuffle with a fixed seed to get a mix of labels
            ds = ds.shuffle(seed=42).select(range(max_samples))
            
        df = ds.to_pandas()
        
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

def main():
    parser = argparse.ArgumentParser(description="Fetch datasets for fact-checking pipeline.")
    parser.add_argument("--max-samples", type=int, default=5000, help="Max rows per dataset to keep")
    args = parser.parse_args()

    # Define absolute path to data directory
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    data_dir.mkdir(exist_ok=True)

    print(f"=== Dataset Acquisition Pipeline (Saving to {data_dir}) ===\n")

    # 1. Fakeddit (Priority 1)
    fakeddit_dir = data_dir / "fakeddit"
    fakeddit_dir.mkdir(exist_ok=True)
    fakeddit_out = fakeddit_dir / "fakeddit_sample.tsv"
    
    if not fakeddit_out.exists():
        download_hf_dataset("rtfarchitect/fakeddit_sample", "train", fakeddit_out, args.max_samples)
    else:
        print(f"✅ Fakeddit already exists at {fakeddit_out}")

    print("\n-----------------------------------\n")

    # 2. FakeNewsNet (Priority 2)
    fnn_dir = data_dir / "fakenewsnet"
    fnn_dir.mkdir(exist_ok=True)
    fnn_out = fnn_dir / "fakenewsnet.csv"
    
    if not fnn_out.exists():
        download_hf_dataset("rickstello/FakeNewsNet", "train", fnn_out, args.max_samples)
    else:
        print(f"✅ FakeNewsNet already exists at {fnn_out}")

    print("\n-----------------------------------\n")

    # 3. MuMiN (Priority 3)
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
