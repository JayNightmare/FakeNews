#!/usr/bin/env python3
"""Upload the generated FakeNews model bundle to Hugging Face Hub."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload FakeNews model bundle")
    parser.add_argument("--repo-id", default="JayNightmare/FakeNews", help="Hub repo id, e.g. username/repo")
    parser.add_argument("--folder", default=str(Path(__file__).resolve().parent), help="Bundle folder to upload")
    parser.add_argument("--private", action="store_true", help="Create the repo as private")
    args = parser.parse_args()

    api = HfApi()
    api.create_repo(repo_id=args.repo_id, private=args.private, exist_ok=True)
    api.upload_folder(
        folder_path=args.folder,
        repo_id=args.repo_id,
        repo_type="model",
        commit_message="Upload FakeNews model bundle",
    )
    print(f"Uploaded model bundle to https://huggingface.co/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
