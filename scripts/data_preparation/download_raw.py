"""
Download raw SpatialCorpus dataset from Hugging Face.

Default: 10xgenomics_xenium_mouse_brain_replicates.h5ad.
Skips download if the file already exists.

Uses Hugging Face API first; on failure (e.g. network/proxy), falls back to
direct URL download.

Usage:
  python scripts/data_preparation/download_raw.py
  python scripts/data_preparation/download_raw.py --dataset 10xgenomics_xenium_mouse_brain_replicates.h5ad
  python scripts/data_preparation/download_raw.py --direct-url  # use direct URL only
"""

import os
import sys
import argparse
import urllib.request

REPO_ID = "theislab/SpatialCorpus-110M"
HF_BASE = "https://huggingface.co/datasets/theislab/SpatialCorpus-110M/resolve/main"
SAVE_DIR = "data/raw"

DEFAULT_DATASET = "10xgenomics_xenium_mouse_brain_replicates.h5ad"
ALTERNATIVE_DATASETS = [
    "10xgenomics_xenium_mouse_brain_replicates.h5ad",
]


def download_direct(dataset: str, local_path: str, force: bool = False) -> str:
    """Download file via direct URL (no Hugging Face API)."""
    url = f"{HF_BASE}/{dataset}"
    if os.path.exists(local_path) and not force:
        print(f"File already exists: {local_path}")
        return local_path
    print(f"Downloading via direct URL...")
    print(f"  {url}")
    # Chunked download for large files
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        total = int(resp.headers.get("Content-Length", 0)) or None
        size_mb = (total / (1024 * 1024)) if total else 0
        if total:
            print(f"  Size: {size_mb:.1f} MB")
        chunk_size = 4 * 1024 * 1024  # 4 MB
        downloaded = 0
        with open(local_path, "wb") as f:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total and downloaded % (50 * 1024 * 1024) < chunk_size:
                    pct = 100 * downloaded / total
                    print(f"  Progress: {pct:.0f}%", end="\r")
        if total:
            print(f"  Progress: 100%")
    print(f"Successfully downloaded to: {local_path}")
    return local_path


def download(dataset: str = DEFAULT_DATASET, force: bool = False, direct_url: bool = False) -> str:
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    local_path = os.path.join(SAVE_DIR, dataset)
    if os.path.exists(local_path) and not force:
        print(f"File already exists: {local_path}")
        print("Skipping download. Use --force to re-download.")
        return local_path

    if direct_url:
        return download_direct(dataset, local_path, force)

    # Try Hugging Face API first
    try:
        from huggingface_hub import hf_hub_download
        print(f"Downloading {dataset} from {REPO_ID}...")
        path = hf_hub_download(
            repo_id=REPO_ID,
            filename=dataset,
            repo_type="dataset",
            local_dir=SAVE_DIR,
        )
        print(f"Successfully downloaded to: {path}")
        return path
    except Exception as e:
        print(f"Hugging Face API failed: {e}")
        print("Falling back to direct URL download...")
        try:
            return download_direct(dataset, local_path, force)
        except Exception as e2:
            print(f"Direct download also failed: {e2}")
            print("")
            print("You can download manually and place the file in data/raw/:")
            print(f"  {HF_BASE}/{dataset}")
            sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download SpatialCorpus raw dataset")
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
        help=f"Dataset filename (default: {DEFAULT_DATASET})",
    )
    parser.add_argument("--force", action="store_true", help="Re-download even if file exists")
    parser.add_argument(
        "--direct-url",
        action="store_true",
        help="Use direct URL download only (skip Hugging Face API)",
    )
    args = parser.parse_args()
    download(args.dataset, args.force, args.direct_url)
