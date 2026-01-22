"""
Download Nicheformer Pretrained Weights

Downloads pretrained Nicheformer weights from Mendeley Data.
These weights are required to use the actual Nicheformer model in the wrapper.

Usage:
    python scripts/milestone3/download_nicheformer_weights.py [output_dir]

The weights will be saved as: output_dir/nicheformer_pretrained.ckpt
"""

import os
import sys
import requests
from pathlib import Path

# Mendeley Data URL (update with actual URL from the repository)
# The actual URL should be obtained from:
# https://data.mendeley.com/preview/87gm9hrgm8?a=d95a6dde-e054-4245-a7eb-0522d6ea7dff
MENDELEY_BASE_URL = "https://data.mendeley.com/preview/87gm9hrgm8"
DEFAULT_OUTPUT_DIR = "data/pretrained"

def download_file(url: str, output_path: str, chunk_size: int = 8192):
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'wb') as f:
        downloaded = 0
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"\rDownloading: {percent:.1f}% ({downloaded}/{total_size} bytes)", end='', flush=True)
    
    print(f"\nDownloaded to: {output_path}")

def main():
    """Download Nicheformer pretrained weights."""
    output_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUTPUT_DIR
    output_path = os.path.join(output_dir, "nicheformer_pretrained.ckpt")
    
    print("=" * 60)
    print("Nicheformer Pretrained Weights Downloader")
    print("=" * 60)
    print(f"\nOutput directory: {output_dir}")
    print(f"Output file: {output_path}")
    
    if os.path.exists(output_path):
        response = input(f"\nFile already exists: {output_path}\nOverwrite? (y/n): ")
        if response.lower() != 'y':
            print("Download cancelled.")
            return
    
    print("\n[INFO] To download the weights:")
    print("1. Visit: https://data.mendeley.com/preview/87gm9hrgm8")
    print("2. Download the pretrained model checkpoint file")
    print("3. Save it to:", output_path)
    print("\nAlternatively, if you have the direct download URL, update this script.")
    
    # If you have the direct download URL, uncomment and use:
    # download_url = "YOUR_DIRECT_DOWNLOAD_URL_HERE"
    # print(f"\nDownloading from: {download_url}")
    # download_file(download_url, output_path)
    
    print("\n[NOTE] Manual download required. See instructions above.")

if __name__ == "__main__":
    main()
