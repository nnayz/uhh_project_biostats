import os
from huggingface_hub import hf_hub_download

REPO_ID = "theislab/SpatialCorpus-110M"
# Use the replicates file for easier federated splitting
FILENAME = "10xgenomics_xenium_mouse_brain_replicates.h5ad" 
SAVE_DIR = "data/raw/"

def download():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    
    print(f"Downloading {FILENAME} (Xenium Mouse Brain Replicates)...")
    path = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        repo_type="dataset",
        local_dir=SAVE_DIR
    )
    print(f"Successfully downloaded to: {path}")

if __name__ == "__main__":
    download()