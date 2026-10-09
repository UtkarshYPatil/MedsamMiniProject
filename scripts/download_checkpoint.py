"""Download MedSAM2 checkpoints into checkpoints/."""
import os
from huggingface_hub import hf_hub_download

MODELS = [
    "MedSAM2_CTLesion.pt",   # CT lesion fine-tune -- primary for this project
    "MedSAM2_latest.pt",     # general recommendation -- benchmark against
]

if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)
    for m in MODELS:
        p = hf_hub_download(
            repo_id="wanglab/MedSAM2",
            filename=m,
            local_dir="checkpoints",
        )
        print(f"downloaded {m} -> {p}")
