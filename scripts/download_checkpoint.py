"""
Download MedSAM2 weights into checkpoints/.

    python scripts/download_checkpoint.py

Tries huggingface_hub first, falls back to a direct HTTPS fetch. Skips files
that are already present and plausibly sized.
"""
import os
import sys
from pathlib import Path

REPO = "wanglab/MedSAM2"
MODELS = [
    "MedSAM2_CTLesion.pt",   # CT lesion fine-tune -- primary for this project
    "MedSAM2_latest.pt",     # general model -- benchmark comparison
]
MIN_BYTES = 10_000_000       # anything smaller is an error page, not a model


def already_there(p: Path) -> bool:
    return p.exists() and p.stat().st_size >= MIN_BYTES


def via_hub(name, dest):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=REPO, filename=name, local_dir=str(dest))


def via_http(name, dest):
    import urllib.request
    url = f"https://huggingface.co/{REPO}/resolve/main/{name}"
    out = dest / name
    tmp = dest / (name + ".part")

    def hook(blocks, bs, total):
        if total > 0:
            pct = min(100, blocks * bs * 100 // total)
            print(f"\r    {pct:3d}%  {blocks*bs/1e6:7.0f} / {total/1e6:.0f} MB",
                  end="", flush=True)

    urllib.request.urlretrieve(url, tmp, reporthook=hook)
    print()
    tmp.rename(out)
    return str(out)


def main():
    dest = Path("checkpoints")
    dest.mkdir(exist_ok=True)
    failed = []

    for name in MODELS:
        target = dest / name
        if already_there(target):
            print(f"ok   {name}  ({target.stat().st_size/1e6:.0f} MB, already present)")
            continue

        print(f"get  {name}")
        for method in (via_hub, via_http):
            try:
                path = method(name, dest)
                p = Path(path)
                if p.stat().st_size < MIN_BYTES:
                    raise RuntimeError(f"only {p.stat().st_size} bytes - not a model file")
                print(f"ok   {name}  ({p.stat().st_size/1e6:.0f} MB) via {method.__name__}")
                break
            except Exception as e:
                print(f"     {method.__name__} failed: {type(e).__name__}: {e}")
        else:
            failed.append(name)

    if failed:
        print(f"\nFAILED: {', '.join(failed)}")
        print("Download manually from https://huggingface.co/wanglab/MedSAM2/tree/main")
        print(f"and place the .pt files in {dest.resolve()}")
        sys.exit(1)


if __name__ == "__main__":
    main()
