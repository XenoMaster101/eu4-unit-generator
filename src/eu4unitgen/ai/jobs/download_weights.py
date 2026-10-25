"""Setup job (Track P4): the AI weights downloaded into ``<AI root>/weights/<name>`` with huggingface_hub, in the
AI environment. A folder already there is kept. DINOv3 is gated: the user accepts its licence on Hugging Face and
the Setup screen passes their token as HF_TOKEN (never stored by the app).

usage: python download_weights.py <AI root> name=repo [name=repo ...]
"""
import sys
from pathlib import Path


def main(root: str, pairs: list[str]) -> None:
    from huggingface_hub import snapshot_download
    for pair in pairs:
        name, repo = pair.split("=", 1)
        dest = Path(root) / "weights" / name
        if dest.is_dir() and any(dest.iterdir()):
            print(f"{name}: already there", flush=True)
            continue
        print(f"{name}: downloading {repo} ...", flush=True)
        snapshot_download(repo, local_dir=str(dest))
        print(f"{name}: done", flush=True)
    print("Weights ready.", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
