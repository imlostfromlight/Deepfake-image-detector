"""
Downloads large model weights from HuggingFace Hub on first start.
Set HF_REPO env var to your HuggingFace model repo (e.g. "yourname/deepverify-weights").
Set HF_TOKEN env var if the repo is private.

Files downloaded if missing:
  - detector_best.pt         (329 MB) — CLIP+LoRA fine-tuned deepfake detector
  - detector/best_voice_model.pt (42 MB) — anti-spoof voice model
"""
import os
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
HF_REPO  = os.environ.get("HF_REPO", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

WEIGHTS = [
    (BASE / "detector_best.pt",                "detector_best.pt"),
    (BASE / "detector" / "best_voice_model.pt", "best_voice_model.pt"),
]

if not HF_REPO:
    print("[weights] HF_REPO not set — skipping download (assuming weights are present locally)")
    sys.exit(0)

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("[weights] huggingface_hub not installed — skipping download")
    sys.exit(0)

for local_path, hf_filename in WEIGHTS:
    if local_path.exists():
        print(f"[weights] {local_path.name} already present — skipping")
        continue
    print(f"[weights] Downloading {hf_filename} from {HF_REPO}...")
    try:
        downloaded = hf_hub_download(
            repo_id=HF_REPO,
            filename=hf_filename,
            token=HF_TOKEN or None,
            local_dir=str(local_path.parent),
        )
        print(f"[weights] {hf_filename} → {downloaded}")
    except Exception as e:
        print(f"[weights] ERROR downloading {hf_filename}: {e}")
        sys.exit(1)

print("[weights] All weights ready.")
