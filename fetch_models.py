#!/usr/bin/env python3
"""
Download the two models the server needs into models/.

    python fetch_models.py                # both
    python fetch_models.py --only vad     # just the VAD

The ASR model is ~2.9 GB and the VAD ~114 MB. Files already present with the
right size are skipped, so an interrupted run can simply be repeated.
"""

import argparse
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

HF = "https://huggingface.co"

# Transcription model: whisper-large-v3 geometry, bf16, Japanese.
ASR_REPO = "TransWithAI/whisper-ja-1.5B-ct2"
ASR_FILES = ["config.json", "model.bin", "preprocessor_config.json", "tokenizer.json", "vocabulary.json"]

# ASMR-tuned VAD: whisper-base encoder + 2 decoder layers, 20 ms frame resolution.
VAD_REPO = "TransWithAI/Whisper-Vad-EncDec-ASMR-onnx"
VAD_FILES = {"whisper_vad.onnx": "whisper_vad.onnx", "whisper_vad_metadata.json": "whisper_vad_metadata.json"}


def remote_size(url: str) -> int | None:
    request = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            length = response.headers.get("Content-Length")
            return int(length) if length else None
    except (urllib.error.URLError, ValueError):
        return None


def download(url: str, target: Path) -> None:
    expected = remote_size(url)
    if target.exists() and expected is not None and target.stat().st_size == expected:
        print(f"  = {target.name} ({expected / 1e6:.0f} MB, already complete)")
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    print(f"  > {target.name}" + (f" ({expected / 1e6:.0f} MB)" if expected else ""))

    with urllib.request.urlopen(url, timeout=60) as response, open(partial, "wb") as f:
        done = 0
        while chunk := response.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if expected:
                pct = done * 100 / expected
                print(f"\r    {done / 1e6:7.0f} / {expected / 1e6:.0f} MB  {pct:5.1f}%", end="", flush=True)
        if expected:
            print()

    if expected is not None and partial.stat().st_size != expected:
        partial.unlink()
        raise RuntimeError(f"{target.name}: expected {expected} bytes, got {partial.stat().st_size}")
    partial.replace(target)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", choices=["asr", "vad"], help="Fetch just one of the two models")
    parser.add_argument("--models-dir", default="models", type=Path, help="Destination (default: models)")
    parser.add_argument("--mirror", default=HF, help=f"Base URL, e.g. https://hf-mirror.com (default: {HF})")
    args = parser.parse_args()

    os.chdir(Path(__file__).resolve().parent)
    root = args.models_dir

    try:
        if args.only != "vad":
            print(f"ASR model: {ASR_REPO}")
            for name in ASR_FILES:
                download(f"{args.mirror}/{ASR_REPO}/resolve/main/{name}", root / name)
        if args.only != "asr":
            print(f"VAD model: {VAD_REPO}")
            for remote, local in VAD_FILES.items():
                download(f"{args.mirror}/{VAD_REPO}/resolve/main/{remote}", root / local)
    except (urllib.error.URLError, RuntimeError) as exc:
        print(f"\nfailed: {exc}", file=sys.stderr)
        print("If huggingface.co is slow or blocked, retry with --mirror https://hf-mirror.com", file=sys.stderr)
        return 1

    print(f"\nModels ready in {root.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
