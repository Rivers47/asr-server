#!/usr/bin/env python3
"""
Compare Whisper inference backends on identical audio chunks.

The question this answers: is an Intel Arc backend worth wiring into the server?
CTranslate2 is NVIDIA-only, so an Arc path means replacing the ASR call with
torch-XPU or OpenVINO. That is only worth doing if it beats the CPU baseline by
enough to matter, at comparable output.

Every backend is fed the *same* chunk boundaries, so the numbers compare ASR
decoding rather than chunking strategy. Chunks come from the repo's ASMR VAD
when it is importable, and from fixed 30 s windows otherwise.

Each backend runs in its own subprocess. Loading CTranslate2, torch-XPU and
OpenVINO into one interpreter invites oneAPI/MKL runtime conflicts, and separate
processes also give honest peak-memory figures.

    python bench/bench_backends.py --list
    python bench/bench_backends.py audio.opus --backends ct2-cpu,torch-xpu,openvino-gpu

See bench/README.md for per-backend install and model conversion.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000
REPO_ROOT = Path(__file__).resolve().parents[1]

MAX_NEW_TOKENS = 440
# faster-whisper truncates hotwords to max_length // 2 - 1 tokens; the transformers
# backends are capped the same way so every backend sees the same hotword content.
HOTWORDS_TOKEN_CAP = 223
# generate() prepends start-of-transcript, language and task tokens to the prompt.
SPECIAL_TOKEN_SLOTS = 4


def prompt_from_hotwords(processor, hotwords: str, max_target_positions: int):
    """Hotwords as Whisper prompt tokens, with the max_new_tokens they leave room for.

    faster-whisper takes hotwords as a string and manages the token budget itself.
    transformers counts prompt + special tokens + max_new_tokens against
    max_target_positions and raises when they exceed it, so the budget is computed
    here rather than passing a constant.
    """
    if not hotwords:
        return None, MAX_NEW_TOKENS
    prompt_ids = processor.get_prompt_ids(hotwords, return_tensors="pt")
    if len(prompt_ids) - 1 > HOTWORDS_TOKEN_CAP:  # index 0 is <|startofprev|>
        prompt_ids = prompt_ids[: HOTWORDS_TOKEN_CAP + 1]
    budget = max_target_positions - len(prompt_ids) - SPECIAL_TOKEN_SLOTS
    return prompt_ids, max(1, min(MAX_NEW_TOKENS, budget))


# backend -> (import probe, human description, install hint)
BACKENDS = {
    "ct2-cpu": ("faster_whisper", "CTranslate2 on CPU (what the server runs today)", "uv sync"),
    "ct2-cuda": ("faster_whisper", "CTranslate2 on NVIDIA CUDA", "uv sync, on an NVIDIA host"),
    "torch-cpu": (
        "transformers",
        "transformers Whisper on CPU (control for torch-xpu)",
        "pip install torch transformers",
    ),
    "torch-xpu": (
        "transformers",
        "transformers Whisper on Intel XPU (Arc)",
        "pip install torch --index-url https://download.pytorch.org/whl/xpu",
    ),
    "torch-cuda": ("transformers", "transformers Whisper on NVIDIA CUDA", "pip install torch transformers"),
    "openvino-cpu": ("optimum.intel", "OpenVINO on CPU", "pip install optimum-intel[openvino]"),
    "openvino-gpu": ("optimum.intel", "OpenVINO on Intel GPU (Arc)", "pip install optimum-intel[openvino]"),
}


@dataclass
class Result:
    backend: str
    ok: bool
    detail: str = ""
    load_s: float = 0.0
    decode_s: float = 0.0
    audio_s: float = 0.0
    peak_rss_mb: float = 0.0
    chunks: int = 0
    text: str = ""

    @property
    def realtime_factor(self) -> float:
        return self.audio_s / self.decode_s if self.decode_s else 0.0


# --------------------------------------------------------------------------
# audio and chunking
# --------------------------------------------------------------------------


def load_audio(path: str) -> np.ndarray:
    """Decode to 16 kHz mono float32, preferring whatever is installed."""
    try:
        # Brings PyAV with it, so this handles opus/mp3/mkv/anything ffmpeg reads.
        from faster_whisper.audio import decode_audio

        return np.asarray(decode_audio(path, sampling_rate=SAMPLE_RATE), dtype=np.float32)
    except ImportError:
        pass

    # Without faster-whisper -- e.g. a torch-only venv -- only plain WAV works.
    import wave

    if not path.lower().endswith(".wav"):
        raise SystemExit(f"cannot decode {path}: install faster-whisper (which brings PyAV), or pass a 16 kHz mono WAV")
    with wave.open(path, "rb") as handle:
        if handle.getframerate() != SAMPLE_RATE or handle.getnchannels() != 1:
            raise SystemExit(f"{path} must be 16 kHz mono; got {handle.getframerate()} Hz, {handle.getnchannels()} ch")
        raw = handle.readframes(handle.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def plan_chunks(audio: np.ndarray, window_s: float, use_vad: bool) -> list[tuple[int, int]]:
    """Chunk boundaries as (start_sample, end_sample), identical for every backend."""
    if use_vad:
        try:
            sys.path.insert(0, str(REPO_ROOT))
            from asr.pipeline import create_contiguous_chunks, vad_segments_to_speech_spans
            from asr.vad_manager import VadConfig, VadModelManager

            config = VadConfig(
                onnx_model_path=str(REPO_ROOT / "models/whisper_vad.onnx"),
                onnx_metadata_path=str(REPO_ROOT / "models/whisper_vad_metadata.json"),
                threshold=0.4,
                min_speech_duration_ms=300,
                min_silence_duration_ms=100,
                speech_pad_ms=200,
            )
            manager = VadModelManager(config=config)
            # A manager whose ONNX session failed to build still answers, with an
            # empty segment list -- and empty segments produce chunk boundaries
            # indistinguishable from fixed windows. Check before trusting them,
            # so the plan is never labelled VAD-selected when no VAD ran.
            if manager.get_device() == "Not initialized":
                raise RuntimeError("VAD model did not load (missing onnxruntime, faster-whisper, or models/)")

            segments = manager.get_speech_timestamps(model_id="whisper_vad", audio=audio, sampling_rate=SAMPLE_RATE)
            if not segments:
                raise RuntimeError("VAD found no speech in this audio")

            spans = vad_segments_to_speech_spans(segments, SAMPLE_RATE)
            total_s = len(audio) / SAMPLE_RATE
            chunks = create_contiguous_chunks(spans, window_s, total_s, 0.4)
            if chunks:
                speech_s = sum(span.duration_s for span in spans)
                print(
                    f"chunk plan: {len(chunks)} VAD-selected boundaries ({speech_s:.1f}s speech in {total_s:.1f}s)",
                    file=sys.stderr,
                )
                return [
                    (int(chunk.start * SAMPLE_RATE), min(int(chunk.end * SAMPLE_RATE), len(audio))) for chunk in chunks
                ]
        except Exception as exc:  # noqa: BLE001 -- the fallback is fine, just say why
            print(f"VAD chunking unavailable ({exc}); falling back to fixed windows", file=sys.stderr)

    step = int(window_s * SAMPLE_RATE)
    bounds = [(start, min(start + step, len(audio))) for start in range(0, len(audio), step)]
    print(f"chunk plan: {len(bounds)} fixed {window_s:g}s windows", file=sys.stderr)
    return bounds


# --------------------------------------------------------------------------
# backends -- each returns (load_seconds, list_of_texts)
# --------------------------------------------------------------------------


def run_ct2(chunks: list[np.ndarray], device: str, args) -> tuple[float, list[str]]:
    from faster_whisper import WhisperModel

    started = time.perf_counter()
    model = WhisperModel(
        args.ct2_model,
        device=device,
        compute_type=args.compute_type,
        # Match the server, which defaults to half the logical count -- the
        # physical core count under SMT. CTranslate2's own default is 4 threads
        # regardless of machine size, which would understate the baseline.
        cpu_threads=args.cpu_threads or max(1, (os.cpu_count() or 2) // 2),
        num_workers=1,
    )
    load_s = time.perf_counter() - started

    def decode(audio: np.ndarray) -> str:
        segments, _ = model.transcribe(
            audio,
            language=args.language,
            task="transcribe",
            beam_size=args.beam_size,
            vad_filter=False,  # chunking is fixed up front; isolate the decoder
            condition_on_previous_text=False,
            hotwords=args.hotwords or None,
        )
        return "".join(segment.text for segment in segments).strip()

    return load_s, _decode_all(chunks, decode, args)


def run_torch(chunks: list[np.ndarray], device: str, args) -> tuple[float, list[str]]:
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    if device == "xpu" and not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        raise RuntimeError("torch.xpu is unavailable; install the XPU build and check your driver")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("torch.cuda is unavailable")

    dtype = torch.float16 if device in ("xpu", "cuda") else torch.float32

    started = time.perf_counter()
    processor = WhisperProcessor.from_pretrained(args.hf_model)
    model = WhisperForConditionalGeneration.from_pretrained(args.hf_model, dtype=dtype).to(device).eval()
    if device != "cpu":
        getattr(torch, device).synchronize()
    load_s = time.perf_counter() - started

    prompt_ids, max_new_tokens = prompt_from_hotwords(
        processor, args.hotwords, getattr(model.config, "max_target_positions", 448)
    )

    def decode(audio: np.ndarray) -> str:
        features = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features
        with torch.inference_mode():
            tokens = model.generate(
                features.to(device, dtype=dtype),
                num_beams=args.beam_size,
                language=args.language,
                task="transcribe",
                max_new_tokens=max_new_tokens,
                prompt_ids=prompt_ids,
            )
        if device != "cpu":
            getattr(torch, device).synchronize()
        return processor.batch_decode(tokens, skip_special_tokens=True)[0].strip()

    return load_s, _decode_all(chunks, decode, args)


def run_openvino(chunks: list[np.ndarray], device: str, args) -> tuple[float, list[str]]:
    from optimum.intel import OVModelForSpeechSeq2Seq
    from transformers import WhisperProcessor

    source = args.ov_model or args.hf_model
    started = time.perf_counter()
    processor = WhisperProcessor.from_pretrained(source)
    model = OVModelForSpeechSeq2Seq.from_pretrained(
        source,
        device=device.upper(),
        export=args.ov_model is None,  # export on the fly only if no IR was given
    )
    load_s = time.perf_counter() - started

    prompt_ids, max_new_tokens = prompt_from_hotwords(
        processor, args.hotwords, getattr(model.config, "max_target_positions", 448)
    )

    def decode(audio: np.ndarray) -> str:
        features = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features
        tokens = model.generate(
            features,
            num_beams=args.beam_size,
            language=args.language,
            task="transcribe",
            max_new_tokens=max_new_tokens,
            prompt_ids=prompt_ids,
        )
        return processor.batch_decode(tokens, skip_special_tokens=True)[0].strip()

    return load_s, _decode_all(chunks, decode, args)


def _decode_all(chunks: list[np.ndarray], decode, args) -> list[str]:
    """Warm up on the first chunk, then decode every chunk for real."""
    if args.warmup and chunks:
        decode(chunks[0])
    texts = []
    for index, chunk in enumerate(chunks, 1):
        texts.append(decode(chunk))
        print(f"  chunk {index}/{len(chunks)}", end="\r", file=sys.stderr, flush=True)
    print(" " * 30, end="\r", file=sys.stderr)
    return texts


RUNNERS = {
    "ct2-cpu": lambda c, a: run_ct2(c, "cpu", a),
    "ct2-cuda": lambda c, a: run_ct2(c, "cuda", a),
    "torch-cpu": lambda c, a: run_torch(c, "cpu", a),
    "torch-xpu": lambda c, a: run_torch(c, "xpu", a),
    "torch-cuda": lambda c, a: run_torch(c, "cuda", a),
    "openvino-cpu": lambda c, a: run_openvino(c, "cpu", a),
    "openvino-gpu": lambda c, a: run_openvino(c, "gpu", a),
}


# --------------------------------------------------------------------------
# worker: one backend, one process
# --------------------------------------------------------------------------


def run_worker(args) -> int:
    payload = np.load(args.chunks_file, allow_pickle=False)
    bounds = np.load(args.bounds_file, allow_pickle=False)
    chunks = [payload[start:end] for start, end in bounds]
    audio_s = sum(len(chunk) for chunk in chunks) / SAMPLE_RATE

    result = Result(backend=args.worker, ok=False, chunks=len(chunks), audio_s=audio_s)
    try:
        started = time.perf_counter()
        load_s, texts = RUNNERS[args.worker](chunks, args)
        result.decode_s = time.perf_counter() - started - load_s
        result.load_s = load_s
        result.text = " ".join(texts).strip()
        result.ok = True
    except Exception as exc:  # noqa: BLE001 -- a missing backend is an expected outcome
        result.detail = f"{type(exc).__name__}: {exc}"

    result.peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print("###RESULT###" + json.dumps(asdict(result)))
    return 0


# --------------------------------------------------------------------------
# parent
# --------------------------------------------------------------------------


def similarity(a: str, b: str) -> float:
    from difflib import SequenceMatcher

    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def probe(module: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(module) is not None
    except ImportError, ValueError:
        return False


def list_backends() -> int:
    print(f"{'backend':<15} {'available':<10} description")
    for name, (module, description, hint) in BACKENDS.items():
        mark = "yes" if probe(module) else "no"
        print(f"{name:<15} {mark:<10} {description}")
        if mark == "no":
            print(f"{'':<26} install: {hint}")
    return 0


def report(results: list[Result], audio_s: float) -> None:
    baseline = next((r for r in results if r.ok), None)

    print(f"\n{'backend':<15} {'load':>8} {'decode':>9} {'realtime':>10} {'peak RSS':>10} {'vs base':>9}")
    print("-" * 66)
    for result in results:
        if not result.ok:
            print(f"{result.backend:<15} {'skipped':>8}   {result.detail[:52]}")
            continue
        speedup = f"{result.realtime_factor / baseline.realtime_factor:.2f}x" if baseline else "-"
        print(
            f"{result.backend:<15} {result.load_s:7.1f}s {result.decode_s:8.1f}s "
            f"{result.realtime_factor:9.2f}x {result.peak_rss_mb:9.0f}M {speedup:>9}"
        )
    print(f"\n{audio_s:.1f}s of audio. 'realtime' is audio seconds per wall second; higher is faster.")

    if baseline:
        print(f"\nText agreement with {baseline.backend} (character ratio):")
        for result in results:
            if result.ok and result is not baseline:
                print(f"  {result.backend:<15} {similarity(baseline.text, result.text):.3f}")
        print("\nA low ratio means a real behaviour difference, not just speed -- read the")
        print("transcripts before trusting a fast backend. Pass --show-text to print them.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audio", nargs="?", help="audio file to benchmark")
    parser.add_argument("--list", action="store_true", help="show which backends are importable and exit")
    parser.add_argument("--backends", default="ct2-cpu", help="comma-separated; first is the baseline")
    parser.add_argument("--ct2-model", default=str(REPO_ROOT / "models"), help="CTranslate2 model directory")
    parser.add_argument("--hf-model", default="efwkjn/whisper-ja-1.5B", help="HF model for torch/OpenVINO")
    parser.add_argument("--ov-model", default=None, help="pre-exported OpenVINO IR directory (skips on-the-fly export)")
    parser.add_argument("--compute-type", default="int8", help="CTranslate2 compute type (default: int8)")
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=0,
        help="CTranslate2 CPU threads; 0 (default) matches the server -- half the logical count. "
        "The curve is steep on both sides of the physical core count, so sweep this before comparing backends.",
    )
    parser.add_argument("--language", default="ja")
    parser.add_argument(
        "--hotwords",
        default="",
        help="domain vocabulary to bias decoding toward. Passed to CTranslate2 as hotwords= and to "
        "the transformers backends as prompt_ids, truncated to the same token cap. The server runs "
        "with hotwords set, so comparisons made without this flag do not reflect production.",
    )
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--window", type=float, default=30.0, help="chunk length in seconds (default: 30)")
    parser.add_argument("--limit", type=int, default=0, help="use only the first N chunks (0 = all)")
    parser.add_argument("--no-vad", action="store_true", help="fixed windows instead of VAD-selected boundaries")
    parser.add_argument("--no-warmup", dest="warmup", action="store_false", help="skip the untimed first pass")
    parser.add_argument("--show-text", action="store_true", help="print each backend's transcript")
    parser.add_argument(
        "--scratch-dir",
        default=None,
        help="where to stage the shared chunk plan (default: $TMPDIR). Audio is written as float32, "
        "so an hour of input needs ~230 MB -- point this somewhere with room if /tmp is small.",
    )
    parser.add_argument(
        "--python",
        action="append",
        default=[],
        metavar="BACKEND=PATH",
        help=(
            "interpreter to run one backend with, e.g. --python torch-xpu=~/xpu-venv/bin/python. "
            "Repeatable. Use it to keep CTranslate2 and torch-XPU in separate environments, "
            "which avoids oneAPI/MKL runtime conflicts."
        ),
    )
    # internal
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--chunks-file", help=argparse.SUPPRESS)
    parser.add_argument("--bounds-file", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.list:
        return list_backends()
    if args.worker:
        return run_worker(args)
    if not args.audio:
        parser.error("an audio file is required (or use --list)")

    requested = [name.strip() for name in args.backends.split(",") if name.strip()]
    unknown = [name for name in requested if name not in RUNNERS]
    if unknown:
        parser.error(f"unknown backend(s): {', '.join(unknown)}; choose from {', '.join(RUNNERS)}")

    interpreters: dict[str, str] = {}
    for mapping in args.python:
        backend, _, path = mapping.partition("=")
        if not path:
            parser.error(f"--python expects BACKEND=PATH, got {mapping!r}")
        if backend not in RUNNERS:
            parser.error(f"--python names unknown backend {backend!r}")
        resolved = os.path.expanduser(path)
        if not os.path.exists(resolved):
            parser.error(f"--python {backend}: no interpreter at {resolved}")
        interpreters[backend] = resolved

    audio = load_audio(args.audio)
    print(f"{args.audio}: {len(audio) / SAMPLE_RATE:.1f}s at {SAMPLE_RATE} Hz", file=sys.stderr)
    bounds = plan_chunks(audio, args.window, use_vad=not args.no_vad)
    if args.limit:
        bounds = bounds[: args.limit]

    import tempfile

    with tempfile.TemporaryDirectory(dir=args.scratch_dir) as scratch:
        chunks_file = os.path.join(scratch, "audio.npy")
        bounds_file = os.path.join(scratch, "bounds.npy")
        np.save(chunks_file, audio)
        np.save(bounds_file, np.array(bounds, dtype=np.int64))

        results = []
        for name in requested:
            print(f"\n=== {name} ===", file=sys.stderr)
            interpreter = interpreters.get(name, sys.executable)
            if interpreter != sys.executable:
                print(f"(using {interpreter})", file=sys.stderr)
            command = [
                interpreter, __file__,
                "--worker", name,
                "--chunks-file", chunks_file,
                "--bounds-file", bounds_file,
                "--ct2-model", args.ct2_model,
                "--hf-model", args.hf_model,
                "--compute-type", args.compute_type,
                "--cpu-threads", str(args.cpu_threads),
                "--language", args.language,
                "--beam-size", str(args.beam_size),
            ]  # fmt: skip
            if args.ov_model:
                command += ["--ov-model", args.ov_model]
            if args.hotwords:
                command += ["--hotwords", args.hotwords]
            if not args.warmup:
                command.append("--no-warmup")

            completed = subprocess.run(command, capture_output=True, text=True)
            marker = [line for line in completed.stdout.splitlines() if line.startswith("###RESULT###")]
            if marker:
                results.append(Result(**json.loads(marker[0][len("###RESULT###") :])))
            else:
                tail = (completed.stderr or completed.stdout).strip().splitlines()
                results.append(Result(name, ok=False, detail=tail[-1] if tail else "worker produced no result"))

        audio_s = sum(end - start for start, end in bounds) / SAMPLE_RATE

    report(results, audio_s)
    if args.show_text:
        for result in results:
            if result.ok:
                print(f"\n--- {result.backend} ---\n{result.text}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
