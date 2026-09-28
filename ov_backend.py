#!/usr/bin/env python3
"""
OpenVINO backend: an exported IR behind faster-whisper's transcribe() shape.

This module lives outside the asr package because it imports transformers, which
asr/ is kept free of -- see tests/test_dependencies.py. Only the OpenVINO image
installs that stack, and asr/server.py imports this file lazily, inside the branch
that selects the backend.

``OpenVinoWhisperModel`` answers the same ``transcribe(audio, **config)`` call
``faster_whisper.WhisperModel`` does and returns the same ``(segments, info)``
pair, so ``TranscriptionService`` and ``Inference._transcribe_smart_chunks`` hold
either one without knowing which.

The IR is built outside the server -- see bench/README.md -- and selected by
directory: models/ov-int8 or models/ov-fp32.
"""

import gc
import logging
import os
import threading
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np

_OPTIMUM_INTEL_IMPORT_ERROR = None

try:
    from optimum.intel import OVModelForSpeechSeq2Seq
    from transformers import WhisperProcessor
except Exception as e:  # pragma: no cover - exercised only on an image without the stack
    _OPTIMUM_INTEL_IMPORT_ERROR = e
    OVModelForSpeechSeq2Seq = None
    WhisperProcessor = None

logger = logging.getLogger(__name__)

WHISPER_SAMPLING_RATE = 16_000
# Whisper's encoder consumes exactly 30 s; the exported encoder IR is fixed at
# [?, 128, 3000] frames, so longer input is decoded one window at a time.
WINDOW_SAMPLES = 30 * WHISPER_SAMPLING_RATE
# faster-whisper truncates hotwords to max_length // 2 - 1 tokens.
HOTWORDS_TOKEN_CAP = 223

# Passed to compile_model unless the caller overrides them. The server decodes one
# request at a time behind a lock, so inference streams beyond the first would size
# device buffers for parallelism it cannot use.
DEFAULT_OV_CONFIG = {
    "PERFORMANCE_HINT": "LATENCY",
    "NUM_STREAMS": "1",
}

# Keys this backend acts on. Everything else in the generation config belongs to
# CTranslate2 or to the pipeline itself and is reported once, then ignored.
HONOURED_KEYS = frozenset({"task", "language", "beam_size", "hotwords", "repetition_penalty", "clip_timestamps"})
PIPELINE_KEYS = frozenset(
    {"vad_filter", "vad_parameters", "smart_split_with_vad", "target_chunk_duration_s", "segment_merge"}
)


def _require_openvino():
    if OVModelForSpeechSeq2Seq is None or WhisperProcessor is None:
        raise RuntimeError(
            "Failed to import optimum.intel / transformers, which the OpenVINO backend needs. "
            f"Original error: {_OPTIMUM_INTEL_IMPORT_ERROR}"
        )
    return OVModelForSpeechSeq2Seq, WhisperProcessor


@dataclass
class RawSegment:
    """The subset of faster-whisper's Segment the pipeline reads: seconds and text."""

    start: float
    end: float
    text: str


def merged_ov_config(ov_config: dict[str, Any] | None) -> dict[str, Any]:
    """DEFAULT_OV_CONFIG with the caller's entries taking precedence."""
    merged = dict(DEFAULT_OV_CONFIG)
    merged.update(ov_config or {})
    return merged


def prompt_from_hotwords(processor, hotwords: str) -> Any:
    """Hotwords as Whisper prompt tokens, capped the way faster-whisper caps them.

    faster-whisper takes hotwords as a string and truncates at 223 tokens; the
    transformers equivalent is ``prompt_ids``, kept to the same cap so both backends
    see the same content.

    No ``max_new_tokens`` goes with it. The model's own ``max_length`` of 448 caps
    prompt, special tokens and generated text together, which is the constraint that
    actually applies, and setting both makes transformers warn on every call.
    """
    if not hotwords:
        return None
    prompt_ids = processor.get_prompt_ids(hotwords, return_tensors="pt")
    if len(prompt_ids) - 1 > HOTWORDS_TOKEN_CAP:  # index 0 is <|startofprev|>
        prompt_ids = prompt_ids[: HOTWORDS_TOKEN_CAP + 1]
    return prompt_ids


def _decode_path(audio_path: str | os.PathLike[Any]) -> np.ndarray:
    try:
        from faster_whisper.audio import decode_audio
    except Exception as e:
        raise RuntimeError(
            f"Decoding {audio_path} needs faster_whisper.audio (which brings PyAV). "
            f"Pass decoded samples instead. Original error: {e}"
        ) from e
    return np.asarray(decode_audio(audio_path, sampling_rate=WHISPER_SAMPLING_RATE), dtype=np.float32)


def _spans_from_clip_timestamps(clip_timestamps: Any, total_s: float) -> list[tuple[float, float]]:
    """Flat [start, end, start, end, ...] seconds, as vad_segments_to_clip_timestamps emits."""
    if not clip_timestamps:
        return [(0.0, total_s)]
    if isinstance(clip_timestamps, str):
        values = [float(part) for part in clip_timestamps.split(",") if part.strip()]
    elif isinstance(clip_timestamps, (list, tuple)) and clip_timestamps and isinstance(clip_timestamps[0], dict):
        return [(float(clip["start"]), float(clip["end"])) for clip in clip_timestamps]
    else:
        values = [float(value) for value in clip_timestamps]

    spans = [(values[i], values[i + 1]) for i in range(0, len(values) - 1, 2)]
    if len(values) % 2:  # trailing start with no end runs to the end of the audio
        spans.append((values[-1], total_s))
    return [(start, end) for start, end in spans if end > start]


class OpenVinoWhisperModel:
    """An exported OpenVINO Whisper IR, shaped like faster_whisper.WhisperModel.

    ``idle_unload_s`` decides when device memory is held. At 0 the model is compiled
    in the constructor and stays resident for the life of the process. Above 0 it is
    compiled on the first request and released again once idle for that long, which
    is what a GPU shared with other containers wants: an idle server holds no device
    memory, so nothing else has to be evicted to make room for it.

    One lock guards compile, decode and release together, so the idle timer can never
    unload a model mid-request.
    """

    def __init__(
        self,
        model_dir: str,
        *,
        device: str = "GPU",
        ov_config: dict[str, Any] | None = None,
        idle_unload_s: float = 0.0,
    ):
        model_cls, processor_cls = _require_openvino()
        self.model_dir = model_dir
        self.device = (device or "GPU").strip().upper()
        self.ov_config = merged_ov_config(ov_config)
        self.idle_unload_s = max(0.0, float(idle_unload_s or 0.0))
        self._model_cls = model_cls
        # The processor is tokeniser and feature extractor only: CPU-side, cheap, and
        # needed to answer requests, so it is loaded once and kept.
        self.processor = processor_cls.from_pretrained(model_dir)
        self.model: Any = None
        self._lock = threading.RLock()
        self._last_used = 0.0
        self._reported_ignored: set[str] = set()

        if self.idle_unload_s:
            logger.info("OpenVINO model loads on first request and unloads after %.0fs idle", self.idle_unload_s)
            threading.Thread(target=self._idle_watch, name="ov-idle-unload", daemon=True).start()
        else:
            self.load()

    def _load_model(self) -> Any:
        return self._model_cls.from_pretrained(self.model_dir, device=self.device, ov_config=self.ov_config)

    def load(self) -> None:
        """Compile for the device if it is not already compiled."""
        with self._lock:
            if self.model is None:
                started = time.monotonic()
                self.model = self._load_model()
                logger.info("OpenVINO model compiled for %s in %.1fs", self.device, time.monotonic() - started)
            self._last_used = time.monotonic()

    def unload(self) -> bool:
        """Release the compiled model, and with it the device memory it holds."""
        with self._lock:
            if self.model is None:
                return False
            self.model = None
            gc.collect()
            logger.info("OpenVINO model unloaded; %s memory released", self.device)
            return True

    def _idle_watch(self) -> None:
        interval = min(5.0, self.idle_unload_s)
        while True:
            time.sleep(interval)
            with self._lock:
                idle_for = time.monotonic() - self._last_used
                if self.model is not None and idle_for >= self.idle_unload_s:
                    self.unload()

    def _report_ignored(self, config: dict[str, Any]) -> None:
        ignored = sorted(set(config) - HONOURED_KEYS - PIPELINE_KEYS - self._reported_ignored)
        if ignored:
            self._reported_ignored.update(ignored)
            logger.info("OpenVINO backend ignores these decoding settings: %s", ", ".join(ignored))

    def _decode_window(self, window: np.ndarray, offset_s: float, params: dict[str, Any]) -> list[RawSegment]:
        features = self.processor(window, sampling_rate=WHISPER_SAMPLING_RATE, return_tensors="pt").input_features
        tokens = self.model.generate(features, return_timestamps=True, **params)
        # tokenizer.decode, not processor.batch_decode: the batched call accepts
        # output_offsets and then returns an empty offsets list, losing every timestamp.
        decoded = self.processor.tokenizer.decode(tokens[0], skip_special_tokens=True, output_offsets=True)

        window_end_s = offset_s + len(window) / WHISPER_SAMPLING_RATE
        segments: list[RawSegment] = []
        for chunk in decoded.get("offsets", []):
            text = chunk.get("text", "").strip()
            if not text:
                continue
            start, end = chunk["timestamp"]
            # The final segment of a window can come back without an end timestamp.
            absolute_start = offset_s + float(start)
            absolute_end = window_end_s if end is None else offset_s + float(end)
            if absolute_end > absolute_start:
                segments.append(RawSegment(absolute_start, min(absolute_end, window_end_s), text))
        return segments

    def transcribe(self, audio: Any, **config: Any) -> tuple[list[RawSegment], SimpleNamespace]:
        """Transcribe samples or a path, returning faster-whisper's (segments, info).

        ``info`` carries ``duration`` only. It deliberately has no
        ``duration_after_vad``: this backend runs no VAD of its own, and the
        pipeline falls back to its own speech total when the attribute is absent.
        """
        samples = _decode_path(audio) if isinstance(audio, (str, os.PathLike)) else np.asarray(audio, dtype=np.float32)
        total_s = len(samples) / WHISPER_SAMPLING_RATE
        self._report_ignored(config)

        params: dict[str, Any] = {
            "language": config.get("language") or "ja",
            "task": config.get("task") or "transcribe",
            "num_beams": max(1, int(config.get("beam_size") or 1)),
            "prompt_ids": prompt_from_hotwords(self.processor, config.get("hotwords") or ""),
        }
        if config.get("repetition_penalty"):
            params["repetition_penalty"] = float(config["repetition_penalty"])

        segments: list[RawSegment] = []
        with self._lock:  # also keeps the idle timer from unloading mid-request
            self.load()
            segments = self._decode_spans(samples, total_s, params, config.get("clip_timestamps"))
            self._last_used = time.monotonic()

        return segments, SimpleNamespace(duration=total_s)

    def _decode_spans(
        self, samples: np.ndarray, total_s: float, params: dict[str, Any], clip_timestamps: Any
    ) -> list[RawSegment]:
        segments: list[RawSegment] = []
        for span_start, span_end in _spans_from_clip_timestamps(clip_timestamps, total_s):
            start_sample = max(0, min(len(samples), int(round(span_start * WHISPER_SAMPLING_RATE))))
            end_sample = max(start_sample, min(len(samples), int(round(span_end * WHISPER_SAMPLING_RATE))))
            for window_start in range(start_sample, end_sample, WINDOW_SAMPLES):
                window = samples[window_start : min(window_start + WINDOW_SAMPLES, end_sample)]
                if not len(window):
                    continue
                segments.extend(self._decode_window(window, window_start / WHISPER_SAMPLING_RATE, params))
        return segments

    def describe(self) -> dict[str, Any]:
        with self._lock:
            loaded = self.model is not None
        return {
            "backend": "openvino",
            "model": self.model_dir,
            "device": self.device,
            "loaded": loaded,
            "idle_unload_s": self.idle_unload_s,
        }
