#!/usr/bin/env python3
"""
Inference script with custom VAD injection support
"""

import json
import logging
import os
from collections import ChainMap
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pyjson5

# Import GPU/runtime-heavy deps defensively so `infer --help` still works on
# machines that don't have the required GPU runtime DLLs installed.
_FASTER_WHISPER_IMPORT_ERROR = None
_CTRANSLATE2_IMPORT_ERROR = None

try:
    from faster_whisper import BatchedInferencePipeline, WhisperModel
    from faster_whisper.audio import decode_audio
except Exception as e:
    _FASTER_WHISPER_IMPORT_ERROR = e
    WhisperModel = None
    BatchedInferencePipeline = None
    decode_audio = None

try:
    import ctranslate2
except Exception as e:
    _CTRANSLATE2_IMPORT_ERROR = e
    ctranslate2 = None

# Import our VAD injection system
from .injection import inject_vad
from .vad_manager import VadConfig, VadModelManager


def format_duration(seconds: float) -> str:
    """Render a duration the way the log lines expect (2h 3m 4s / 3m 4.5s / 4.50s)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    if hours > 0:
        return f"{hours}h {minutes}m {secs:0.0f}s"
    if minutes > 0:
        return f"{minutes}m {secs:0.1f}s"
    return f"{secs:0.2f}s"


def format_percentage(value: float) -> str:
    return f"{value * 100:0.1f}%"


WHISPER_TASKS = ("transcribe", "translate")
WHISPER_SAMPLING_RATE = 16_000
MAX_SMART_CHUNK_DURATION_S = 30.0


def _normalize_whisper_task(task: Any) -> str:
    if not isinstance(task, str):
        raise ValueError(f"Whisper task must be one of {', '.join(WHISPER_TASKS)}")

    normalized = task.strip().lower()
    if normalized not in WHISPER_TASKS:
        raise ValueError(f"Invalid Whisper task '{task}'. Expected one of: {', '.join(WHISPER_TASKS)}")
    return normalized


def _require_ctranslate2():
    if ctranslate2 is None:
        raise RuntimeError(
            f"Failed to import ctranslate2. This build may be missing required GPU runtime libraries. "
            f"Original error: {_CTRANSLATE2_IMPORT_ERROR}"
        )
    return ctranslate2


def _require_faster_whisper():
    if WhisperModel is None or BatchedInferencePipeline is None or decode_audio is None:
        raise RuntimeError(
            f"Failed to import faster_whisper. This build may be missing required runtime libraries. "
            f"Original error: {_FASTER_WHISPER_IMPORT_ERROR}"
        )
    return WhisperModel, BatchedInferencePipeline


def select_best_compute_type(device: str) -> str:
    """
    Automatically select the best compute type based on device and available types.

    Preference order:
    - bfloat16 > float16 > int8 types > float32
    - Prefer int8 over float32 for better memory usage

    Args:
        device: The device to use ('cpu', 'cuda', or 'auto')

    Returns:
        The best available compute type for the device
    """
    ct2 = _require_ctranslate2()

    # Normalize and accept friendly aliases.
    device = (device or "auto").strip().lower()
    if device in {"amd", "rocm", "hip"}:
        device = "cuda"  # HIP builds still use the public device name "cuda".

    # Determine the actual device if 'auto' is specified.
    actual_device = device
    if device == "auto":
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        if cuda_visible in {"", "-1"}:
            actual_device = "cpu"
        else:
            try:
                actual_device = "cuda" if ct2.get_cuda_device_count() > 0 else "cpu"
            except Exception:
                actual_device = "cpu"
        logger.info(f"Auto-detected device: {actual_device}")

    # Get supported compute types for the device.
    try:
        supported_types = ct2.get_supported_compute_types(actual_device)
    except Exception as e:
        logger.warning(f"Could not get supported compute types for {actual_device}: {e}")
        # Fallback to safe default
        return "int8" if actual_device == "cpu" else "float16"

    # Define preference order
    # Prefer bfloat16 > float16 > int8 types > float32
    preference_order = [
        "bfloat16",
        "float16",
        "int16",  # For CPU
        "int8_bfloat16",
        "int8_float16",
        "int8_float32",
        "int8",
        "float32",  # Least preferred due to memory usage
    ]

    # Select the best available type based on preference
    for compute_type in preference_order:
        if compute_type in supported_types:
            logger.info(f"Auto-selected compute type '{compute_type}' for device '{actual_device}'")
            return compute_type

    # If nothing matched (shouldn't happen), use a safe default
    default = "int8" if actual_device == "cpu" else "float16"
    logger.warning(f"No preferred compute type found, using default '{default}'")
    return default


def _coerce_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    raise ValueError(f"Expected boolean value, got {value!r}")


@dataclass
class Segment:
    start: int  # ms
    end: int  # ms
    text: str


@dataclass(frozen=True)
class SpeechSpan:
    start: float
    end: float

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(frozen=True)
class AudioChunk:
    index: int
    start: float
    end: float

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(frozen=True)
class SegmentMergeOptions:
    enabled: bool = True
    max_gap_ms: int = 2_000
    max_duration_ms: int = 20_000


@dataclass(frozen=True)
class SmartSplitOptions:
    enabled: bool = True
    target_chunk_duration_s: float = MAX_SMART_CHUNK_DURATION_S
    split_window_factor: float = 0.4


def _normalize_merge_text(text: str) -> str:
    return " ".join(text.strip().split())


def vad_segments_to_clip_timestamps(
    vad_segments: list[dict[str, Any]], sampling_rate: int = WHISPER_SAMPLING_RATE, *, batched: bool = False
) -> list[float] | list[dict[str, float]]:
    if batched:
        clips: list[dict[str, float]] = []
        for segment in vad_segments:
            start = float(segment["start"]) / sampling_rate
            end = float(segment["end"]) / sampling_rate
            if end > start:
                clips.append({"start": start, "end": end})
        return clips

    timestamps: list[float] = []
    for segment in vad_segments:
        start = float(segment["start"]) / sampling_rate
        end = float(segment["end"]) / sampling_rate
        if end > start:
            timestamps.extend([start, end])
    return timestamps


def vad_segments_to_speech_spans(
    vad_segments: list[dict[str, Any]], sampling_rate: int = WHISPER_SAMPLING_RATE
) -> list[SpeechSpan]:
    spans: list[SpeechSpan] = []
    for segment in vad_segments:
        start = float(segment["start"]) / sampling_rate
        end = float(segment["end"]) / sampling_rate
        if end > start:
            spans.append(SpeechSpan(start=start, end=end))
    return spans


def create_contiguous_chunks(
    segments: list[SpeechSpan],
    max_duration: float,
    total_duration: float,
    split_window_factor: float = 0.4,
) -> list[AudioChunk]:
    if max_duration <= 0:
        raise ValueError("max_duration must be greater than 0")
    if total_duration <= 0:
        return []
    if total_duration <= max_duration:
        return [AudioChunk(0, 0.0, total_duration)]

    chunks: list[AudioChunk] = []
    current_start = 0.0
    sorted_segments = sorted((span for span in segments if span.end > span.start), key=lambda span: span.start)

    while current_start < total_duration:
        potential_end = current_start + max_duration
        if potential_end >= total_duration:
            chunks.append(AudioChunk(len(chunks), current_start, total_duration))
            break

        decision_zone_start = current_start + (max_duration * (1 - split_window_factor))
        best_split: float | None = None
        best_gap_duration = 0.0

        for previous, current in zip(sorted_segments, sorted_segments[1:], strict=False):
            gap_start = previous.end
            gap_end = current.start
            if decision_zone_start <= gap_start and gap_end <= potential_end:
                gap_duration = gap_end - gap_start
                if gap_duration > 0.1 and gap_duration > best_gap_duration:
                    best_gap_duration = gap_duration
                    best_split = gap_start + (gap_duration / 2)

        split_point = best_split if best_split is not None else potential_end
        split_point = max(current_start, min(split_point, potential_end, total_duration))
        chunks.append(AudioChunk(len(chunks), current_start, split_point))
        current_start = split_point

    return chunks


def _max_segment_end(start: int, end: int, max_duration_ms: int | None) -> int:
    if max_duration_ms is None or max_duration_ms <= 0:
        return end
    return min(end, start + max_duration_ms)


def enforce_segment_timeline(segments: list[Segment], max_duration_ms: int | None = None) -> list[Segment]:
    normalized: list[Segment] = []

    for segment in sorted((s for s in segments if s.text.strip()), key=lambda s: (s.start, s.end)):
        start = max(segment.start, normalized[-1].end if normalized else segment.start)
        end = _max_segment_end(start, segment.end, max_duration_ms)
        if end <= start:
            continue
        normalized.append(Segment(start=start, end=end, text=segment.text))

    return normalized


def merge_segments(segments: list[Segment], options: SegmentMergeOptions | None = None) -> list[Segment]:
    if options is None:
        options = SegmentMergeOptions()

    segments = [s for s in segments if s.text.strip()]
    segments.sort(key=lambda s: (s.start, s.end))
    if not options.enabled:
        return segments

    merged: list[Segment] = []

    for seg in segments:
        if not merged:
            merged.append(seg)
            continue

        last = merged[-1]

        gap_ms = seg.start - last.end
        if gap_ms > options.max_gap_ms:
            merged.append(seg)
            continue

        merged_duration_ms = seg.end - last.start
        if merged_duration_ms > options.max_duration_ms:
            merged.append(seg)
            continue

        last_norm = _normalize_merge_text(last.text)
        seg_norm = _normalize_merge_text(seg.text)

        if seg_norm.startswith(last_norm):
            merged[-1] = Segment(start=last.start, end=max(last.end, seg.end), text=seg.text)
            continue

        if last_norm.startswith(seg_norm) or last_norm.endswith(seg_norm):
            merged[-1] = Segment(start=last.start, end=max(last.end, seg.end), text=last.text)
            continue

        if seg_norm.endswith(last_norm):
            merged[-1] = Segment(start=last.start, end=max(last.end, seg.end), text=seg.text)
            continue

        merged.append(seg)

    return merged


class SubWriter:
    @classmethod
    def txt(cls, segments: list[Segment], path: str):
        lines = []
        for _idx, segment in enumerate(segments):
            lines.append(f"{segment.text}\n")
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(lines)

    @classmethod
    def lrc(cls, segments: list[Segment], path: str):
        lines = []
        for idx, segment in enumerate(segments):
            start_ts = cls.lrc_timestamp(segment.start)
            end_es = cls.lrc_timestamp(segment.end)
            lines.append(f"[{start_ts}]{segment.text}\n")
            if idx != len(segments) - 1:
                next_start = segments[idx + 1].start
                if next_start is not None and end_es == cls.lrc_timestamp(next_start):
                    continue
            lines.append(f"[{end_es}]\n")
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(lines)

    @staticmethod
    def lrc_timestamp(ms: int) -> str:
        m = ms // 60_000
        ms = ms - m * 60_000
        s = ms // 1_000
        ms = ms - s * 1_000
        ms = ms // 10
        return f"{m:02d}:{s:02d}.{ms:02d}"

    @classmethod
    def vtt(cls, segments: list[Segment], path: str):
        lines = ["WebVTT\n\n"]
        for idx, segment in enumerate(segments):
            lines.append(f"{idx + 1}\n")
            lines.append(f"{cls.vtt_timestamp(segment.start)} --> {cls.vtt_timestamp(segment.end)}\n")
            lines.append(f"{segment.text}\n\n")
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(lines)

    @classmethod
    def vtt_timestamp(cls, ms: int):
        return cls._timestamp(ms, ".")

    @classmethod
    def srt(cls, segments: list[Segment], path: str):
        lines = []
        for idx, segment in enumerate(segments):
            lines.append(f"{idx + 1}\n")
            lines.append(f"{cls.srt_timestamp(segment.start)} --> {cls.srt_timestamp(segment.end)}\n")
            lines.append(f"{segment.text}\n\n")
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(lines)

    @classmethod
    def srt_timestamp(cls, ms: int):
        return cls._timestamp(ms, ",")

    @classmethod
    def _timestamp(cls, ms: int, delim: str):
        h = ms // 3600_000
        ms -= h * 3600_000
        m = ms // 60_000
        ms -= m * 60_000
        s = ms // 1_000
        ms -= s * 1_000
        return f"{h:02d}:{m:02d}:{s:02d}{delim}{ms:03d}"


@dataclass
class InferenceTask:
    audio_path: str
    sub_prefix: str
    sub_formats: list[str]


logger = logging.getLogger(__name__)
log_handler = logging.StreamHandler()
log_handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(log_handler)


class Inference:
    sub_writers = {
        "lrc": SubWriter.lrc,
        "srt": SubWriter.srt,
        "vtt": SubWriter.vtt,
        "txt": SubWriter.txt,
    }

    def __init__(self, args):
        self.args = args
        self.model_name_or_path = args.model_name_or_path
        self.vad_injected = False
        self.vad_manager = None
        self.device = (args.device or "auto").strip().lower()
        if self.device in {"amd", "rocm", "hip"}:
            # CTranslate2 HIP backend still uses the public device name "cuda".
            self.device = "cuda"
        # Auto-select compute type if 'auto' or 'default' is specified
        if args.compute_type in ["auto", "default"]:
            self.compute_type = select_best_compute_type(self.device)
        else:
            self.compute_type = args.compute_type

        # Load generation config
        self.generation_config, self.segment_merge_options, self.smart_split_options = self._load_generation_config(
            args
        )

        # Setup VAD injection (whisper_vad is always used)
        self._setup_vad_injection()

        logger.info(f"Generation config: {self.generation_config}")
        logger.info(
            "Segment merge: enabled=%s, max_gap_ms=%s, max_duration_ms=%s",
            self.segment_merge_options.enabled,
            self.segment_merge_options.max_gap_ms,
            self.segment_merge_options.max_duration_ms,
        )
        logger.info(
            "Smart VAD split: enabled=%s, target_chunk_duration_s=%s",
            self.smart_split_options.enabled,
            self.smart_split_options.target_chunk_duration_s,
        )

    def _load_generation_config(self, args) -> tuple[dict[str, Any], SegmentMergeOptions, SmartSplitOptions]:
        """Load and process generation configuration"""
        # Default config
        config: dict[str, Any] = {
            "language": "ja",
            "task": "translate",
            "vad_filter": True,
        }

        segment_merge_options = SegmentMergeOptions()
        smart_split_options = SmartSplitOptions()

        # Load from file if exists
        if os.path.exists(args.generation_config):
            with open(args.generation_config, encoding="utf-8") as f:
                file_config = pyjson5.decode_io(f)
                file_segment_merge = file_config.pop("segment_merge", None)
                if isinstance(file_segment_merge, dict):
                    segment_merge_options = SegmentMergeOptions(
                        enabled=bool(file_segment_merge.get("enabled", segment_merge_options.enabled)),
                        max_gap_ms=int(file_segment_merge.get("max_gap_ms", segment_merge_options.max_gap_ms)),
                        max_duration_ms=int(
                            file_segment_merge.get("max_duration_ms", segment_merge_options.max_duration_ms)
                        ),
                    )
                file_smart_split = file_config.pop("smart_split_with_vad", None)
                file_target_chunk_duration = file_config.pop("target_chunk_duration_s", None)
                smart_split_options = SmartSplitOptions(
                    enabled=_coerce_bool(file_smart_split, default=smart_split_options.enabled),
                    target_chunk_duration_s=(
                        float(file_target_chunk_duration)
                        if file_target_chunk_duration is not None
                        else smart_split_options.target_chunk_duration_s
                    ),
                    split_window_factor=smart_split_options.split_window_factor,
                )
                config = dict(**ChainMap(file_config, config))

        config["task"] = _normalize_whisper_task(args.task if args.task is not None else config.get("task"))

        # Process VAD parameters from config file
        if "vad_parameters" in config:
            vad_params = config.pop("vad_parameters")

            # Convert to VadOptions format
            vad_options = {}

            # Map common parameters
            if "threshold" in vad_params:
                vad_options["threshold"] = vad_params["threshold"]
            if "neg_threshold" in vad_params:
                vad_options["neg_threshold"] = vad_params["neg_threshold"]
            if "min_speech_duration_ms" in vad_params:
                vad_options["min_speech_duration_ms"] = vad_params["min_speech_duration_ms"]
            if "max_speech_duration_s" in vad_params:
                vad_options["max_speech_duration_s"] = vad_params["max_speech_duration_s"]
            if "min_silence_duration_ms" in vad_params:
                vad_options["min_silence_duration_ms"] = vad_params["min_silence_duration_ms"]
            if "speech_pad_ms" in vad_params:
                vad_options["speech_pad_ms"] = vad_params["speech_pad_ms"]

            config["vad_parameters"] = vad_options

        # Override with command line arguments
        if args.vad_threshold is not None:
            if "vad_parameters" not in config:
                config["vad_parameters"] = {}
            config["vad_parameters"]["threshold"] = args.vad_threshold

        if args.vad_min_speech_duration_ms is not None:
            if "vad_parameters" not in config:
                config["vad_parameters"] = {}
            config["vad_parameters"]["min_speech_duration_ms"] = args.vad_min_speech_duration_ms

        if args.vad_min_silence_duration_ms is not None:
            if "vad_parameters" not in config:
                config["vad_parameters"] = {}
            config["vad_parameters"]["min_silence_duration_ms"] = args.vad_min_silence_duration_ms

        if args.vad_speech_pad_ms is not None:
            if "vad_parameters" not in config:
                config["vad_parameters"] = {}
            config["vad_parameters"]["speech_pad_ms"] = args.vad_speech_pad_ms

        # Override segment merge options with command line arguments
        segment_merge_options = SegmentMergeOptions(
            enabled=args.merge_segments if args.merge_segments is not None else segment_merge_options.enabled,
            max_gap_ms=args.merge_max_gap_ms if args.merge_max_gap_ms is not None else segment_merge_options.max_gap_ms,
            max_duration_ms=(
                args.merge_max_duration_ms
                if args.merge_max_duration_ms is not None
                else segment_merge_options.max_duration_ms
            ),
        )

        smart_split_options = SmartSplitOptions(
            enabled=(
                _coerce_bool(args.smart_split_with_vad, default=smart_split_options.enabled)
                if args.smart_split_with_vad is not None
                else smart_split_options.enabled
            ),
            target_chunk_duration_s=(
                args.target_chunk_duration_s
                if args.target_chunk_duration_s is not None
                else smart_split_options.target_chunk_duration_s
            ),
            split_window_factor=smart_split_options.split_window_factor,
        )
        if smart_split_options.target_chunk_duration_s <= 0:
            raise ValueError("target_chunk_duration_s must be greater than 0")
        if smart_split_options.target_chunk_duration_s > MAX_SMART_CHUNK_DURATION_S:
            smart_split_options = SmartSplitOptions(
                enabled=smart_split_options.enabled,
                target_chunk_duration_s=MAX_SMART_CHUNK_DURATION_S,
                split_window_factor=smart_split_options.split_window_factor,
            )

        return config, segment_merge_options, smart_split_options

    def _vad_progress_callback(self, chunk_idx, total_chunks, device):
        """Progress callback for VAD processing."""
        progress_pct = (chunk_idx / total_chunks) * 100
        # Use carriage return to update the same line
        print(
            "\r  " + f"VAD Progress: {chunk_idx}/{total_chunks} chunks ({progress_pct:0.1f}%) on {device}",
            end="",
            flush=True,
        )
        if chunk_idx == total_chunks:
            print()  # New line when done

    def _setup_vad_injection(self):
        """Setup whisper_vad injection - always enforced"""
        # Always use whisper_vad model
        vad_model = "whisper_vad"

        logger.info("Initializing enhanced VAD model...")

        # Create VAD config with progress callback
        vad_config = VadConfig(default_model=vad_model)

        # Apply VAD parameters from generation config
        if "vad_parameters" in self.generation_config:
            vad_params = self.generation_config["vad_parameters"]
            if "threshold" in vad_params:
                vad_config.threshold = vad_params["threshold"]
            if "neg_threshold" in vad_params:
                vad_config.neg_threshold = vad_params["neg_threshold"]
            if "min_speech_duration_ms" in vad_params:
                vad_config.min_speech_duration_ms = vad_params["min_speech_duration_ms"]
            if "max_speech_duration_s" in vad_params:
                vad_config.max_speech_duration_s = vad_params["max_speech_duration_s"]
            if "min_silence_duration_ms" in vad_params:
                vad_config.min_silence_duration_ms = vad_params["min_silence_duration_ms"]
            if "speech_pad_ms" in vad_params:
                vad_config.speech_pad_ms = vad_params["speech_pad_ms"]

        # Load ONNX VAD configuration from metadata
        vad_metadata_path = "models/whisper_vad_metadata.json"
        vad_config.onnx_model_path = "models/whisper_vad.onnx"
        vad_config.onnx_metadata_path = vad_metadata_path

        # Read model configuration from metadata JSON if it exists
        if os.path.exists(vad_metadata_path):
            try:
                with open(vad_metadata_path) as f:
                    metadata = json.load(f)

                # Load model configuration from metadata
                vad_config.frame_duration_ms = metadata.get("frame_duration_ms", 20)
                vad_config.chunk_duration_ms = metadata.get("total_duration_ms", 30000)

                logger.info(f"Loaded VAD configuration from {vad_metadata_path}")
            except Exception as e:
                logger.warning(f"Failed to load VAD metadata from {vad_metadata_path}: {e}")
                logger.warning("Using default VAD configuration")
                # Fallback to defaults
                vad_config.frame_duration_ms = 20
                vad_config.chunk_duration_ms = 30000
        else:
            # Use defaults if metadata file doesn't exist
            logger.warning(f"VAD metadata file not found at {vad_metadata_path}")
            logger.warning("Using default VAD configuration")
            vad_config.frame_duration_ms = 20
            vad_config.chunk_duration_ms = 30000

        # Hardcoded runtime configuration
        vad_config.force_cpu = False
        vad_config.num_threads = 8

        # Inject VAD with progress callback
        self.vad_manager = VadModelManager(
            config=vad_config,
            ttl=vad_config.ttl,
            progress_callback=self._vad_progress_callback,
        )
        inject_vad(
            model_id=vad_model,
            config=vad_config,
            progress_callback=self._vad_progress_callback,
        )
        self.vad_injected = True
        logger.info(f"✓ Enhanced VAD activated (threshold={vad_config.threshold})")

    def _prepare_transcription(
        self, audio_path: str, *, batched: bool, overrides: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any], float | None]:
        config = dict(self.generation_config)
        if overrides:
            config.update(overrides)

        if self.smart_split_options.enabled or not config.get("vad_filter") or "clip_timestamps" in config:
            return audio_path, config, None

        if self.vad_manager is None:
            return audio_path, config, None

        audio = decode_audio(audio_path, sampling_rate=WHISPER_SAMPLING_RATE)
        vad_parameters = config.get("vad_parameters") or {}
        vad_segments = self.vad_manager.get_speech_timestamps(
            model_id="whisper_vad",
            audio=audio,
            sampling_rate=WHISPER_SAMPLING_RATE,
            **vad_parameters,
        )
        duration_after_vad = sum(segment["end"] - segment["start"] for segment in vad_segments) / WHISPER_SAMPLING_RATE

        config["vad_filter"] = False
        config["clip_timestamps"] = vad_segments_to_clip_timestamps(
            vad_segments,
            WHISPER_SAMPLING_RATE,
            batched=batched,
        )
        config.setdefault("beam_size", 1)
        config.setdefault("condition_on_previous_text", False)

        return audio, config, duration_after_vad

    def _plan_smart_chunks(self, audio_path: str) -> tuple[Any, list[AudioChunk], float | None]:
        audio = decode_audio(audio_path, sampling_rate=WHISPER_SAMPLING_RATE)
        duration = len(audio) / WHISPER_SAMPLING_RATE

        if not self.smart_split_options.enabled or not self.generation_config.get("vad_filter"):
            return audio, [AudioChunk(0, 0.0, duration)], None

        if self.vad_manager is None:
            return audio, [AudioChunk(0, 0.0, duration)], None

        vad_parameters = self.generation_config.get("vad_parameters") or {}
        vad_segments = self.vad_manager.get_speech_timestamps(
            model_id="whisper_vad",
            audio=audio,
            sampling_rate=WHISPER_SAMPLING_RATE,
            **vad_parameters,
        )
        duration_after_vad = sum(segment["end"] - segment["start"] for segment in vad_segments) / WHISPER_SAMPLING_RATE

        spans = vad_segments_to_speech_spans(vad_segments, WHISPER_SAMPLING_RATE)
        chunks = create_contiguous_chunks(
            spans,
            min(self.smart_split_options.target_chunk_duration_s, MAX_SMART_CHUNK_DURATION_S),
            duration,
            self.smart_split_options.split_window_factor,
        )
        if not chunks:
            chunks = [AudioChunk(0, 0.0, duration)]
        logger.info("Smart VAD split planned %s chunk(s)", len(chunks))

        return audio, chunks, duration_after_vad

    def _transcribe_smart_chunks(
        self, model, task: InferenceTask, overrides: dict[str, Any] | None = None
    ) -> tuple[list[Segment], Any]:
        audio, chunks, outer_duration_after_vad = self._plan_smart_chunks(task.audio_path)
        duration = len(audio) / WHISPER_SAMPLING_RATE
        config = dict(self.generation_config)
        if overrides:
            config.update(overrides)
        config.pop("clip_timestamps", None)
        config["vad_filter"] = bool(config.get("vad_filter", True))
        config.setdefault("beam_size", 1)
        config.setdefault("condition_on_previous_text", False)

        if outer_duration_after_vad == 0:
            return [], SimpleNamespace(duration=duration, duration_after_vad=0)

        segments: list[Segment] = []
        inner_duration_after_vad = 0.0
        has_inner_duration = False

        for chunk in chunks:
            start_sample = max(0, min(len(audio), int(round(chunk.start * WHISPER_SAMPLING_RATE))))
            end_sample = max(start_sample, min(len(audio), int(round(chunk.end * WHISPER_SAMPLING_RATE))))
            if end_sample <= start_sample:
                continue
            chunk_audio = audio[start_sample:end_sample]
            logger.debug(
                "Smart VAD chunk %s/%s: %s --> %s",
                chunk.index + 1,
                len(chunks),
                SubWriter.srt_timestamp(int(round(chunk.start * 1_000))),
                SubWriter.srt_timestamp(int(round(chunk.end * 1_000))),
            )
            chunk_segments_iter, chunk_info = model.transcribe(chunk_audio, **config)
            if hasattr(chunk_info, "duration_after_vad"):
                inner_duration_after_vad += float(chunk_info.duration_after_vad)
                has_inner_duration = True
            chunk_offset_ms = int(round(chunk.start * 1_000))
            chunk_end_ms = int(round(chunk.end * 1_000))
            for _segment in chunk_segments_iter:
                segment = Segment(
                    start=chunk_offset_ms + int(round(_segment.start * 1_000)),
                    end=chunk_offset_ms + int(round(_segment.end * 1_000)),
                    text=_segment.text.strip(),
                )
                if segment.start >= chunk_end_ms:
                    continue
                segment = Segment(segment.start, min(segment.end, chunk_end_ms), segment.text)
                if segment.end > segment.start:
                    segments.append(segment)
                    logger.debug(
                        f"[{SubWriter.lrc_timestamp(segment.start)} --> "
                        f"{SubWriter.lrc_timestamp(segment.end)}] {segment.text}"
                    )

        duration_after_vad = inner_duration_after_vad if has_inner_duration else outer_duration_after_vad
        return segments, SimpleNamespace(duration=duration, duration_after_vad=duration_after_vad)

    def _log_duration(self, duration: float, duration_after_vad: float) -> None:
        if duration == duration_after_vad or duration_after_vad == 0:
            logger.info(f"Duration: {format_duration(duration)}")
            return

        rate = duration_after_vad / duration
        logger.info(
            f"Duration: {format_duration(duration)} → {format_duration(duration_after_vad)} ({format_percentage(rate)} speech detected)"
        )

    def _should_use_smart_split(self) -> bool:
        return bool(self.smart_split_options.enabled and self.generation_config.get("vad_filter", True))
