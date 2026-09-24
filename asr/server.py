"""
HTTP transcription server.

Loads the VAD and Whisper models once at startup and serves them over HTTP, so
callers pay the multi-second model load only on boot instead of on every file.

Endpoints:
    GET  /health              -- liveness plus the loaded model configuration
    POST /transcribe          -- audio in, transcript out

Drives the same ``Inference`` object the CLI uses, so VAD injection, smart
chunking and segment merging behave identically.
"""

import argparse
import json
import logging
import os
import re
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from typing import Any
from urllib.parse import parse_qs, urlparse

from .pipeline import (
    WHISPER_SAMPLING_RATE,
    WHISPER_TASKS,
    Inference,
    InferenceTask,
    Segment,
    _require_faster_whisper,
    enforce_segment_timeline,
    merge_segments,
)

logger = logging.getLogger(__name__)

# Subtitle formats the /transcribe endpoint can return, plus the content type
# each one is served with.
SUB_FORMATS = {
    "srt": "application/x-subrip; charset=utf-8",
    "vtt": "text/vtt; charset=utf-8",
    "lrc": "text/plain; charset=utf-8",
    "txt": "text/plain; charset=utf-8",
}

# Extensions accepted on an upload. Anything else falls back to .bin and lets
# ffmpeg sniff the container.
KNOWN_SUFFIXES = {
    ".wav",
    ".flac",
    ".mp3",
    ".m4a",
    ".aac",
    ".ogg",
    ".opus",
    ".wma",
    ".aiff",
    ".alac",
    ".mp4",
    ".mkv",
    ".avi",
    ".mov",
    ".webm",
    ".flv",
    ".ts",
    ".wmv",
}


# --------------------------------------------------------------------------
# multipart/form-data
# --------------------------------------------------------------------------


@dataclass
class FormPart:
    name: str
    filename: str | None
    content: bytes


def parse_content_type(header: str) -> tuple[str, dict[str, str]]:
    """Split a Content-Type header into its media type and parameters."""
    pieces = header.split(";")
    media_type = pieces[0].strip().lower()
    params: dict[str, str] = {}
    for piece in pieces[1:]:
        if "=" not in piece:
            continue
        key, _, value = piece.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
            value = value[1:-1]
        params[key.strip().lower()] = value
    return media_type, params


def _strip_part_edges(chunk: bytes) -> bytes:
    """Drop the CRLF that follows a boundary and the one that precedes the next."""
    if chunk.startswith(b"\r\n"):
        chunk = chunk[2:]
    elif chunk.startswith(b"\n"):
        chunk = chunk[1:]
    if chunk.endswith(b"\r\n"):
        chunk = chunk[:-2]
    elif chunk.endswith(b"\n"):
        chunk = chunk[:-1]
    return chunk


def parse_multipart(body: bytes, boundary: str) -> list[FormPart]:
    """Parse a multipart/form-data body.

    Deliberately minimal: form uploads only, no nested multiparts and no
    transfer encodings, which is all a file upload needs. Like every other
    boundary-splitting parser this trusts the client to pick a boundary that
    does not occur in the payload, as RFC 2046 requires.
    """
    delimiter = b"--" + boundary.encode("utf-8", "strict")
    parts: list[FormPart] = []

    for chunk in body.split(delimiter)[1:]:
        if chunk.startswith(b"--"):  # closing delimiter; the rest is epilogue
            break
        chunk = _strip_part_edges(chunk)
        if not chunk:
            continue

        head, sep, content = chunk.partition(b"\r\n\r\n")
        if not sep:
            head, sep, content = chunk.partition(b"\n\n")
        if not sep:
            continue

        name = None
        filename = None
        for line in head.decode("utf-8", "replace").splitlines():
            if not line.lower().startswith("content-disposition:"):
                continue
            _, params = parse_content_type(line.partition(":")[2])
            name = params.get("name")
            filename = params.get("filename")
            break

        if name is not None:
            parts.append(FormPart(name=name, filename=filename or None, content=content))

    return parts


def safe_suffix(filename: str | None) -> str:
    """Pick a temp-file extension from an upload's name without trusting it."""
    if not filename:
        return ".bin"
    suffix = os.path.splitext(os.path.basename(filename))[1].lower()
    if suffix in KNOWN_SUFFIXES and re.fullmatch(r"\.[a-z0-9]{1,5}", suffix):
        return suffix
    return ".bin"


# Decoding settings a caller may override per request. Everything else --
# language, task, the VAD parameters, chunking -- is process-wide and comes from
# generation_config.json5, because changing it per request would mean reloading
# or re-planning work the server does once at startup.
MAX_HOTWORDS_CHARS = 1024  # Whisper's prompt window is 224 tokens; this is well past it
MAX_BEAM_SIZE = 10  # decode cost scales with beam width, so cap what a caller can ask for


def parse_overrides(query: dict[str, list[str]]) -> dict[str, Any]:
    """Read per-request decoding overrides off the query string.

    Raises ValueError with a caller-facing message for anything malformed.
    """
    overrides: dict[str, Any] = {}

    if "hotwords" in query:
        hotwords = query["hotwords"][0]
        if len(hotwords) > MAX_HOTWORDS_CHARS:
            raise ValueError(f"hotwords is {len(hotwords)} characters; the limit is {MAX_HOTWORDS_CHARS}")
        # An explicit empty value is meaningful: it clears whatever the config set.
        overrides["hotwords"] = hotwords

    if "beam_size" in query:
        raw = query["beam_size"][0]
        try:
            beam_size = int(raw)
        except ValueError:
            raise ValueError(f"beam_size must be an integer, got {raw!r}") from None
        if not 1 <= beam_size <= MAX_BEAM_SIZE:
            raise ValueError(f"beam_size must be between 1 and {MAX_BEAM_SIZE}, got {beam_size}")
        overrides["beam_size"] = beam_size

    return overrides


# --------------------------------------------------------------------------
# transcription
# --------------------------------------------------------------------------


class TranscriptionError(Exception):
    """Raised when a single request fails; reported to the client as a 500."""


class MissingModelError(RuntimeError):
    """Raised at startup when a required model file is absent."""


def _require_models(inference) -> None:
    """Refuse to start without the models, rather than degrading silently.

    A missing VAD is the dangerous one: ``Inference`` only logs a warning, the
    server comes up, ``/health`` reports ok -- and then every request returns an
    empty transcript, because with no speech spans the chunk planner reports
    zero speech and the decoder is never reached. A server answering 200 with
    "" is worse than one that does not come up.
    """
    if not inference.vad_manager or inference.vad_manager.get_device() == "Not initialized":
        raise MissingModelError(
            f"VAD model not loaded. Expected models/whisper_vad.onnx relative to {os.getcwd()}.\n"
            "Run: python fetch_models.py --only vad"
        )

    # model_name_or_path may be a HuggingFace repo id, which faster-whisper
    # downloads itself -- only check it when it names a local directory.
    asr_path = inference.model_name_or_path
    if os.path.isdir(asr_path) and not os.path.exists(os.path.join(asr_path, "model.bin")):
        raise MissingModelError(f"No model.bin in {os.path.abspath(asr_path)}.\nRun: python fetch_models.py --only asr")


class TranscriptionService:
    """Owns the loaded models and runs one transcription at a time.

    CTranslate2 and the ONNX VAD session are both driven through process-global
    state -- ``injection.py`` patches ``faster_whisper.vad`` for the whole
    interpreter -- so requests are serialised behind a lock rather than run
    concurrently.
    """

    def __init__(self, args: argparse.Namespace, max_queue: int = 8):
        self.inference = Inference(args)
        _require_models(self.inference)
        self.max_queue = max_queue
        self._lock = threading.Lock()
        self._waiting = 0
        self._waiting_lock = threading.Lock()

        logger.info("Loading Whisper model from %s", self.inference.model_name_or_path)
        started = time.monotonic()
        whisper_model_cls, _batched_cls = _require_faster_whisper()
        self.model = whisper_model_cls(
            self.inference.model_name_or_path,
            device=self.inference.device,
            compute_type=self.inference.compute_type,
            cpu_threads=self.inference.cpu_threads,
            # Transcriptions are serialised behind a lock, so a second worker
            # would only duplicate the model in memory.
            num_workers=1,
        )
        logger.info(
            "Model ready in %.1fs (device=%s, compute_type=%s, task=%s)",
            time.monotonic() - started,
            self.inference.device,
            self.inference.compute_type,
            self.inference.generation_config.get("task"),
        )

    @property
    def queue_depth(self) -> int:
        with self._waiting_lock:
            return self._waiting

    def describe(self) -> dict[str, Any]:
        config = self.inference.generation_config
        return {
            "model": self.inference.model_name_or_path,
            "device": self.inference.device,
            "compute_type": self.inference.compute_type,
            "cpu_threads": self.inference.cpu_threads,
            "task": config.get("task"),
            "language": config.get("language"),
            "vad_device": self.inference.vad_manager.get_device() if self.inference.vad_manager else None,
            "smart_split": self.inference.smart_split_options.enabled,
            "queued": self.queue_depth,
        }

    def transcribe(
        self, audio_path: str, overrides: dict[str, Any] | None = None
    ) -> tuple[list[Segment], float, float]:
        """Transcribe one file, returning its segments and duration accounting.

        ``overrides`` are decoding settings layered over the process-wide config
        for this call only. Raises ``TranscriptionError`` if too many requests
        are already queued.
        """
        with self._waiting_lock:
            if self._waiting >= self.max_queue:
                raise TranscriptionError(f"server busy: {self._waiting} request(s) already queued")
            self._waiting += 1
        try:
            with self._lock:
                return self._transcribe_locked(audio_path, overrides)
        finally:
            with self._waiting_lock:
                self._waiting -= 1

    def _transcribe_locked(
        self, audio_path: str, overrides: dict[str, Any] | None = None
    ) -> tuple[list[Segment], float, float]:
        # Mirrors the per-file body of Inference.generates(), minus the batching
        # path (which auto-tunes against a sample file) and minus writing to disk.
        inference = self.inference
        merge_options = inference.segment_merge_options

        if inference._should_use_smart_split():
            task = InferenceTask(audio_path=audio_path, sub_prefix="", sub_formats=[])
            segments, info = inference._transcribe_smart_chunks(self.model, task, overrides)
            duration_after_vad = info.duration_after_vad
        else:
            audio_input, config, manual_duration_after_vad = inference._prepare_transcription(audio_path, batched=False)
            if manual_duration_after_vad == 0:
                raw_segments: Any = []
                info = SimpleNamespace(duration=len(audio_input) / WHISPER_SAMPLING_RATE, duration_after_vad=0)
            else:
                raw_segments, info = self.model.transcribe(audio_input, **config)

            duration_after_vad = (
                manual_duration_after_vad if manual_duration_after_vad is not None else info.duration_after_vad
            )
            segments = [
                Segment(
                    start=int(raw.start * 1_000),
                    end=int(raw.end * 1_000),
                    text=raw.text.strip(),
                )
                for raw in raw_segments
            ]

        segments = enforce_segment_timeline(segments, merge_options.max_duration_ms)
        segments = merge_segments(segments, merge_options)
        segments = enforce_segment_timeline(segments, merge_options.max_duration_ms)
        return segments, float(info.duration), float(duration_after_vad)


def render_subtitle(fmt: str, segments: list[Segment]) -> str:
    """Render segments with the CLI's own writers, so formats cannot drift."""
    writer = Inference.sub_writers[fmt]
    handle, path = tempfile.mkstemp(suffix=f".{fmt}")
    os.close(handle)
    try:
        writer(segments, path)
        with open(path, encoding="utf-8") as f:
            return f.read()
    finally:
        os.unlink(path)


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------


class TranscribeHandler(BaseHTTPRequestHandler):
    server_version = "ChickenRiceASR"
    protocol_version = "HTTP/1.1"

    service: TranscriptionService  # set on the server instance below
    max_upload_bytes: int

    # -- plumbing ---------------------------------------------------------

    def log_message(self, format: str, *args: Any) -> None:
        logger.info("%s - %s", self.address_string(), format % args)

    def _send(self, status: HTTPStatus, body: bytes, content_type: str, close: bool = False) -> None:
        if close:
            self.close_connection = True
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if close:
            self.send_header("Connection", "close")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any], close: bool = False) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8", close=close)

    def _send_error(self, status: HTTPStatus, message: str) -> None:
        # Errors can fire before the body has been read, and unread bytes would be
        # parsed as the next request on a keep-alive connection.
        logger.warning("%s: %s", status.phrase, message)
        self._send_json(status, {"error": message, "status": int(status)}, close=True)

    # -- routing ----------------------------------------------------------

    def do_GET(self) -> None:
        route = urlparse(self.path).path.rstrip("/") or "/"
        if route in ("/health", "/"):
            self._send_json(HTTPStatus.OK, {"status": "ok", **self.service.describe()})
        else:
            self._send_error(HTTPStatus.NOT_FOUND, f"no such route: {route}")

    do_HEAD = do_GET

    def do_POST(self) -> None:
        route = urlparse(self.path).path.rstrip("/") or "/"
        if route != "/transcribe":
            self._send_error(HTTPStatus.NOT_FOUND, f"no such route: {route}")
            return

        query = parse_qs(urlparse(self.path).query)
        fmt = (query.get("format") or ["json"])[0].lower()
        if fmt != "json" and fmt not in SUB_FORMATS:
            self._send_error(
                HTTPStatus.BAD_REQUEST,
                f"unsupported format {fmt!r}; expected json or one of {', '.join(sorted(SUB_FORMATS))}",
            )
            return

        try:
            overrides = parse_overrides(query)
        except ValueError as exc:
            self._send_error(HTTPStatus.BAD_REQUEST, str(exc))
            return

        try:
            audio_path, cleanup = self._receive_upload()
        except ValueError as exc:
            self._send_error(HTTPStatus.BAD_REQUEST, str(exc))
            return

        try:
            logger.info(
                "Transcribing upload (%s bytes)%s",
                os.path.getsize(audio_path),
                f" with {overrides}" if overrides else "",
            )
            started = time.monotonic()
            segments, duration, duration_after_vad = self.service.transcribe(audio_path, overrides)
            elapsed = time.monotonic() - started
            logger.info("Transcribed %.1fs of audio in %.1fs -> %d segment(s)", duration, elapsed, len(segments))
        except TranscriptionError as exc:
            self._send_error(HTTPStatus.SERVICE_UNAVAILABLE, str(exc))
            return
        except Exception as exc:  # noqa: BLE001 -- one bad file must not kill the server
            logger.exception("Transcription failed")
            self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, f"transcription failed: {exc}")
            return
        finally:
            cleanup()

        if fmt == "json":
            config = {**self.service.inference.generation_config, **overrides}
            self._send_json(
                HTTPStatus.OK,
                {
                    "text": "".join(segment.text for segment in segments),
                    "segments": [
                        {
                            "start": segment.start / 1_000,
                            "end": segment.end / 1_000,
                            "text": segment.text,
                        }
                        for segment in segments
                    ],
                    "duration": round(duration, 3),
                    "duration_after_vad": round(duration_after_vad, 3),
                    "language": config.get("language"),
                    "task": config.get("task"),
                    # Echoed so a caller can confirm what actually applied.
                    "hotwords": config.get("hotwords", ""),
                    "beam_size": config.get("beam_size"),
                    "processing_time": round(elapsed, 3),
                },
            )
        else:
            self._send(HTTPStatus.OK, render_subtitle(fmt, segments).encode("utf-8"), SUB_FORMATS[fmt])

    # -- upload handling --------------------------------------------------

    def _receive_upload(self):
        """Write the uploaded audio to a temp file; return its path and a cleaner.

        Accepts a multipart/form-data upload under the field ``file``, or a raw
        request body (``curl --data-binary @audio.wav``), which streams straight
        to disk and is the cheaper path for large files.
        """
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            raise ValueError("Content-Length header is required") from None
        if length <= 0:
            raise ValueError("empty request body")
        if length > self.max_upload_bytes:
            raise ValueError(f"upload of {length} bytes exceeds the {self.max_upload_bytes} byte limit")

        media_type, params = parse_content_type(self.headers.get("Content-Type", ""))

        if media_type == "multipart/form-data":
            boundary = params.get("boundary")
            if not boundary:
                raise ValueError("multipart/form-data request is missing its boundary parameter")
            parts = parse_multipart(self.rfile.read(length), boundary)
            uploads = [part for part in parts if part.filename or part.name in ("file", "audio")]
            if not uploads:
                raise ValueError("no file part found; send the audio as the 'file' field")
            upload = uploads[0]
            return self._spool(upload.content, safe_suffix(upload.filename))

        suffix = safe_suffix(self.headers.get("X-Filename"))
        if suffix == ".bin":
            suffix = safe_suffix(f"x.{media_type.rpartition('/')[2]}") if "/" in media_type else ".bin"
        return self._stream_to_file(length, suffix)

    def _spool(self, content: bytes, suffix: str):
        handle, path = tempfile.mkstemp(suffix=suffix, prefix="asr-")
        try:
            with os.fdopen(handle, "wb") as f:
                f.write(content)
        except BaseException:
            os.unlink(path)
            raise
        return path, lambda: self._discard(path)

    def _stream_to_file(self, length: int, suffix: str):
        handle, path = tempfile.mkstemp(suffix=suffix, prefix="asr-")
        try:
            remaining = length
            with os.fdopen(handle, "wb") as f:
                while remaining > 0:
                    block = self.rfile.read(min(1 << 20, remaining))
                    if not block:
                        raise ValueError("client disconnected before the upload finished")
                    f.write(block)
                    remaining -= len(block)
        except BaseException:
            os.unlink(path)
            raise
        return path, lambda: self._discard(path)

    @staticmethod
    def _discard(path: str) -> None:
        try:
            os.unlink(path)
        except OSError:
            logger.warning("Could not remove temp file %s", path)


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve the ASMR-VAD + Whisper transcription pipeline over HTTP.",
    )

    server = parser.add_argument_group("server")
    server.add_argument("--host", default="127.0.0.1", help="Interface to bind (default: 127.0.0.1)")
    server.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    server.add_argument(
        "--max_upload_mb", type=int, default=2048, help="Reject uploads larger than this many MB (default: 2048)"
    )
    server.add_argument(
        "--max_queue",
        type=int,
        default=8,
        help="Requests allowed to wait for the model before returning 503 (default: 8)",
    )
    server.add_argument("--log_level", default="INFO", help="Logging level (default: INFO)")

    model = parser.add_argument_group("model")
    model.add_argument("--model_name_or_path", default="models", help="CTranslate2 model directory (default: models)")
    model.add_argument("--device", default="auto", help="cpu, cuda, auto (amd/rocm/hip alias to cuda)")
    model.add_argument("--compute_type", default="auto", help="float16, int8_float16, int8, auto, ...")
    model.add_argument(
        "--generation_config",
        default="generation_config.json5",
        help="Decoding and VAD settings (default: generation_config.json5)",
    )
    model.add_argument(
        "--cpu_threads",
        type=int,
        default=0,
        help="CTranslate2 CPU threads; 0 (default) uses half the CPU budget, which is the "
        "physical core count under SMT. Pass the real core count if SMT is off.",
    )
    model.add_argument(
        "--task",
        choices=WHISPER_TASKS,
        default=None,
        help="transcribe (default) or translate; overrides the config file",
    )

    vad = parser.add_argument_group("VAD overrides (otherwise taken from the config file)")
    vad.add_argument("--vad_threshold", type=float, default=None, help="Speech probability threshold")
    vad.add_argument("--vad_min_speech_duration_ms", type=int, default=None, help="Shortest kept speech run")
    vad.add_argument("--vad_min_silence_duration_ms", type=int, default=None, help="Silence needed to split speech")
    vad.add_argument("--vad_speech_pad_ms", type=int, default=None, help="Padding added around speech")
    vad.add_argument(
        "--vad_threads",
        type=int,
        default=0,
        help="ONNX threads for the VAD; 0 (default) uses half the CPU budget",
    )
    vad.add_argument(
        "--vad_force_cpu",
        action="store_true",
        help="Run the VAD on CPU even when a CUDA execution provider is available",
    )

    chunking = parser.add_argument_group("chunking and subtitle merge overrides")
    chunking.add_argument(
        "--smart_split_with_vad", default=None, help="true/false: pick chunk boundaries in VAD-detected silence"
    )
    chunking.add_argument("--target_chunk_duration_s", type=float, default=None, help="Chunk target, capped at 30 s")
    chunking.add_argument("--merge_segments", dest="merge_segments", action="store_true", default=None)
    chunking.add_argument("--no_merge_segments", dest="merge_segments", action="store_false", default=None)
    chunking.add_argument("--merge_max_gap_ms", type=int, default=None, help="Largest gap that still merges")
    chunking.add_argument("--merge_max_duration_ms", type=int, default=None, help="Cap on a merged subtitle")

    return parser


def resolve_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line and apply the server's own defaults."""
    args = build_parser().parse_args(argv)
    # generation_config.json5 ships task="translate"; a transcription server should
    # default the other way, while still honouring an explicit flag.
    if args.task is None:
        args.task = "transcribe"
    return args


def main(argv: list[str] | None = None) -> int:
    args = resolve_args(argv)

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        stream=sys.stderr,
    )
    logging.getLogger("asr").setLevel(args.log_level)

    service = TranscriptionService(args, max_queue=args.max_queue)

    handler = type(
        "BoundTranscribeHandler",
        (TranscribeHandler,),
        {"service": service, "max_upload_bytes": args.max_upload_mb * 1024 * 1024},
    )

    httpd = ThreadingHTTPServer((args.host, args.port), handler)
    httpd.daemon_threads = True
    logger.info("Listening on http://%s:%d  (POST /transcribe, GET /health)", args.host, args.port)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down")
    finally:
        httpd.server_close()
    return 0
