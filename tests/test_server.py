import io
import json
import os
import sys
import threading
import time
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

# A stand-in module, so importing asr.pipeline does not need the real pyjson5.
_pyjson5 = types.ModuleType("pyjson5")
_pyjson5.decode_io = json.load  # type: ignore[attr-defined]
sys.modules.setdefault("pyjson5", _pyjson5)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from asr.pipeline import Segment, SegmentMergeOptions  # noqa: E402
from asr.server import (  # noqa: E402
    MAX_BEAM_SIZE,
    MAX_HOTWORDS_CHARS,
    TranscribeHandler,
    TranscriptionError,
    TranscriptionService,
    parse_content_type,
    parse_multipart,
    parse_overrides,
    render_subtitle,
    resolve_args,
    safe_suffix,
)

BOUNDARY = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
MERGE_OPTIONS = SegmentMergeOptions(enabled=True, max_gap_ms=2_000, max_duration_ms=20_000)


def build_multipart(parts, boundary: str = BOUNDARY) -> bytes:
    """Assemble a multipart/form-data body from (name, filename, content) triples."""
    body = b""
    for name, filename, content in parts:
        disposition = f'form-data; name="{name}"'
        if filename is not None:
            disposition += f'; filename="{filename}"'
        body += f"--{boundary}\r\nContent-Disposition: {disposition}\r\n\r\n".encode()
        body += content + b"\r\n"
    return body + f"--{boundary}--\r\n".encode()


class ParseContentTypeTest(unittest.TestCase):
    def test_splits_media_type_and_parameters(self):
        media_type, params = parse_content_type('multipart/form-data; boundary="abc"; charset=utf-8')
        self.assertEqual(media_type, "multipart/form-data")
        self.assertEqual(params, {"boundary": "abc", "charset": "utf-8"})

    def test_bare_media_type(self):
        self.assertEqual(parse_content_type("audio/wav"), ("audio/wav", {}))


class SafeSuffixTest(unittest.TestCase):
    def test_known_extension_is_lowercased(self):
        self.assertEqual(safe_suffix("Track 01.WAV"), ".wav")

    def test_unknown_and_hostile_names_fall_back(self):
        for name in (None, "", "../../etc/passwd", "payload.exe", "no-extension"):
            self.assertEqual(safe_suffix(name), ".bin", name)


class ParseMultipartTest(unittest.TestCase):
    def test_binary_payload_round_trips(self):
        payload = b"RIFF\x00\x00\xff\xfe\r\n\r\ndata\x00\r\n"
        parts = parse_multipart(build_multipart([("file", "a.wav", payload)]), BOUNDARY)
        self.assertEqual(len(parts), 1)
        self.assertEqual(parts[0].name, "file")
        self.assertEqual(parts[0].filename, "a.wav")
        self.assertEqual(parts[0].content, payload)

    def test_file_field_after_a_plain_field(self):
        audio = b"OggS" + bytes(range(256))
        body = build_multipart([("model", None, b"ja-1.5B"), ("file", "b.opus", audio)])
        parts = parse_multipart(body, BOUNDARY)
        self.assertEqual([part.name for part in parts], ["model", "file"])
        self.assertIsNone(parts[0].filename)
        self.assertEqual(parts[1].content, audio)

    def test_empty_file_part(self):
        parts = parse_multipart(build_multipart([("file", "c.wav", b"")]), BOUNDARY)
        self.assertEqual(len(parts), 1)
        self.assertEqual(parts[0].content, b"")

    def test_bare_linefeed_client(self):
        body = (
            f"--{BOUNDARY}\n".encode()
            + b'Content-Disposition: form-data; name="file"; filename="e.wav"\n\n'
            + b"abc\n"
            + f"--{BOUNDARY}--\n".encode()
        )
        parts = parse_multipart(body, BOUNDARY)
        self.assertEqual(len(parts), 1)
        self.assertEqual(parts[0].content, b"abc")

    def test_preamble_and_epilogue_are_ignored(self):
        body = b"preamble\r\n" + build_multipart([("file", "f.wav", b"q")]) + b"epilogue"
        parts = parse_multipart(body, BOUNDARY)
        self.assertEqual(len(parts), 1)
        self.assertEqual(parts[0].content, b"q")

    def test_large_payload_integrity(self):
        payload = os.urandom(2 * 1024 * 1024)
        parts = parse_multipart(build_multipart([("file", "g.wav", payload)]), BOUNDARY)
        self.assertEqual(parts[0].content, payload)


class FakeService:
    """Stands in for TranscriptionService so the handler can be tested without models."""

    def __init__(self):
        self.inference = SimpleNamespace(generation_config={"task": "transcribe", "language": "ja"})
        self.mode = "ok"
        self.seen_path = None
        self.seen_bytes = None
        self.seen_overrides = None

    def describe(self):
        return {"model": "models", "device": "cpu", "task": "transcribe", "queued": 0}

    def transcribe(self, audio_path, overrides=None):
        self.seen_path = audio_path
        self.seen_overrides = overrides
        with open(audio_path, "rb") as f:
            self.seen_bytes = f.read()
        if self.mode == "busy":
            raise TranscriptionError("server busy: 8 request(s) already queued")
        if self.mode == "boom":
            raise RuntimeError("ffmpeg exploded")
        return ([Segment(1_230, 4_560, "こんにちは"), Segment(4_560, 7_000, "ありがとう")], 12.5, 6.25)


class FakeSocket:
    """Minimal socket for driving BaseHTTPRequestHandler without binding a port."""

    def __init__(self, data: bytes):
        self.data = data
        self.out = io.BytesIO()

    def makefile(self, mode="rb", bufsize=-1):
        return io.BytesIO(self.data)

    def sendall(self, chunk):
        self.out.write(chunk)

    def close(self):
        pass


AUDIO = b"RIFF\x00\x01\x02fake wav bytes\r\n\x00"
AUDIO_HEADERS = {"Content-Type": "audio/wav", "Content-Length": str(len(AUDIO))}


class HandlerTestCase(unittest.TestCase):
    def setUp(self):
        self.service = FakeService()
        self.handler_cls = type(
            "BoundHandler",
            (TranscribeHandler,),
            {"service": self.service, "max_upload_bytes": 1024 * 1024},
        )

    @staticmethod
    def wire(method, path, body=b"", headers=None):
        request = f"{method} {path} HTTP/1.1\r\nHost: test\r\n"
        for key, value in (headers or {}).items():
            request += f"{key}: {value}\r\n"
        return request.encode() + b"\r\n" + body

    def serve(self, wire: bytes) -> bytes:
        sock = FakeSocket(wire)
        self.handler_cls(sock, ("127.0.0.1", 1234), SimpleNamespace())
        return sock.out.getvalue()

    @staticmethod
    def split_responses(blob: bytes):
        responses = []
        while blob:
            head, _, rest = blob.partition(b"\r\n\r\n")
            lines = head.split(b"\r\n")
            headers = {}
            for line in lines[1:]:
                key, _, value = line.partition(b": ")
                headers[key.decode().lower()] = value.decode()
            length = int(headers["content-length"])
            responses.append((int(lines[0].split()[1]), headers, rest[:length]))
            blob = rest[length:]
        return responses

    def request(self, method, path, body=b"", headers=None):
        responses = self.split_responses(self.serve(self.wire(method, path, body, headers)))
        self.assertEqual(len(responses), 1, "expected exactly one response")
        return responses[0]


class HealthEndpointTest(HandlerTestCase):
    def test_health_reports_configuration(self):
        status, headers, body = self.request("GET", "/health")
        self.assertEqual(status, 200)
        self.assertTrue(headers["content-type"].startswith("application/json"))
        payload = json.loads(body)
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["task"], "transcribe")

    def test_head_omits_body_but_keeps_length(self):
        status, headers, body = self.request("HEAD", "/health")
        self.assertEqual(status, 200)
        self.assertEqual(body, b"")
        self.assertGreater(int(headers["content-length"]), 0)

    def test_unknown_routes_are_404(self):
        self.assertEqual(self.request("GET", "/nope")[0], 404)
        self.assertEqual(self.request("POST", "/nope", AUDIO, AUDIO_HEADERS)[0], 404)


class TranscribeEndpointTest(HandlerTestCase):
    def test_raw_body_upload_returns_json(self):
        status, headers, body = self.request("POST", "/transcribe", AUDIO, AUDIO_HEADERS)
        self.assertEqual(status, 200)
        self.assertTrue(headers["content-type"].startswith("application/json"))
        payload = json.loads(body)
        self.assertEqual(self.service.seen_bytes, AUDIO)
        self.assertTrue(self.service.seen_path.endswith(".wav"))
        self.assertEqual(payload["segments"][0], {"start": 1.23, "end": 4.56, "text": "こんにちは"})
        self.assertEqual(payload["text"], "こんにちはありがとう")
        self.assertEqual(payload["duration"], 12.5)
        self.assertEqual(payload["duration_after_vad"], 6.25)
        self.assertIn("processing_time", payload)

    def test_upload_temp_file_is_removed(self):
        self.request("POST", "/transcribe", AUDIO, AUDIO_HEADERS)
        self.assertFalse(os.path.exists(self.service.seen_path))

    def test_multipart_upload_returns_srt(self):
        body = build_multipart([("file", "x.opus", AUDIO)])
        headers = {
            "Content-Type": f"multipart/form-data; boundary={BOUNDARY}",
            "Content-Length": str(len(body)),
        }
        status, response_headers, payload = self.request("POST", "/transcribe?format=srt", body, headers)
        self.assertEqual(status, 200)
        self.assertTrue(response_headers["content-type"].startswith("application/x-subrip"))
        self.assertTrue(payload.decode("utf-8").startswith("1\n00:00:01,230 --> 00:00:04,560\nこんにちは\n"))
        self.assertEqual(self.service.seen_bytes, AUDIO)
        self.assertTrue(self.service.seen_path.endswith(".opus"))

    def test_subtitle_formats(self):
        for fmt, marker in (("vtt", "WebVTT"), ("txt", "こんにちは\nありがとう"), ("lrc", "[00:01.23]こんにちは")):
            with self.subTest(fmt=fmt):
                status, _, body = self.request("POST", f"/transcribe?format={fmt}", AUDIO, AUDIO_HEADERS)
                self.assertEqual(status, 200)
                self.assertIn(marker, body.decode("utf-8"))

    def test_bad_requests(self):
        oversize = b"x" * (1024 * 1024 + 10)
        no_file = build_multipart([("model", None, b"ja")])
        cases = [
            ("unsupported format", "/transcribe?format=docx", AUDIO, AUDIO_HEADERS),
            ("empty request body", "/transcribe", b"", {"Content-Type": "audio/wav", "Content-Length": "0"}),
            ("Content-Length", "/transcribe", b"", {"Content-Type": "audio/wav"}),
            (
                "exceeds",
                "/transcribe",
                oversize,
                {"Content-Type": "audio/wav", "Content-Length": str(len(oversize))},
            ),
            (
                "boundary",
                "/transcribe",
                AUDIO,
                {"Content-Type": "multipart/form-data", "Content-Length": str(len(AUDIO))},
            ),
            (
                "no file part",
                "/transcribe",
                no_file,
                {
                    "Content-Type": f"multipart/form-data; boundary={BOUNDARY}",
                    "Content-Length": str(len(no_file)),
                },
            ),
        ]
        for expected, path, body, headers in cases:
            with self.subTest(expected=expected):
                status, _, payload = self.request("POST", path, body, headers)
                self.assertEqual(status, 400)
                self.assertIn(expected, json.loads(payload)["error"])

    def test_queue_full_returns_503(self):
        self.service.mode = "busy"
        status, _, body = self.request("POST", "/transcribe", AUDIO, AUDIO_HEADERS)
        self.assertEqual(status, 503)
        self.assertIn("busy", json.loads(body)["error"])
        self.assertFalse(os.path.exists(self.service.seen_path))

    def test_pipeline_failure_returns_500_and_cleans_up(self):
        self.service.mode = "boom"
        with self.assertLogs("asr.server", level="ERROR"):
            status, _, body = self.request("POST", "/transcribe", AUDIO, AUDIO_HEADERS)
        self.assertEqual(status, 500)
        self.assertIn("ffmpeg exploded", json.loads(body)["error"])
        self.assertFalse(os.path.exists(self.service.seen_path))

    def test_keep_alive_serves_several_requests_per_connection(self):
        wire = (
            self.wire("GET", "/health")
            + self.wire("POST", "/transcribe", AUDIO, AUDIO_HEADERS)
            + self.wire("GET", "/health")
        )
        responses = self.split_responses(self.serve(wire))
        self.assertEqual([status for status, _, _ in responses], [200, 200, 200])

    def test_error_closes_the_connection_instead_of_desyncing(self):
        # The body of a rejected request is never read, so the connection must
        # close -- otherwise those bytes are parsed as the next request.
        wire = self.wire("POST", "/transcribe?format=docx", AUDIO, AUDIO_HEADERS) + self.wire("GET", "/health")
        responses = self.split_responses(self.serve(wire))
        self.assertEqual(len(responses), 1)
        self.assertEqual(responses[0][0], 400)
        self.assertEqual(responses[0][1].get("connection"), "close")


class ParseOverridesTest(unittest.TestCase):
    def test_absent_parameters_produce_no_overrides(self):
        self.assertEqual(parse_overrides({}), {})
        self.assertEqual(parse_overrides({"format": ["srt"]}), {})

    def test_hotwords_and_beam_size(self):
        self.assertEqual(
            parse_overrides({"hotwords": ["花子, お兄さん"], "beam_size": ["5"]}),
            {"hotwords": "花子, お兄さん", "beam_size": 5},
        )

    def test_empty_hotwords_is_kept_so_it_can_clear_the_config(self):
        self.assertEqual(parse_overrides({"hotwords": [""]}), {"hotwords": ""})

    def test_hotwords_length_is_capped(self):
        parse_overrides({"hotwords": ["x" * MAX_HOTWORDS_CHARS]})  # at the limit, fine
        with self.assertRaises(ValueError) as caught:
            parse_overrides({"hotwords": ["x" * (MAX_HOTWORDS_CHARS + 1)]})
        self.assertIn("limit", str(caught.exception))

    def test_beam_size_must_be_an_integer(self):
        for bad in ("wide", "3.5", ""):
            with self.subTest(value=bad), self.assertRaises(ValueError) as caught:
                parse_overrides({"beam_size": [bad]})
            self.assertIn("must be an integer", str(caught.exception))

    def test_beam_size_is_bounded(self):
        self.assertEqual(parse_overrides({"beam_size": ["1"]}), {"beam_size": 1})
        self.assertEqual(parse_overrides({"beam_size": [str(MAX_BEAM_SIZE)]}), {"beam_size": MAX_BEAM_SIZE})
        for bad in ("0", "-1", str(MAX_BEAM_SIZE + 1), "9999"):
            with self.subTest(value=bad), self.assertRaises(ValueError) as caught:
                parse_overrides({"beam_size": [bad]})
            self.assertIn("between 1 and", str(caught.exception))


class OverrideEndpointTest(HandlerTestCase):
    def test_overrides_reach_the_service_and_are_echoed(self):
        status, _, body = self.request(
            "POST", "/transcribe?hotwords=%E8%8A%B1%E5%AD%90&beam_size=3", AUDIO, AUDIO_HEADERS
        )
        self.assertEqual(status, 200)
        self.assertEqual(self.service.seen_overrides, {"hotwords": "花子", "beam_size": 3})
        payload = json.loads(body)
        self.assertEqual(payload["hotwords"], "花子")
        self.assertEqual(payload["beam_size"], 3)

    def test_without_overrides_the_response_echoes_the_process_config(self):
        self.service.inference.generation_config["beam_size"] = 5
        status, _, body = self.request("POST", "/transcribe", AUDIO, AUDIO_HEADERS)
        self.assertEqual(status, 200)
        self.assertIsNone(self.service.seen_overrides or None)
        payload = json.loads(body)
        self.assertEqual(payload["beam_size"], 5)
        self.assertEqual(payload["hotwords"], "")

    def test_overrides_combine_with_format(self):
        status, headers, body = self.request("POST", "/transcribe?format=srt&beam_size=2", AUDIO, AUDIO_HEADERS)
        self.assertEqual(status, 200)
        self.assertTrue(headers["content-type"].startswith("application/x-subrip"))
        self.assertEqual(self.service.seen_overrides, {"beam_size": 2})

    def test_bad_override_is_rejected_before_the_upload_is_read(self):
        for query, expected in (
            ("beam_size=wide", "must be an integer"),
            ("beam_size=99", "between 1 and"),
            (f"hotwords={'x' * (MAX_HOTWORDS_CHARS + 1)}", "limit"),
        ):
            with self.subTest(query=query):
                status, _, body = self.request("POST", f"/transcribe?{query}", AUDIO, AUDIO_HEADERS)
                self.assertEqual(status, 400)
                self.assertIn(expected, json.loads(body)["error"])
                self.assertIsNone(self.service.seen_path, "upload should not have been spooled")


def bare_service(inference, model, max_queue: int = 8) -> TranscriptionService:
    """Build a TranscriptionService without running __init__ (which loads models)."""
    service = TranscriptionService.__new__(TranscriptionService)
    service.inference = inference
    service.model = model
    service.max_queue = max_queue
    service._lock = threading.Lock()
    service._waiting = 0
    service._waiting_lock = threading.Lock()
    return service


class TranscriptionServiceTest(unittest.TestCase):
    def test_smart_split_path_is_used_when_enabled(self):
        seen = {}

        def smart_chunks(model, task, overrides=None):
            seen["model"] = model
            seen["audio_path"] = task.audio_path
            seen["overrides"] = overrides
            return (
                [Segment(0, 1_000, "あ"), Segment(1_000, 2_000, "い")],
                SimpleNamespace(duration=30.0, duration_after_vad=2.0),
            )

        inference = SimpleNamespace(
            _should_use_smart_split=lambda: True,
            _transcribe_smart_chunks=smart_chunks,
            segment_merge_options=MERGE_OPTIONS,
        )
        service = bare_service(inference, "MODEL")
        segments, duration, duration_after_vad = service.transcribe("/tmp/x.wav")

        self.assertEqual(seen, {"model": "MODEL", "audio_path": "/tmp/x.wav", "overrides": None})
        self.assertEqual((duration, duration_after_vad), (30.0, 2.0))
        self.assertEqual([segment.text for segment in segments], ["あ", "い"])
        self.assertEqual(service.queue_depth, 0)

    def test_plain_path_converts_seconds_to_milliseconds(self):
        raw = [
            SimpleNamespace(start=1.2345, end=2.5, text="  hello  "),
            SimpleNamespace(start=10.0, end=12.0, text="world"),
        ]
        inference = SimpleNamespace(
            _should_use_smart_split=lambda: False,
            _prepare_transcription=lambda path, batched, overrides=None: ("AUDIO", {"language": "ja"}, None),
            segment_merge_options=MERGE_OPTIONS,
        )
        model = SimpleNamespace(
            transcribe=lambda audio, **kwargs: (raw, SimpleNamespace(duration=20.0, duration_after_vad=5.0))
        )
        segments, duration, duration_after_vad = bare_service(inference, model).transcribe("/tmp/y.wav")

        self.assertEqual(
            [(segment.start, segment.end, segment.text) for segment in segments],
            [(1_234, 2_500, "hello"), (10_000, 12_000, "world")],
        )
        self.assertEqual((duration, duration_after_vad), (20.0, 5.0))

    def test_silent_file_never_reaches_the_model(self):
        import numpy as np

        def must_not_run(*args, **kwargs):
            raise AssertionError("model.transcribe ran even though the VAD found no speech")

        inference = SimpleNamespace(
            _should_use_smart_split=lambda: False,
            _prepare_transcription=lambda path, batched, overrides=None: (np.zeros(16_000 * 7), {}, 0),
            segment_merge_options=MERGE_OPTIONS,
        )
        service = bare_service(inference, SimpleNamespace(transcribe=must_not_run))
        segments, duration, duration_after_vad = service.transcribe("/tmp/z.wav")

        self.assertEqual(segments, [])
        self.assertEqual((duration, duration_after_vad), (7.0, 0.0))

    def test_merge_post_processing_runs(self):
        duplicates = [Segment(0, 1_000, "same"), Segment(1_000, 2_000, "same"), Segment(30_000, 31_000, "other")]
        inference = SimpleNamespace(
            _should_use_smart_split=lambda: True,
            _transcribe_smart_chunks=lambda model, task, overrides=None: (
                list(duplicates),
                SimpleNamespace(duration=40.0, duration_after_vad=3.0),
            ),
            segment_merge_options=MERGE_OPTIONS,
        )
        segments, _, _ = bare_service(inference, None).transcribe("/tmp/w.wav")
        self.assertLess(len(segments), len(duplicates))

    def test_requests_are_serialised_and_the_queue_is_capped(self):
        gate = threading.Event()
        first_started = threading.Event()
        in_flight = []
        guard = threading.Lock()

        def slow_chunks(model, task, overrides=None):
            with guard:
                in_flight.append(1)
                self.assertEqual(len(in_flight), 1, "two transcriptions ran concurrently")
            first_started.set()
            gate.wait(5)
            with guard:
                in_flight.pop()
            return ([], SimpleNamespace(duration=1.0, duration_after_vad=0.0))

        inference = SimpleNamespace(
            _should_use_smart_split=lambda: True,
            _transcribe_smart_chunks=slow_chunks,
            segment_merge_options=MERGE_OPTIONS,
        )
        service = bare_service(inference, None, max_queue=3)
        rejections = []

        def worker():
            try:
                service.transcribe("/tmp/q.wav")
            except TranscriptionError as exc:
                rejections.append(str(exc))

        threads = [threading.Thread(target=worker) for _ in range(6)]
        threads[0].start()
        first_started.wait(5)
        for thread in threads[1:]:
            thread.start()
        time.sleep(0.3)
        gate.set()
        for thread in threads:
            thread.join(10)

        self.assertEqual(len(rejections), 3)
        self.assertTrue(all("busy" in message for message in rejections))
        self.assertEqual(service.queue_depth, 0)


class RenderSubtitleTest(unittest.TestCase):
    def test_formats_match_the_cli_writers(self):
        segments = [Segment(1_230, 4_560, "こんにちは")]
        self.assertEqual(render_subtitle("srt", segments), "1\n00:00:01,230 --> 00:00:04,560\nこんにちは\n\n")
        self.assertTrue(render_subtitle("vtt", segments).startswith("WebVTT\n\n1\n00:00:01.230 --> 00:00:04.560"))
        self.assertEqual(render_subtitle("txt", segments), "こんにちは\n")


class ArgumentPlumbingTest(unittest.TestCase):
    def test_defaults(self):
        args = resolve_args([])
        self.assertEqual((args.host, args.port, args.max_upload_mb, args.max_queue), ("127.0.0.1", 8000, 2048, 8))
        self.assertEqual(args.model_name_or_path, "models")
        self.assertEqual(args.generation_config, "generation_config.json5")
        self.assertEqual((args.device, args.compute_type), ("auto", "auto"))

    def test_task_defaults_to_transcribe(self):
        self.assertEqual(resolve_args([]).task, "transcribe")

    def test_explicit_task_wins(self):
        self.assertEqual(resolve_args(["--task", "translate"]).task, "translate")

    def test_invalid_task_is_rejected(self):
        with self.assertRaises(SystemExit):
            resolve_args(["--task", "summarize"])

    def test_every_flag_the_pipeline_reads_is_defined(self):
        # Inference and _load_generation_config read these off the namespace; a
        # missing one is an AttributeError at startup rather than a parse error.
        args = resolve_args([])
        for name in (
            "model_name_or_path", "device", "compute_type", "generation_config", "task",
            "vad_threshold", "vad_min_speech_duration_ms", "vad_min_silence_duration_ms", "vad_speech_pad_ms",
            "merge_segments", "merge_max_gap_ms", "merge_max_duration_ms",
            "smart_split_with_vad", "target_chunk_duration_s",
        ):  # fmt: skip
            self.assertTrue(hasattr(args, name), f"parser does not define --{name}")

    def test_overrides_parse(self):
        args = resolve_args(
            [
                "--host", "0.0.0.0",
                "--port", "9000",
                "--max_queue", "2",
                "--model_name_or_path", "models/other",
                "--device", "cuda",
                "--compute_type", "float16",
                "--vad_threshold", "0.35",
                "--target_chunk_duration_s", "20",
                "--no_merge_segments",
            ]
        )  # fmt: skip
        self.assertEqual((args.host, args.port, args.max_queue), ("0.0.0.0", 9000, 2))
        self.assertEqual(args.model_name_or_path, "models/other")
        self.assertEqual((args.device, args.compute_type), ("cuda", "float16"))
        self.assertEqual(args.vad_threshold, 0.35)
        self.assertEqual(args.target_chunk_duration_s, 20.0)
        self.assertIs(args.merge_segments, False)

    def test_merge_flags_default_to_none_so_the_config_file_wins(self):
        args = resolve_args([])
        for name in ("merge_segments", "merge_max_gap_ms", "vad_threshold", "smart_split_with_vad"):
            self.assertIsNone(getattr(args, name), name)


if __name__ == "__main__":
    unittest.main()
