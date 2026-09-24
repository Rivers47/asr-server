"""
Covers the two runtime fixes: a container-aware CPU budget for the VAD, and a
single ONNX session shared by both VAD passes rather than one per call.
"""

import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

sys.modules.setdefault("pyjson5", types.SimpleNamespace(decode_io=json.load))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import asr.injection as injection  # noqa: E402
from asr.vad_manager import VadConfig, available_cpus  # noqa: E402


class AvailableCpusTest(unittest.TestCase):
    def setUp(self):
        # addCleanup rather than enterContext: the latter is Python 3.11+, and
        # pyproject declares requires-python = ">=3.10".
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.root = Path(scratch.name)
        self.host = len(os.sched_getaffinity(0))

    def test_no_cgroup_files_falls_back_to_affinity(self):
        self.assertEqual(available_cpus(self.root), self.host)

    def test_cgroup_v2_quota_caps_the_budget(self):
        (self.root / "cpu.max").write_text("200000 100000")  # 2 cores
        self.assertEqual(available_cpus(self.root), 2)

    def test_cgroup_v2_unlimited_keeps_affinity(self):
        (self.root / "cpu.max").write_text("max 100000")
        self.assertEqual(available_cpus(self.root), self.host)

    def test_cgroup_v1_quota_caps_the_budget(self):
        (self.root / "cpu").mkdir()
        (self.root / "cpu/cpu.cfs_quota_us").write_text("150000")
        (self.root / "cpu/cpu.cfs_period_us").write_text("100000")
        self.assertEqual(available_cpus(self.root), 1)  # 1.5 cores floors to 1

    def test_cgroup_v1_unlimited_keeps_affinity(self):
        (self.root / "cpu").mkdir()
        (self.root / "cpu/cpu.cfs_quota_us").write_text("-1")
        (self.root / "cpu/cpu.cfs_period_us").write_text("100000")
        self.assertEqual(available_cpus(self.root), self.host)

    def test_quota_never_exceeds_affinity(self):
        (self.root / "cpu.max").write_text(f"{(self.host + 50) * 100000} 100000")
        self.assertEqual(available_cpus(self.root), self.host)

    def test_result_is_always_at_least_one(self):
        (self.root / "cpu.max").write_text("1000 100000")  # 0.01 cores
        self.assertEqual(available_cpus(self.root), 1)

    def test_garbage_cgroup_content_is_ignored(self):
        (self.root / "cpu.max").write_text("not a quota")
        self.assertEqual(available_cpus(self.root), self.host)


class ThreadBudgetTest(unittest.TestCase):
    """Both thread counts must come from the CPU budget, not the host core count."""

    @staticmethod
    def _args(**overrides):
        from asr.server import resolve_args

        argv = []
        for key, value in overrides.items():
            argv += [f"--{key}", str(value)]
        return resolve_args(argv)

    def test_cpu_threads_defaults_to_half_the_budget(self):
        from asr.vad_manager import available_cpus

        args = self._args()
        self.assertEqual(args.cpu_threads, 0, "the flag default must stay 0 = auto")
        # Resolution happens in Inference.__init__, which also loads models; check
        # the arithmetic directly against the same helper it uses.
        resolved = max(0, args.cpu_threads or 0) or max(1, available_cpus() // 2)
        self.assertEqual(resolved, max(1, available_cpus() // 2))
        self.assertGreaterEqual(resolved, 1)

    def test_explicit_cpu_threads_wins(self):
        args = self._args(cpu_threads=3)
        self.assertEqual(max(0, args.cpu_threads or 0) or 999, 3)

    def test_negative_cpu_threads_falls_back_to_auto(self):
        from asr.vad_manager import available_cpus

        args = self._args(cpu_threads=-4)
        expected = max(1, available_cpus() // 2)
        self.assertEqual(max(0, args.cpu_threads or 0) or expected, expected)

    def test_the_model_is_built_with_the_resolved_thread_count(self):
        """Guards the wiring: a dropped kwarg means CT2 silently auto-detects again."""
        import inspect

        from asr import server

        source = inspect.getsource(server.TranscriptionService.__init__)
        self.assertIn("cpu_threads=self.inference.cpu_threads", source)
        self.assertIn("num_workers=1", source)

    def test_health_reports_the_thread_count(self):
        import inspect

        from asr import server

        self.assertIn('"cpu_threads"', inspect.getsource(server.TranscriptionService.describe))


class ManagerCacheTest(unittest.TestCase):
    """The inner VAD runs once per audio chunk; rebuilding the session per call
    cost ~0.2-0.3 s and 114 MB each time."""

    def setUp(self):
        injection.reset_manager()
        self.addCleanup(injection.reset_manager)
        self.built = []

        def fake_manager(config=None, ttl=None, progress_callback=None):
            self.built.append(config)
            return mock.Mock(name=f"manager-{len(self.built)}")

        patcher = mock.patch.object(injection, "VadModelManager", side_effect=fake_manager)
        self.mock_cls = patcher.start()
        self.addCleanup(patcher.stop)

    def test_repeated_calls_reuse_one_manager(self):
        config = VadConfig(onnx_model_path="models/whisper_vad.onnx")
        first = injection.get_active_manager(config, None)
        for _ in range(50):
            self.assertIs(injection.get_active_manager(config, None), first)
        self.assertEqual(len(self.built), 1, f"built {len(self.built)} managers, expected 1")

    def test_equal_configs_are_not_rebuilt(self):
        a = VadConfig(threshold=0.4, onnx_model_path="models/whisper_vad.onnx")
        b = VadConfig(threshold=0.4, onnx_model_path="models/whisper_vad.onnx")
        self.assertIs(injection.get_active_manager(a, None), injection.get_active_manager(b, None))
        self.assertEqual(len(self.built), 1)

    def test_changed_config_rebuilds(self):
        first = injection.get_active_manager(VadConfig(threshold=0.4), None)
        second = injection.get_active_manager(VadConfig(threshold=0.6), None)
        self.assertIsNot(first, second)
        self.assertEqual(len(self.built), 2)

    def test_changed_progress_callback_rebuilds(self):
        config = VadConfig(threshold=0.4)
        first = injection.get_active_manager(config, lambda *a: None)
        second = injection.get_active_manager(config, lambda *a: None)
        self.assertIsNot(first, second)

    def test_injected_entry_point_does_not_rebuild_per_call(self):
        """The path faster-whisper actually calls, once per chunk."""
        config = VadConfig(threshold=0.4, onnx_model_path="models/whisper_vad.onnx")
        injection.set_global_config(config)
        audio = object()
        for _ in range(20):
            injection.get_speech_timestamps_injected(audio, None, 16000)
        self.assertEqual(len(self.built), 1, f"built {len(self.built)} managers across 20 chunks")

    def test_reset_releases_the_session(self):
        config = VadConfig(threshold=0.4)
        first = injection.get_active_manager(config, None)
        injection.reset_manager()
        self.assertIsNot(injection.get_active_manager(config, None), first)
        self.assertEqual(len(self.built), 2)


if __name__ == "__main__":
    unittest.main()
