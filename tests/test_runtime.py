"""
Covers the two runtime fixes: a container-aware CPU budget for the VAD, and a
single ONNX session shared by both VAD passes rather than one per call.
"""

import json
import os
import sys
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
        self.root = Path(self.enterContext(__import__("tempfile").TemporaryDirectory()))
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
