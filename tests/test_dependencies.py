"""
Guards the trimmed dependency set.

This package exists to run the pipeline without transformers (69 MB) or librosa
(347 MB, via numba/llvmlite/scipy/scikit-learn). Both were removable because
faster-whisper already ships an equivalent log-mel extractor and because every
caller decodes at 16 kHz, so nothing ever resamples. These tests fail if either
creeps back in.
"""

import ast
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = sorted((ROOT / "asr").glob("*.py")) + [ROOT / "serve.py", ROOT / "fetch_models.py"]
BANNED = {"transformers", "librosa", "torch", "torchaudio", "scipy", "sklearn", "soundfile", "numba"}


def imported_modules(path: Path) -> set[str]:
    """Every top-level module name imported anywhere in a file, lazy imports included."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


class DependencyHygieneTest(unittest.TestCase):
    def test_no_heavy_imports_anywhere(self):
        for path in SOURCES:
            with self.subTest(file=path.name):
                offenders = imported_modules(path) & BANNED
                self.assertEqual(offenders, set(), f"{path.name} imports {offenders}")

    def test_openvino_backend_stays_outside_the_package(self):
        """ov_backend.py may import transformers; asr/ may not, so the import is lazy.

        A module-level `import ov_backend` in asr/ would pull transformers into the
        CPU image at startup, which is exactly what BANNED exists to prevent.
        """
        self.assertTrue((ROOT / "ov_backend.py").exists(), "ov_backend.py is missing")
        for path in sorted((ROOT / "asr").glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            module_level = set()
            for node in tree.body:  # top level only
                if isinstance(node, ast.Import):
                    module_level.update(alias.name.split(".")[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    module_level.add(node.module.split(".")[0])
            with self.subTest(file=path.name):
                self.assertNotIn("ov_backend", module_level, f"{path.name} imports ov_backend at module level")

    def test_pyproject_declares_only_the_short_list(self):
        text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        block = text.split("dependencies = [", 1)[1].split("]", 1)[0]
        declared = {re.split(r"[><=~!\[]", line.strip().strip('",'))[0] for line in block.splitlines() if '"' in line}
        self.assertEqual(declared, {"faster-whisper", "ctranslate2", "onnxruntime", "numpy", "pyjson5"})

    def test_the_openvino_stack_is_opt_in(self):
        """The heavy stack may appear in the openvino extra, never in the default install."""
        import tomllib

        data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

        def names(specs):
            return {re.split(r"[><=~!\[;]", spec.strip())[0].lower() for spec in specs}

        self.assertEqual(BANNED & names(data["project"]["dependencies"]), set())
        extra = names(data["project"]["optional-dependencies"]["openvino"])
        self.assertLessEqual({"optimum-intel", "transformers", "torch"}, extra)

    def test_torch_is_pinned_to_the_cpu_index(self):
        """From the default index torch is the CUDA build, and torchvision then fails to load."""
        import tomllib

        data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        indexes = {entry["name"]: entry for entry in data["tool"]["uv"]["index"]}
        self.assertEqual(indexes["pytorch-cpu"]["url"], "https://download.pytorch.org/whl/cpu")
        self.assertTrue(indexes["pytorch-cpu"]["explicit"], "the index must not shadow other packages")
        for package in ("torch", "torchvision"):
            with self.subTest(package=package):
                self.assertEqual(data["tool"]["uv"]["sources"][package]["index"], "pytorch-cpu")

    def test_openvino_releases_the_gpu_when_idle_by_default(self):
        """A shared GPU is the expected case, so an idle server should not hold it."""
        import sys

        sys.path.insert(0, str(ROOT))
        import json
        import types

        sys.modules.setdefault("pyjson5", types.SimpleNamespace(decode_io=json.load))
        from asr.server import resolve_args

        self.assertEqual(resolve_args([]).ov_idle_unload, 300.0)

    def test_vad_uses_the_faster_whisper_extractor(self):
        source = (ROOT / "asr" / "vad_manager.py").read_text(encoding="utf-8")
        self.assertIn("from faster_whisper.feature_extractor import FeatureExtractor", source)
        # The parameters must stay the ones the ONNX graph was exported against.
        for parameter in ("feature_size=80", "sampling_rate=16000", "hop_length=160", "chunk_length=30", "n_fft=400"):
            self.assertIn(parameter, source)

    def test_resampling_is_a_hard_error_not_a_silent_fallback(self):
        source = (ROOT / "asr" / "vad_manager.py").read_text(encoding="utf-8")
        self.assertIn('raise ValueError(f"expected {self.sample_rate} Hz audio', source)


class FetchModelsTest(unittest.TestCase):
    """The local filenames fetch_models writes must be the ones the code opens."""

    @staticmethod
    def _fetch_module():
        import importlib.util

        spec = importlib.util.spec_from_file_location("fetch_models", ROOT / "fetch_models.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_vad_local_names_match_what_the_pipeline_opens(self):
        fetch = self._fetch_module()
        pipeline = (ROOT / "asr" / "pipeline.py").read_text(encoding="utf-8")
        for local in fetch.VAD_FILES.values():
            with self.subTest(file=local):
                self.assertIn(f"models/{local}", pipeline)

    def test_asr_list_includes_the_file_startup_checks_for(self):
        fetch = self._fetch_module()
        self.assertIn("model.bin", fetch.ASR_FILES)

    def test_vad_remote_and_local_names_differ(self):
        """The repo publishes model.onnx; the pipeline wants whisper_vad.onnx."""
        fetch = self._fetch_module()
        self.assertEqual(
            fetch.VAD_FILES,
            {"model.onnx": "whisper_vad.onnx", "model_metadata.json": "whisper_vad_metadata.json"},
        )


class ParserCoverageTest(unittest.TestCase):
    def test_parser_defines_every_attribute_the_pipeline_reads(self):
        """Derived from the source, so a new args.* in pipeline.py fails here first."""
        import sys

        sys.path.insert(0, str(ROOT))
        import json
        import types

        sys.modules.setdefault("pyjson5", types.SimpleNamespace(decode_io=json.load))
        from asr.server import resolve_args

        source = (ROOT / "asr" / "pipeline.py").read_text(encoding="utf-8")
        needed = set(re.findall(r"\bargs\.([a-z_]+)", source))
        args = resolve_args([])
        missing = {name for name in needed if not hasattr(args, name)}
        self.assertEqual(missing, set(), f"parser is missing {missing}")


if __name__ == "__main__":
    unittest.main()
