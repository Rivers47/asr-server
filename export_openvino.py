#!/usr/bin/env python3
"""
Convert the Whisper checkpoint to the OpenVINO IR the openvino backend loads.

    python export_openvino.py                      # models/ov-int8
    python export_openvino.py --precision fp32     # models/ov-fp32
    python export_openvino.py --precision both

There is nothing to download: no OpenVINO export of this model is published, so the
transformers checkpoint is fetched and converted here. That costs ~3 GB of download,
around 8.5 GB of RSS while tracing, and 6.2 GB of disk for the fp32 IR -- which is
written even when only int8 is wanted, because the compression pass reads it.

Runs inside the OpenVINO image, which already has the stack:

    podman run --rm -v ./models:/srv/models:Z asmr-asr-ov python export_openvino.py
"""

import argparse
import functools
import inspect
import resource
import shutil
import sys
import tempfile
import time
from pathlib import Path

MODEL = "efwkjn/whisper-ja-1.5B"
# The task that exports a KV-cached decoder. Without -with-past the decoder
# recomputes the whole prefix at every step.
TASK = "automatic-speech-recognition-with-past"
IR_FILES = ("openvino_encoder_model.xml", "openvino_decoder_model.xml")


def rebind_partials() -> int:
    """Make optimum's functools.partial class attributes readable again on Python 3.14.

    3.14 gave functools.partial a __get__, so a partial held as a class attribute binds
    like a method and prepends the instance to the positional arguments. Every optimum
    exporter config is built that way -- NORMALIZED_CONFIG_CLASS = X.with_args(...) --
    and the exporter then calls it as self.NORMALIZED_CONFIG_CLASS(self._config),
    which raises "got multiple values for argument 'allow_new'". Wrapping each partial
    in staticmethod restores the access every optimum release up to 2.3.0 assumes, and
    is a no-op on earlier interpreters.
    """
    import optimum.exporters.openvino.model_configs  # noqa: F401 -- fills the task registry

    patched = 0
    for name, module in list(sys.modules.items()):
        if not name.startswith("optimum") or module is None:
            continue
        for _, obj in inspect.getmembers(module, inspect.isclass):
            for attr, value in list(vars(obj).items()):
                if isinstance(value, functools.partial):
                    setattr(obj, attr, staticmethod(value))
                    patched += 1
    return patched


def is_complete(directory: Path) -> bool:
    return all((directory / name).exists() for name in IR_FILES) and (directory / "config.json").exists()


def export_fp32(model: str, output: Path, cache_dir: str | None) -> None:
    from optimum.exporters.openvino import main_export

    started = time.perf_counter()
    print(f"exporting {model} -> {output}", flush=True)
    main_export(model_name_or_path=model, output=str(output), task=TASK, cache_dir=cache_dir)
    print(f"  fp32 IR written in {time.perf_counter() - started:.0f}s", flush=True)


def compress_int8(source: Path, destination: Path) -> None:
    """Weight-only int8, matching `optimum-cli export openvino --weight-format int8`.

    main_export ignores ov_config.quantization_config -- it runs its quantization pass
    only when that config is falsy -- so compression is applied to the finished IR.
    """
    import nncf
    import openvino as ov

    destination.mkdir(parents=True, exist_ok=True)
    core = ov.Core()
    for name in IR_FILES:
        started = time.perf_counter()
        compressed = nncf.compress_weights(core.read_model(source / name), mode=nncf.CompressWeightsMode.INT8_ASYM)
        ov.save_model(compressed, destination / name)
        del compressed
        print(f"  compressed {name} in {time.perf_counter() - started:.0f}s", flush=True)

    for extra in sorted(source.glob("*.json")):  # config, generation config, tokenizer
        shutil.copy2(extra, destination / extra.name)


def peak_rss_mb() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=MODEL, help=f"HF checkpoint to convert (default: {MODEL})")
    parser.add_argument(
        "--precision",
        default="int8",
        choices=["int8", "fp32", "both"],
        help="int8 (default) writes ov-int8 only, using a scratch copy of the fp32 IR; "
        "fp32 writes ov-fp32; both keeps each.",
    )
    parser.add_argument("--models-dir", default="models", type=Path, help="Destination (default: models)")
    parser.add_argument("--cache-dir", default=None, help="HF cache for the checkpoint (default: $HF_HOME)")
    parser.add_argument("--force", action="store_true", help="Re-export even if the IR is already there")
    args = parser.parse_args()

    fp32_dir = args.models_dir / "ov-fp32"
    int8_dir = args.models_dir / "ov-int8"
    wanted = {"int8": [int8_dir], "fp32": [fp32_dir], "both": [fp32_dir, int8_dir]}[args.precision]

    if not args.force:
        wanted = [directory for directory in wanted if not is_complete(directory)]
        if not wanted:
            print("IR already present; pass --force to re-export")
            return 0

    print(f"re-bound {rebind_partials()} partial class attributes", flush=True)
    args.models_dir.mkdir(parents=True, exist_ok=True)

    if fp32_dir in wanted:
        export_fp32(args.model, fp32_dir, args.cache_dir)
        if int8_dir in wanted:
            compress_int8(fp32_dir, int8_dir)
    elif int8_dir in wanted:
        # The fp32 IR is an intermediate here. Keeping it on the destination filesystem
        # means the compressed copy is written beside it rather than across a mount.
        with tempfile.TemporaryDirectory(dir=args.models_dir, prefix="ov-export-") as scratch:
            export_fp32(args.model, Path(scratch) / "fp32", args.cache_dir)
            compress_int8(Path(scratch) / "fp32", int8_dir)

    for directory in wanted:
        total = sum(path.stat().st_size for path in directory.glob("*.bin"))
        print(f"{directory}: {total / 1e9:.2f} GB of weights")
    print(f"peak RSS {peak_rss_mb()} MB")
    print(f"\nServe it with: OV_PRECISION={'int8' if int8_dir in wanted else 'fp32'} python serve.py")
    return 0


if __name__ == "__main__":
    # nncf starts loky workers, and a spawn or forkserver child re-imports this module.
    # Without the guard the child runs a second concurrent export into the same output
    # directory, doubling peak memory and racing on every file.
    sys.exit(main())
