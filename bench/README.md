# Backend benchmark

Answers one question: **is an Intel Arc backend worth wiring into the server?**

CTranslate2 is [NVIDIA-only](https://opennmt.net/CTranslate2/hardware_support.html)
— no Intel, no ROCm, no SYCL — so an Arc path means replacing the ASR call with
torch-XPU or OpenVINO and giving up faster-whisper. That is worth doing only if
it beats the CPU baseline by enough to matter, at comparable output quality.

Measure before committing to the work.

## How it compares fairly

All backends get the **same chunk boundaries**, planned once from the repo's
ASMR VAD (or fixed 30 s windows with `--no-vad`). Each backend then decodes
those identical arrays with `vad_filter=False`, so the numbers reflect the
decoder alone rather than differing chunking strategies.

Each backend runs in its own subprocess. That keeps peak-memory figures honest
and stops CTranslate2, torch-XPU and OpenVINO from fighting over oneAPI/MKL
runtimes in one interpreter.

The first chunk is decoded untimed as a warm-up (`--no-warmup` to skip).

## Usage

```bash
python bench/bench_backends.py --list
python bench/bench_backends.py four_short.opus --backends ct2-cpu
python bench/bench_backends.py four_short.opus \
    --backends ct2-cpu,torch-xpu,openvino-gpu --show-text
```

Useful flags: `--limit N` (first N chunks only, for a quick loop), `--beam-size`,
`--compute-type`, `--window`, `--no-vad`, `--hotwords`.

## Hotwords

The server runs with hotwords set, so a comparison without them measures a
configuration you do not deploy. Pass the same list to every backend with
`--hotwords`:

```bash
python bench/bench_backends.py four_short.opus \
    --backends ct2-cpu,openvino-gpu --ov-model bench/ov-whisper-ja \
    --hotwords "$(cat hotwords.txt)" --show-text
```

`hotwords` is faster-whisper's own parameter, and it is prompt injection:
`tokenizer.encode(" " + hotwords)` appended to the decoder prompt, truncated to
`max_length // 2 - 1` = 223 tokens. The transformers backends (`torch-*`,
`openvino-*`) have no such parameter, so the harness converts the string with
`WhisperProcessor.get_prompt_ids` and passes it as `prompt_ids` — the same
mechanism, reachable because optimum-intel dispatches Whisper to
`_OVModelForWhisper(OVModelForSpeechSeq2Seq, WhisperForConditionalGeneration)`.

Two differences the harness absorbs, and that a server port would have to as well:

- transformers counts prompt + special tokens + `max_new_tokens` against the model's
  448 `max_target_positions` and **raises** rather than truncating, so
  `max_new_tokens` is computed per run instead of fixed.
- the prompt is capped at the same 223 tokens faster-whisper uses, so every backend
  sees identical hotword content.

Per-chunk `generate()` calls re-assert the prompt at the head of every chunk, which
is the behaviour the server's smart-split path relies on. A whole-file call would
need `prompt_condition_type="all-segments"` and `condition_on_prev_tokens`.

Output is realtime factor (audio seconds per wall second — higher is faster),
model load time, peak RSS, and a character-level agreement ratio against the
first backend listed.

**Read the agreement ratio.** A fast backend that disagrees with the baseline is
not a win. Use `--show-text` and look at the transcripts before believing a
number.

## Separate environments per backend

torch-XPU and CTranslate2 in one environment is the oneAPI/MKL conflict this
harness is built to avoid. Give each its own venv and point the harness at them:

```bash
python -m venv ~/venv-xpu
~/venv-xpu/bin/pip install torch --index-url https://download.pytorch.org/whl/xpu
~/venv-xpu/bin/pip install transformers numpy

python bench/bench_backends.py four_short.opus \
    --backends ct2-cpu,torch-xpu \
    --python torch-xpu=~/venv-xpu/bin/python
```

The chunk plan is still built once, in the parent, and passed to every worker as
a `.npy` file — so separate environments do not change what is being compared.

## Backend setup

### `ct2-cpu` — the baseline

Already installed: `uv sync`. Uses `models/` as it stands.

### `torch-xpu` / `torch-cpu` — Intel Arc via PyTorch

[Native XPU support since PyTorch 2.5](https://pytorch.org/blog/intel-gpu-support-pytorch-2-5/).
Least conversion work: runs `efwkjn/whisper-ja-1.5B` as published.

```bash
pip install torch --index-url https://download.pytorch.org/whl/xpu
pip install transformers
```

First run downloads ~3 GB from the Hub. Pass `--hf-model /path/to/local` to use
a local copy. `torch-cpu` is the control — run it to separate "the Arc is fast"
from "torch is slow".

### `openvino-gpu` / `openvino-cpu` — Intel Arc via OpenVINO

Most native Intel path. Export the IR once rather than converting on every run:

```bash
pip install optimum-intel[openvino]
optimum-cli export openvino --model efwkjn/whisper-ja-1.5B \
    --task automatic-speech-recognition-with-past ov-whisper-ja

python bench/bench_backends.py four_short.opus \
    --backends openvino-gpu --ov-model ov-whisper-ja
```

Without `--ov-model` the model is exported on the fly, which inflates the load
time and tells you nothing useful.

Add `--weight-format int8` to the export for a quantized comparison — worth
running as its own entry, since quantization is most of OpenVINO's advantage.

## whisper.cpp is not in this harness

whisper.cpp with the [SYCL backend](https://github.com/ggml-org/whisper.cpp/blob/master/README_sycl.md)
also targets Arc, but it is a separate binary with its own chunking and seek
logic. Benchmarking it per-chunk would mean reloading the model on every
subprocess call, which would measure the wrong thing.

Adopting it would mean replacing the whole server rather than swapping a
backend, so measure it separately if you get that far:

```bash
./build/bin/whisper-cli -m ggml-whisper-ja-1.5B.bin -f audio.wav -l ja
```

One trap: whisper.cpp's *OpenVINO* backend accelerates only the encoder and
leaves the decoder on CPU. Build the **SYCL** backend, not the OpenVINO one.

## Known install traps

Hit while trying to benchmark `openvino-cpu` in this repo's environment
(Python 3.14, September 2026). The CTranslate2 baseline installed and ran first
try; the OpenVINO stack needed all three fixes below.

**1. `torchvision` from the wrong index.** `pip install optimum-intel[openvino]`
pulls `torchvision` from the default index, which will not match a CPU or XPU
torch build:

```
RuntimeError: operator torchvision::nms does not exist
```

It blocks `from optimum.intel import ...` entirely. Whisper needs no vision
stack, so `pip uninstall torchvision` fixes it — optimum-intel still lists it as
a hard requirement, so pip will warn.

**2. `forced_decoder_ids` is gone in transformers 5.x.** `generate()` no longer
accepts it. Pass `language=` and `task=` directly instead; that works on
transformers 4.39+ and 5.x alike. This harness already does.

**3. The OpenVINO export fails on Python 3.14.** With `optimum 2.3.0` +
`optimum-intel 2.2.0` + `transformers 5.5.4`, both the CLI and the in-process
`export=True` path die identically:

```
TypeError: NormalizedConfig.__init__() got multiple values for argument 'allow_new'
```

The cause is a Python 3.14 change, not a version mismatch: `functools.partial`
now implements `__get__`, so a partial stored as a class attribute binds like a
method and prepends the instance to the positional arguments. optimum builds
every exporter config that way —

```python
NORMALIZED_CONFIG_CLASS = NormalizedSeq2SeqConfig.with_args(..., allow_new=True)
```

— and `optimum/exporters/base.py:151` calls it as
`self.NORMALIZED_CONFIG_CLASS(self._config)`. On 3.14 that becomes
`NormalizedSeq2SeqConfig(self, self._config, allow_new=True, ...)`, so
`self._config` lands on the `allow_new` parameter that `with_args` already bound.
Whisper fails loudly because its config binds `allow_new=True`; configs without
it get the instance as their `config` instead and fail later or silently.

Repro with no optimum involved:

```python
import functools

def f(config, allow_new=False, **kw): ...

class C:
    NORMALIZED = functools.partial(f, allow_new=True)
    def go(self): self.NORMALIZED("the-real-config")

C().go()   # TypeError on 3.14
```

Neither `optimum` nor `optimum-intel` claims 3.14 support, and 2.3.0 / 2.2.0 are
the current releases, so there is no version to upgrade to.

Two ways out. The IR is a build artifact and the server never imports optimum, so
**export in a throwaway Python 3.13 venv** and point `--ov-model` at the result.
Or re-bind the partials before exporting:

```python
import functools, inspect, sys
import optimum.exporters.openvino.model_configs

for name, module in list(sys.modules.items()):
    if name.startswith("optimum") and module is not None:
        for _, obj in inspect.getmembers(module, inspect.isclass):
            for attr, value in list(vars(obj).items()):
                if isinstance(value, functools.partial):
                    setattr(obj, attr, staticmethod(value))
```

Verified on 3.14: 43 attributes re-bound, `openai/whisper-tiny` exported to IR,
and the IR transcribed `four_short.opus` through `OVModelForSpeechSeq2Seq` on CPU.
The 3.13 route follows from the same mechanism — `staticmethod` restores exactly
the pre-3.14 attribute access — but was not run, since no 3.13 interpreter was
on the box.

The earlier note here blamed an empty `TasksManager` registry. That was wrong —
the `KeyError: Only []` comes from querying `TasksManager` before
`optimum.exporters.openvino.model_configs` is imported. After importing it, 179
model types are registered and `whisper` carries all five tasks under the
`openvino` exporter key.

`optimum-onnx` is not needed: optimum 2.x moved the ONNX exporters there, but
optimum-intel 2.2.0 defines its own OpenVINO configs. It could not be installed
anyway — `optimum-onnx 0.1.0` pins `optimum~=2.1.0`.

**4. `optimum-cli` will not start where a venv's `lib64` is a symlink to `lib`**
(Fedora/RHEL-family layouts; Debian/Ubuntu venvs have no `lib64`):

```
argparse.ArgumentError: argument {openvino}: conflicting subparser: openvino
```

`optimum.commands.register` is a PEP 420 namespace package, and a venv puts both
`lib/pythonX.Y/site-packages` and `lib64/pythonX.Y/site-packages` on `sys.path`.
When `lib64` is a symlink to `lib`, the namespace reports the same directory under
both spellings, and `load_optimum_namespace_cli_commands()` dedupes with `set()`
over path strings rather than resolved paths. So optimum-intel's
`register_openvino.py` is imported twice and `OVExportCommand` registers twice
under `export`. Fingerprint:

```bash
python -c "import importlib.util; print(list(importlib.util.find_spec('optimum.commands.register').submodule_search_locations))"
```

Two entries differing only by `lib` vs `lib64` confirms it. Either call
`main_export` from a script and skip the CLI, or add a `sitecustomize.py` to the
venv that rewrites `sys.path` to one entry per `os.path.realpath`. Deleting the
`lib64` symlink also works, but a later `pip install` may recreate it as a real
directory and split the packages across two trees.

Export peak memory is the other limit: the fp32 torch model, the trace, and the
OpenVINO model are live at once, and the 1.5B IR is ~6 GB at fp32. Run the export
under `/usr/bin/time -v` to capture peak RSS. `--weight-format fp16` on the CLI, or
`ov_config=OVConfig(dtype="fp16")` in a script, roughly halves both.

`torch-xpu` is still the shorter path to a first Arc number: it runs the
published safetensors with no conversion step at all.

## Reference numbers

From this repo's environment, so only the shape matters — your Arc box will
differ:

| backend | load | decode | realtime | peak RSS |
|---|---|---|---|---|
| `ct2-cpu` (int8, beam 5, 8 threads) | 5.8 s | 39.6 s | 2.25× | 3048 MB |

89 s of VAD-selected audio across 3 chunks, on 8 physical / 16 logical cores.

A later run in a container (31 GB, 16 cores as the container reports them), on
143.8 s of audio — 104.4 s of speech across 5 chunks — with the fp32 IR exported
as described above:

| backend | load | decode | realtime | peak RSS | agreement |
|---|---|---|---|---|---|
| `ct2-cpu` (int8, beam 5) | 4.3 s | 58.1 s | 2.47× | 3050 MB | 1.00 |
| `openvino-cpu` (fp32, beam 5) | 13.7 s | 128.8 s | 1.12× | 15818 MB | 0.740 |

Both tables above were measured **without** `--hotwords`, so neither reflects the
server's real configuration. Re-run with it before drawing conclusions.

OpenVINO is 2.2× slower at 5× the memory, but the comparison is not
precision-matched: the baseline is int8 and the IR is fp32. Export with
`--weight-format int8` before concluding anything about the framework.

The agreement ratio is the finding that matters. At 0.740 the OpenVINO transcript
drops most of the non-verbal utterances CTranslate2 captures — whole `ちゅぷっ` and
`れろぉ` runs disappear — and contains one garbled span where the baseline is
intelligible. Both ran beam 5 over identical chunk boundaries, so chunking is not
the cause. For ASMR those utterances are content, not noise.

Exporting the fp32 IR took 96.9 s and peaked at 8548 MB RSS for 5.8 GB of output.

**Sweep `--cpu-threads` before comparing anything.** On this host the CPU
baseline ranges from 1.08× to 3.38× depending on thread count alone, with a
45% cliff one thread past the physical core count — a wider spread than most
backend changes would buy. Benchmarking a GPU backend against a mis-tuned CPU
baseline will tell you whatever you want to hear.
