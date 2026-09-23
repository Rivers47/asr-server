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
`--compute-type`, `--window`, `--no-vad`.

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
try; the OpenVINO stack took three tries and still did not convert.

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

**3. The OpenVINO export fails outright.** With `optimum 2.3.0` +
`optimum-intel 2.2.0` (the versions pip resolves), both the CLI and the
in-process `export=True` path die identically:

```
TypeError: NormalizedConfig.__init__() got multiple values for argument 'allow_new'
```

Not a model problem — `WhisperOpenVINOConfig.NORMALIZED_CONFIG_CLASS(config)`
constructs fine when called directly, so the exporter is resolving a different
config class. Asking `TasksManager` shows why:

```
KeyError: 'whisper is not supported yet for transformers.
Only [] are supported for the library transformers.'
```

An empty task registry. In optimum 2.x the exporters moved into separate
packages (`optimum-onnx`, `optimum-intel`) and the registry the backend is meant
to populate is coming up empty. Downgrading transformers to 4.57.6 did not help.

Minimal repro — no export, no model weights, just the config:

```python
from optimum.exporters.openvino.model_configs import WhisperOpenVINOConfig
from transformers import AutoConfig

cfg = AutoConfig.from_pretrained("efwkjn/whisper-ja-1.5B")
WhisperOpenVINOConfig(cfg, task="automatic-speech-recognition")
# TypeError: NormalizedConfig.__init__() got multiple values for argument 'allow_new'
```

`WhisperOpenVINOConfig.NORMALIZED_CONFIG_CLASS(cfg)` succeeds on its own, so the
partial that `WhisperOnnxConfig` binds `allow_new=True` into is being applied
twice somewhere in construction. Reproduced on:

| optimum | optimum-intel | transformers | result |
|---|---|---|---|
| 2.3.0 | 2.2.0 | 5.5.4 | fails |
| 2.3.0 | 2.2.0 | 4.57.6 | fails |
| 2.1.0 | 1.27.0 | 4.57.6 | fails |

Both the `optimum-cli` and in-process `export=True` paths fail identically, so
there is no conversion route through optimum on these versions.

**Start with `torch-xpu` instead.** It needs no conversion at all — it runs the
published safetensors directly — so it cannot hit any of this. Only come back to
OpenVINO if torch-XPU is too slow and you have appetite for version archaeology.

## Reference numbers

From this repo's environment, so only the shape matters — your Arc box will
differ:

| backend | load | decode | realtime | peak RSS |
|---|---|---|---|---|
| `ct2-cpu` (int8, beam 5, 8 threads) | 5.8 s | 39.6 s | 2.25× | 3048 MB |

89 s of VAD-selected audio across 3 chunks, on 8 physical / 16 logical cores.

**Sweep `--cpu-threads` before comparing anything.** On this host the CPU
baseline ranges from 1.08× to 3.38× depending on thread count alone, with a
45% cliff one thread past the physical core count — a wider spread than most
backend changes would buy. Benchmarking a GPU backend against a mis-tuned CPU
baseline will tell you whatever you want to hear.
