# ASMR ASR Server

HTTP transcription for Japanese ASMR audio: an ASMR-tuned VAD picking chunk
boundaries, `whisper-ja-1.5B` doing the transcribing, results as JSON or subtitles.

Trimmed from [Faster-Whisper-TransWithAI-ChickenRice](https://github.com/TransWithAI/Faster-Whisper-TransWithAI-ChickenRice)
down to the server path: no translation model, no Windows bundling, no Modal
cloud inference, no batch CLI, no i18n.

## Install

```bash
uv sync                  # or: pip install -e .
python fetch_models.py   # ~3.0 GB into models/
python serve.py
```

Binds `127.0.0.1:8000`. There is no authentication — put it behind a reverse
proxy or firewall before using `--host 0.0.0.0`.

## Use

```bash
# raw body -> JSON (default); streams to disk, best for large files
curl -X POST --data-binary @audio.wav http://127.0.0.1:8000/transcribe

# multipart form -> SRT
curl -X POST -F file=@audio.opus "http://127.0.0.1:8000/transcribe?format=srt" -o audio.srt

curl http://127.0.0.1:8000/health
```

| method | path | |
|---|---|---|
| `GET` | `/health` | liveness plus the loaded configuration |
| `POST` | `/transcribe` | audio in, transcript out |

### Query parameters

| | values | default |
|---|---|---|
| `format` | `json`, `srt`, `vtt`, `lrc`, `txt` | `json` |
| `hotwords` | comma-separated words, up to 1024 chars | from the config file |
| `beam_size` | 1–10 | from the config file |

`hotwords` biases the decoder toward particular spellings — the lever for proper
nouns and homophones the model would otherwise render with its broadcast-corpus
prior. An explicit empty value clears whatever the config set. `beam_size` trades
runtime for accuracy on homophones; 1 is greedy, 5 is Whisper's own default.
Everything else — language, task, VAD parameters, chunking — is process-wide.

```bash
curl -X POST --data-binary @track.opus \
     "http://127.0.0.1:8000/transcribe?hotwords=%E6%9F%9A%E5%A7%AB,%E7%88%B6%E3%81%95%E3%81%BE&beam_size=5"
```

Both are echoed in the JSON response so a caller can confirm what applied, and
logged alongside the request. Malformed values are rejected before the upload is
read, so a bad `beam_size` never costs you a 2 GB transfer.

A raw-body upload can name itself with an `X-Filename` header so the container
extension survives.

```json
{
  "text": "こんにちはお父さま",
  "segments": [{ "start": 1.23, "end": 4.56, "text": "こんにちは" }],
  "duration": 12.5,
  "duration_after_vad": 6.25,
  "language": "ja",
  "task": "transcribe",
  "hotwords": "柚姫",
  "beam_size": 5,
  "processing_time": 8.1
}
```

Transcriptions run one at a time — CTranslate2 and the ONNX VAD session are both
driven through process-global state. Requests past `--max_queue` (default 8) get
a `503`; uploads past `--max_upload_mb` (default 2048) get a `400`. Health checks
answer immediately regardless.

## Container

```bash
podman build -t asmr-asr --format oci -f containerfile .
podman run --rm -v ./models:/srv/models:Z asmr-asr python fetch_models.py   # once
podman run --rm -v ./models:/srv/models:ro,Z -p 127.0.0.1:8000:8000 asmr-asr
```

`docker build -f containerfile .` works too. Only OCI-spec instructions are
used, so `--format oci` drops nothing — which is why there is no `HEALTHCHECK`
(a Docker config extension the OCI image spec does not define). Probe
`GET /health` from the orchestrator instead.

Weights are a bind mount rather than image content: 3 GB that versions
independently of the code. The dependency layer is separate from the source
layer, so editing `asr/` rebuilds in seconds.

**Nothing downloads the models for you.** `fetch_models.py` has to be run once
against the volume — the first command above — and the image has no network
fallback at runtime. If the volume is empty or half-populated the server refuses
to start and names the missing file, rather than coming up and returning empty
transcripts.

The image runs as uid 10001 and binds `0.0.0.0` inside the container — publish
the port only where you want it reachable, since there is no authentication.
`--vad_threads` is left at its default on purpose: it reads the cgroup quota, so
`--cpus 2` gives the VAD one thread without being told.

CI builds this image on every push and smoke-tests it (`serve.py --help`,
`fetch_models.py --help`, `import asr.server`). What CI cannot check is model
loading or transcription — the image carries no weights — so a change to the
pipeline still wants one real request against a running container.

## How it works

1. **Decode** — ffmpeg via `av`, to 16 kHz mono
2. **VAD** — `whisper_vad.onnx`, a whisper-base encoder plus 2 decoder layers,
   emitting a speech probability every 20 ms, run under ONNX Runtime
3. **Chunk** — boundaries chosen inside the longest silence in the last 40 % of
   each 30 s window, never mid-utterance
4. **Transcribe** — one `model.transcribe()` per chunk, with the VAD running
   again *inside* each chunk to keep silence away from the decoder
5. **Merge** — overlapping and duplicate segments collapsed, timeline clamped

Step 3 is the point. Whisper decodes 30 s windows autoregressively and, with
`condition_on_previous_text`, seeds each window from the last one's output.
Stock Whisper advances the window using its own predicted timestamp tokens, so a
bad decode picks a bad boundary and the error compounds — that is how a silent
stretch becomes twenty seconds of `ご視聴ありがとうございました`. Measuring the
boundary acoustically, ahead of decoding, breaks that loop, and a per-chunk
`transcribe()` call keeps a bad chunk from seeding the next one.

## Configuration

`generation_config.json5` holds decoding and VAD settings, with comments. Any
[faster-whisper `transcribe()`](https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/transcribe.py)
parameter is accepted. The flags on `serve.py --help` override the file.

`hotwords` ships empty, since the server takes arbitrary uploads. Set it in the
config as a process-wide default (`"hotwords": "柚姫, 父さま"`), or per request
with the `hotwords` query parameter.

### CPU threads

`--cpu_threads` sets CTranslate2's thread count for the ASR model, where nearly
all the compute goes. CTranslate2's own default is **4 threads regardless of
machine size**, which leaves a larger host idle.

Measured on an 8-physical-core / 16-thread host, 30 s of audio at beam 5:

| `cpu_threads` | realtime | |
|---|---|---|
| 4 | 2.40× | CTranslate2's default |
| 6 | 2.93× | |
| 8 | **3.38×** | = physical cores, **this default** |
| 9 | 1.86× | one thread more, 45% slower |
| 12 | 1.63× | |
| 16 | 1.08× | all logical threads |

The cliff sits exactly at the physical core count. The 9th thread lands on an SMT
sibling and shares one core's execution units, and because a GEMM thread team is
barrier-synchronised, that single contended pair holds up every other thread.
This is the usual rule for dense linear algebra: **threads = physical cores, not
logical**. It is not a CTranslate2 quirk, and it is not disk — the model is
resident in RAM and the audio is page-cached after the warm-up pass.

So the default is half the CPU budget (`sched_getaffinity` ∩ cgroup quota), which
is the physical core count whenever SMT is on. It stays correct even when a
hypervisor hides the topology, as it did on the host above: `lscpu` there claimed
"Thread(s) per core: 1" and `thread_siblings_list` reported no siblings, so
topology detection would have returned 16 and picked the worst setting on the
chart. That is why this is a flag rather than an auto-detect.

**If your host has SMT disabled, pass the real core count** — halving it would
idle half the machine. Either way the curve is steep on both sides, so measure:

```bash
for n in 4 8 12 16; do python serve.py --cpu_threads $n & sleep 30; \
  time curl -sX POST --data-binary @sample.opus \
       localhost:8000/transcribe > /dev/null; kill %1; done
```

`/health` reports the resolved value.

### VAD runtime

The VAD runs under ONNX Runtime, on the `CUDAExecutionProvider` when one is
available and `CPUExecutionProvider` otherwise. `pyproject.toml` declares plain
`onnxruntime`, so by default it is CPU even when the ASR model is on CUDA.
`/health` reports which, as `vad_device`.

`--vad_threads` defaults to half the CPU budget, inherited from upstream. Pass a
number to override, or `--vad_force_cpu` to pin the provider to CPU regardless.

## Why the dependency list is short

Five runtime dependencies, ~400 MB installed, against ~835 MB for the upstream set.

**No `transformers` (−69 MB).** It was there for `WhisperFeatureExtractor`, which
turns 30 s of waveform into the VAD's 80×3000 log-mel input. faster-whisper —
already installed — ships a pure-numpy extractor whose defaults are exactly what
this graph was exported against.

Validated end to end: both extractors were run against the real
`whisper_vad.onnx` on speech-like, whisper-level, mixed speech/silence, and
silent input, and the resulting speech probabilities compared.

| input | max abs Δp | mean abs Δp | frames crossing the 0.4 threshold |
|---|---|---|---|
| speech-like | 2.6e-05 | 4.5e-07 | 0 |
| whisper-level (0.004 amplitude) | 9.5e-06 | 3.6e-07 | 0 |
| speech then silence | 3.7e-08 | 4.4e-09 | 0 |
| digital silence | 0 | 0 | 0 |

No frame changes classification, so segmentation is unchanged. This also deletes
`models/whisper-base/`, which existed only to carry that extractor's config.

**Python 3.14 floor.** It drops `sympy`, `mpmath`, `coloredlogs`,
`exceptiongroup`, `tomli`, `humanfriendly` and `pyreadline3` from the lock —
backports and compat shims that older interpreters or older `onnxruntime` needed
— and lifts `onnxruntime` from 1.23.2 to 1.30.0, whose `<1.24` cap existed only
because that was the last release with cp310 wheels.

**No `librosa` (−347 MB).** It pulled numba → llvmlite (172 MB), scipy (119 MB),
and scikit-learn (34 MB) to serve one resample call that never fired, since every
caller decodes at 16 kHz. Mismatched input is now a `ValueError` rather than a
silent reinterpretation.

`tests/test_dependencies.py` fails if either returns.

## Models

Fetched once, explicitly. There is no lazy download at first request, and no
fallback if the directory is missing.

| | source | size |
|---|---|---|
| ASR | [`TransWithAI/whisper-ja-1.5B-ct2`](https://huggingface.co/TransWithAI/whisper-ja-1.5B-ct2) | 2.9 GB |
| VAD | [`TransWithAI/Whisper-Vad-EncDec-ASMR-onnx`](https://huggingface.co/TransWithAI/Whisper-Vad-EncDec-ASMR-onnx) | 114 MB |

Not committed — `fetch_models.py` pulls them (`--mirror https://hf-mirror.com` if
huggingface.co is slow). Point `--model_name_or_path` elsewhere to swap the ASR
model; any CTranslate2 Whisper model works.

## Development

```bash
python -m unittest discover -s tests -t .
ruff check . && ruff format --check .
mypy --config-file pyproject.toml --no-site-packages asr serve.py fetch_models.py
```

`.gitlab-ci.yml` runs those three in a `check` stage, then builds the container
and smoke-tests it. The push to a registry is conditional — it happens only if
GitLab's Container Registry is enabled, so CI is useful without one. See
[docs/ci.md](docs/ci.md) for runner and registry setup.

## Credits

[SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) ·
[AI汉化组](https://t.me/transWithAI) for the pipeline and both models ·
[efwkjn/whisper-ja-1.5B](https://huggingface.co/efwkjn/whisper-ja-1.5B) upstream of the ASR model.
MIT, as upstream.
