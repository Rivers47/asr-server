# CPU image: both the ASR model and the VAD run on the CPU.
#
#   podman build -t asmr-asr --format oci -f containerfile .
#   buildah bud  -t asmr-asr --format oci -f containerfile .
#   docker build -t asmr-asr -f containerfile .
#
# Only OCI-spec instructions are used, so `--format oci` produces a complete
# image with nothing dropped. Notably absent: HEALTHCHECK, which is a Docker
# config extension the OCI image spec does not define -- probe GET /health from
# the orchestrator instead.
#
# Weights are a bind mount, not image content: they are 3 GB and version
# independently of the code.
#
#   podman run --rm -v ./models:/srv/models:ro,Z -p 127.0.0.1:8000:8000 asmr-asr
#
# Populate them once with:
#
#   podman run --rm -v ./models:/srv/models:Z asmr-asr python fetch_models.py

FROM docker.io/library/python:3.14-slim

# uv resolves from the committed lock. Pinned so a rebuild cannot drift.
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv

# libgomp is the OpenMP runtime CTranslate2 and ONNX Runtime link against.
# PyAV bundles its own ffmpeg, so no system ffmpeg is needed.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

ENV UV_PROJECT_ENVIRONMENT=/srv/.venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/srv/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Dependency layer first: rebuilt only when the lock changes, not on every edit.
# pyproject sets `package = false`, so this installs dependencies without
# needing the source tree.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY asr/ asr/
COPY serve.py fetch_models.py generation_config.json5 ./

# No USER on purpose. Under rootless podman the container's root is already an
# unprivileged host user, so a baked-in uid drops no privilege -- it only stops
# the process matching the owner of a bind-mounted models directory, which is
# the one thing here that needs write access.
#
# Running this under ROOTFUL docker or podman does give the process real root.
# Pass --user there:  podman run --user 10001:10001 ...
# Nothing in the image needs to write outside /tmp and the models mount.

# Without this, `fetch_models.py` run without -v writes 3 GB into the container's
# writable layer; with it, that lands in a volume instead. Forgetting -v still
# gets you an empty models directory, but the server refuses to start and names
# the missing file rather than serving empty transcripts.
VOLUME /srv/models

EXPOSE 8000

# 0.0.0.0 because the container's loopback is unreachable from the host. Publish
# the port only where you want it reachable -- there is no authentication.
#
# --cpu_threads and --vad_threads are omitted deliberately: both defaults read
# the cgroup CPU quota, so `--cpus 2` sizes them correctly without being told.
CMD ["python", "serve.py", "--host", "0.0.0.0", "--port", "8000"]
