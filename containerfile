# CPU image: both the ASR model and the VAD run on the CPU.
#
#   podman build -t asmr-asr --format oci -f containerfile .
#   podman run --rm -v ./models:/srv/models:Z asmr-asr python fetch_models.py
#   podman run --rm -v ./models:/srv/models:ro,Z -p 127.0.0.1:8000:8000 asmr-asr
#
# Weights are a bind mount, not image content: 3 GB that versions independently
# of the code. Under a rootful runtime, pass --user 10001:10001.

FROM docker.io/library/python:3.14-slim

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv

# libgomp is the OpenMP runtime CTranslate2 and ONNX Runtime link against.
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
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY asr/ asr/
COPY serve.py fetch_models.py generation_config.json5 ./

VOLUME /srv/models
EXPOSE 8000

# 0.0.0.0 because the container's loopback is unreachable from the host.
CMD ["python", "serve.py", "--host", "0.0.0.0", "--port", "8000"]
