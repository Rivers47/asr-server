# CPU image. Both the ASR model and the VAD run on the CPU.
#
#   docker build -t asmr-asr .
#   docker run --rm -v "$PWD/models:/srv/models" -p 8000:8000 asmr-asr
#
# Models are not baked in -- they are 3 GB and change independently of the code.
# Populate the volume once with:
#   docker run --rm -v "$PWD/models:/srv/models" asmr-asr python fetch_models.py

FROM python:3.10-slim

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv

# libgomp is the OpenMP runtime CTranslate2 and ONNX Runtime link against. PyAV
# bundles its own ffmpeg, so no system ffmpeg is needed.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /srv
ENV UV_PROJECT_ENVIRONMENT=/srv/.venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/srv/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

# Dependency layer first: rebuilt only when the lock changes, not on every edit.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY asr/ asr/
COPY serve.py fetch_models.py generation_config.json5 ./

RUN useradd --create-home --uid 10001 asr && chown -R asr:asr /srv
USER asr

VOLUME /srv/models
EXPOSE 8000

# --vad_threads defaults to half the container's CPU budget, read from the
# cgroup quota rather than the host's core count.
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5)"

# 0.0.0.0 because the container's loopback is unreachable from the host. Publish
# the port only where you want it reachable -- there is no authentication.
CMD ["python", "serve.py", "--host", "0.0.0.0", "--port", "8000"]
