# syntax=docker/dockerfile:1
ARG PYTHON_IMAGE=python:3.12-slim
FROM ${PYTHON_IMAGE} AS build
ENV CXX=g++ CC=gcc
# Read-only context mounts avoid full VFS copies for each COPY/RUN layer.
# Only the explicit source inputs enter the disposable build directory.
RUN --mount=type=bind,source=.,target=/source,readonly \
    --mount=type=secret,id=proxy_ca \
    if [ -f /run/secrets/proxy_ca ]; then export PIP_CERT=/run/secrets/proxy_ca; fi; \
    apt-get update && apt-get install -y --no-install-recommends g++ \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /build /wheels \
    && cp /source/pyproject.toml /source/setup.py /source/README.md /build/ \
    && cp -R /source/src /source/configs /build/ \
    && cd /build \
    && pip wheel --no-cache-dir --wheel-dir /wheels ".[web,research,calendar]"

FROM ${PYTHON_IMAGE}
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 GIG_BROKER_CONNECTIVITY_ENABLED=false
RUN --mount=type=bind,from=build,source=/wheels,target=/wheels,readonly \
    --mount=type=bind,source=configs,target=/source-configs,readonly \
    --mount=type=secret,id=proxy_ca \
    apt-get update && apt-get install -y --no-install-recommends libstdc++6 libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir --no-index --find-links=/wheels "gig[web,research,calendar]" \
    && groupadd --system --gid 10001 gig && useradd --system --uid 10001 --gid gig gig \
    && mkdir -p /app/data /app/results /app/configs \
    && cp -R /source-configs/. /app/configs/ \
    && chown -R gig:gig /app/data /app/results
WORKDIR /app
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3)"
CMD ["python", "-m", "gig", "serve", "--host", "0.0.0.0", "--port", "8000"]
