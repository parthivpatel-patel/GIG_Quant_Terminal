# syntax=docker/dockerfile:1
FROM python:3.12-slim AS build
WORKDIR /app
ENV CXX=g++ CC=gcc
RUN --mount=type=secret,id=proxy_ca \
    apt-get update && apt-get install -y --no-install-recommends g++ && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml setup.py README.md ./
COPY src ./src
COPY configs ./configs
RUN --mount=type=secret,id=proxy_ca \
    if [ -f /run/secrets/proxy_ca ]; then export PIP_CERT=/run/secrets/proxy_ca; fi; \
    pip wheel --no-cache-dir --wheel-dir /wheels ".[web]"

FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 GIG_BROKER_CONNECTIVITY_ENABLED=false
RUN --mount=type=secret,id=proxy_ca \
    apt-get update && apt-get install -y --no-install-recommends libstdc++6 && rm -rf /var/lib/apt/lists/*
COPY --from=build /wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links=/wheels "gig[web]" && rm -rf /wheels \
    && groupadd --system --gid 10001 gig && useradd --system --uid 10001 --gid gig gig \
    && mkdir -p /app/data /app/results && chown -R gig:gig /app/data /app/results
COPY configs ./configs
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3)"
CMD ["python", "-m", "gig", "serve", "--host", "0.0.0.0", "--port", "8000"]
