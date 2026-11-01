FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 GIG_BROKER_CONNECTIVITY_ENABLED=false
COPY pyproject.toml setup.py README.md ./
COPY src ./src
COPY configs ./configs
RUN pip install --no-cache-dir ".[web]" && useradd --system --uid 10001 gig && mkdir -p /app/data /app/results && chown -R gig:gig /app/data /app/results
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3)"
CMD ["python", "-m", "gig", "serve", "--host", "0.0.0.0", "--port", "8000"]
