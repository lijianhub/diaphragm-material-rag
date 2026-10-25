# The Python RAG API (FastAPI). The index is built at image build time from the tracked
# sample documents only: .dockerignore keeps the private PDF, .env and local data out.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first, so code changes do not reinstall them.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[api,llm]" && rm -rf build

COPY scripts ./scripts
COPY data/raw ./data/raw
RUN python scripts/index.py \
    && useradd --create-home --uid 10001 app \
    && mkdir -p logs runs \
    && chown -R app:app logs runs

USER app
EXPOSE 8000
# Readiness, not liveness: the container is healthy once the index is loaded.
HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=5 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=2).status == 200 else 1)"
CMD ["python", "scripts/serve.py", "--host", "0.0.0.0", "--port", "8000"]
