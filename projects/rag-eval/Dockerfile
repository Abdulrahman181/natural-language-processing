FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    RAGEVAL_HOST=0.0.0.0 \
    RAGEVAL_INDEX_DIR=/app/artifacts

WORKDIR /app
RUN useradd --create-home --uid 10001 appuser

COPY pyproject.toml README.md requirements.txt ./
COPY rageval ./rageval

RUN python -m pip install --upgrade pip \
    && python -m pip install ".[serve]" \
    && chown -R appuser:appuser /app

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz')"

CMD ["python", "-m", "rageval.api"]
