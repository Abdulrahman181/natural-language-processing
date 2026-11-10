FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/huggingface \
    CWE_QLORA_HOST=0.0.0.0

WORKDIR /app

RUN useradd --create-home --uid 10001 appuser

COPY pyproject.toml README.md requirements.txt ./
COPY cwe_qlora ./cwe_qlora

RUN python -m pip install --upgrade pip \
    && python -m pip install ".[serve]" \
    && mkdir -p "$HF_HOME" \
    && chown -R appuser:appuser /app "$HF_HOME"

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz')"

CMD ["python", "-m", "cwe_qlora.api"]
