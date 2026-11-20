# Productionization Guide

## Architecture

- `scripts/train_production_model.py`: trains and serializes the approved TF-IDF + Linear SVM artifact.
- `src/sentiment_service`: FastAPI service with validation, health/readiness endpoints, structured request logging, batch inference, and model versioning.
- `tests`: API contract and validation tests.
- `Dockerfile` / `docker-compose.yml`: reproducible local deployment.
- `artifacts/production_model_manifest.json`: training metadata.

## Local run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-prod.txt
python scripts/train_production_model.py
make test
make api
```

Then open `http://localhost:8000/docs`.

## API example

```bash
curl -X POST http://localhost:8000/v1/predict \
  -H 'Content-Type: application/json' \
  -d '{"text":"الفندق ممتاز والخدمة رائعة"}'
```

The confidence field is a calibrated softmax proxy over LinearSVC decision scores; it must not be interpreted as a formally calibrated probability without a calibration study. In a real company deployment, add authentication, rate limiting, TLS, centralized logs, model registry, drift monitoring, and a CI/CD pipeline before public exposure.

## Full Transformer run

The official full-data Transformer entrypoint is:

```bash
python scripts/train_transformer.py
```

It uses every row by default. It creates:

- `artifacts/transformer_metrics.json`
- `artifacts/arabert_full_final/`
- `artifacts/arabert_full_final/model_manifest.json`

It automatically detects CUDA, uses mixed precision when available, applies dynamic padding, uses stratified train/validation/test splits, and handles both old and new Transformers API names. To perform a smoke test only, set `MAX_SAMPLES=5000`; remove that variable for the official run.

The company inference API intentionally keeps the lightweight SVM as the default low-latency artifact. After approving the Transformer metrics, promote the Transformer artifact through a separate model-serving image or add a controlled backend switch; do not silently replace the production model without an evaluation and rollback record.

## End-to-end pipeline

```bash
bash scripts/run_full_pipeline.sh
```
