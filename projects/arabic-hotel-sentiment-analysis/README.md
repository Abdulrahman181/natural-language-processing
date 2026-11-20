# Arabic Hotel Reviews — Sentiment Analysis Submission

This project is catalogued under the NLP repository as a company-ready Arabic sentiment-analysis system. It includes reproducible data-quality notebooks, traditional ML, BiLSTM, a full-data Arabic Transformer pipeline, and a FastAPI inference service.

The raw and derived CSV files are intentionally not stored in this public category repository because GitHub rejects regular files larger than 100 MB. See [`data/README.md`](./data/README.md) and [`DATA_DOWNLOAD.md`](./DATA_DOWNLOAD.md) for the data placement contract. The private full bundle remains available in the standalone private repository.

This submission contains an engineering-grade notebook for the supplied Arabic hotel-review dataset.

## Data integrity
- Original source preserved byte-for-byte in `data/Hotel Arabic-Reviews Dataset.txt`.
- SHA-256: `fc3bb8210d4f8a658b77c92fe659e532173432679992d5ae1e6c91969681748b`
- Encoding: UTF-16LE with BOM.
- Logical records: 409,562.
- Seven fields per record; no malformed rows found.
- Original data is never overwritten.

## Findings before modeling
- Ratings: 1=14,382, 2=38,467, 3=80,326, 4=132,208, 5=144,179.
- Strong positive-class skew; macro-F1 is reported in addition to accuracy.
- 2,693 repeated review-text groups (9,461 extra copies) were observed. They are reported, not silently deleted.
- Some metadata values are `-`, treated as unknown categories.

## Run
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook Arabic_Hotel_Sentiment_Analysis.ipynb
```

The notebook defaults to a stratified 60,000-record experiment for ordinary hardware. Set `MAX_SAMPLES=None` for all records. Set `RUN_TRANSFORMER=True` to execute the Arabic BERT fine-tuning section when a GPU or adequate RAM is available.


## Staged data pipeline

The project is intentionally split into stages:

1. `01_full_data_loading_and_quality_audit.ipynb` loads the complete source and writes reproducible reports under `reports/`.
2. `02_non_destructive_preprocessing.ipynb` creates `data/processed/hotel_reviews_processed.csv` with 409,562 rows and 30 columns. It preserves every original field as `original_*` and adds cleaned fields, derived labels, numeric nights, and explicit quality flags.
3. The modeling notebooks consume only the processed derived dataset.

Both stage notebooks were executed successfully on the full dataset. The processed CSV is a derived artifact; the original UTF-16LE text file remains unchanged.


## Full cleaning completed

Notebook 02 now performs the complete documented cleaning workflow. It produced:

- Full traceable processed dataset: `data/processed/hotel_reviews_processed.csv` (409,562 rows).
- Final model-ready dataset: `data/processed/hotel_reviews_model_ready.csv` (397,993 rows, no duplicate normalized reviews, no empty/short reviews, and no invalid ratings).
- Quarantine dataset: `data/processed/hotel_reviews_quarantine.csv` (11,569 rows) with an explicit exclusion reason for every row.
- Cleaning log: `data/processed/model_ready_cleaning_log.json`.

The model-ready exclusions were: 11,529 normalized duplicate reviews, 32 empty reviews after normalization, and 8 too-short reviews. The 72 room-like `user type` values were corrected under the documented rule: moved into the room field only when the room field was missing, and the user field was set to `Unknown`. No original source row was overwritten; excluded rows are retained in quarantine.

## Company-ready production layer

The project now includes a production inference service in addition to the research notebooks:

- `src/sentiment_service/`: FastAPI service with validation, `/healthz`, `/readyz`, `/v1/predict`, `/v1/batch-predict`, request IDs, latency logs, and model versioning.
- `scripts/train_production_model.py`: trains and serializes the approved model artifact from all 397,993 model-ready rows.
- `artifacts/tfidf_svm.joblib`: production model artifact.
- `tests/`: API contract tests; all tests pass.
- `Dockerfile` and `docker-compose.yml`: reproducible container execution.
- `.github/workflows/ci.yml`: CI test workflow.
- `PRODUCTION_README.md`: local, Docker, API, and operational guidance.

The service was verified locally with health, readiness, single-prediction, and batch-prediction requests. Before exposing it publicly, a company deployment should add authentication, TLS, rate limiting, centralized logs, model registry/version promotion, drift monitoring, and a secrets manager.

## Full Transformer and end-to-end pipeline

The full-data Transformer implementation is now available in both `scripts/train_transformer.py` and `05_transformer_arabic_bert.ipynb`. It defaults to all 397,993 model-ready records and saves a complete model directory plus metrics. Run `make train-transformer` or `bash scripts/run_full_pipeline.sh` on a machine with sufficient RAM/GPU. The pipeline never substitutes a sample for the official run unless `MAX_SAMPLES` is explicitly set.
