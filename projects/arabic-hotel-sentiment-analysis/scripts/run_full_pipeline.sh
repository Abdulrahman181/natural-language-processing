#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
cd "$ROOT"

echo '[1/5] Run data audit notebook'
"$PYTHON" -m jupyter nbconvert --to notebook --execute 01_full_data_loading_and_quality_audit.ipynb --output 01_full_data_loading_and_quality_audit_executed.ipynb --ExecutePreprocessor.kernel_name=sentiment-venv --ExecutePreprocessor.timeout=3600

echo '[2/5] Run non-destructive cleaning notebook'
"$PYTHON" -m jupyter nbconvert --to notebook --execute 02_non_destructive_preprocessing.ipynb --output 02_non_destructive_preprocessing_executed.ipynb --ExecutePreprocessor.kernel_name=sentiment-venv --ExecutePreprocessor.timeout=3600

echo '[3/5] Train production SVM artifact'
"$PYTHON" scripts/train_production_model.py

echo '[4/5] Train full Arabic Transformer (all model-ready rows)'
"$PYTHON" scripts/train_transformer.py

echo '[5/5] Run API tests'
PYTHONPATH=src "$PYTHON" -m pytest -q

echo 'Full pipeline completed successfully.'
