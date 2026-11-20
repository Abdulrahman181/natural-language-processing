#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$ROOT/.venv}"

usage() {
  cat <<'USAGE'
Usage:
  ./run_ubuntu.sh setup              Create venv and install all requirements
  ./run_ubuntu.sh test               Run API tests
  ./run_ubuntu.sh train-svm          Train production SVM on all model-ready rows
  ./run_ubuntu.sh train-transformer  Train Arabic BERT on all 397,993 rows
  ./run_ubuntu.sh pipeline           Run audit, cleaning, SVM, Transformer, and tests
  ./run_ubuntu.sh api                Start the FastAPI service on port 8000
  ./run_ubuntu.sh smoke               Run a small Transformer smoke test only

Official full Transformer training must be run with:
  ./run_ubuntu.sh train-transformer
Do not set MAX_SAMPLES for the official run.
USAGE
}

setup() {
  command -v "$PYTHON_BIN" >/dev/null || { echo "Python 3 is required" >&2; exit 1; }
  "$PYTHON_BIN" -m venv "$VENV_DIR"
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  python -m pip install --upgrade pip
  python -m pip install -r "$ROOT/requirements.txt"
  python -m pip install -r "$ROOT/requirements-prod.txt"
  # Register the exact virtualenv kernel used by all project notebooks.
  python -m ipykernel install --user --name sentiment-venv --display-name "Python (sentiment-venv)"
  echo "Environment is ready: $VENV_DIR"
}

activate_env() {
  if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    echo "Virtual environment is missing; running setup first."
    setup
  fi
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
  cd "$ROOT"
}

cmd="${1:-help}"
case "$cmd" in
  setup)
    setup
    ;;
  test)
    activate_env
    python -m pytest -q
    ;;
  train-svm)
    activate_env
    python scripts/train_production_model.py
    ;;
  train-transformer)
    activate_env
    echo "Starting official Transformer training on all model-ready rows."
    echo "This can take hours on CPU; a CUDA GPU is strongly recommended."
    unset MAX_SAMPLES
    python scripts/train_transformer.py
    ;;
  pipeline)
    activate_env
    bash scripts/run_full_pipeline.sh
    ;;
  api)
    activate_env
    if [[ ! -f "$ROOT/artifacts/tfidf_svm.joblib" ]]; then
      python scripts/train_production_model.py
    fi
    exec uvicorn sentiment_service.app:app --host 0.0.0.0 --port "${PORT:-8000}"
    ;;
  smoke)
    activate_env
    rm -rf /tmp/arabic_sentiment_transformer_smoke
    ARTIFACTS_ROOT=/tmp/arabic_sentiment_transformer_smoke MAX_SAMPLES=32 EPOCHS=1 MAX_LENGTH=64 \
      TRAIN_BATCH_SIZE=8 EVAL_BATCH_SIZE=8 GRADIENT_ACCUMULATION_STEPS=1 \
      python scripts/train_transformer.py
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    echo "Unknown command: $cmd" >&2
    usage >&2
    exit 2
    ;;
esac
