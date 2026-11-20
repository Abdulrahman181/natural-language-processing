install:
python -m pip install -r requirements-prod.txt
install-ml:
python -m pip install -r requirements.txt
train:
python scripts/train_production_model.py
train-transformer:
python scripts/train_transformer.py
pipeline:
bash scripts/run_full_pipeline.sh
api:
PYTHONPATH=src uvicorn sentiment_service.app:app --host 0.0.0.0 --port 8000
test:
PYTHONPATH=src pytest -q
compose-up:
docker compose up --build
