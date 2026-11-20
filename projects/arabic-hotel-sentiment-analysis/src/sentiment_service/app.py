import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from .config import settings
from .model import model
from .schemas import BatchPredictRequest, HealthResponse, PredictRequest, Prediction

logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("sentiment_service")

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        model.load()
        logger.info("model_loaded path=%s version=%s", settings.model_path, settings.model_version)
    except FileNotFoundError:
        logger.warning("model_not_loaded path=%s; train the artifact before serving", settings.model_path)
    yield

app = FastAPI(title="Arabic Hotel Sentiment API", version="1.0.0", lifespan=lifespan)

@app.middleware("http")
async def request_logging(request: Request, call_next):
    request_id = str(uuid.uuid4())
    started = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - started) * 1000
    response.headers["X-Request-ID"] = request_id
    logger.info("request_id=%s method=%s path=%s status=%s latency_ms=%.2f", request_id, request.method, request.url.path, response.status_code, elapsed_ms)
    return response

@app.get("/healthz", response_model=HealthResponse)
def healthz():
    return HealthResponse(status="ok", model_loaded=model.loaded, model_version=settings.model_version)

@app.get("/readyz", response_model=HealthResponse)
def readyz():
    if not model.loaded:
        raise HTTPException(status_code=503, detail="model is not loaded")
    return HealthResponse(status="ready", model_loaded=True, model_version=settings.model_version)

@app.post("/v1/predict", response_model=Prediction)
def predict(payload: PredictRequest):
    if not model.loaded:
        raise HTTPException(status_code=503, detail="model is not loaded")
    return model.predict_one(payload.text)

@app.post("/v1/batch-predict", response_model=list[Prediction])
def batch_predict(payload: BatchPredictRequest):
    if not model.loaded:
        raise HTTPException(status_code=503, detail="model is not loaded")
    return [model.predict_one(item.text) for item in payload.items]

@app.exception_handler(Exception)
async def unhandled_exception(request: Request, exc: Exception):
    logger.exception("unhandled_error path=%s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "internal server error"})
