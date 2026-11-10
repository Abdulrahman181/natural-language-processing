"""Optional FastAPI service for CVE-to-CWE inference."""

import os
from contextlib import asynccontextmanager
from dataclasses import asdict

from .parsing import parse_prediction
from .service import CWEInferenceEngine, InferenceConfig


def create_app(config: InferenceConfig | None = None):
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field

    engine = CWEInferenceEngine(config or InferenceConfig())

    class PredictionRequest(BaseModel):
        description: str = Field(min_length=1, max_length=10000)

    class BatchPredictionRequest(BaseModel):
        descriptions: list[str] = Field(min_length=1, max_length=256)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.engine = engine
        yield

    app = FastAPI(
        title="CVE-to-CWE Inference API",
        version="0.3.0",
        lifespan=lifespan,
    )

    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz():
        if not engine.loaded:
            return {"status": "not_ready", "model_loaded": False}
        return {"status": "ready", "model_loaded": True, "metadata": engine.metadata()}

    @app.post("/v1/predict")
    def predict(request: PredictionRequest):
        try:
            result = engine.predict(request.description)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail="inference backend unavailable") from exc
        return result

    @app.post("/v1/predict:batch")
    def predict_batch(request: BatchPredictionRequest):
        try:
            raw_outputs = engine.predict_raw(request.descriptions)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail="inference backend unavailable") from exc
        return {
            "predictions": [
                {"prediction": parse_prediction(raw), "raw": raw}
                for raw in raw_outputs
            ],
            "count": len(raw_outputs),
        }

    @app.get("/v1/metadata")
    def metadata():
        return asdict(config or InferenceConfig()) | {"runtime": engine.metadata()}

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "cwe_qlora.api:app",
        host=os.getenv("CWE_QLORA_HOST", "127.0.0.1"),
        port=int(os.getenv("CWE_QLORA_PORT", "8000")),
    )
