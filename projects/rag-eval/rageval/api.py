"""Optional FastAPI serving layer for evaluated retrievers."""

import os
from contextlib import asynccontextmanager
from pathlib import Path

from .index import load_index
from .retrievers import Bm25Retriever


def create_app(index_dir: str | Path = "artifacts"):
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field

    class SearchRequest(BaseModel):
        query: str = Field(min_length=1, max_length=2000)
        k: int = Field(default=5, ge=1, le=100)

    index_path = Path(index_dir)
    state = {"index": None, "retriever": None}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if (index_path / "meta.json").exists():
            state["index"] = load_index(index_path)
            state["retriever"] = Bm25Retriever(state["index"].chunks)
        yield

    app = FastAPI(title="RAG Evaluation Retrieval API", version="0.2.0", lifespan=lifespan)

    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz():
        ready = state["retriever"] is not None
        return {"status": "ready" if ready else "not_ready", "index_loaded": ready}

    @app.get("/v1/metadata")
    def metadata():
        if state["index"] is None:
            return {"index_loaded": False, "index_dir": str(index_path)}
        return {"index_loaded": True, "index_dir": str(index_path), "index": state["index"].meta}

    @app.post("/v1/search")
    def search(request: SearchRequest):
        if state["retriever"] is None:
            raise HTTPException(status_code=503, detail="index is not loaded")
        return {
            "query": request.query,
            "retriever": state["retriever"].name,
            "results": state["retriever"].search(request.query, request.k),
        }

    return app


app = create_app(os.getenv("RAGEVAL_INDEX_DIR", "artifacts"))

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "rageval.api:app",
        host=os.getenv("RAGEVAL_HOST", "127.0.0.1"),
        port=int(os.getenv("RAGEVAL_PORT", "8000")),
    )
