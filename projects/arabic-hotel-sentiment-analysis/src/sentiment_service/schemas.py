from typing import List
from pydantic import BaseModel, Field, field_validator

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000, description="Arabic hotel review")

    @field_validator("text")
    @classmethod
    def non_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("text must not be blank")
        return value

class BatchPredictRequest(BaseModel):
    items: List[PredictRequest] = Field(..., min_length=1, max_length=100)

class Prediction(BaseModel):
    sentiment: str
    confidence: float = Field(..., ge=0, le=1)
    probabilities: dict[str, float]
    model_version: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
