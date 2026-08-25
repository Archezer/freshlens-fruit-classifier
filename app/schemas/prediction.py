from enum import StrEnum

from pydantic import BaseModel, Field


class FruitQuality(StrEnum):
    good = "good"
    rotten = "rotten"


class PredictionResponse(BaseModel):
    predicted_class: FruitQuality
    confidence: float = Field(ge=0.0, le=1.0)
    probabilities: dict[FruitQuality, float]
    heatmap_url: str | None = None
