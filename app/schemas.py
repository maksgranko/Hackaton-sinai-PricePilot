from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, confloat, conint


class PlatformEnum(str, Enum):
    android = "android"
    ios = "ios"
    web = "web"


class OrderRequest(BaseModel):
    order_timestamp: int = Field(..., gt=0, description="Unix timestamp of the order in seconds.")
    distance_in_meters: int = Field(..., ge=0)
    duration_in_seconds: int = Field(..., ge=0)
    pickup_in_meters: int = Field(..., ge=0)
    pickup_in_seconds: int = Field(..., ge=0)
    driver_rating: confloat(ge=1.0, le=5.0) = Field(..., description="Rating on a 1.0-5.0 scale.")
    platform: PlatformEnum
    price_start_local: confloat(ge=0)


class PriceRange(BaseModel):
    min: float = Field(..., ge=0)
    max: float = Field(..., ge=0)


class ZoneMetrics(BaseModel):
    avg_probability_percent: float = Field(..., ge=0, le=100)
    avg_normalized_probability_percent: float = Field(..., ge=0, le=100)
    avg_expected_value: float = Field(..., ge=0)


class Zone(BaseModel):
    zone_id: int = Field(..., ge=0)
    zone_name: str
    price_range: PriceRange
    metrics: ZoneMetrics


class OptimalPrice(BaseModel):
    price: float = Field(..., ge=0)
    probability_percent: float = Field(..., ge=0, le=100)
    normalized_probability_percent: float = Field(..., ge=0, le=100)
    expected_value: float = Field(..., ge=0)
    zone_id: int = Field(..., ge=0)
    zone: Optional[str] = None
    score: Optional[int] = None
    zone_name: Optional[str] = None


class ScanRange(BaseModel):
    min: float = Field(..., ge=0)
    max: float = Field(..., ge=0)


class ModelAnalysis(BaseModel):
    start_price: float = Field(..., ge=0)
    max_probability_percent: float = Field(..., ge=0, le=100)
    max_probability_price: float = Field(..., ge=0)
    scan_range: ScanRange
    timestamp: str
    price_increment: Optional[float] = Field(None, ge=0)


class PriceProbability(BaseModel):
    prob: float = Field(..., ge=0)
    ev: float = Field(..., ge=0)
    norm: float = Field(..., ge=0, le=100)
    zone: str


class Recommendation(BaseModel):
    score: int = Field(..., ge=0)
    zone: str
    price_range: PriceRange
    avg_probability_percent: float = Field(..., ge=0, le=100)
    normalized_probability_percent: float = Field(..., ge=0, le=100)
    avg_expected_value: float = Field(..., ge=0)


class ModelResponse(BaseModel):
    zones: List[Zone]
    optimal_price: OptimalPrice
    analysis: ModelAnalysis
    price_probabilities: Dict[str, PriceProbability]
    recommendations: List[Recommendation]


class User(BaseModel):
    email: str


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    sub: Optional[str] = None
