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
    
    # Optional fields that ML model can use
    carname: Optional[str] = None
    carmodel: Optional[str] = None
    driver_reg_date: Optional[str] = None


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
    net_profit: float = Field(..., description="Чистая прибыль после вычета топлива")


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


class ZoneThresholds(BaseModel):
    """Пороги вероятности для определения зон"""
    green_zone: str
    yellow_low_zone: str
    yellow_high_zone: str
    red_zone: str


class FuelEconomics(BaseModel):
    """Экономика топлива для поездки"""
    fuel_cost: float = Field(..., ge=0, description="Стоимость топлива в рублях")
    fuel_liters: float = Field(..., ge=0, description="Расход топлива в литрах")
    distance_km: float = Field(..., ge=0, description="Расстояние в километрах")
    fuel_price_per_liter: float = Field(..., ge=0, description="Цена за литр топлива")
    consumption_per_100km: float = Field(..., ge=0, description="Расход на 100 км")
    min_profitable_price: float = Field(..., ge=0, description="Минимальная рентабельная цена")
    net_profit_from_optimal: float = Field(..., description="Чистая прибыль от оптимальной цены")


class ModelResponse(BaseModel):
    zones: List[Zone]
    optimal_price: OptimalPrice
    zone_thresholds: Optional[ZoneThresholds] = None
    fuel_economics: FuelEconomics
    analysis: ModelAnalysis


class User(BaseModel):
    email: str


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    sub: Optional[str] = None
