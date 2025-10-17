from __future__ import annotations

import json
import os
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Tuple, Union

import joblib
import numpy as np
import pandas as pd

DEFAULT_MODEL_PATH = Path(os.getenv("PRICING_MODEL_PATH", "model_enhanced.joblib"))
DEFAULT_PRICE_STEPS = max(int(os.getenv("PRICING_SCAN_POINTS", "200")), 20)

OrderLike = Union[Mapping[str, Any], "schemas.OrderRequest"]  # type: ignore[name-defined]
ZoneConfig = Dict[str, Any]


def _ensure_series(
    value: Union[pd.Series, Any], index: pd.Index, default: float = 0.0
) -> pd.Series:
    if isinstance(value, pd.Series):
        return value
    if isinstance(value, (list, tuple, np.ndarray)):
        return pd.Series(value, index=index)
    if np.isscalar(value):
        return pd.Series(value, index=index)
    return pd.Series(default, index=index)


def build_features_with_order(
    order_dict: Mapping[str, Any], feat_cols: Iterable[str]
) -> pd.DataFrame:
    """Строит признаки для одного заказа"""
    frame = pd.DataFrame([order_dict])

    if isinstance(frame["order_timestamp"].iloc[0], (int, float)):
        frame["order_timestamp"] = pd.to_datetime(
            frame["order_timestamp"], unit="s", errors="coerce"
        )
    else:
        frame["order_timestamp"] = pd.to_datetime(
            frame["order_timestamp"], errors="coerce"
        )

    ts = frame["order_timestamp"]
    hour = ts.dt.hour.fillna(0)
    wday = ts.dt.weekday.fillna(0)

    is_morning_rush = ((hour >= 7) & (hour <= 9)).astype(int)
    is_evening_rush = ((hour >= 15) & (hour <= 17)).astype(int)
    is_night_rush = ((hour >= 19) & (hour <= 21)).astype(int)
    is_peak_hour = ((is_morning_rush + is_evening_rush + is_night_rush) > 0).astype(
        float
    )
    is_weekend = (wday >= 5).astype(float)
    is_night = ((hour < 6) | (hour >= 22)).astype(float)

    dist_km = frame["distance_in_meters"] / 1000.0
    dur_min = frame["duration_in_seconds"] / 60.0
    pickup_km = frame["pickup_in_meters"] / 1000.0
    pickup_min = frame["pickup_in_seconds"] / 60.0
    log_start = np.log1p(frame["price_start_local"])
    price_per_km = frame["price_start_local"] / (dist_km + 0.1)

    if "driver_reg_date" in frame.columns and pd.notna(frame["driver_reg_date"].iloc[0]):
        driver_reg = pd.to_datetime(frame["driver_reg_date"], errors="coerce")
        days_since_reg = (ts - driver_reg).dt.days.fillna(180)
        driver_experience_months = days_since_reg / 30.0
        is_new_driver = (days_since_reg < 30).astype(float)
    else:
        driver_experience_months = pd.Series(6.0, index=frame.index)
        is_new_driver = pd.Series(0.0, index=frame.index)

    premium_brands = {"Toyota", "Volkswagen", "Hyundai", "Nissan", "Skoda"}
    car_series = _ensure_series(frame.get("carname", ""), frame.index, "")
    is_premium_car = car_series.isin(premium_brands).astype(float)

    is_frequent_user = pd.Series(0.5, index=frame.index)
    response_time_minutes = pd.Series(1.0, index=frame.index)
    log_response_time = pd.Series(0.0, index=frame.index)

    base = pd.DataFrame(
        {
            "dist_km": dist_km,
            "dur_min": dur_min,
            "pickup_km": pickup_km,
            "pickup_min": pickup_min,
            "rating": _ensure_series(frame.get("driver_rating", 4.5), frame.index, 4.5),
            "log_start": log_start,
            "hour_sin": np.sin(2 * np.pi * hour / 24.0),
            "hour_cos": np.cos(2 * np.pi * hour / 24.0),
            "wday_sin": np.sin(2 * np.pi * wday / 7.0),
            "wday_cos": np.cos(2 * np.pi * wday / 7.0),
            "is_peak_hour": is_peak_hour,
            "is_morning_rush": is_morning_rush.astype(float),
            "is_evening_rush": is_evening_rush.astype(float),
            "is_night_rush": is_night_rush.astype(float),
            "is_weekend": is_weekend,
            "is_night": is_night,
            "speed_kmh": dist_km / ((dur_min + 1) / 60.0),
            "pickup_speed": pickup_km / ((pickup_min + 1) / 60.0),
            "total_time_min": dur_min + pickup_min,
            "price_per_km": price_per_km,
            "price_per_min": frame["price_start_local"] / (dur_min + 0.1),
            "driver_experience_months": driver_experience_months,
            "is_new_driver": is_new_driver,
            "is_premium_car": is_premium_car,
            "is_frequent_user": is_frequent_user,
            "response_time_minutes": response_time_minutes,
            "log_response_time": log_response_time,
            "rush_x_rating": is_peak_hour
            * _ensure_series(frame.get("driver_rating", 4.5), frame.index, 4.5),
            "weekend_x_dist": is_weekend * dist_km,
            "peak_x_price_per_km": is_peak_hour * price_per_km,
            "hour_x_weekend": hour * is_weekend / 24.0,
            "morning_x_dist": is_morning_rush.astype(float) * dist_km,
            "evening_x_dist": is_evening_rush.astype(float) * dist_km,
            "night_x_price": is_night * log_start,
            "night_x_dist": is_night * dist_km,
            "night_x_price_per_km": is_night * price_per_km,
            "hour_x_price_per_km": hour * price_per_km / 24.0,
            "hour_x_dist": hour * dist_km / 24.0,
            "peak_x_weekend_x_dist": is_peak_hour * is_weekend * dist_km,
            "peak_x_weekend_x_price": is_peak_hour * is_weekend * log_start,
            "premium_x_price": is_premium_car * log_start,
            "experience_x_rating": driver_experience_months
            * _ensure_series(frame.get("driver_rating", 4.5), frame.index, 4.5),
            "new_driver_x_price": is_new_driver * log_start,
            "frequent_user_x_price": is_frequent_user * log_start,
        }
    ).fillna(0.0)

    base = base.replace([np.inf, -np.inf], 0)
    base = base.reindex(columns=list(feat_cols), fill_value=0.0)
    return base


@lru_cache(maxsize=1)
def _load_model(model_path: Path = DEFAULT_MODEL_PATH) -> Tuple[Any, Iterable[str]]:
    if not model_path.exists():
        raise FileNotFoundError(
            f"ML model artifact not found at '{model_path}'. "
            "Train the model or adjust PRICING_MODEL_PATH."
        )
    model_bundle = joblib.load(model_path)
    return model_bundle["model"], model_bundle["feat_cols"]


def _resolve_order_dict(order: OrderLike) -> Dict[str, Any]:
    if hasattr(order, "model_dump"):
        return dict(order.model_dump())  # type: ignore[call-arg]
    if isinstance(order, Mapping):
        return dict(order)
    raise TypeError(f"Unsupported order payload type: {type(order)!r}")


def _derive_scan_range(start_price: float, current_dt: datetime) -> Tuple[float, float]:
    hour = current_dt.hour
    wday = current_dt.weekday()

    is_peak = (7 <= hour <= 9) or (15 <= hour <= 17) or (19 <= hour <= 21)
    is_night = (hour < 6) or (hour >= 22)
    is_weekend = wday >= 5

    if is_night:
        max_coef = 2.00
    elif is_peak and not is_weekend:
        max_coef = 2.20
    elif is_weekend:
        max_coef = 1.80
    else:
        max_coef = 1.60

    min_scan = max(start_price * 0.6, 1.0)
    max_scan = max(start_price * max_coef, min_scan + 1.0)
    return min_scan, max_scan


def _build_zone_config(
    min_scan: float, max_scan: float, optimal_price: float
) -> Tuple[ZoneConfig, ...]:
    return (
        {
            "zone_id": 1,
            "zone_name": "zone_1_red_low",
            "zone_label": "red",
            "score": 1,
            "price_range": (min_scan, max(optimal_price * 0.75, min_scan)),
        },
        {
            "zone_id": 2,
            "zone_name": "zone_2_yellow_low",
            "zone_label": "yellow",
            "score": 2,
            "price_range": (
                max(optimal_price * 0.75, min_scan),
                max(optimal_price * 0.90, min_scan),
            ),
        },
        {
            "zone_id": 3,
            "zone_name": "zone_3_green",
            "zone_label": "green",
            "score": 3,
            "price_range": (
                max(optimal_price * 0.90, min_scan),
                min(optimal_price * 1.10, max_scan),
            ),
        },
        {
            "zone_id": 4,
            "zone_name": "zone_4_yellow_high",
            "zone_label": "yellow",
            "score": 2,
            "price_range": (min(optimal_price * 1.10, max_scan), max_scan),
        },
    )


def _assign_zone(price: float, zones: Tuple[ZoneConfig, ...]) -> ZoneConfig:
    for zone in zones:
        min_price, max_price = zone["price_range"]
        upper_inclusive = zone["zone_id"] == zones[-1]["zone_id"]
        if (
            min_price <= price <= max_price
            if upper_inclusive
            else min_price <= price < max_price
        ):
            return zone
    return zones[-1]


def compute_price_recommendations(order: OrderLike) -> Dict[str, Any]:
    classifier, feature_columns = _load_model()
    order_dict = _resolve_order_dict(order)
    order_copy = dict(order_dict)

    timestamp_raw = order_dict.get("order_timestamp", datetime.utcnow().timestamp())
    if isinstance(timestamp_raw, (int, float)):
        current_timestamp = int(timestamp_raw)
        current_dt = datetime.fromtimestamp(current_timestamp)
    else:
        current_dt = datetime.fromisoformat(str(timestamp_raw))
        current_timestamp = int(current_dt.timestamp())

    start_price = float(order_dict.get("price_start_local", 0.0))
    min_scan, max_scan = _derive_scan_range(start_price, current_dt)

    price_candidates = np.linspace(min_scan, max_scan, DEFAULT_PRICE_STEPS)
    price_increment = (
        float(price_candidates[1] - price_candidates[0])
        if len(price_candidates) > 1
        else 0.0
    )

    price_results = []
    for candidate_price in price_candidates:
        order_copy["price_start_local"] = float(candidate_price)
        X_candidate = build_features_with_order(order_copy, feature_columns)
        prob_done = float(classifier.predict_proba(X_candidate)[0, 1])
        expected_value = float(candidate_price * prob_done)
        price_results.append(
            {
                "price": float(candidate_price),
                "probability": prob_done,
                "expected_value": expected_value,
            }
        )

    optimal_result = max(price_results, key=lambda x: x["expected_value"])
    optimal_price = optimal_result["price"]
    optimal_prob = optimal_result["probability"]
    optimal_ev = optimal_result["expected_value"]

    max_prob_result = max(price_results, key=lambda x: x["probability"])
    max_prob = max_prob_result["probability"]
    safe_max_prob = max(max_prob, 1e-9)

    zones = _build_zone_config(min_scan, max_scan, optimal_price)

    zone_price_map = {zone["zone_id"]: [] for zone in zones}
    price_probabilities: Dict[str, Dict[str, float]] = {}

    for result in price_results:
        zone = _assign_zone(result["price"], zones)
        zone_price_map[zone["zone_id"]].append(result)
        price_probabilities[f"{result['price']:.2f}"] = {
            "prob": round(result["probability"] * 100, 4),
            "ev": round(result["expected_value"], 4),
            "norm": round((result["probability"] / safe_max_prob) * 100, 4),
            "zone": zone["zone_label"],
        }

    zones_payload = []
    recommendations = []
    optimal_zone_id = zones[-1]["zone_id"]
    optimal_zone_label = zones[-1]["zone_label"]

    for zone in zones:
        prices = zone_price_map.get(zone["zone_id"], [])
        if not prices:
            continue

        avg_prob = float(np.mean([p["probability"] for p in prices]))
        avg_ev = float(np.mean([p["expected_value"] for p in prices]))
        min_price_zone = float(min(p["price"] for p in prices))
        max_price_zone = float(max(p["price"] for p in prices))

        zone_payload = {
            "zone_id": zone["zone_id"],
            "zone_name": zone["zone_name"],
            "price_range": {
                "min": round(min_price_zone, 2),
                "max": round(max_price_zone, 2),
            },
            "metrics": {
                "avg_probability_percent": round(avg_prob * 100, 2),
                "avg_normalized_probability_percent": round(
                    (avg_prob / safe_max_prob) * 100, 2
                ),
                "avg_expected_value": round(avg_ev, 2),
            },
        }
        zones_payload.append(zone_payload)

        recommendations.append(
            {
                "zone": zone["zone_label"],
                "score": zone["score"],
                "price_range": {
                    "min": round(min_price_zone, 2),
                    "max": round(max_price_zone, 2),
                },
                "avg_probability_percent": round(avg_prob * 100, 2),
                "normalized_probability_percent": round(
                    (avg_prob / safe_max_prob) * 100, 2
                ),
                "avg_expected_value": round(avg_ev, 2),
            }
        )

        zone_min, zone_max = zone["price_range"]
        if zone_min <= optimal_price <= zone_max:
            optimal_zone_id = zone["zone_id"]
            optimal_zone_label = zone["zone_label"]

    result = {
        "zones": zones_payload,
        "optimal_price": {
            "price": round(optimal_price, 2),
            "probability_percent": round(optimal_prob * 100, 2),
            "normalized_probability_percent": round(
                (optimal_prob / safe_max_prob) * 100, 2
            ),
            "expected_value": round(optimal_ev, 2),
            "zone_id": optimal_zone_id,
            "zone": optimal_zone_label,
            "score": next(
                (z["score"] for z in zones if z["zone_id"] == optimal_zone_id), 0
            ),
            "zone_name": next(
                (z["zone_name"] for z in zones if z["zone_id"] == optimal_zone_id), ""
            ),
        },
        "analysis": {
            "start_price": round(start_price, 2),
            "max_probability_percent": round(max_prob * 100, 2),
            "max_probability_price": round(max_prob_result["price"], 2),
            "scan_range": {
                "min": round(min_scan, 2),
                "max": round(max_scan, 2),
            },
            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            "price_increment": round(price_increment, 2),
        },
        "price_probabilities": price_probabilities,
        "recommendations": sorted(
            recommendations, key=lambda r: (r["score"], r["avg_probability_percent"]), reverse=True
        ),
    }
    return result


def recommend_price(order: OrderLike, output_format: str = "json") -> Union[str, Dict[str, Any]]:
    """Рекомендует оптимальные зоны цен. Совместимо со старым тестовым скриптом."""
    payload = compute_price_recommendations(order)
    if output_format == "json":
        return json.dumps(payload, ensure_ascii=False, indent=2)
    return payload


def predict(order: OrderLike) -> Dict[str, Any]:
    """Callable для FastAPI: принимает dict/Pydantic OrderRequest, возвращает структуру ModelResponse."""
    return compute_price_recommendations(order)
