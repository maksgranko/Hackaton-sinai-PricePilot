from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import importlib
import inspect
import logging
from functools import lru_cache
from typing import Any, Awaitable, Callable, Dict, Optional, Union

from . import schemas
from .config import settings

logger = logging.getLogger(__name__)

ModelPayload = Union[schemas.ModelResponse, Dict[str, Any]]
ModelCallable = Callable[[schemas.OrderRequest], Union[ModelPayload, Awaitable[ModelPayload]]]

DUMMY_RESPONSE: Dict[str, Any] = {
    "price_probabilities": {
        "180": {"prob": 30.0, "ev": 54.0, "norm": 53.43, "zone": "red"},
        "200": {"prob": 35.0, "ev": 70.0, "norm": 62.33, "zone": "yellow"},
        "250": {"prob": 40.0, "ev": 100.0, "norm": 71.24, "zone": "yellow"},
        "300": {"prob": 45.0, "ev": 135.0, "norm": 80.14, "zone": "green"},
        "350": {"prob": 48.0, "ev": 168.0, "norm": 85.5, "zone": "green"},
        "378": {"prob": 42.72, "ev": 161.64, "norm": 76.08, "zone": "yellow"},
        "400": {"prob": 38.0, "ev": 152.0, "norm": 67.68, "zone": "yellow"},
        "450": {"prob": 25.0, "ev": 112.5, "norm": 44.52, "zone": "red"},
    },
    "recommendations": [
        {
            "score": 4,
            "zone": "green",
            "price_range": {"min": 300.0, "max": 380.0},
            "avg_probability_percent": 45.36,
            "normalized_probability_percent": 80.8,
            "avg_expected_value": 154.88,
        },
        {
            "score": 3,
            "zone": "yellow",
            "price_range": {"min": 200.0, "max": 295.0},
            "avg_probability_percent": 38.33,
            "normalized_probability_percent": 68.27,
            "avg_expected_value": 90.95,
        },
        {
            "score": 2,
            "zone": "red",
            "price_range": {"min": 180.0, "max": 195.0},
            "avg_probability_percent": 32.5,
            "normalized_probability_percent": 57.88,
            "avg_expected_value": 60.15,
        },
    ],
    "zones": [
        {
            "zone_id": 1,
            "zone_name": "zone_1_red_low",
            "price_range": {"min": 54.0, "max": 225.14},
            "metrics": {
                "avg_probability_percent": 41.15,
                "avg_normalized_probability_percent": 73.72,
                "avg_expected_value": 55.96,
            },
        },
        {
            "zone_id": 2,
            "zone_name": "zone_2_yellow_low",
            "price_range": {"min": 227.13, "max": 320.65},
            "metrics": {
                "avg_probability_percent": 44.59,
                "avg_normalized_probability_percent": 79.89,
                "avg_expected_value": 122.15,
            },
        },
        {
            "zone_id": 3,
            "zone_name": "zone_3_green",
            "price_range": {"min": 322.64, "max": 434.08},
            "metrics": {
                "avg_probability_percent": 41.25,
                "avg_normalized_probability_percent": 73.9,
                "avg_expected_value": 155.02,
            },
        },
        {
            "zone_id": 4,
            "zone_name": "zone_4_yellow_high",
            "price_range": {"min": 436.07, "max": 450.0},
            "metrics": {
                "avg_probability_percent": 34.06,
                "avg_normalized_probability_percent": 61.02,
                "avg_expected_value": 150.89,
            },
        },
    ],
    "optimal_price": {
        "price": 378.36,
        "probability_percent": 45.3,
        "normalized_probability_percent": 81.16,
        "expected_value": 171.4,
        "zone_id": 3,
        "zone": "yellow",
        "score": 3,
        "zone_name": "zone_3_green",
    },
    "analysis": {
        "start_price": 180.0,
        "max_probability_percent": 55.82,
        "max_probability_price": 54.0,
        "scan_range": {"min": 180.0, "max": 450.0},
        "price_increment": 5.0,
        "timestamp": "2025-10-17 16:19:26",
    },
}

@lru_cache(maxsize=1)
def _load_ml_callable() -> Optional[ModelCallable]:
    module_path = settings.ml_module_path
    callable_name = settings.ml_callable_name

    if not module_path or not callable_name:
        return None

    module = importlib.import_module(module_path)
    handler = getattr(module, callable_name, None)
    if handler is None:
        raise AttributeError(f"Callable '{callable_name}' not found in module '{module_path}'")
    if not callable(handler):
        raise TypeError(f"Attribute '{callable_name}' in '{module_path}' is not callable")

    logger.info("Loaded ML handler %s.%s", module_path, callable_name)
    return handler  # type: ignore[return-value]


def _build_stub_response(order: schemas.OrderRequest) -> schemas.ModelResponse:
    response_copy = deepcopy(DUMMY_RESPONSE)
    analysis = response_copy["analysis"]
    analysis["timestamp"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    analysis["start_price"] = order.price_start_local
    analysis["scan_range"]["min"] = min(
        order.price_start_local,
        analysis["scan_range"]["max"],
    )
    return schemas.ModelResponse(**response_copy)


def _coerce_model_response(result: ModelPayload) -> schemas.ModelResponse:
    if isinstance(result, schemas.ModelResponse):
        return result
    if isinstance(result, dict):
        return schemas.ModelResponse(**result)
    raise TypeError(
        "ML handler must return schemas.ModelResponse or dict compatible payload; "
        f"received {type(result)!r}"
    )


async def call_pricing_model(order: schemas.OrderRequest) -> schemas.ModelResponse:
    """
    Call real ML module if configured, otherwise return stub response.
    """
    try:
        handler = _load_ml_callable()
    except Exception as exc:
        if settings.ml_allow_stub_fallback:
            logger.warning("Failed to load ML handler, falling back to stub: %s", exc)
            return _build_stub_response(order)
        raise

    if handler is None:
        return _build_stub_response(order)

    try:
        result = handler(order)
        if inspect.isawaitable(result):
            result = await result  # type: ignore[assignment]
        return _coerce_model_response(result)
    except Exception as exc:
        if settings.ml_allow_stub_fallback:
            logger.error("ML handler failed, falling back to stub: %s", exc, exc_info=True)
            return _build_stub_response(order)
        raise
