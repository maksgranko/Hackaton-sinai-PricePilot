from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from typing import Any, Dict

from . import schemas

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


async def call_ml_model_stub(order: schemas.OrderRequest) -> schemas.ModelResponse:
    """
    Fake async call to ML model endpoint.
    Replace with real HTTP call or RPC integration once the model is ready.
    """
    # Update timestamp to current time to show request context.
    response_copy = deepcopy(DUMMY_RESPONSE)
    analysis = response_copy["analysis"]
    analysis["timestamp"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    analysis["start_price"] = order.price_start_local
    analysis["scan_range"]["min"] = min(
        order.price_start_local,
        analysis["scan_range"]["max"],
    )

    return schemas.ModelResponse(**response_copy)
