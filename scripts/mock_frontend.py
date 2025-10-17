from __future__ import annotations

import json
import os
from datetime import datetime

import httpx

API_BASE = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
API_URL = f"{API_BASE}/api/v1/orders/price-recommendation"
TOKEN_URL = f"{API_BASE}/auth/token"
USERNAME = os.getenv("API_USERNAME", "demo@example.com")
PASSWORD = os.getenv("API_PASSWORD", "demo")


def build_order_payload() -> dict:
    return {
        "order_timestamp": int(datetime.now().timestamp()),
        "distance_in_meters": 3404,
        "duration_in_seconds": 486,
        "pickup_in_meters": 790,
        "pickup_in_seconds": 169,
        "driver_rating": 5,
        "platform": "android",
        "price_start_local": 180.0,
    }


def obtain_access_token(client: httpx.Client) -> str:
    response = client.post(
        TOKEN_URL,
        data={"username": USERNAME, "password": PASSWORD},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    response.raise_for_status()
    token_payload = response.json()
    return token_payload["access_token"]


def main() -> None:
    payload = build_order_payload()
    with httpx.Client(timeout=10.0) as client:
        token = obtain_access_token(client)
        response = client.post(
            API_URL,
            json=payload,
            headers={"Authorization": f"Bearer {token}"},
        )
        response.raise_for_status()
        print(json.dumps(response.json(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
