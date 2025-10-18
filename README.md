# FastAPI Pricing Backend Skeleton

Backend skeleton that accepts ride order parameters from the frontend, forwards them to an (as yet unimplemented) ML pricing model, and returns the calculated price recommendations. The ML integration is stubbed out so the service can be wired up and iterated on before the model goes live.

## Prerequisites

- Python 3.10+
- `pip` (or another dependency manager)

## Getting Started

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

The API will be available on `http://127.0.0.1:8000`. You can use the interactive docs at `/docs` or `/redoc`.

## Authentication

All application endpoints are protected with bearer tokens issued via `/auth/token`. A demo account is provided for local development and can be overridden with environment variables (`TEST_USER_EMAIL`, `TEST_USER_PASSWORD`).

```bash
curl -X POST "http://127.0.0.1:8000/auth/token" \
     -H "Content-Type: application/x-www-form-urlencoded" \
     -d "username=demo@example.com&password=demo"
```

The response contains an `access_token`. Send it as `Authorization: Bearer <token>` when calling the pricing endpoint.

## Web UI

The driver-facing screen is bundled with the backend and served from the root path once the server is running. Open `http://127.0.0.1:8000` to access the interface; static assets live under `/assets/*`.

You can tweak behaviour with environment variables before starting the app:

- `WEBUI_BACKEND_BASE` – override the base URL used by the browser (default: same origin)
- `WEBUI_TOKEN_PATH` / `WEBUI_PRICING_PATH` – override endpoint paths if they change
- `WEBUI_USERNAME` / `WEBUI_PASSWORD` – demo credentials for the automatic login
- `WEBUI_INCLUDE_CREDENTIALS` – set to `true` when the API relies on cookie-based auth

If you expose the UI from a different origin, remember to allow it via `BACKEND_ALLOW_ORIGINS`, e.g. `BACKEND_ALLOW_ORIGINS="http://127.0.0.1:3000"`.

Press the bug icon in the header to open the built-in debugger: set `carname`, inject raw JSON overrides, toggle lottery-mode client simulation, or refresh the JWT token without reloading the page.

## ML Integration

By default the backend returns a mocked response. To plug in the bundled Python model (`src/recommend_price.py`) or your own implementation:

- `PRICING_ML_MODULE` – dotted path to the module containing the entrypoint (e.g. `src.recommend_price`)
- `PRICING_ML_CALLABLE` – callable inside that module (sync or async) that accepts `OrderRequest`/dict and returns a payload compatible with `ModelResponse` (default for bundled module: `predict`)
- `PRICING_MODEL_PATH` – override path to the serialized model artifact (defaults to `model_enhanced.joblib`)
- `PRICING_ML_ALLOW_STUB_FALLBACK` – set to `false` to disable fallback to the dummy payload when import/execution fails
- `PRICING_SCAN_POINTS` – number of price points to scan when building the curve (defaults to 200)

Example configuration for the bundled pipeline:

```bash
export PRICING_ML_MODULE=src.recommend_price
export PRICING_ML_CALLABLE=predict
export PRICING_MODEL_PATH=model_enhanced.joblib
```

When both `PRICING_ML_MODULE` and `PRICING_ML_CALLABLE` are present the app will load and cache that callable; otherwise the stub remains active.

## API

- `POST /api/v1/orders/price-recommendation`  
  Request body:

  ```json
  {
    "order_timestamp": 1718558240,
    "distance_in_meters": 3404,
    "duration_in_seconds": 486,
    "pickup_in_meters": 790,
    "pickup_in_seconds": 169,
    "driver_rating": 5,
    "platform": "android",
    "price_start_local": 180
  }
  ```

  Response mirrors the structure expected from the ML team. Currently the data is mocked with a static payload and dynamic timestamp/start price updates.

## Mock Frontend Script

Use the helper script to simulate the frontend call once the server is running:

```bash
python scripts/mock_frontend.py
```

The script requests a JWT using the demo credentials (override with `API_USERNAME` / `API_PASSWORD`) and prints the JSON response.

## Replacing the Stub

- Update `app/services.py` to call the real ML model (HTTP, RPC, etc.).
- Adjust response parsing in `call_pricing_model` if the ML contract changes.
