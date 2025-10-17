from __future__ import annotations

import json
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import OAuth2PasswordRequestForm

from . import auth, schemas, services
from .config import settings

WEBUI_DIR = Path(__file__).resolve().parent.parent / "webui"
WEBUI_STATIC_DIR = WEBUI_DIR / "static"
WEBUI_INDEX_FILE = WEBUI_DIR / "templates" / "index.html"
CONFIG_PLACEHOLDER = "<!--__WEBUI_CONFIG__-->"


def create_app() -> FastAPI:
    application = FastAPI(
        title="Pricing Recommendation API",
        version="0.1.0",
        description="Backend service that proxies frontend requests to the ML pricing model.",
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if WEBUI_STATIC_DIR.exists():
        application.mount("/assets", StaticFiles(directory=WEBUI_STATIC_DIR), name="webui-assets")

    @application.get("/health", tags=["health"])
    async def health_check() -> dict[str, str]:
        return {"status": "ok"}

    @application.post(
        "/auth/token",
        response_model=schemas.Token,
        tags=["auth"],
    )
    async def login(form_data: OAuth2PasswordRequestForm = Depends()) -> schemas.Token:
        user = auth.authenticate_user(form_data.username, form_data.password)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        access_token = auth.create_access_token({"sub": user.email})
        return schemas.Token(access_token=access_token, token_type="bearer")

    @application.post(
        "/api/v1/orders/price-recommendation",
        response_model=schemas.ModelResponse,
        status_code=status.HTTP_200_OK,
        tags=["pricing"],
    )
    async def price_recommendation(
        order: schemas.OrderRequest,
        _current_user: schemas.User = Depends(auth.get_current_user),
    ) -> schemas.ModelResponse:
        try:
            return await services.call_ml_model_stub(order)
        except Exception as exc:  # pragma: no cover - defensive until real integration
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Failed to retrieve recommendation: {exc}",
            ) from exc

    @application.get("/", include_in_schema=False, response_class=HTMLResponse)
    async def serve_frontend() -> HTMLResponse:
        if not WEBUI_INDEX_FILE.exists():
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Web UI index file not found",
            )

        raw_html = WEBUI_INDEX_FILE.read_text(encoding="utf-8")
        config_payload = {
            "backendBase": settings.webui_backend_base,
            "tokenPath": settings.webui_token_path,
            "pricingPath": settings.webui_pricing_path,
            "username": settings.webui_username,
            "password": settings.webui_password,
            "includeCredentials": settings.webui_include_credentials,
        }
        config_script = f"<script>window.__WEBUI_CONFIG__ = {json.dumps(config_payload)};</script>"
        if CONFIG_PLACEHOLDER in raw_html:
            rendered_html = raw_html.replace(CONFIG_PLACEHOLDER, config_script, 1)
        else:
            rendered_html = raw_html.replace("</head>", f"{config_script}</head>", 1)
        return HTMLResponse(rendered_html)

    return application


app = create_app()
