from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import List


def _parse_origins(raw: str) -> List[str]:
    if not raw or raw.strip() in {"", "*"}:
        return ["*"]
    items = [chunk.strip() for chunk in raw.split(",")]
    return [item for item in items if item] or ["*"]


def _env_bool(key: str, default: bool) -> bool:
    value = os.getenv(key)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    secret_key: str = os.getenv("SECRET_KEY", "super-secret-key")
    access_token_expire_minutes: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
    test_user_email: str = os.getenv("TEST_USER_EMAIL", "demo@example.com")
    test_user_password: str = os.getenv("TEST_USER_PASSWORD", "demo")
    cors_allow_origins: List[str] = field(
        default_factory=lambda: _parse_origins(os.getenv("BACKEND_ALLOW_ORIGINS", "*"))
    )
    webui_backend_base: str = os.getenv("WEBUI_BACKEND_BASE", "")
    webui_token_path: str = os.getenv("WEBUI_TOKEN_PATH", "/auth/token")
    webui_pricing_path: str = os.getenv(
        "WEBUI_PRICING_PATH", "/api/v1/orders/price-recommendation"
    )
    webui_username: str = os.getenv("WEBUI_USERNAME", "demo@example.com")
    webui_password: str = os.getenv("WEBUI_PASSWORD", "demo")
    webui_include_credentials: bool = _env_bool("WEBUI_INCLUDE_CREDENTIALS", False)
    ml_module_path: str = os.getenv("PRICING_ML_MODULE", "src.recommend_price").strip()
    ml_callable_name: str = os.getenv("PRICING_ML_CALLABLE", "recommend_price").strip()
    ml_allow_stub_fallback: bool = _env_bool("PRICING_ML_ALLOW_STUB_FALLBACK", False)


settings = Settings()
