from __future__ import annotations

from datetime import datetime, timedelta
from hashlib import sha256
from typing import Dict, Optional

import jwt
from jwt import PyJWTError
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from . import schemas
from .config import settings

ALGORITHM = "HS256"
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")


def _normalize_identifier(identifier: str) -> str:
    return identifier.strip().lower()


def hash_password(password: str) -> str:
    return sha256(password.encode("utf-8")).hexdigest()


def _load_fake_user_store() -> Dict[str, Dict[str, str]]:
    email = _normalize_identifier(settings.test_user_email)
    return {
        email: {
            "email": settings.test_user_email,
            "hashed_password": hash_password(settings.test_user_password),
        }
    }


_USER_STORE = _load_fake_user_store()


def authenticate_user(username: str, password: str) -> Optional[schemas.User]:
    normalized = _normalize_identifier(username)
    record = _USER_STORE.get(normalized)
    if not record:
        return None
    if record["hashed_password"] != hash_password(password):
        return None
    return schemas.User(email=record["email"])


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (
        expires_delta
        if expires_delta is not None
        else timedelta(minutes=settings.access_token_expire_minutes)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.secret_key, algorithm=ALGORITHM)


def get_current_user(token: str = Depends(oauth2_scheme)) -> schemas.User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[ALGORITHM])
        subject: Optional[str] = payload.get("sub")
        if subject is None:
            raise credentials_exception
        normalized = _normalize_identifier(subject)
    except PyJWTError as exc:
        raise credentials_exception from exc

    record = _USER_STORE.get(normalized)
    if not record:
        raise credentials_exception
    return schemas.User(email=record["email"])
