import functools
from typing import Any, Callable
from fastapi import Depends, HTTPException, Query, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from src.database import get_db
from src.auth.service import AuthService
from src.utils.helper import decode_token


security = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
    token_query: str | None = Query(None, alias="token"),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """
    Centralized authentication dependency.
    Extracts and validates JWT Bearer token from header or optional query parameter.
    """
    token = credentials.credentials if credentials else token_query

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication token required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        payload = decode_token(token)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if payload.get("type") != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Access token required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    jti = payload.get("jti")
    if jti and await AuthService(db).is_token_blacklisted(jti):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has been revoked",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return payload


async def get_optional_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
    token_query: str | None = Query(None, alias="token"),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any] | None:
    """Optional authentication dependency that returns None if token is absent."""
    token = credentials.credentials if credentials else token_query
    if not token:
        return None

    try:
        payload = decode_token(token)
        if payload.get("type") != "access":
            return None
        jti = payload.get("jti")
        if jti and await AuthService(db).is_token_blacklisted(jti):
            return None
        return payload
    except Exception:
        return None


def require_role(*allowed_roles: str):
    """Dependency factory that checks role permissions."""
    async def dependency(current_user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
        user_role = current_user.get("role")

        if user_role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Required role: {', '.join(allowed_roles)}",
            )

        return current_user

    return dependency


def require_auth(*allowed_roles: str) -> Callable:
    """
    Centralized decorator for endpoint functions.
    Can be used as a Python decorator:
        @require_auth("user", "admin")
        async def my_endpoint(...):
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # When called inside FastAPI, if current_user is in kwargs, validate role
            current_user = kwargs.get("current_user")
            if current_user and allowed_roles:
                user_role = current_user.get("role")
                if user_role not in allowed_roles:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"Access denied. Required role: {', '.join(allowed_roles)}",
                    )
            return await func(*args, **kwargs)
        return wrapper
    return decorator
