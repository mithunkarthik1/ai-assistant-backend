from typing import Any
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

import logging
# from src.core.config import logger
from src.database import get_db
from src.auth.dependancy import get_current_user, require_role
from src.auth.schema import (
    APIResponse,
    LoginRequest,
    SigninRequest,
    TokenResponse,
    RefreshTokenRequest,
    UserResponse,
)
from src.auth.service import AuthService

# logger = get_logger(__name__)
logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/auth",
    tags=["Authentication"],
)


@router.post(
    "/signin",
    status_code=status.HTTP_201_CREATED,
    response_model=APIResponse[UserResponse],
)
async def signin(signin_data: SigninRequest, db: AsyncSession = Depends(get_db)):
    """Create a new standard user account."""
    try:
        user = await AuthService(db).create_user(
            email_address=signin_data.email_address,
            password=signin_data.password,
            name=signin_data.name,
            phone_number=signin_data.phone_number,
        )
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="An account with this email address already exists",
            )
        return {
            "status_code": status.HTTP_201_CREATED,
            "message": "User created successfully",
            "data": {
                "user_id": str(user.user_id),
                "name": user.name,
                "email_address": user.email_address,
                "role": user.role,
                "phone_number": user.phone_number,
            },
        }
    except HTTPException:
        raise
    except Exception:
        logger.exception("Error creating user: %s", signin_data.email_address)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.get("/profile")
async def get_profile(current_user: dict[str, Any] = Depends(get_current_user)):
    try:
        return {
            "status_code": status.HTTP_200_OK,
            "message": "User information retrieved successfully",
            "data": {
                "user_id": current_user.get("sub"),
                "email": current_user.get("email"),
                "role": current_user.get("role"),
            },
        }
    except Exception as e:
        logger.exception("Error retrieving user profile")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.post("/login", response_model=APIResponse[TokenResponse])
async def login(login_data: LoginRequest, db: AsyncSession = Depends(get_db)):
    try:
        auth_service = AuthService(db)
        user = await auth_service.authenticate_user(
            email_address=login_data.email_address,
            password=login_data.password,
        )

        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email address or password",
            )

        access_token, refresh_token = await auth_service.create_tokens(user)
        logger.info("User logged in: %s", login_data.email_address)

        return {
            "status_code": status.HTTP_200_OK,
            "message": "login successfully",
            "data": {
                "access_token": access_token,
                "refresh_token": refresh_token,
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error during login for: %s", login_data.email_address)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.post("/refresh", response_model=APIResponse[dict])
async def refresh_token(refresh_data: RefreshTokenRequest, db: AsyncSession = Depends(get_db)):
    try:
        access_token = await AuthService(db).refresh_access_token(refresh_data.refresh_token)

        if not access_token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired refresh token",
            )

        logger.info("Access token refreshed")
        return {
            "status_code": status.HTTP_200_OK,
            "message": "Access token refreshed successfully",
            "data": {
                "access_token": access_token,
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error refreshing access token")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.post("/logout")
async def logout(refresh_data: RefreshTokenRequest, db: AsyncSession = Depends(get_db)):
    try:
        revoked = await AuthService(db).revoke_refresh_token(refresh_data.refresh_token)

        if not revoked:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or already revoked refresh token",
            )

        logger.info("User logged out")
        return {
            "status_code": status.HTTP_200_OK,
            "message": "Logout successfully",
            "data": None,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error during logout")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


# @router.get("/admin/profile")
# async def admin_profile(
#     current_user: dict[str, Any] = Depends(require_role("admin")),
# ):
#     try:
#         return {
#             "status_code": status.HTTP_200_OK,
#             "message": "Admin profile retrieved successfully",
#             "data": {
#                 "user_id": current_user.get("sub"),
#                 "email": current_user.get("email"),
#                 "role": current_user.get("role"),
#             },
#         }
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.exception("Error accessing admin profile")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail="Internal server error",
#         )


# @router.get("/user/profile")
# async def user_profile(
#     current_user: dict[str, Any] = Depends(require_role("user")),
# ):
#     try:
#         return {
#             "status_code": status.HTTP_200_OK,
#             "message": "User profile retrieved successfully",
#             "data": {
#                 "user_id": current_user.get("sub"),
#                 "email": current_user.get("email"),
#                 "role": current_user.get("role"),
#             },
#         }
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.exception("Error accessing user profile")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail="Internal server error",
#         )
