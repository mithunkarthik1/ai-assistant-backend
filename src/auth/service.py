"""Async, class-based authentication business logic."""

from datetime import datetime, timezone
import logging

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.auth.model import AccessTokenBlacklist, RefreshToken, User
from src.utils.helper import (
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_password,
)

logger = logging.getLogger(__name__)


class AuthService:
    """Handles authentication and token persistence for one database session."""

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def create_user(
        self,
        *,
        email_address: str,
        password: str,
        name: str | None = None,
        phone_number: str | None = None,
    ) -> User | None:
        """Create a standard user, returning ``None`` when the email exists."""
        normalized_email = email_address.strip().lower()
        result = await self.db.execute(
            select(User).where(User.email_address == normalized_email, User.phone_number == phone_number, User.is_active.is_(True))
        )
        if result.scalar_one_or_none() is not None:
            return None

        try:
            user = User(
                name=name.strip() if name else None,
                email_address=normalized_email,
                password_hash=hash_password(password),
                role="user",
                phone_number=phone_number.strip() if phone_number else None,
                created_by=normalized_email,
                updated_by=normalized_email,
                is_active=True,
            )
            self.db.add(user)
            await self.db.commit()
            await self.db.refresh(user)
            logger.info("User created: %s", normalized_email)
            return user
        except Exception:
            await self.db.rollback()
            logger.exception("Error creating user: %s", normalized_email)
            raise

    async def authenticate_user(self, email_address: str, password: str) -> User | None:
        try:
            result = await self.db.execute(select(User).where(User.email_address == email_address, User.is_active.is_(True)))
            user = result.scalar_one_or_none()
            if not user:
                logger.warning("Authentication failed: user not found (%s)", email_address)
                return None
            if not user.is_active:
                logger.warning("Authentication failed: inactive user (%s)", email_address)
                return None
            if not verify_password(password, user.password_hash):
                logger.warning("Authentication failed: invalid password (%s)", email_address)
                return None
            logger.info("User authenticated: %s", email_address)
            return user
        except Exception:
            logger.exception("Error authenticating user: %s", email_address)
            raise

    async def create_tokens(self, user: User) -> tuple[str, str]:
        try:
            access_token = create_access_token(
                user_id=str(user.user_id), email=user.email_address, role=user.role
            )
            refresh_token = create_refresh_token(
                user_id=str(user.user_id), email=user.email_address, role=user.role
            )
            access_payload = decode_token(access_token)
            refresh_payload = decode_token(refresh_token)
            self.db.add(
                RefreshToken(
                    user_id=user.user_id,
                    jti=refresh_payload["jti"],
                    token=refresh_token,
                    access_token_jti=access_payload.get("jti"),
                    expires_at=datetime.fromtimestamp(refresh_payload["exp"], tz=timezone.utc),
                    is_revoked=False,
                    created_by=user.email_address,
                    updated_by=user.email_address,
                )
            )
            await self.db.commit()
            logger.info("Tokens created for user: %s", user.email_address)
            return access_token, refresh_token
        except Exception:
            await self.db.rollback()
            logger.exception("Error creating tokens for user: %s", user.email_address)
            raise

    async def refresh_access_token(self, refresh_token: str) -> str | None:
        try:
            payload = decode_token(refresh_token)
        except Exception:
            logger.warning("Refresh failed: invalid token")
            return None
        if payload.get("type") != "refresh" or not (jti := payload.get("jti")):
            logger.warning("Refresh failed: invalid refresh-token claims")
            return None

        result = await self.db.execute(
            select(RefreshToken).where(
                RefreshToken.jti == jti, RefreshToken.is_revoked.is_(False)
            )
        )
        token_record = result.scalar_one_or_none()
        if not token_record:
            logger.warning("Refresh failed: token not found or revoked (jti=%s)", jti)
            return None

        user_id, email, role = payload.get("sub"), payload.get("email"), payload.get("role")
        if not user_id or not email or not role:
            logger.warning("Refresh failed: missing claims in token")
            return None
        try:
            await self._blacklist_access_token(token_record.access_token_jti, token_record.expires_at, email)
            new_access_token = create_access_token(user_id=user_id, email=email, role=role)
            token_record.access_token_jti = decode_token(new_access_token).get("jti")
            token_record.updated_at = datetime.now(timezone.utc)
            token_record.updated_by = email
            await self.db.commit()
        except Exception:
            await self.db.rollback()
            logger.exception("Error refreshing access token for user: %s", email)
            raise
        logger.info("Access token refreshed for user: %s", email)
        return new_access_token

    async def revoke_refresh_token(self, refresh_token: str) -> bool:
        try:
            payload = decode_token(refresh_token)
        except Exception:
            logger.warning("Revoke failed: invalid token")
            return False
        if payload.get("type") != "refresh" or not (jti := payload.get("jti")):
            logger.warning("Revoke failed: invalid refresh-token claims")
            return False

        result = await self.db.execute(select(RefreshToken).where(RefreshToken.jti == jti))
        token_record = result.scalar_one_or_none()
        if not token_record or token_record.is_revoked:
            logger.warning("Revoke failed: token not found or already revoked (jti=%s)", jti)
            return False

        email = payload.get("email")
        try:
            token_record.is_revoked = True
            token_record.revoked_at = datetime.now(timezone.utc)
            token_record.updated_at = datetime.now(timezone.utc)
            token_record.updated_by = email
            await self._blacklist_access_token(token_record.access_token_jti, token_record.expires_at, email)
            await self.db.commit()
        except Exception:
            await self.db.rollback()
            logger.exception("Error revoking refresh token (jti=%s)", jti)
            raise
        logger.info("Refresh token revoked (jti=%s)", jti)
        return True

    async def is_token_blacklisted(self, jti: str) -> bool:
        result = await self.db.execute(
            select(AccessTokenBlacklist.blacklist_id).where(AccessTokenBlacklist.jti == jti)
        )
        return result.scalar_one_or_none() is not None

    async def _blacklist_access_token(
        self, access_token_jti: str | None, expires_at: datetime, email: str | None
    ) -> None:
        if not access_token_jti:
            return
        result = await self.db.execute(
            select(AccessTokenBlacklist.blacklist_id).where(
                AccessTokenBlacklist.jti == access_token_jti
            )
        )
        if result.scalar_one_or_none() is None:
            self.db.add(
                AccessTokenBlacklist(
                    jti=access_token_jti,
                    token="",
                    expires_at=expires_at,
                    created_by=email,
                )
            )
