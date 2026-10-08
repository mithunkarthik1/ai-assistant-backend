from datetime import datetime, timedelta, timezone
import re
import uuid
import jwt
from pwdlib import PasswordHash

# from src.core.config import (
#     JWT_ACCESS_TOKEN_EXPIRE_MINUTES,
#     JWT_ALGORITHM,
#     JWT_REFRESH_TOKEN_EXPIRE_DAYS,
#     JWT_SECRET_KEY,
# )

from src.core.config import get_settings


_EMAIL_PATTERN = re.compile(
    r"^(?=.{1,254}$)(?=.{1,64}@)[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@"
    r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?(?:\."
    r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)+$"
)
_MOBILE_NUMBER_PATTERN = re.compile(r"^\+?[1-9]\d{7,14}$")


def validate_email(email_address: str) -> str:
    """Validate and normalize an email address for account creation."""
    normalized_email = email_address.strip().lower()
    if not _EMAIL_PATTERN.fullmatch(normalized_email):
        raise ValueError("Enter a valid email address")
    return normalized_email


def validate_mobile_number(phone_number: str) -> str:
    """Validate an international mobile number (8–15 digits, optional + prefix)."""
    normalized_number = re.sub(r"[\s()-]", "", phone_number)
    if not _MOBILE_NUMBER_PATTERN.fullmatch(normalized_number):
        raise ValueError(
            "Enter a valid mobile number with 8 to 15 digits and an optional + prefix"
        )
    return normalized_number


def validate_password(password: str) -> str:
    """Enforce a password suitable for user account authentication."""
    if not 8 <= len(password) <= 128:
        raise ValueError("Password must be between 8 and 128 characters")
    if password != password.strip():
        raise ValueError("Password must not begin or end with whitespace")
    if not re.search(r"[a-z]", password):
        raise ValueError("Password must include a lowercase letter")
    if not re.search(r"[A-Z]", password):
        raise ValueError("Password must include an uppercase letter")
    if not re.search(r"\d", password):
        raise ValueError("Password must include a number")
    if not re.search(r"[^A-Za-z0-9\s]", password):
        raise ValueError("Password must include a special character")
    return password

def create_access_token(user_id: str, email: str, role: str) -> str:

    now = datetime.now(timezone.utc)

    payload = {
        "sub": user_id,
        "email": email,
        "role": role,
        "type": "access",
        "jti": str(uuid.uuid4()),
        "iat": now,
        "exp": now + timedelta(minutes = get_settings().jwt_access_token_expire_minutes),
    }

    return jwt.encode(payload, get_settings().jwt_secret_key, algorithm = get_settings().jwt_algorithm)

def create_refresh_token(user_id: str, email: str, role: str) -> str:

    now = datetime.now(timezone.utc)
    payload = {
        "sub": user_id,
        "email": email,
        "role": role,
        "type": "refresh",
        "jti": str(uuid.uuid4()),
        "iat": now,
        "exp": now + timedelta(days = get_settings().jwt_refresh_token_expire_days)
    }

    return jwt.encode(payload, get_settings().jwt_secret_key, algorithm = get_settings().jwt_algorithm)

def decode_token(token: str) -> dict:
    return jwt.decode(token, get_settings().jwt_secret_key, algorithms = [get_settings().jwt_algorithm])



password_hash = PasswordHash.recommended()

def hash_password(password: str) -> str:
    return password_hash.hash(password)


def verify_password(password: str, hashed_password: str) -> bool:
    return password_hash.verify(password, hashed_password)
