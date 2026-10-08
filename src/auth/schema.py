from typing import Generic, TypeVar
from pydantic import BaseModel, Field, field_validator

from src.utils.helper import validate_email, validate_mobile_number, validate_password

T = TypeVar("T")

class LoginRequest(BaseModel):
    email_address: str
    password: str


class SigninRequest(BaseModel):
    """Public payload for creating a standard user account."""

    name: str | None = Field(default=None, max_length=255)
    email_address: str = Field(min_length=3, max_length=255)
    password: str = Field(min_length=8, max_length=128)
    phone_number: str | None = Field(default=None, max_length=20)

    @field_validator("email_address")
    @classmethod
    def validate_email_address(cls, value: str) -> str:
        return validate_email(value)

    @field_validator("phone_number")
    @classmethod
    def validate_phone_number(cls, value: str | None) -> str | None:
        return validate_mobile_number(value) if value else None

    @field_validator("password")
    @classmethod
    def validate_user_password(cls, value: str) -> str:
        return validate_password(value)


class UserResponse(BaseModel):
    user_id: str
    name: str | None
    email_address: str
    role: str
    phone_number: str | None

class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str

class APIResponse(BaseModel, Generic[T]):
    status_code: int
    message: str
    data: T

class RefreshTokenRequest(BaseModel):
    refresh_token: str
