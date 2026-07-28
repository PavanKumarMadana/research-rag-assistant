"""
Security Module.

Provides authentication, authorization, and API key validation utilities.
"""

import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from loguru import logger

from backend.app.core.config import settings

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Security schemes
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
bearer_scheme = HTTPBearer(auto_error=False)

# In-memory API keys store (in production, use database)
_api_keys: dict[str, dict] = {}


def generate_api_key() -> str:
    """Generate a unique API key.

    Returns:
        str: A new UUID-based API key.
    """
    return f"ra_{uuid.uuid4().hex}"


def create_api_key(owner: str = "default") -> str:
    """Create and store a new API key.

    Args:
        owner: Owner identifier for the key.

    Returns:
        str: The generated API key.
    """
    api_key = generate_api_key()
    _api_keys[api_key] = {
        "owner": owner,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "is_active": True,
    }
    logger.info(f"API key created for owner: {owner}")
    return api_key


def validate_api_key(api_key: str) -> bool:
    """Validate an API key.

    Args:
        api_key: The API key to validate.

    Returns:
        bool: True if valid, False otherwise.
    """
    key_data = _api_keys.get(api_key)
    if not key_data:
        return False
    return key_data.get("is_active", False)


def revoke_api_key(api_key: str) -> bool:
    """Revoke an API key.

    Args:
        api_key: The API key to revoke.

    Returns:
        bool: True if revoked, False if not found.
    """
    if api_key in _api_keys:
        _api_keys[api_key]["is_active"] = False
        logger.info(f"API key revoked: {api_key[:8]}...")
        return True
    return False


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain password against a hashed password.

    Args:
        plain_password: The plain text password.
        hashed_password: The hashed password.

    Returns:
        bool: True if passwords match.
    """
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password.

    Args:
        password: The plain text password.

    Returns:
        str: The hashed password.
    """
    return pwd_context.hash(password)


def create_access_token(
    data: dict,
    expires_delta: Optional[timedelta] = None,
) -> str:
    """Create a JWT access token.

    Args:
        data: Claims to encode in the token.
        expires_delta: Token expiration time.

    Returns:
        str: Encoded JWT token.
    """
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(hours=24)
    )
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(
        to_encode,
        settings.GEMINI_API_KEY or "secret-key-change-in-production",
        algorithm="HS256",
    )
    return encoded_jwt


def decode_access_token(token: str) -> Optional[dict]:
    """Decode a JWT access token.

    Args:
        token: The JWT token to decode.

    Returns:
        Optional[dict]: Decoded claims if valid, None otherwise.
    """
    try:
        payload = jwt.decode(
            token,
            settings.GEMINI_API_KEY or "secret-key-change-in-production",
            algorithms=["HS256"],
        )
        return payload
    except JWTError as e:
        logger.warning(f"JWT decode error: {e}")
        return None


async def verify_api_key_dependency(
    api_key: Optional[str] = Security(api_key_header),
    credentials: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme),
) -> str:
    """Dependency to verify API key or Bearer token.

    Args:
        api_key: API key from header.
        credentials: Bearer token from header.

    Returns:
        str: The validated API key or token.

    Raises:
        HTTPException: If authentication fails.
    """
    # Check API key first
    if api_key and validate_api_key(api_key):
        return api_key

    # Check Bearer token
    if credentials:
        token = credentials.credentials
        payload = decode_access_token(token)
        if payload:
            return token

    # In development mode, allow unauthenticated access
    if settings.ENVIRONMENT == "development":
        return "dev-mode"

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing authentication credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )