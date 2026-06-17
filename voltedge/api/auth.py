"""
VoltEdge API — JWT Authentication

Provides:
  - Password hashing / verification (bcrypt)
  - Access token creation and validation (HS256 JWT)
  - Role-based scope enforcement (read, write, admin)
  - FastAPI dependency: require_auth, require_scope

Usage in routes:
    from voltedge.api.auth import require_auth, require_scope, UserPrincipal

    @router.get("/protected")
    def protected(user: UserPrincipal = Depends(require_auth)):
        return {"user": user.username}

    @router.post("/admin-only")
    def admin_only(user: UserPrincipal = Depends(require_scope("admin"))):
        return {"ok": True}
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
import bcrypt
from pydantic import BaseModel

from voltedge.utils.logger import get_logger

log = get_logger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

# In production: set VOLTEDGE_SECRET_KEY in environment (min 32 chars)
SECRET_KEY: str = os.getenv(
    "VOLTEDGE_SECRET_KEY",
    "voltedge-dev-secret-change-in-production-min-32-chars",
)
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))

# Supported scopes (least-to-most privileged)
VALID_SCOPES = {"read", "write", "admin"}

# ── Password hashing ──────────────────────────────────────────────────────────


def hash_password(plain: str) -> str:
    """Hash a plain-text password with bcrypt."""
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    """Verify a plain-text password against a bcrypt hash."""
    return bcrypt.checkpw(plain.encode(), hashed.encode())


# ── Token models ──────────────────────────────────────────────────────────────

class TokenData(BaseModel):
    username: str
    scopes:   list[str] = []
    exp:      datetime | None = None


class UserPrincipal(BaseModel):
    """Resolved identity attached to a request after successful auth."""
    username: str
    scopes:   list[str]


class Token(BaseModel):
    access_token: str
    token_type:   str = "bearer"
    expires_in:   int          # seconds


# ── Token creation ────────────────────────────────────────────────────────────

def create_access_token(
    username: str,
    scopes: list[str] | None = None,
    expires_delta: timedelta | None = None,
) -> str:
    """
    Create a signed JWT access token.

    Args:
        username:      Subject (user identifier).
        scopes:        List of granted permission scopes.
        expires_delta: Token lifetime. Defaults to ACCESS_TOKEN_EXPIRE_MINUTES.

    Returns:
        Signed JWT string.
    """
    if not scopes:
        scopes = ["read"]

    invalid = set(scopes) - VALID_SCOPES
    if invalid:
        raise ValueError(f"Invalid scopes: {invalid}. Must be one of {VALID_SCOPES}")

    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    payload = {
        "sub":    username,
        "scopes": scopes,
        "exp":    expire,
        "iat":    datetime.now(timezone.utc),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


# ── Token validation ──────────────────────────────────────────────────────────

def decode_token(token: str) -> TokenData:
    """
    Decode and validate a JWT. Raises HTTPException on any failure.

    Raises:
        HTTPException 401: Token is invalid, expired, or malformed.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str | None = payload.get("sub")
        if not username:
            raise credentials_exception
        return TokenData(
            username=username,
            scopes=payload.get("scopes", []),
            exp=datetime.fromtimestamp(payload["exp"], tz=timezone.utc),
        )
    except JWTError as exc:
        log.warning("auth.jwt_error", error=str(exc))
        raise credentials_exception


# ── FastAPI dependencies ──────────────────────────────────────────────────────

_oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token", auto_error=True)


def require_auth(token: Annotated[str, Depends(_oauth2_scheme)]) -> UserPrincipal:
    """
    FastAPI dependency: validates the Bearer token and returns the caller's identity.
    Inject with `Depends(require_auth)`.
    """
    data = decode_token(token)
    return UserPrincipal(username=data.username, scopes=data.scopes)


def require_scope(required_scope: str):
    """
    FastAPI dependency factory: validates token AND checks the caller has
    the required scope.

    Usage:
        @router.post("/admin-endpoint")
        def endpoint(user = Depends(require_scope("admin"))):
            ...
    """
    def _checker(
        token: Annotated[str, Depends(_oauth2_scheme)]
    ) -> UserPrincipal:
        data = decode_token(token)
        if required_scope not in data.scopes:
            log.warning(
                "auth.insufficient_scope",
                user=data.username,
                required=required_scope,
                granted=data.scopes,
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Scope '{required_scope}' required.",
            )
        return UserPrincipal(username=data.username, scopes=data.scopes)
    return _checker


# ── Simple in-memory user store (replace with DB in production) ───────────────

_USERS: dict[str, dict] = {
    "admin": {
        "hashed_password": hash_password("admin-secret"),
        "scopes": ["read", "write", "admin"],
    },
    "analyst": {
        "hashed_password": hash_password("analyst-secret"),
        "scopes": ["read"],
    },
    "operator": {
        "hashed_password": hash_password("operator-secret"),
        "scopes": ["read", "write"],
    },
}


def authenticate_user(username: str, password: str) -> UserPrincipal | None:
    """
    Validate username + password against the user store.
    Returns UserPrincipal on success, None on failure.
    """
    user = _USERS.get(username)
    if not user:
        return None
    if not verify_password(password, user["hashed_password"]):
        return None
    return UserPrincipal(username=username, scopes=user["scopes"])


# ── /auth/token endpoint (wired into app by auth_router) ─────────────────────

from fastapi import APIRouter
from fastapi.security import OAuth2PasswordRequestForm

auth_router = APIRouter(prefix="/auth", tags=["Auth"])


@auth_router.post("/token", response_model=Token)
def login(form: Annotated[OAuth2PasswordRequestForm, Depends()]) -> Token:
    """
    OAuth2 password flow — exchange username + password for a JWT.
    curl -X POST /auth/token -d "username=admin&password=admin-secret"
    """
    user = authenticate_user(form.username, form.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = create_access_token(username=user.username, scopes=user.scopes)
    log.info("auth.token_issued", user=user.username, scopes=user.scopes)
    return Token(
        access_token=token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )
