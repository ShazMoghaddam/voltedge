"""Auth router — /auth endpoints."""
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from voltedge.api.auth import TokenResponse, authenticate_user, create_access_token

router = APIRouter(prefix="/auth", tags=["auth"])

@router.post("/token", response_model=TokenResponse)
def login(form: OAuth2PasswordRequestForm = Depends()):
    """Exchange credentials for a JWT access token."""
    user = authenticate_user(form.username, form.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = create_access_token(sub=form.username, role=user["role"])
    return TokenResponse(access_token=token, role=user["role"])
