import hmac
import os
from fastapi import HTTPException

def require_api_key(authorization: str | None) -> None:
    keys = [value.strip() for value in os.getenv("OCTA_API_KEYS", "").split(",") if value.strip()]
    if not keys:
        raise HTTPException(status_code=503, detail="API authentication is not configured")
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Bearer token required")
    token = authorization[7:]
    if not any(hmac.compare_digest(token, key) for key in keys):
        raise HTTPException(status_code=403, detail="Invalid token")
