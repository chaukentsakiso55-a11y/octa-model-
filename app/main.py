import hmac
import os
from typing import Any
import httpx
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="OCTA Cloud API", version="0.1.0")
MODEL = os.getenv("OCTA_MODEL", "Qwen/Qwen3-8B")
UPSTREAM = os.getenv("OCTA_INFERENCE_URL", "http://inference:8000/v1").rstrip("/")
TIMEOUT = float(os.getenv("OCTA_TIMEOUT_SECONDS", "120"))

class ChatRequest(BaseModel):
    messages: list[dict[str, Any]] = Field(min_length=1)
    model: str | None = None
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int = Field(default=512, ge=1, le=8192)

def authorize(value: str | None):
    keys = [x.strip() for x in os.getenv("OCTA_API_KEYS", "").split(",") if x.strip()]
    if not keys:
        raise HTTPException(503, "Server authentication is not configured")
    if not value or not value.startswith("Bearer "):
        raise HTTPException(401, "Bearer token required")
    supplied = value[7:]
    if not any(hmac.compare_digest(supplied, key) for key in keys):
        raise HTTPException(403, "Invalid token")

@app.get("/health")
async def health():
    return {"status": "ok", "service": "octa-cloud"}

@app.post("/v1/chat/completions")
async def chat(payload: ChatRequest, authorization: str | None = Header(default=None)):
    authorize(authorization)
    upstream_key = os.getenv("OCTA_INFERENCE_API_KEY", "")
    headers = {"Authorization": f"Bearer {upstream_key}"} if upstream_key else {}
    body = payload.model_dump(exclude_none=True)
    body["model"] = MODEL  # Server-controlled model allowlist: one configured model
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.post(f"{UPSTREAM}/chat/completions", json=body, headers=headers)
        response.raise_for_status()
        return response.json()
    except httpx.TimeoutException:
        raise HTTPException(504, "Inference timed out")
    except (httpx.HTTPError, ValueError):
        raise HTTPException(502, "Inference service unavailable")
