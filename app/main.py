import os
import httpx
from fastapi import FastAPI, Header, HTTPException
from app.schemas import ChatRequest
from app.security import require_api_key

app = FastAPI(title="OCTA Cloud API", version="0.1.0")
MODEL = os.getenv("OCTA_MODEL", "Qwen/Qwen3-8B")
UPSTREAM = os.getenv("OCTA_INFERENCE_URL", "http://inference:8000/v1").rstrip("/")
TIMEOUT = float(os.getenv("OCTA_TIMEOUT_SECONDS", "120"))

@app.get("/health")
async def health():
    return {"status": "ok", "service": "octa-cloud"}

@app.post("/v1/chat/completions")
async def chat(payload: ChatRequest, authorization: str | None = Header(default=None)):
    require_api_key(authorization)
    upstream_key = os.getenv("OCTA_INFERENCE_API_KEY", "")
    headers = {"Authorization": f"Bearer {upstream_key}"} if upstream_key else {}
    body = payload.model_dump(exclude_none=True)
    body["model"] = MODEL
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.post(f"{UPSTREAM}/chat/completions", json=body, headers=headers)
        response.raise_for_status()
        return response.json()
    except httpx.TimeoutException:
        raise HTTPException(504, "Inference timed out")
    except (httpx.HTTPError, ValueError):
        raise HTTPException(502, "Inference service unavailable")
