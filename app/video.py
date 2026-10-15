import os
import re
from urllib.parse import urlparse
import httpx
from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field

class VideoRenderRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(min_length=3, max_length=3000)
    duration_seconds: int = Field(default=5, ge=2, le=20)
    aspect_ratio: str = Field(default="16:9", pattern="^(16:9|9:16|1:1)$")

def provider_config() -> tuple[str, dict]:
    base = os.getenv("OCTA_VIDEO_API_URL", "").rstrip("/")
    key = os.getenv("OCTA_VIDEO_API_KEY", "")
    parsed = urlparse(base)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or not key:
        raise HTTPException(status_code=503, detail="Video provider is not configured")
    return base, {"Authorization": f"Bearer {key}", "Accept": "application/json"}

async def request_provider(method: str, path: str, body: dict | None = None) -> dict:
    base, headers = provider_config()
    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=False) as client:
            response = await client.request(method, f"{base}/{path}", json=body, headers=headers)
            response.raise_for_status()
            data = response.json()
        if not isinstance(data, dict):
            raise ValueError("Invalid provider response")
        return data
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Video provider timed out")
    except (httpx.HTTPError, ValueError):
        raise HTTPException(status_code=502, detail="Video provider unavailable")

async def start_render(request: VideoRenderRequest) -> dict:
    data = await request_provider("POST", "videos/generations", {
        "model": os.getenv("OCTA_VIDEO_MODEL", "video-model"),
        "prompt": request.prompt,
        "duration_seconds": request.duration_seconds,
        "aspect_ratio": request.aspect_ratio,
    })
    job_id = data.get("id")
    if not isinstance(job_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", job_id):
        raise HTTPException(status_code=502, detail="Invalid provider job ID")
    return {"id": job_id, "status": data.get("status", "submitted")}

async def render_status(job_id: str) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", job_id):
        raise HTTPException(status_code=422, detail="Invalid job ID")
    data = await request_provider("GET", f"videos/generations/{job_id}")
    return {"id": job_id, "status": data.get("status", "unknown"), "output": data.get("output")}
