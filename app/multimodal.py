import os
import httpx
from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field

class ImageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(min_length=3, max_length=4000)
    size: str = Field(default="1024x1024", pattern="^(512x512|1024x1024)$")

class VisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=1, max_length=2000)
    image_url: str = Field(min_length=10, max_length=8000000)

class VideoScriptRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    concept: str = Field(min_length=3, max_length=3000)
    duration_seconds: int = Field(default=60, ge=10, le=600)

async def invoke_provider(endpoint_env: str, key_env: str, path: str, payload: dict) -> dict:
    endpoint = os.getenv(endpoint_env, "").rstrip("/")
    key = os.getenv(key_env, "")
    if not endpoint or not endpoint.startswith("https://") or not key:
        raise HTTPException(status_code=503, detail="Multimodal provider is not configured with HTTPS and credentials")
    try:
        async with httpx.AsyncClient(timeout=120.0, follow_redirects=False) as client:
            response = await client.post(
                f"{endpoint}/{path}", json=payload,
                headers={"Authorization": f"Bearer {key}"}
            )
            response.raise_for_status()
            return response.json()
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Multimodal provider timed out")
    except (httpx.HTTPError, ValueError):
        raise HTTPException(status_code=502, detail="Multimodal provider unavailable")

async def generate_image(request: ImageRequest) -> dict:
    return await invoke_provider(
        "OCTA_IMAGE_API_URL", "OCTA_IMAGE_API_KEY", "images/generations",
        {"model": os.getenv("OCTA_IMAGE_MODEL", "image-model"), "prompt": request.prompt, "size": request.size}
    )

async def understand_image(request: VisionRequest) -> dict:
    if not request.image_url.startswith("data:image/"):
        raise HTTPException(status_code=422, detail="Only inline image data URLs are supported")
    media_type = request.image_url.split(";", 1)[0]
    if media_type not in ("data:image/jpeg", "data:image/png", "data:image/webp"):
        raise HTTPException(status_code=422, detail="Unsupported image format")
    if ";base64," not in request.image_url:
        raise HTTPException(status_code=422, detail="Base64 image data required")
    return await invoke_provider(
        "OCTA_VISION_API_URL", "OCTA_VISION_API_KEY", "chat/completions",
        {"model": os.getenv("OCTA_VISION_MODEL", "vision-model"),
         "messages": [{"role": "user", "content": [
             {"type": "text", "text": request.question},
             {"type": "image_url", "image_url": {"url": request.image_url}}
         ]}], "max_tokens": 1024}
    )

async def create_video_script(request: VideoScriptRequest) -> dict:
    return await invoke_provider(
        "OCTA_INFERENCE_URL", "OCTA_INFERENCE_API_KEY", "chat/completions",
        {"model": os.getenv("OCTA_MODEL", "Qwen/Qwen3-8B"),
         "messages": [
             {"role": "system", "content": "Create an original, practical video script with timed scenes, narration, visuals and production notes. Do not claim to have rendered video."},
             {"role": "user", "content": f"Concept: {request.concept}. Duration: {request.duration_seconds} seconds."}
         ], "max_tokens": 1400}
    )
