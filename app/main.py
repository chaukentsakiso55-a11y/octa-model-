import os
import httpx
from fastapi import FastAPI, Header, HTTPException
from app.schemas import ChatRequest
from app.security import require_api_key
from app.research import SearchRequest, search_web
from app.agent import ResearchAnswerRequest, research_answer
from app.multimodal import ImageRequest, VisionRequest, VideoScriptRequest, generate_image, understand_image, create_video_script
from app.video import VideoRenderRequest, start_render, render_status

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

@app.post("/v1/research/search")
async def research(request: SearchRequest, authorization: str | None = Header(default=None)):
    require_api_key(authorization)
    return await search_web(request)

@app.post("/v1/research/answer")
async def answer_with_sources(request: ResearchAnswerRequest, authorization: str | None = Header(default=None)):
    require_api_key(authorization)
    return await research_answer(request)

@app.post("/v1/images/generations")
async def images(request: ImageRequest, authorization: str | None = Header(default=None)):
    require_api_key(authorization)
    return await generate_image(request)

@app.post("/v1/vision/analyze")
async def vision(request: VisionRequest, authorization: str | None = Header(default=None)):
    require_api_key(authorization)
    return await understand_image(request)

@app.post("/v1/video/script")
async def video_script(request: VideoScriptRequest, authorization: str | None = Header(default=None)):
    require_api_key(authorization)
    return await create_video_script(request)

@app.post("/v1/video/generations")
async def video_generate(request: VideoRenderRequest, authorization: str | None = Header(default=None)):
    require_api_key(authorization)
    return await start_render(request)

@app.get("/v1/video/generations/{job_id}")
async def video_status(job_id: str, authorization: str | None = Header(default=None)):
    require_api_key(authorization)
    return await render_status(job_id)
