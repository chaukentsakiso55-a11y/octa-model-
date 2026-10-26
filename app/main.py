from fastapi import FastAPI, Header, HTTPException
from app.schemas import ChatRequest
from app.router import complete
from app.security import require_api_key
from app.research import SearchRequest, search_web
from app.agent import ResearchAnswerRequest, research_answer
from app.multimodal import ImageRequest, VisionRequest, VideoScriptRequest, generate_image, understand_image, create_video_script
from app.video import VideoRenderRequest, start_render, render_status
from app.reasoning import ReasonRequest, reason

app = FastAPI(title="OCTA Cloud API", version="0.1.0")
@app.get("/health")
async def health():
    return {"status": "ok", "service": "octa-cloud"}

@app.post("/v1/chat/completions")
async def chat(payload: ChatRequest, authorization: str | None = Header(default=None)):
    require_api_key(authorization)
    return await complete(payload.model_dump(exclude_none=True))

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

@app.post("/v1/reason")
async def reason_endpoint(request: ReasonRequest, authorization: str | None = Header(default=None)):
    require_api_key(authorization)
    return await reason(request)
