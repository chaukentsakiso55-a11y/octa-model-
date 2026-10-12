"""OCTA native inference API: serves a trusted OCTA byte checkpoint."""
import os
from contextlib import asynccontextmanager
from pathlib import Path
import hmac
import torch
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from .model import OctaConfig, OctaForCausalLM
from .tokenizer import ByteTokenizer

model = None
tokenizer = ByteTokenizer()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    path = os.environ.get("OCTA_CHECKPOINT")
    if not path or not Path(path).is_file():
        raise RuntimeError("OCTA_CHECKPOINT must point to a trusted checkpoint")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = OctaConfig(**checkpoint["config"])
    if config.vocab_size != 256:
        raise RuntimeError("Byte-level checkpoints only")
    model = OctaForCausalLM(config)
    model.load_state_dict(checkpoint.get("model", checkpoint.get("state_dict")))
    model.eval()
    try:
        yield
    finally:
        model = None

app = FastAPI(title="OCTA Native Inference", version="0.5.0", lifespan=lifespan)

class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4096)
    max_new_tokens: int = Field(default=32, ge=1, le=256)
    temperature: float = Field(default=0.8, gt=0, le=2)

@app.get("/health")
def health():
    return {"status": "ready" if model is not None else "unloaded", "model": "OCTA"}

@app.post("/v1/generate")
def generate(body: GenerateRequest, authorization: str | None = Header(default=None)):
    expected = os.environ.get("OCTA_API_KEY")
    if not expected:
        raise HTTPException(503, "OCTA_API_KEY is not configured")
    if not authorization or not hmac.compare_digest(authorization, "Bearer " + expected):
        raise HTTPException(401, "Unauthorized")
    if model is None:
        raise HTTPException(503, "Checkpoint not loaded")
    prompt_ids = tokenizer.encode(body.prompt)[-model.cfg.context_length:]
    with torch.inference_mode():
        generated = model.generate(torch.tensor([prompt_ids], dtype=torch.long), max_new_tokens=body.max_new_tokens, temperature=body.temperature)
    new_tokens = generated[0, len(prompt_ids):].tolist()
    return {"model": "OCTA", "text": tokenizer.decode(new_tokens), "generated_tokens": len(new_tokens)}
