"""Cloud inference routing with bounded failover and explicit provider configuration."""
import json
import os
from urllib.parse import urlparse

import httpx
from fastapi import HTTPException


def providers() -> list[dict]:
    raw = os.getenv("OCTA_CLOUD_PROVIDERS", "").strip()
    if raw:
        try:
            entries = json.loads(raw)
        except (ValueError, TypeError):
            raise HTTPException(503, "Invalid cloud provider configuration")
    else:
        entries = [{
            "name": "default",
            "url": os.getenv("OCTA_INFERENCE_URL", "http://inference:8000/v1"),
            "model": os.getenv("OCTA_MODEL", "Qwen/Qwen3-8B"),
            "key_env": "OCTA_INFERENCE_API_KEY",
        }]
    if not isinstance(entries, list) or not 1 <= len(entries) <= 5:
        raise HTTPException(503, "Cloud providers must contain 1 to 5 entries")
    seen = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise HTTPException(503, "Invalid cloud provider entry")
        name, url, model, key_env = (entry.get(k) for k in ("name", "url", "model", "key_env"))
        if not all(isinstance(v, str) and v for v in (name, url, model, key_env)):
            raise HTTPException(503, "Cloud provider fields are required")
        parsed = urlparse(url)
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise HTTPException(503, "Invalid cloud provider URL")
        if parsed.scheme != "https" and not (parsed.scheme == "http" and parsed.hostname == "inference"):
            raise HTTPException(503, "Cloud provider must use HTTPS")
        if name in seen:
            raise HTTPException(503, "Duplicate cloud provider")
        seen.add(name)
    return entries


async def complete(body: dict) -> dict:
    configured = providers()
    requested = body.get("model")
    if requested and requested not in ("auto", *[p["model"] for p in configured]):
        raise HTTPException(422, "Requested model is not configured")
    candidates = [p for p in configured if p["model"] == requested] if requested and requested != "auto" else configured
    timeout = max(1.0, min(float(os.getenv("OCTA_TIMEOUT_SECONDS", "120")), 180.0))
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
        for index, provider in enumerate(candidates):
            key = os.getenv(provider["key_env"], "")
            headers = {"Authorization": f"Bearer {key}"} if key else {}
            payload = {**body, "model": provider["model"]}
            try:
                response = await client.post(provider["url"].rstrip("/") + "/chat/completions", json=payload, headers=headers)
                if response.status_code in (408, 429, 500, 502, 503, 504):
                    if index < len(candidates) - 1:
                        continue
                    raise HTTPException(502, "All inference providers are unavailable")
                if response.status_code >= 400:
                    raise HTTPException(502, "Inference provider rejected the request")
                result = response.json()
                if not isinstance(result, dict) or not isinstance(result.get("choices"), list):
                    raise ValueError("Invalid inference response")
                return result
            except (httpx.TimeoutException, httpx.TransportError, ValueError):
                if index == len(candidates) - 1:
                    raise HTTPException(502, "All inference providers are unavailable")
    raise HTTPException(502, "No inference provider available")
