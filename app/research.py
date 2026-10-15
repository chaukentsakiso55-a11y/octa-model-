import os
import httpx
from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field

class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=2, max_length=500)
    count: int = Field(default=5, ge=1, le=10)

async def search_web(request: SearchRequest) -> dict:
    api_key = os.getenv("OCTA_BRAVE_SEARCH_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=503, detail="Web search provider is not configured")
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(
                "https://api.search.brave.com/res/v1/web/search",
                params={"q": request.query, "count": request.count},
                headers={"X-Subscription-Token": api_key, "Accept": "application/json"},
            )
            response.raise_for_status()
            results = response.json().get("web", {}).get("results", [])
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Search provider timed out")
    except (httpx.HTTPError, ValueError, TypeError, AttributeError):
        raise HTTPException(status_code=502, detail="Search provider unavailable")
    sources = []
    for result in results[:request.count]:
        url = result.get("url", "")
        if not url.startswith(("https://", "http://")):
            continue
        sources.append({
            "id": len(sources) + 1,
            "title": str(result.get("title", ""))[:300],
            "url": url,
            "snippet": str(result.get("description", ""))[:1200],
        })
    return {"query": request.query, "sources": sources}
