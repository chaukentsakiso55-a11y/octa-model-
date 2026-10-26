import json
import asyncio

import httpx
import pytest
from fastapi import HTTPException

from app.router import complete, providers


def test_rejects_insecure_external_provider(monkeypatch):
    monkeypatch.setenv("OCTA_CLOUD_PROVIDERS", json.dumps([
        {"name": "bad", "url": "http://example.com/v1", "model": "test", "key_env": "TEST_KEY"}
    ]))
    with pytest.raises(HTTPException) as error:
        providers()
    assert error.value.status_code == 503


def test_fallback_after_rate_limit(monkeypatch):
    monkeypatch.setenv("OCTA_CLOUD_PROVIDERS", json.dumps([
        {"name": "first", "url": "https://first.example/v1", "model": "first-model", "key_env": "FIRST_KEY"},
        {"name": "second", "url": "https://second.example/v1", "model": "second-model", "key_env": "SECOND_KEY"},
    ]))
    calls = []

    def handler(request):
        calls.append(request.url.host)
        if request.url.host == "first.example":
            return httpx.Response(429)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    transport = httpx.MockTransport(handler)
    real_client = httpx.AsyncClient
    monkeypatch.setattr("app.router.httpx.AsyncClient", lambda **kwargs: real_client(transport=transport, **kwargs))
    response = asyncio.run(complete({"messages": [{"role": "user", "content": "hello"}]}))
    assert response["choices"][0]["message"]["content"] == "ok"
    assert calls == ["first.example", "second.example"]


def test_unknown_model_is_rejected(monkeypatch):
    monkeypatch.delenv("OCTA_CLOUD_PROVIDERS", raising=False)
    with pytest.raises(HTTPException) as error:
        asyncio.run(complete({"model": "unapproved", "messages": [{"role": "user", "content": "hello"}]}))
    assert error.value.status_code == 422
