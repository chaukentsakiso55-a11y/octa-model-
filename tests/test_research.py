from fastapi.testclient import TestClient
from app.main import app

def test_research_requires_authentication(monkeypatch):
    monkeypatch.setenv("OCTA_API_KEYS", "test-secret")
    with TestClient(app) as client:
        response = client.post("/v1/research/search", json={"query": "Octa AI"})
    assert response.status_code == 401

def test_research_requires_provider(monkeypatch):
    monkeypatch.setenv("OCTA_API_KEYS", "test-secret")
    monkeypatch.delenv("OCTA_BRAVE_SEARCH_API_KEY", raising=False)
    with TestClient(app) as client:
        response = client.post("/v1/research/search", headers={"Authorization": "Bearer test-secret"}, json={"query": "Octa AI"})
    assert response.status_code == 503

def test_research_validates_query(monkeypatch):
    monkeypatch.setenv("OCTA_API_KEYS", "test-secret")
    with TestClient(app) as client:
        response = client.post("/v1/research/search", headers={"Authorization": "Bearer test-secret"}, json={"query": ""})
    assert response.status_code == 422
