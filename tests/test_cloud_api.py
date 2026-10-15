from fastapi.testclient import TestClient
from app.main import app

def test_authentication_required(monkeypatch):
    monkeypatch.setenv("OCTA_API_KEYS", "test-secret")
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json={"messages":[{"role":"user","content":"Hello"}]})
    assert response.status_code == 401

def test_rejects_invalid_messages(monkeypatch):
    monkeypatch.setenv("OCTA_API_KEYS", "test-secret")
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", headers={"Authorization":"Bearer test-secret"}, json={"messages":[{"role":"tool","content":"Hello"}]})
    assert response.status_code == 422

def test_health():
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["service"] == "octa-cloud"
