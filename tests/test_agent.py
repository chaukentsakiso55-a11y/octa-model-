from fastapi.testclient import TestClient
from app.main import app

def test_agent_requires_auth(monkeypatch):
    monkeypatch.setenv("OCTA_API_KEYS", "test-secret")
    with TestClient(app) as client:
        response = client.post("/v1/research/answer", json={"question": "What is Octa?"})
    assert response.status_code == 401

def test_agent_validates_input(monkeypatch):
    monkeypatch.setenv("OCTA_API_KEYS", "test-secret")
    with TestClient(app) as client:
        response = client.post("/v1/research/answer", headers={"Authorization": "Bearer test-secret"}, json={"question": ""})
    assert response.status_code == 422
