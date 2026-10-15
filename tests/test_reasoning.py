from fastapi.testclient import TestClient
from app.main import app
import app.reasoning as reasoning

def test_reason_requires_auth(monkeypatch):
    monkeypatch.setenv("OCTA_API_KEYS", "test-secret")
    with TestClient(app) as client:
        response = client.post("/v1/reason", json={"task": "Explain the water cycle"})
    assert response.status_code == 401

def test_reason_invalid_mode(monkeypatch):
    monkeypatch.setenv("OCTA_API_KEYS", "test-secret")
    with TestClient(app) as client:
        response = client.post("/v1/reason", headers={"Authorization": "Bearer test-secret"}, json={"task": "Explain the water cycle", "mode": "unlimited"})
    assert response.status_code == 422

def test_reason_standard(monkeypatch):
    monkeypatch.setenv("OCTA_API_KEYS", "test-secret")
    async def fake_infer(messages, max_tokens=1200):
        return "Water evaporates, condenses and falls as precipitation."
    monkeypatch.setattr(reasoning, "infer", fake_infer)
    with TestClient(app) as client:
        response = client.post("/v1/reason", headers={"Authorization": "Bearer test-secret"}, json={"task": "Explain the water cycle"})
    assert response.status_code == 200
    assert response.json()["passes"] == 1

def test_reason_deep(monkeypatch):
    monkeypatch.setenv("OCTA_API_KEYS", "test-secret")
    calls = []
    async def fake_infer(messages, max_tokens=1200):
        calls.append(messages)
        return f"Revision {len(calls)}"
    monkeypatch.setattr(reasoning, "infer", fake_infer)
    with TestClient(app) as client:
        response = client.post("/v1/reason", headers={"Authorization": "Bearer test-secret"}, json={"task": "Explain the water cycle", "mode": "deep"})
    assert response.status_code == 200
    assert response.json()["passes"] == 3
    assert response.json()["answer"] == "Revision 3"
