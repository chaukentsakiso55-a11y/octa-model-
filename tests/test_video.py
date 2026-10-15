from fastapi.testclient import TestClient
from app.main import app

def test_video_render_requires_auth(monkeypatch):
    monkeypatch.setenv("OCTA_API_KEYS", "test-secret")
    with TestClient(app) as client:
        assert client.post("/v1/video/generations", json={"prompt": "A flying robot"}).status_code == 401
        assert client.get("/v1/video/generations/test-job").status_code == 401

def test_video_render_provider_missing(monkeypatch):
    monkeypatch.setenv("OCTA_API_KEYS", "test-secret")
    monkeypatch.delenv("OCTA_VIDEO_API_URL", raising=False)
    with TestClient(app) as client:
        response = client.post("/v1/video/generations", headers={"Authorization": "Bearer test-secret"}, json={"prompt": "A flying robot"})
        assert response.status_code == 503

def test_video_render_job_id_validation(monkeypatch):
    monkeypatch.setenv("OCTA_API_KEYS", "test-secret")
    with TestClient(app) as client:
        response = client.get("/v1/video/generations/invalid.id", headers={"Authorization": "Bearer test-secret"})
        assert response.status_code == 422
