from fastapi.testclient import TestClient
from app.main import app

def test_multimodal_authentication(monkeypatch):
    monkeypatch.setenv("OCTA_API_KEYS", "test-secret")
    with TestClient(app) as client:
        for path, payload in (
            ("/v1/images/generations", {"prompt": "A futuristic city"}),
            ("/v1/vision/analyze", {"question": "What is this?", "image_url": "data:image/png;base64,aGVsbG8="}),
            ("/v1/video/script", {"concept": "An AI science documentary"}),
        ):
            assert client.post(path, json=payload).status_code == 401

def test_missing_image_provider(monkeypatch):
    monkeypatch.setenv("OCTA_API_KEYS", "test-secret")
    monkeypatch.delenv("OCTA_IMAGE_API_URL", raising=False)
    with TestClient(app) as client:
        response = client.post("/v1/images/generations", headers={"Authorization": "Bearer test-secret"}, json={"prompt": "A futuristic city"})
    assert response.status_code == 503

def test_vision_rejects_remote_urls(monkeypatch):
    monkeypatch.setenv("OCTA_API_KEYS", "test-secret")
    with TestClient(app) as client:
        response = client.post("/v1/vision/analyze", headers={"Authorization": "Bearer test-secret"}, json={"question": "Describe this", "image_url": "https://example.com/image.png"})
    assert response.status_code == 422
