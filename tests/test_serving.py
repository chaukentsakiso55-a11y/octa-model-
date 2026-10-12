import torch
from fastapi.testclient import TestClient
from octa.model import OctaConfig, OctaForCausalLM
from octa.serve import app

def test_serving_checkpoint(tmp_path, monkeypatch):
    cfg = OctaConfig(vocab_size=256, d_model=32, n_heads=4, n_layers=1, context_length=16)
    m = OctaForCausalLM(cfg)
    path = tmp_path / "test.pt"
    torch.save({"config": cfg.__dict__, "model": m.state_dict()}, path)
    monkeypatch.setenv("OCTA_CHECKPOINT", str(path))
    monkeypatch.setenv("OCTA_API_KEY", "test-only-token")
    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "ready"
        assert client.post("/v1/generate", json={"prompt":"hi"}).status_code == 401
        response = client.post("/v1/generate", headers={"Authorization":"Bearer test-only-token"}, json={"prompt":"hi", "max_new_tokens":2})
        assert response.status_code == 200, response.text
        assert response.json()["generated_tokens"] == 2
