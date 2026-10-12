# OCTA 0.5 — Native inference server

This endpoint serves OCTA's own checkpoint, not Qwen or OpenAI weights. It does not turn an experimental byte-level model into a capable assistant.

## Install and test
```bash
pip install -r requirements-model.txt
python -m pytest tests/ -q
```

## Run after training
```bash
export OCTA_CHECKPOINT=checkpoints/octa-gpu.pt
export OCTA_API_KEY='set-a-long-random-secret'
uvicorn octa.serve:app --host 127.0.0.1 --port 8001
```

```bash
curl -X POST http://127.0.0.1:8001/v1/generate -H 'Authorization: Bearer set-a-long-random-secret' -H 'Content-Type: application/json' -d '{"prompt":"Hello OCTA","max_new_tokens":32}'
```

The native endpoint is /v1/generate. The earlier CyberPulse gateway's /v1/chat/completions endpoint is **not yet connected** to OCTA. Do not expose this service publicly without TLS, rate limits, monitoring, and user-specific authorization. Checkpoints are not bundled.
