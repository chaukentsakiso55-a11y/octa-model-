# OCTA — CyberPulse Cloud AI

Shared, authenticated cloud AI gateway for CyberPulse applications (Studylog, GuardianAI, StudyLock, Obsidian).

## Architecture
Applications -> HTTPS reverse proxy + rate limiter -> OCTA FastAPI gateway -> private OpenAI-compatible inference server (e.g. vLLM running an official Qwen checkpoint).

## Quick start
1. Copy `.env.example` to `.env` and set a long random `OCTA_API_KEYS` value.
2. Configure `OCTA_INFERENCE_URL` and `OCTA_MODEL` to match your separately deployed inference server.
3. Run `docker compose up --build -d`.
4. Test `curl http://127.0.0.1:8080/health`.
5. Send POST `/v1/chat/completions` with `Authorization: Bearer <your key>` and JSON `{"messages":[{"role":"user","content":"Hello"}]}`.

## Important
This repository contains a functional API gateway, **not pretrained model weights**. Start an OpenAI-compatible inference service separately; the default upstream hostname is illustrative. No inference is available until it is configured. Keep the API behind HTTPS and a rate-limiting reverse proxy. Do not embed master API keys in Android applications: use per-user short-lived tokens issued by a trusted backend before public deployment. Add per-app authorization, logging controls, privacy/data retention, and abuse prevention before production use. Do not send private student records to unapproved providers.

The local OCTA repository will be separate. Local Qwen inference performance depends on device RAM, CPU/GPU, quantization, and context length.
