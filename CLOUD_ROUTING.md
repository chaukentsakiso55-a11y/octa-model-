# Octa Cloud model routing

The cloud gateway supports a prioritized list of OpenAI-compatible inference providers. The first available provider handles requests; Octa retries the next provider for connection errors, timeouts, HTTP 408, 429, and common server errors.

## Configuration

Set `OCTA_CLOUD_PROVIDERS` to a JSON array (maximum five entries):

```json
[
  {"name":"primary","url":"https://your-primary-provider.example/v1","model":"your-model-id","key_env":"OCTA_PRIMARY_KEY"},
  {"name":"backup","url":"https://your-backup-provider.example/v1","model":"your-backup-model-id","key_env":"OCTA_BACKUP_KEY"}
]
```

Set the referenced environment variables to provider-issued API keys. Never commit keys or ship master keys inside Android apps.

If `OCTA_CLOUD_PROVIDERS` is unset, the gateway uses the existing `OCTA_INFERENCE_URL`, `OCTA_MODEL` and `OCTA_INFERENCE_API_KEY` settings.

The chat endpoint is still `POST /v1/chat/completions`. Omitting `model` or using `auto` enables fallback across the configured providers. Supplying a configured model name targets only matching providers.

## Limitations

This is inference routing, not training or hosting model weights. A running compatible inference service is still required. Fallback is best-effort and only covers request failures before a successful response. Deploy behind HTTPS with per-app authorization, rate limiting and secure token issuance. Test provider-specific response formats before production use.

Run `python -m pytest tests -q` and verify the cloud workflow before merging.
