# OCTA 0.3 — Current engineering status

OCTA is an original experimental decoder-only transformer. It has randomly initialized weights until trained. No claim of superiority to Qwen, GPT, or other models is supported.

## Available
- Original decoder-only transformer implementation
- Byte tokenizer with lossless UTF-8 round trips
- Tiny next-token training experiment with train/validation splits
- Checkpoint save/resume and validation loss
- Basic text sampler
- GitHub Actions CPU unit-test workflow

## Important limitations
- The 256-token byte tokenizer is a baseline, not an efficient production tokenizer.
- The model does not yet have a trained, evaluated checkpoint.
- No measured performance, intelligence, accuracy, or inference latency is available.
- Current cloud gateway is not connected to OCTA's own weights.
- No instruction tuning, retrieval, or safety evaluation has been completed.

## Next priorities
1. Run CI and repair any failing tests.
2. Train on authorized data, save checkpoints, compare held-out validation metrics.
3. Build a learned subword tokenizer and configurable model scale.
4. Implement safe model serving and connect the CyberPulse cloud API.
5. Benchmark against baselines using independent tasks.
