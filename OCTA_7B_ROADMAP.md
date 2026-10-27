# OCTA 7B Foundation Plan

OCTA is an independent experimental language model. This branch does not contain a trained 7B checkpoint and does not claim GPT-level intelligence.

## Phase 1: Verify foundations
- Preserve the current tiny training path and CPU tests.
- Count parameters analytically before allocating GPU memory.
- Train and evaluate small checkpoints on authorized text.
- Record validation loss, reproducibility and token throughput.

## Phase 2: Architecture research
- Benchmark the existing decoder-only transformer against RoPE, grouped-query attention and gated MLP alternatives.
- Introduce a trained subword tokenizer, carefully versioned with checkpoints.
- Test attention masks, checkpoint loading and long-context correctness.
- Select architecture only after small-scale experiments.

## Phase 3: Scaling
- Run 100M and 500M experiments before committing to 7B.
- Estimate GPU training memory including optimizer, gradients and activations.
- Add distributed training, resumable checkpoints and dataset manifests.
- Train the original OCTA weights on properly licensed or authorized data.

## Phase 4: Quality
- Hold out evaluation data and test language, reasoning, code, safety and calibration.
- Add instruction tuning and preference optimization only after a competent pretrained checkpoint exists.
- Benchmark against public baselines without overstating performance.

## Phase 5: Cloud
- Serve OCTA's own checkpoint behind authenticated streaming inference.
- Track latency, throughput, cost, reliability and abuse protections.
- Keep checkpoints out of Git; use secured model artifact storage.

The 7B config in octa/scale_plan.py is a sizing proposal only. The current architecture uses learned absolute positional embeddings and a simple MLP; architecture changes should be tested before training at scale.
