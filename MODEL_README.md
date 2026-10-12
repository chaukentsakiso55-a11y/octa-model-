# OCTA Original Model Research Track

This is an **independently initialized** decoder-only transformer research implementation, not Qwen/OpenAI weights, a pretrained chatbot, or a claim of state-of-the-art performance.

## Architecture
- Learned token and position embeddings
- Pre-normalized causal multi-head attention
- RMSNorm and SiLU feed-forward layers
- Tied output and input embeddings
- Autoregressive next-token training and sampling

## First experiment
```bash
pip install -r requirements-model.txt
python -m pytest tests/test_model.py
python -m octa.train --text path/to/licensed_training_text.txt --steps 100
```
This first pipeline trains a tiny **byte-level** model from scratch and saves a checkpoint locally. Only use datasets you have permission to train on. The default model configuration in `octa/model.py` is not yet pretrained. The experimental byte vocabulary uses 256 tokens.

## Next milestones
1. Train a proper subword tokenizer on permitted data.
2. Add reproducible dataset preparation, validation split, checkpoint resume, mixed precision, and distributed training.
3. Train a small baseline and measure held-out loss/perplexity.
4. Benchmark reasoning, coding, language, latency, and safety against published baselines.
5. Deploy only trained, evaluated OCTA checkpoints to the cloud inference server.

The existing `app/` directory is a separate gateway and still points to a configurable upstream model. It does **not** serve this untrained OCTA model automatically.
