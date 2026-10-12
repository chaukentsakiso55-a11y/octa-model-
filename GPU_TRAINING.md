# OCTA 0.4 — GPU training and benchmarking

This stage supplies training code; it does not supply trained weights or rented GPU resources.

## Run
```bash
pip install -r requirements-model.txt
python -m pytest tests/ -q
python -m octa.train_gpu --text path/to/authorized_training.txt --steps 200
python -m octa.benchmark --checkpoint checkpoints/octa-gpu.pt --text path/to/authorized_training.txt
```

The training script uses CUDA when available, and CPU otherwise. On CUDA devices supporting bfloat16, it uses automatic mixed precision. Gradient accumulation reduces the microbatch memory requirement. It records a SHA-256 dataset fingerprint and prevents resuming against changed data.

The benchmark compares validation loss with a uniform-byte baseline. It does not establish general intelligence or superiority to other LLMs.

## Limitations
- Byte tokenizer and tiny default architecture are experimental.
- Evaluation is sampled and not yet a fixed benchmark suite.
- Data must be legally usable for model training.
- Checkpoints are generated locally and should not be committed to Git.
- For large-scale training, implement distributed data loading, efficient attention, optimizer scaling, stable learning-rate schedules, and rigorous evaluation.
