# OCTA 0.2 — Experimental training

OCTA is not yet a capable assistant. These commands run a small byte-level language-model experiment using text you have permission to use.

```bash
pip install -r requirements-model.txt
python -m pytest tests/
python -m octa.train_v2 --text path/to/authorized_text.txt --steps 200
python -m octa.train_v2 --text path/to/authorized_text.txt --steps 400 --resume checkpoints/octa-v2.pt
python -m octa.sample --checkpoint checkpoints/octa-v2.pt --prompt "OCTA" --tokens 80
```

The script uses a contiguous 90/10 train/validation split, saves model and optimizer checkpoints, and reports validation perplexity. A tiny dataset can lead to memorization, so validation loss is not a measure of general intelligence. Do not commit private data or checkpoints. The cloud gateway is not yet connected to OCTA's own weights.

Before production: introduce a licensed high-quality dataset, a trained subword tokenizer, deduplication, evaluation benchmarks, data governance, more efficient training, distributed GPU support, safety testing, and a secure model-serving layer.
