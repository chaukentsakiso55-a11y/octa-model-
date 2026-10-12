# OCTA 0.7 — Document-level evaluation

OCTA 0.7 adds document-level train/validation splitting, deterministic seeds, whitespace-normalized exact deduplication, and held-out validation loss.

Input: UTF-8 JSONL, one document per line, e.g. `{"text": "Authorized training document"}`.

```bash
pip install -r requirements-model.txt
python -m pytest tests/ -q
python -m octa.train_documents --data path/to/authorized_documents.jsonl --steps 100 --output checkpoints/octa-0.7.pt
```

Only train on text you have permission to use. This deduplicates exact normalized documents, not paraphrases or near-duplicates. Small-scale byte-level validation is not evidence of general conversational intelligence. No dataset or trained checkpoint is included.
