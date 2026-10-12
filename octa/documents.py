"""Document-level dataset preparation for OCTA."""
import hashlib
import json
import random
from pathlib import Path

def load_jsonl(path):
    docs = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at line {line_number}") from exc
        if not isinstance(item, dict) or not isinstance(item.get("text"), str):
            raise ValueError(f"Line {line_number}: expected a string text field")
        if item["text"].strip():
            docs.append(item["text"].strip())
    return docs

def split_documents(documents, seed=42, validation_fraction=0.2):
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    unique = {}
    for doc in documents:
        cleaned = " ".join(doc.split())
        if cleaned:
            unique[hashlib.sha256(cleaned.encode("utf-8")).hexdigest()] = cleaned
    docs = [unique[k] for k in sorted(unique)]
    if len(docs) < 2:
        raise ValueError("At least two distinct documents required")
    random.Random(seed).shuffle(docs)
    n_val = max(1, min(len(docs) - 1, round(len(docs) * validation_fraction)))
    return docs[n_val:], docs[:n_val]

def tokenize_documents(documents):
    return [list(doc.encode("utf-8")) for doc in documents if len(doc.encode("utf-8")) >= 2]
