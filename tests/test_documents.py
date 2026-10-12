import json
import pytest
from octa.documents import load_jsonl, split_documents, tokenize_documents

def test_no_duplicate_leakage(tmp_path):
    path = tmp_path / "docs.jsonl"
    docs = ["first document", "first  document", "second document", "third document", "fourth document"]
    path.write_text("".join(json.dumps({"text": d}) + "\n" for d in docs), encoding="utf-8")
    train, val = split_documents(load_jsonl(path))
    assert len(train) + len(val) == 4
    assert not set(train).intersection(val)
    assert tokenize_documents(train)

def test_bad_input(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"wrong": 1}\n')
    with pytest.raises(ValueError):
        load_jsonl(path)
