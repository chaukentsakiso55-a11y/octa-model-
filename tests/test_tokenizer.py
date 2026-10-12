import pytest
from octa.tokenizer import ByteTokenizer

@pytest.mark.parametrize("text", ["Hello OCTA", "Sawubona!", "你好", "🤖", ""])
def test_round_trip(text):
    tokenizer = ByteTokenizer()
    assert tokenizer.decode(tokenizer.encode(text)) == text

def test_invalid_tokens():
    with pytest.raises(ValueError):
        ByteTokenizer().decode([256])
