import pytest
import torch
from octa.model import OctaConfig, OctaForCausalLM
from octa.tokenizer import ByteTokenizer


def test_tokenizer_round_trip():
    tokenizer = ByteTokenizer()
    text = "Octa AI 🌍 multilingual"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_causal_model_forward_and_loss():
    cfg = OctaConfig(vocab_size=256, d_model=32, n_heads=4, n_layers=2, context_length=16)
    model = OctaForCausalLM(cfg)
    ids = torch.randint(0, 256, (2, 8))
    result = model(ids, labels=ids)
    assert result["logits"].shape == (2, 8, 256)
    assert torch.isfinite(result["loss"])


def test_context_length_rejected():
    cfg = OctaConfig(vocab_size=256, d_model=32, n_heads=4, n_layers=1, context_length=8)
    model = OctaForCausalLM(cfg)
    with pytest.raises(ValueError):
        model(torch.zeros((1, 9), dtype=torch.long))


def test_generation_preserves_prompt():
    torch.manual_seed(42)
    cfg = OctaConfig(vocab_size=256, d_model=32, n_heads=4, n_layers=1, context_length=16)
    model = OctaForCausalLM(cfg)
    prompt = torch.tensor([[79, 67, 84, 65]], dtype=torch.long)
    result = model.generate(prompt, max_new_tokens=3)
    assert result.shape == (1, 7)
    assert torch.equal(result[:, :4], prompt)
    assert torch.all((result >= 0) & (result < 256))
