import torch
from octa.model import OctaConfig, OctaForCausalLM

def test_forward_backward_and_generation():
    cfg = OctaConfig(vocab_size=64, d_model=32, n_heads=4, n_layers=2, context_length=16)
    model = OctaForCausalLM(cfg)
    ids = torch.randint(0, 64, (2, 8))
    output = model(ids, labels=ids)
    assert output["logits"].shape == (2, 8, 64)
    assert torch.isfinite(output["loss"])
    output["loss"].backward()
    assert model.embed.weight.grad is not None
    generated = model.generate(ids[:1], max_new_tokens=2)
    assert generated.shape == (1, 10)
