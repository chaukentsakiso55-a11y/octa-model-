import torch
from octa.model import OctaConfig, OctaForCausalLM
from octa.train_gpu import evaluate

def test_cpu_training_and_validation():
    cfg = OctaConfig(vocab_size=256, d_model=32, n_heads=4, n_layers=1, context_length=16)
    model = OctaForCausalLM(cfg)
    data = torch.randint(0, 256, (200,), dtype=torch.long)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    opt.zero_grad()
    x = data[:16].unsqueeze(0)
    loss = model(x, labels=x)["loss"]
    loss.backward()
    opt.step()
    assert torch.isfinite(loss)
    assert evaluate(model, data, cfg, "cpu", batches=2, batch_size=2) >= 0
