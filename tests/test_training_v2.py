import torch
from octa.train_v2 import batch_from, load_data, evaluate
from octa.model import OctaConfig, OctaForCausalLM

def test_data_split_and_training(tmp_path):
    p = tmp_path / "authorized.txt"
    p.write_bytes(bytes(range(256)) * 3)
    train, val = load_data(str(p), 42)
    assert len(train) + len(val) == 768
    x, y = batch_from(train, 16, 2, "cpu")
    assert x.shape == (2, 16)
    assert torch.equal(x[:, 1:], y[:, :-1])
    cfg = OctaConfig(vocab_size=256, d_model=32, n_heads=4, n_layers=1, context_length=16)
    model = OctaForCausalLM(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = model(x, labels=x)["loss"]
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss)
    assert evaluate(model, val, cfg, 2, 2, "cpu") >= 0
