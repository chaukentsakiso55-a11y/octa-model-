"""Reproducible OCTA 0.6 learning smoke test; synthetic text, no external data."""
import math
import torch
from pathlib import Path
from .model import OctaConfig, OctaForCausalLM

def main():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    cfg = OctaConfig(vocab_size=256, d_model=32, n_heads=4, n_layers=1, context_length=32)
    model = OctaForCausalLM(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    raw = b"OCTA learns patterns. OCTA learns patterns. " * 150
    cut = int(len(raw) * 0.9)
    train = torch.tensor(list(raw[:cut]), dtype=torch.long)
    val = torch.tensor(list(raw[cut:]), dtype=torch.long)
    rng = torch.Generator().manual_seed(42)

    def batch(data):
        starts = torch.randint(0, len(data) - cfg.context_length + 1, (8,), generator=rng)
        return torch.stack([data[i:i + cfg.context_length] for i in starts])

    model.eval()
    with torch.no_grad():
        x = batch(val)
        before = model(x, labels=x)["loss"].item()
    model.train()
    for step in range(100):
        x = batch(train)
        loss = model(x, labels=x)["loss"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        losses = []
        for _ in range(10):
            x = batch(val)
            losses.append(model(x, labels=x)["loss"].item())
    after = sum(losses) / len(losses)
    output = Path("checkpoints/octa-0.6-synthetic.pt")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": cfg.__dict__, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "step": 100}, output)
    print(f"Parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Initial loss: {before:.4f}")
    print(f"Validation loss: {after:.4f}; perplexity: {math.exp(after):.4f}")
    print(f"Checkpoint saved: {output}")
    assert after < before, "Synthetic training did not improve validation loss"

if __name__ == "__main__":
    main()
