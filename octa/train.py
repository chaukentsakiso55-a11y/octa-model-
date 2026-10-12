"""Small experimental training entry point. Requires a UTF-8 text file."""
import argparse
import json
from pathlib import Path
import torch
from .model import OctaConfig, OctaForCausalLM

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--text", required=True)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--out", default="checkpoints/octa-tiny.pt")
    args = p.parse_args()
    if args.steps < 1:
        p.error("--steps must be positive")
    # Byte-level vocabulary is deliberately simple for a first end-to-end experiment.
    cfg = OctaConfig(vocab_size=256, d_model=128, n_heads=4, n_layers=2, context_length=128)
    raw = Path(args.text).read_bytes()
    if len(raw) < 2:
        raise ValueError("Training text must contain at least 2 bytes")
    data = torch.tensor(list(raw), dtype=torch.long)
    model = OctaForCausalLM(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    model.train()
    for step in range(args.steps):
        length = min(cfg.context_length, len(data))
        start = torch.randint(0, len(data) - length + 1, (1,)).item()
        batch = data[start:start + length].unsqueeze(0)
        loss = model(batch, labels=batch)["loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 10 == 0:
            print(f"step={step} loss={loss.item():.4f}")
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": cfg.__dict__, "state_dict": model.state_dict()}, output)
    print(f"Saved {output}")

if __name__ == "__main__":
    main()
