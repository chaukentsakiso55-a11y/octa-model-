"""Reproducible OCTA byte-level training experiment.

Example:
 python -m octa.train_v2 --text data/authorized.txt --steps 200 --eval-every 20
 python -m octa.train_v2 --text data/authorized.txt --steps 300 --resume checkpoints/octa-v2.pt
"""
import argparse
import math
import random
from pathlib import Path

import torch

from .model import OctaConfig, OctaForCausalLM


def load_data(path: str, seed: int, validation_fraction: float = 0.1):
    raw = Path(path).read_bytes()
    if len(raw) < 128:
        raise ValueError("Provide at least 128 bytes of authorized training text")
    # Split by contiguous chunks, not overlapping windows across partitions.
    cut = int(len(raw) * (1 - validation_fraction))
    if min(cut, len(raw) - cut) < 2:
        raise ValueError("Both training and validation splits need at least two bytes")
    return torch.tensor(list(raw[:cut]), dtype=torch.long), torch.tensor(list(raw[cut:]), dtype=torch.long)


def batch_from(data, context, batch_size, device):
    # Require next-token targets and never sample outside this split.
    length = min(context + 1, len(data))
    if length < 2:
        raise ValueError("Not enough tokens for a next-token batch")
    starts = torch.randint(0, len(data) - length + 1, (batch_size,))
    windows = torch.stack([data[i:i + length] for i in starts])
    return windows[:, :-1].to(device), windows[:, 1:].to(device)


@torch.no_grad()
def evaluate(model, data, cfg, batch_size, eval_batches, device):
    model.eval()
    losses = []
    for _ in range(eval_batches):
        x, y = batch_from(data, cfg.context_length, batch_size, device)
        # model's shifted loss expects labels aligned with input sequence.
        # Include one extra input token to predict next token at each position.
        tokens = torch.cat((x, y[:, -1:]), dim=1)
        if tokens.shape[1] > cfg.context_length:
            tokens = tokens[:, :cfg.context_length]
        output = model(tokens, labels=tokens)
        losses.append(output["loss"].item())
    model.train()
    return sum(losses) / len(losses)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--text", required=True)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--eval-batches", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume")
    p.add_argument("--out", default="checkpoints/octa-v2.pt")
    args = p.parse_args()
    if min(args.steps, args.batch_size, args.eval_every, args.eval_batches) < 1:
        p.error("steps, batch-size, eval-every and eval-batches must be positive")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train, val = load_data(args.text, args.seed)
    cfg = OctaConfig(vocab_size=256, d_model=128, n_heads=4, n_layers=2, context_length=64)
    model = OctaForCausalLM(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    start_step = 0
    if args.resume:
        # Only load checkpoints you created or trust: torch.load uses pickle.
        checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
        cfg = OctaConfig(**checkpoint["config"])
        model = OctaForCausalLM(cfg).to(device)
        model.load_state_dict(checkpoint["model"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"])
        torch.set_rng_state(checkpoint["rng_cpu"].cpu())
        if device == "cuda" and checkpoint.get("rng_cuda") is not None:
            torch.cuda.set_rng_state(checkpoint["rng_cuda"].cpu())
    model.train()
    for step in range(start_step + 1, args.steps + 1):
        x, _ = batch_from(train, cfg.context_length - 1, args.batch_size, device)
        optimizer.zero_grad(set_to_none=True)
        loss = model(x, labels=x)["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % args.eval_every == 0 or step == args.steps:
            validation_loss = evaluate(model, val, cfg, args.batch_size, args.eval_batches, device)
            print(f"step={step} train_loss={loss.item():.4f} val_loss={validation_loss:.4f} val_ppl={math.exp(min(validation_loss, 30)):.2f}", flush=True)
            path = Path(args.out)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "config": cfg.__dict__, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "step": step,
                "rng_cpu": torch.get_rng_state(),
                "rng_cuda": torch.cuda.get_rng_state() if device == "cuda" else None,
            }, path)
    print(f"Checkpoint: {args.out}")


if __name__ == "__main__":
    main()
