"""OCTA 0.4 GPU-capable byte-level training experiment.

Usage: python -m octa.train_gpu --text data/authorized.txt --steps 200
Resume: python -m octa.train_gpu --text data/authorized.txt --steps 400 --resume checkpoints/octa-gpu.pt
"""
import argparse
import hashlib
import math
import random
from pathlib import Path
import torch
from .model import OctaConfig, OctaForCausalLM
from .train_v2 import load_data, batch_from

def evaluate(model, data, cfg, device, batches=5, batch_size=4):
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(batches):
            x, _ = batch_from(data, cfg.context_length, batch_size, device)
            losses.append(model(x, labels=x)["loss"].item())
    model.train()
    return sum(losses) / len(losses)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--text", required=True)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--accumulation", type=int, default=4)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--out", default="checkpoints/octa-gpu.pt")
    p.add_argument("--resume")
    args = p.parse_args()
    if min(args.steps, args.batch_size, args.accumulation, args.eval_every) < 1 or args.lr <= 0:
        p.error("Steps, batch size, accumulation, eval frequency and learning rate must be positive")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train, val = load_data(args.text, args.seed)
    dataset_hash = hashlib.sha256(Path(args.text).read_bytes()).hexdigest()
    cfg = OctaConfig(vocab_size=256, d_model=128, n_heads=4, n_layers=2, context_length=64)
    model = OctaForCausalLM(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    start = 0
    if args.resume:
        # Load checkpoints only from trusted sources.
        state = torch.load(args.resume, map_location=device, weights_only=True)
        if state["dataset_hash"] != dataset_hash:
            raise ValueError("Dataset changed since checkpoint was created")
        cfg = OctaConfig(**state["config"])
        model = OctaForCausalLM(cfg).to(device)
        model.load_state_dict(state["model"])
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
        opt.load_state_dict(state["optimizer"])
        start = int(state["step"])
        torch.set_rng_state(state["rng_cpu"].cpu())
        if device == "cuda" and state.get("rng_cuda") is not None:
            torch.cuda.set_rng_state(state["rng_cuda"].cpu())
    for step in range(start + 1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        running = 0.0
        for _ in range(args.accumulation):
            x, _ = batch_from(train, cfg.context_length, args.batch_size, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda" and torch.cuda.is_bf16_supported()):
                loss = model(x, labels=x)
                scaled_loss = loss["loss"] / args.accumulation
            scaled_loss.backward()
            running += loss["loss"].item() / args.accumulation
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % args.eval_every == 0 or step == args.steps:
            val_loss = evaluate(model, val, cfg, device)
            print(f"step={step} train_loss={running:.4f} val_loss={val_loss:.4f} val_ppl={math.exp(min(val_loss, 30)):.2f}", flush=True)
            path = Path(args.out)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "config": cfg.__dict__, "model": model.state_dict(),
                "optimizer": opt.state_dict(), "step": step,
                "dataset_hash": dataset_hash,
                "rng_cpu": torch.get_rng_state(),
                "rng_cuda": torch.cuda.get_rng_state() if device == "cuda" else None,
            }, path)
    print(f"Saved {args.out} on {device}")

if __name__ == "__main__":
    main()
