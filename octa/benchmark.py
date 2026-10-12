"""Evaluate OCTA against a uniform byte-prediction baseline on held-out text."""
import argparse
import math
import torch
from .model import OctaConfig, OctaForCausalLM
from .train_v2 import load_data, batch_from

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--text", required=True)
    p.add_argument("--batches", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4)
    args = p.parse_args()
    if args.batches < 1 or args.batch_size < 1:
        p.error("batches and batch-size must be positive")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    cfg = OctaConfig(**checkpoint["config"])
    model = OctaForCausalLM(cfg).to(device)
    model.load_state_dict(checkpoint.get("model", checkpoint.get("state_dict")))
    _, validation = load_data(args.text, 42)
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(args.batches):
            x, _ = batch_from(validation, cfg.context_length, args.batch_size, device)
            losses.append(model(x, labels=x)["loss"].item())
    loss = sum(losses) / len(losses)
    baseline = math.log(cfg.vocab_size)
    print(f"OCTA held-out loss: {loss:.4f}")
    print(f"OCTA perplexity: {math.exp(min(loss, 30)):.2f}")
    print(f"Uniform-token baseline loss: {baseline:.4f}")
    print(f"Improvement over uniform baseline (positive is better): {baseline - loss:.4f}")
    print("Note: a uniform baseline is not a comparison against Qwen, GPT, or human performance.")

if __name__ == "__main__":
    main()
