"""OCTA 0.7 document-level training experiment."""
import argparse
import hashlib
import json
import math
import random
from pathlib import Path
import torch
from .documents import load_jsonl, split_documents, tokenize_documents
from .model import OctaConfig, OctaForCausalLM

def windows(documents, context):
    for doc in documents:
        for start in range(0, len(doc) - 1, context):
            chunk = doc[start:start + context]
            if len(chunk) >= 2:
                yield torch.tensor(chunk, dtype=torch.long)

@torch.no_grad()
def validation_loss(model, documents, context):
    model.eval()
    nll, targets = 0.0, 0
    for tokens in windows(documents, context):
        n = len(tokens) - 1
        nll += model(tokens.unsqueeze(0), labels=tokens)["loss"].item() * n
        targets += n
    if not targets:
        raise ValueError("Validation documents have no token pairs")
    return nll / targets

def run(args):
    if args.steps < 1 or args.context < 2 or args.batch_size < 1 or args.width % 4:
        raise ValueError("Invalid training parameters")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    raw = Path(args.data).read_bytes()
    training_docs, validation_docs = split_documents(load_jsonl(args.data), args.seed)
    train = [d for d in tokenize_documents(training_docs) if len(d) >= args.context]
    val = tokenize_documents(validation_docs)
    if not train or not val:
        raise ValueError("Need a train document >= context bytes and a validation document >= 2 bytes")
    cfg = OctaConfig(vocab_size=256, d_model=args.width, n_heads=4, n_layers=1, context_length=args.context)
    model = OctaForCausalLM(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
    initial = validation_loss(model, val, args.context)
    for _ in range(args.steps):
        sequences = []
        for _ in range(args.batch_size):
            doc = random.choice(train)
            start = random.randrange(len(doc) - args.context + 1)
            sequences.append(doc[start:start + args.context])
        x = torch.tensor(sequences, dtype=torch.long)
        loss = model(x, labels=x)["loss"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    final = validation_loss(model, val, args.context)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": cfg.__dict__, "model": model.state_dict(),
                "step": args.steps, "data_sha256": hashlib.sha256(raw).hexdigest()}, output)
    result = {"training_documents": len(training_docs), "validation_documents": len(validation_docs),
              "initial_validation_loss": round(initial, 5), "final_validation_loss": round(final, 5),
              "final_validation_perplexity": round(math.exp(min(final, 30)), 5),
              "steps": args.steps, "checkpoint": str(output)}
    print(json.dumps(result, indent=2))
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--context", type=int, default=32)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="checkpoints/octa-0.7.pt")
    run(parser.parse_args())

if __name__ == "__main__":
    main()
