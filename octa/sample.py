"""Generate raw UTF-8 bytes from a trained OCTA byte-level checkpoint."""
import argparse
import torch
from .model import OctaConfig, OctaForCausalLM

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/octa-v2.pt")
    parser.add_argument("--prompt", default="OCTA")
    parser.add_argument("--tokens", type=int, default=100)
    args = parser.parse_args()
    if args.tokens < 1:
        parser.error("--tokens must be positive")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = OctaForCausalLM(OctaConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt.get("model", ckpt.get("state_dict")))
    if model.cfg.vocab_size != 256:
        raise ValueError("This sampler requires a byte-level 256-token model")
    prompt = args.prompt.encode("utf-8") or b" "
    ids = torch.tensor([list(prompt)], dtype=torch.long, device=device)
    result = model.generate(ids, max_new_tokens=args.tokens)
    print(bytes(result[0].tolist()).decode("utf-8", errors="replace"))

if __name__ == "__main__":
    main()
