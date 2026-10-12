"""OCTA: independently initialized decoder-only language model."""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F

@dataclass
class OctaConfig:
    vocab_size: int = 8192
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    context_length: int = 512
    dropout: float = 0.0

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.dropout = cfg.dropout

    def forward(self, x):
        b, t, c = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [z.view(b, t, self.n_heads, c // self.n_heads).transpose(1, 2) for z in (q, k, v)]
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0)
        return self.proj(y.transpose(1, 2).contiguous().view(b, t, c))

class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = nn.RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = nn.RMSNorm(cfg.d_model)
        self.mlp = nn.Sequential(nn.Linear(cfg.d_model, 4 * cfg.d_model), nn.SiLU(), nn.Linear(4 * cfg.d_model, cfg.d_model))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))

class OctaForCausalLM(nn.Module):
    def __init__(self, cfg: OctaConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.context_length, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = nn.RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.embed.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, input_ids, labels=None):
        b, t = input_ids.shape
        if t > self.cfg.context_length:
            raise ValueError("Sequence exceeds OCTA context length")
        positions = torch.arange(t, device=input_ids.device)
        x = self.embed(input_ids) + self.pos(positions)[None, :, :]
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.norm(x))
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, self.cfg.vocab_size), labels[:, 1:].reshape(-1), ignore_index=-100) if labels is not None and t > 1 else None
        return {"logits": logits, "loss": loss}

    @torch.no_grad()
    def generate(self, ids, max_new_tokens=32, temperature=1.0):
        self.eval()
        for _ in range(max_new_tokens):
            logits = self(ids[:, -self.cfg.context_length:])["logits"][:, -1] / max(temperature, 1e-5)
            next_id = torch.multinomial(F.softmax(logits, dim=-1), 1)
            ids = torch.cat([ids, next_id], dim=1)
        return ids
