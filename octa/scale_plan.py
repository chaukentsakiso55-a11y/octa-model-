from dataclasses import dataclass
from .model import OctaConfig

@dataclass(frozen=True)
class ScaleEstimate:
    parameters: int
    weight_gb_bf16: float
    weight_gb_fp32: float

def count_parameters(cfg: OctaConfig) -> int:
    d = cfg.d_model
    v = cfg.vocab_size
    l = cfg.n_layers
    c = cfg.context_length
    embedding = v * d + c * d
    attention = 4 * d * d + 4 * d
    mlp = 8 * d * d + 5 * d
    norms = 2 * d
    final_norm = d
    return embedding + l * (attention + mlp + norms) + final_norm

def estimate(cfg: OctaConfig) -> ScaleEstimate:
    if min(cfg.vocab_size, cfg.d_model, cfg.n_heads, cfg.n_layers, cfg.context_length) <= 0:
        raise ValueError("All configuration dimensions must be positive")
    if cfg.d_model % cfg.n_heads:
        raise ValueError("d_model must be divisible by n_heads")
    n = count_parameters(cfg)
    return ScaleEstimate(n, n * 2 / 1e9, n * 4 / 1e9)

def proposed_7b_config() -> OctaConfig:
    return OctaConfig(vocab_size=65536, d_model=4096, n_heads=32, n_layers=34, context_length=4096)
