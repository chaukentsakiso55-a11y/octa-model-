from octa.model import OctaConfig
from octa.scale_plan import count_parameters, estimate, proposed_7b_config

def test_tiny_matches_actual_parameters():
    from octa.model import OctaForCausalLM
    cfg = OctaConfig(vocab_size=256, d_model=128, n_heads=4, n_layers=2, context_length=128)
    model = OctaForCausalLM(cfg)
    assert count_parameters(cfg) == sum(p.numel() for p in model.parameters())

def test_proposed_scale_is_near_seven_billion():
    result = estimate(proposed_7b_config())
    assert 6_500_000_000 <= result.parameters <= 7_500_000_000

def test_storage_estimates():
    result = estimate(proposed_7b_config())
    assert result.weight_gb_fp32 == 2 * result.weight_gb_bf16
