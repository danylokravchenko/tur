"""Regenerate the offline P8 MoE oracle with Transformers 5.20.2 / torch 2.14.1.

Weights use the same deterministic LCG and BF16 conversion as tests/common/mod.rs.
No weights or network access are needed. All layers and final logits come from the
independent Transformers Qwen3_5MoeForCausalLM on CPU in F32.

python scripts/qwen3_5_moe_reference.py
"""
import json
from pathlib import Path

import torch
import transformers
from safetensors.torch import save_file
from transformers import Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "crates/tur-engine/tests/fixtures"


def values(count, seed, scale):
    state = (seed * 6364136223846793005 + 1) & ((1 << 64) - 1)
    result = []
    for _ in range(count):
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        result.append(state >> 40)
    return ((torch.tensor(result, dtype=torch.float32) / (1 << 24)) * 2 - 1) * scale


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    raw = json.loads((FIXTURES / "qwen3_5_moe_config.json").read_text())
    config = Qwen3_5MoeTextConfig(**{k: v for k, v in raw.items() if k not in ("model_type", "architectures")})
    config._attn_implementation = "eager"
    model = Qwen3_5MoeForCausalLM(config).eval()
    for seed, (name, tensor) in enumerate(sorted(model.state_dict().items()), 1):
        scale = 0.1 if "norm" in name else 0.03
        tensor.copy_(values(tensor.numel(), seed, scale).reshape(tensor.shape).bfloat16().float())
    ids = torch.tensor([[2 + (i * 37 + 11) % 88 for i in range(19)]])
    hidden = []
    hooks = [layer.register_forward_hook(lambda _m, _i, o: hidden.append(o[0] if isinstance(o, tuple) else o)) for layer in model.model.layers]
    output = model(ids, use_cache=False).logits[0]
    for hook in hooks:
        hook.remove()
    save_file({"ids": ids[0].to(torch.int64), "hidden": torch.stack([h[0] for h in hidden]).contiguous(), "logits": output.contiguous()}, FIXTURES / "qwen3_5_moe_golden.safetensors")
    (FIXTURES / "qwen3_5_moe_golden.json").write_text(json.dumps({
        "transformers": transformers.__version__, "torch": torch.__version__, "device": "cpu", "dtype": "float32",
        "weight_format": "BF16 values promoted to F32", "scale": 0.03, "norm_scale": 0.1,
        "upstream_reference": "ea52711b4850c332be3f778396ce882ab84a9011",
        "shapes": {k: list(v.shape) for k, v in sorted(model.state_dict().items())},
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
