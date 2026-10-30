"""Pinned reference for tur's multi-token prediction head over routed experts.

Transformers 5.22.1, torch 2.15.0, CPU F32. Transformers drops `mtp.*` tensors, so the head
is assembled from its own Qwen3.5 MoE modules, with the conventions
scripts/qwen3_5_mtp_reference.py established on the real dense checkpoint:

    h'_p = norm(layer(fc([norm_e(embed(x_{p+1})), norm_h(h_p)])))   predicts x_{p+2}

Random weights: an offline numerical oracle, not evidence that a real head predicts text.
  python -I scripts/qwen3_5_moe_mtp_reference.py crates/tur/tests/fixtures/qwen3_5_moe_mtp
"""
import json
import sys
from pathlib import Path

import torch
import transformers
from safetensors.torch import save_file
from transformers import Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeDecoderLayer, Qwen3_5MoeRMSNorm

CONFIG = {
    "architectures": ["Qwen3_5MoeForCausalLM"], "model_type": "qwen3_5_moe_text",
    "hidden_size": 64, "vocab_size": 96, "num_hidden_layers": 2,
    "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 64,
    "linear_num_key_heads": 2, "linear_num_value_heads": 4, "linear_key_head_dim": 64,
    "linear_value_head_dim": 64, "linear_conv_kernel_dim": 4, "max_position_embeddings": 512,
    "rms_norm_eps": 1e-6, "hidden_act": "silu", "attention_bias": False, "attention_dropout": 0.0,
    "attn_output_gate": True, "tie_word_embeddings": False, "eos_token_id": 1,
    "num_experts": 4, "num_experts_per_tok": 2, "moe_intermediate_size": 64,
    "shared_expert_intermediate_size": 64, "layer_types": ["linear_attention", "full_attention"],
    "mtp_num_hidden_layers": 1, "mtp_use_dedicated_embeddings": False,
    "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25},
}


@torch.no_grad()
def main():
    output = Path(sys.argv[1])
    output.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(35)
    settings = {k: v for k, v in CONFIG.items() if k not in ("model_type", "architectures", "mtp_num_hidden_layers", "mtp_use_dedicated_embeddings")}
    config = Qwen3_5MoeTextConfig(**settings)
    config._attn_implementation = "eager"

    def randomize(module):
        for name, tensor in sorted(module.state_dict().items()):
            scale = 0.1 if "norm" in name else 0.05
            value = torch.randn(tensor.shape, generator=generator) * scale
            tensor.copy_(value.bfloat16().float())

    model = Qwen3_5MoeForCausalLM(config).eval()
    randomize(model)
    hidden = config.hidden_size
    fc = torch.nn.Linear(2 * hidden, hidden, bias=False)
    norm_e = Qwen3_5MoeRMSNorm(hidden, eps=config.rms_norm_eps)
    norm_h = Qwen3_5MoeRMSNorm(hidden, eps=config.rms_norm_eps)
    layer = Qwen3_5MoeDecoderLayer(config, config.layer_types.index("full_attention")).eval()
    norm = Qwen3_5MoeRMSNorm(hidden, eps=config.rms_norm_eps)
    parts = {
        "mtp.fc.": fc, "mtp.pre_fc_norm_embedding.": norm_e, "mtp.pre_fc_norm_hidden.": norm_h,
        "mtp.layers.0.": layer, "mtp.norm.": norm,
    }
    weights = dict(model.state_dict())
    for prefix, module in parts.items():
        randomize(module)
        weights.update({prefix + k: v for k, v in module.state_dict().items()})
    ids = torch.randint(2, config.vocab_size, (1, 18), generator=generator)
    count = ids.shape[1] - 1
    text = model.model
    states = text(input_ids=ids[:, :count]).last_hidden_state
    x = fc(torch.cat([norm_e(text.embed_tokens(ids[:, 1:])), norm_h(states)], dim=-1))
    positions = torch.arange(count)[None]
    mask = torch.full((count, count), float("-inf")).triu(1)[None, None]
    rotary = text.rotary_emb(x, positions[None].expand(3, 1, -1))
    out = layer(x, position_embeddings=rotary, attention_mask=mask, position_ids=positions)
    out = out[0] if isinstance(out, tuple) else out
    logits = model.lm_head(norm(out))[0]
    (output / "config.json").write_text(json.dumps(CONFIG, indent=2) + "\n")
    save_file({k: v.bfloat16().contiguous() for k, v in weights.items()}, output / "model.safetensors", metadata={"format": "pt"})
    save_file(
        {"tokens": ids[0].to(torch.int64), "logits": logits.contiguous()}, output / "golden.safetensors",
        metadata={"transformers": transformers.__version__, "torch": torch.__version__},
    )


if __name__ == "__main__":
    main()
