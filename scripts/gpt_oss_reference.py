"""Pinned gpt-oss references: Transformers 5.22.1, torch 2.15.0, CPU F32.

The tiny fixture is an offline numerical oracle, not evidence that a real model works. Its
weights are random but stored in the published layout: packed MXFP4 expert blocks beside BF16
attention, router and embedding tensors. Transformers' own decoder unpacks them.
  python -I scripts/gpt_oss_reference.py --tiny crates/tur/tests/fixtures/gpt_oss
The real checkpoint runs Transformers' decoder layers one at a time, so its 76 GB of decoded
F32 experts never share memory (immutable openai/gpt-oss-20b revision):
  python -I scripts/gpt_oss_reference.py --model <dir> --golden <file>
"""
import argparse
import json
from pathlib import Path

import torch
import transformers
from safetensors.torch import save_file
from safetensors import safe_open
from transformers import AutoTokenizer, GptOssConfig, GptOssForCausalLM
from transformers.integrations.mxfp4 import convert_moe_packed_tensors
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssDecoderLayer, GptOssRMSNorm, GptOssRotaryEmbedding

REVISION = "6cee5e81ee83917806bbde320786a8fb61efebee"
PROMPT = (
    "fn main() {\n    let primes: Vec<u32> = (2..50).filter(|n| (2..*n).all(|d| n % d != 0)).collect();\n"
    '    println!("{primes:?}");\n}\n\nThe capital of France is Paris, and the capital of Japan is'
)


def tiny_config():
    return dict(
        architectures=["GptOssForCausalLM"], model_type="gpt_oss",
        attention_bias=True, attention_dropout=0.0, eos_token_id=1, pad_token_id=0,
        experts_per_token=2, head_dim=64, hidden_act="silu", hidden_size=64, intermediate_size=64,
        layer_types=["sliding_attention", "full_attention"] * 2, max_position_embeddings=2048,
        num_attention_heads=4, num_experts_per_tok=2, num_hidden_layers=4, num_key_value_heads=2,
        num_local_experts=4, output_router_logits=False, rms_norm_eps=1e-5,
        router_aux_loss_coef=0.9, sliding_window=4, swiglu_limit=7.0, tie_word_embeddings=False,
        use_cache=True, vocab_size=96, rope_theta=150000,
        quantization_config=dict(quant_method="mxfp4", modules_to_not_convert=[]),
        rope_scaling=dict(
            beta_fast=32.0, beta_slow=1.0, factor=32.0, original_max_position_embeddings=64,
            rope_type="yarn", truncate=False,
        ),
    )


def tiny_weights(config, generator):
    def normal(*shape, scale=0.03):
        return (torch.randn(*shape, generator=generator) * scale).to(torch.bfloat16)

    hidden, inner, heads = config["hidden_size"], config["intermediate_size"], config["num_attention_heads"]
    kv, dim, experts = config["num_key_value_heads"], config["head_dim"], config["num_local_experts"]
    vocab = config["vocab_size"]

    def packed(name, rows):
        groups = (hidden if name == "gate_up_proj" else inner) // 32
        blocks = torch.randint(0, 256, (experts, rows, groups, 16), generator=generator, dtype=torch.uint8)
        scales = torch.randint(118, 124, blocks.shape[:-1], generator=generator, dtype=torch.uint8)
        return {f"{name}_blocks": blocks, f"{name}_scales": scales}

    data = {
        "model.embed_tokens.weight": normal(vocab, hidden, scale=0.5),
        "model.norm.weight": (1 + normal(hidden)).to(torch.bfloat16),
        "lm_head.weight": normal(vocab, hidden),
    }
    for i in range(config["num_hidden_layers"]):
        p = f"model.layers.{i}"
        data[f"{p}.input_layernorm.weight"] = (1 + normal(hidden)).to(torch.bfloat16)
        data[f"{p}.post_attention_layernorm.weight"] = (1 + normal(hidden)).to(torch.bfloat16)
        data[f"{p}.self_attn.q_proj.weight"] = normal(heads * dim, hidden)
        data[f"{p}.self_attn.q_proj.bias"] = normal(heads * dim)
        for name in ("k_proj", "v_proj"):
            data[f"{p}.self_attn.{name}.weight"] = normal(kv * dim, hidden)
            data[f"{p}.self_attn.{name}.bias"] = normal(kv * dim)
        data[f"{p}.self_attn.o_proj.weight"] = normal(hidden, heads * dim)
        data[f"{p}.self_attn.o_proj.bias"] = normal(hidden)
        data[f"{p}.self_attn.sinks"] = normal(heads, scale=1.0)
        data[f"{p}.mlp.router.weight"] = normal(experts, hidden, scale=0.5)
        data[f"{p}.mlp.router.bias"] = normal(experts)
        for name, rows, bias in (("gate_up_proj", 2 * inner, 2 * inner), ("down_proj", hidden, hidden)):
            for key, value in packed(name, rows).items():
                data[f"{p}.mlp.experts.{key}"] = value
            data[f"{p}.mlp.experts.{name}_bias"] = normal(experts, bias)
    return data


def f32_state(data):
    """Published tensors as the F32 parameters Transformers multiplies by."""
    state = {}
    for name, value in data.items():
        if name.endswith("_blocks"):
            base = name.removesuffix("_blocks")
            state[base] = convert_moe_packed_tensors(value, data[f"{base}_scales"], dtype=torch.float32)
        elif not name.endswith("_scales"):
            state[name] = value.float()
    return state


@torch.inference_mode()
def golden(config, data, tokens):
    settings = {k: v for k, v in config.items() if k != "quantization_config"}
    settings["attn_implementation"] = "eager"
    model = GptOssForCausalLM(GptOssConfig(**settings)).float().eval()
    model.load_state_dict(f32_state(data), strict=True)
    result = {"tokens": tokens.to(torch.int64)}
    handles = []
    for index, layer in enumerate(model.model.layers):
        def record(_module, _inputs, output, index=index):
            result[f"layer.{index}"] = output.detach().float().squeeze(0).contiguous()
        handles.append(layer.register_forward_hook(record))
    result["logits"] = model(input_ids=tokens.unsqueeze(0), use_cache=False).logits.squeeze(0).contiguous()
    for handle in handles:
        handle.remove()
    rotary = model.model.rotary_emb
    result["rope_inv_freq"] = rotary.inv_freq.detach().float().contiguous()
    result["rope_scaling"] = torch.tensor([float(rotary.attention_scaling)], dtype=torch.float32)
    return result


def tiny(output):
    output.mkdir(parents=True, exist_ok=True)
    config = tiny_config()
    generator = torch.Generator().manual_seed(20)
    data = tiny_weights(config, generator)
    tokens = torch.randint(2, config["vocab_size"], (24,), generator=generator)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    save_file({k: v.contiguous() for k, v in data.items()}, output / "model.safetensors", metadata={"format": "pt"})
    save_file(
        golden(config, data, tokens), output / "golden.safetensors",
        metadata={"transformers": transformers.__version__, "torch": torch.__version__},
    )


def causal_mask(tokens, window=None):
    """The 4-D additive mask Transformers' eager attention adds to its scores."""
    allowed = torch.ones(tokens, tokens).tril()
    if window is not None:
        allowed = allowed.triu(-window + 1)
    return torch.where(allowed > 0, 0.0, -torch.inf)[None, None]


class Tensors:
    """Every tensor of a checkpoint by name, read on demand."""

    def __init__(self, directory):
        index = json.loads((directory / "model.safetensors.index.json").read_text())["weight_map"]
        self.files = {name: safe_open(directory / file, "pt") for name, file in index.items()}

    def get(self, name):
        return self.files[name].get_tensor(name)

    def layer(self, index):
        prefix = f"model.layers.{index}."
        return {n[len(prefix):]: self.get(n) for n in self.files if n.startswith(prefix)}


@torch.inference_mode()
def real(directory, output):
    settings = json.loads((directory / "config.json").read_text())
    settings.pop("quantization_config")
    settings["attn_implementation"] = "eager"
    config = GptOssConfig(**settings)
    tensors = Tensors(directory)
    ids = AutoTokenizer.from_pretrained(directory)(PROMPT, return_tensors="pt").input_ids
    count = ids.shape[1]
    embed = tensors.get("model.embed_tokens.weight").float()
    state = embed[ids]
    del embed
    positions = torch.arange(count)[None]
    rotary = GptOssRotaryEmbedding(config)
    position_embeddings = rotary(state, positions)
    result = {"tokens": ids[0].to(torch.int64)}
    for index, kind in enumerate(config.layer_types):
        layer = GptOssDecoderLayer(config, index).float().eval()
        layer.load_state_dict(f32_state(tensors.layer(index)), strict=True)
        mask = causal_mask(count, config.sliding_window if kind == "sliding_attention" else None)
        state = layer(state, attention_mask=mask, position_embeddings=position_embeddings, position_ids=positions)
        del layer
        result[f"layer.{index}"] = state.squeeze(0).contiguous().clone()
        print(f"layer {index}", flush=True)
    norm = GptOssRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    norm.weight.data = tensors.get("model.norm.weight").float()
    head = tensors.get("lm_head.weight").float()
    result["logits"] = (norm(state) @ head.T).squeeze(0).contiguous()
    save_file(result, output, metadata={
        "transformers": transformers.__version__, "torch": torch.__version__, "revision": REVISION,
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiny", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--golden", type=Path)
    args = parser.parse_args()
    if args.tiny:
        tiny(args.tiny)
    else:
        real(args.model, args.golden)


if __name__ == "__main__":
    main()
