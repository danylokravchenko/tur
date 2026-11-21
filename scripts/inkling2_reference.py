"""Pinned Inkling 2 references: Transformers 5.24.0, torch 2.15.0, CPU F32.

The tiny fixture is an offline numerical oracle, not evidence that the real model works. Its
weights are random but stored in the published layout: original tensor names, NVFP4 experts in
the early layers and MXFP8 experts in the late ones, a padded unembedding. Transformers' own
loader renames and dequantizes them, and its cache produces the decode logits.
  python -I scripts/inkling2_reference.py --tiny crates/tur/tests/fixtures/inkling2
The real checkpoint runs one decoder layer at a time, so its decoded F32 experts (about 4 GB a
layer) never share memory (immutable thinkingmachines/Inkling-2-Mini revision):
  python -I scripts/inkling2_reference.py --model <dir> --golden <file>
The tiny mode also runs that layer-at-a-time path and checks it against `from_pretrained`.
"""
import argparse
import json
import shutil
import tempfile
from pathlib import Path

import torch
import transformers
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoTokenizer
from transformers.core_model_loading import Inkling2LoopDequantizeExperts
from transformers.models.inkling2_loop.configuration_inkling2_loop import Inkling2LoopTextConfig
from transformers.models.inkling2_loop.modeling_inkling2_loop import (
    Inkling2LoopBlockAttentionResidual,
    Inkling2LoopForCausalLM,
    Inkling2LoopFullDecoderLayer,
    Inkling2LoopFullLoopDecoderLayer,
    Inkling2LoopMessageAttentionResidual,
    Inkling2LoopRMSNorm,
    Inkling2LoopSlidingDecoderLayer,
    Inkling2LoopSlidingLoopDecoderLayer,
)

REVISION = "ed40e7da625f594fc9c4990f5f66be24873c22d7"
PROMPT = (
    "fn main() {\n    let primes: Vec<u32> = (2..50).filter(|n| (2..*n).all(|d| n % d != 0)).collect();\n"
    '    println!("{primes:?}");\n}\n\nThe capital of France is Paris, and the capital of Japan is'
)
# Tokens the tiny golden prefills before decoding the rest one at a time.
PREFILL = 13


def tiny_config():
    text = dict(
        model_type="inkling2_loop_text", torch_dtype="bfloat16", model_max_length=4096,
        hidden_size=64, num_hidden_layers=6, vocab_size=112, unpadded_vocab_size=100,
        num_attention_heads=4, num_key_value_heads=2, head_dim=64, d_rel=16, rel_extent=7,
        q_bias=False, o_bias=False, log_scaling_n_floor=8, log_scaling_alpha=0.1, rms_norm_eps=1e-6,
        use_value_norm_no_gamma=True, qk_l2_clip_threshold=6.0, rel_l2_clip_threshold=8.0,
        local_layer_ids=[0, 2, 4], sconv_kernel_size=4, final_logit_softcapping=8.0,
        swa_head_dim=64, swa_num_attention_heads=4, swa_num_key_value_heads=2, sliding_window_size=5,
        n_layers_per_attn_res_block=2, attn_res_normalize_values=True, n_routed_experts=8,
        num_experts_per_tok=3, n_shared_experts=0, intermediate_size=32, mlp_gate_softcap=4.0,
        mlp_up_softcap=25.0, route_scale=8.0, use_gate_bias=True,
        loop_config=dict(core_start=2, core_end=4, num_passes=3, gate_rank=8, is_varloop=False),
    )
    return dict(
        dtype="bfloat16", architectures=["Inkling2LoopForConditionalGeneration"], model_type="inkling2_loop",
        eos_token_id=1, text_config=text,
        quantization_config=dict(quant_method="modelopt", quant_algo="NVFP4"),
    )


def transformers_text_config(text):
    """The published text settings as Transformers reads them, with `local_layer_ids` (which
    Transformers ignores) turned into the `layer_types` it reads instead."""
    settings = dict(text)
    local = set(settings.pop("local_layer_ids"))
    settings["layer_types"] = [
        "hybrid_sliding" if i in local else "hybrid" for i in range(settings["num_hidden_layers"])
    ]
    settings["eos_token_id"] = 1
    settings["attn_implementation"] = "eager"
    return Inkling2LoopTextConfig(**settings)


E2M1 = Inkling2LoopDequantizeExperts.E2M1_VALUES


def float8_bytes(generator, shape, low, high):
    """Random E4M3 values with exponent fields in `low..=high`, never NaN."""
    sign = torch.randint(0, 2, shape, generator=generator, dtype=torch.uint8) << 7
    exponent = torch.randint(low, high + 1, shape, generator=generator, dtype=torch.uint8) << 3
    mantissa = torch.randint(0, 8, shape, generator=generator, dtype=torch.uint8)
    return (sign | exponent | mantissa).view(torch.float8_e4m3fn)


def tiny_experts(prefix, experts, rows, cols, nvfp4, generator):
    """One expert projection in its published format."""
    if nvfp4:
        codes = torch.randint(0, 256, (experts, rows, cols // 2), generator=generator, dtype=torch.uint8)
        # Positive scales around 8..64, so weights land near 0.05..0.5 after the expert factor.
        scale = float8_bytes(generator, (experts, rows, cols // 16), 9, 12).float().abs().to(torch.float8_e4m3fn)
        amax = (1.0 + 2.0 * torch.rand(experts, generator=generator)).float()
        return {
            f"{prefix}": codes, f"{prefix}_scale": scale, f"{prefix}_amax": amax,
            f"{prefix}_outer_scale_denominator": torch.tensor([2688.0]),
        }
    values = float8_bytes(generator, (experts, rows, cols), 4, 10)
    scales = torch.randint(118, 123, (experts, rows, cols // 32), generator=generator, dtype=torch.uint8)
    return {f"{prefix}": values, f"{prefix}_scale": scales}


def tiny_weights(config, generator):
    t = config["text_config"]
    hidden, heads, kv, dim = t["hidden_size"], t["num_attention_heads"], t["num_key_value_heads"], t["head_dim"]
    layers, experts, inner, rel = t["num_hidden_layers"], t["n_routed_experts"], t["intermediate_size"], t["d_rel"]
    taps, rank, vocab = t["sconv_kernel_size"], t["loop_config"]["gate_rank"], t["vocab_size"]
    loop = range(t["loop_config"]["core_start"], t["loop_config"]["core_end"])
    local = set(t["local_layer_ids"])

    def normal(*shape, scale=0.1, dtype=torch.bfloat16):
        return (torch.randn(*shape, generator=generator) * scale).to(dtype)

    def gain(*shape):
        return (1 + normal(*shape, scale=0.2, dtype=torch.float32)).to(torch.bfloat16)

    data = {
        "model.llm.embed.weight": normal(vocab, hidden, scale=1.0),
        "model.llm.embed_norm.weight": gain(hidden),
        "model.llm.norm.weight": gain(hidden),
        "model.llm.unembed.weight": normal(vocab, hidden, scale=0.3),
        "model.llm.attn_res.output_query": normal(hidden, scale=0.5, dtype=torch.float32),
        "model.llm.middle_cycle_cross_pass_queries": normal(2 * len(loop), hidden, scale=0.5, dtype=torch.float32),
        "model.llm.middle_cycle_cross_pass_kv_queries": normal(2 * len(loop), hidden, scale=0.5, dtype=torch.float32),
    }
    for i in range(layers):
        p = f"model.llm.layers.{i}"
        sliding = i in local
        if i > 0:
            shape = (hidden,) if sliding else (3, hidden)
            data[f"model.llm.attn_res.attn_queries.{i - 1}"] = normal(*shape, scale=0.5, dtype=torch.float32)
        data[f"model.llm.attn_res.mlp_queries.{i}"] = normal(hidden, scale=0.5, dtype=torch.float32)
        a = f"{p}.attn"
        extent = t["sliding_window_size"] if sliding else t["rel_extent"]
        data[f"{a}.wq_du.weight"] = normal(heads * dim, hidden)
        data[f"{a}.wk_dv.weight"] = normal(kv * dim, hidden)
        data[f"{a}.wv_dv.weight"] = normal(kv * dim, hidden)
        data[f"{a}.wo_ud.weight"] = normal(hidden, heads * dim)
        data[f"{a}.gate_attn.weight"] = normal(heads * dim, hidden, scale=0.3)
        data[f"{a}.wr_du.weight"] = normal(heads * rel, hidden, scale=0.3)
        data[f"{a}.rel_logits_proj.proj"] = normal(rel, extent, scale=0.5)
        data[f"{a}.q_l2_clip.scale"] = gain(heads, dim)
        data[f"{a}.rel_l2_clip.scale"] = gain(heads, rel)
        data[f"{a}.attn_sink_proj.weight"] = normal(heads, hidden, scale=0.3)
        if sliding:
            data[f"{p}.attn_norm.weight"] = gain(hidden)
            for name in ("k_sconv", "v_sconv"):
                data[f"{a}.{name}.weight"] = normal(kv * dim, 1, taps, scale=0.4)
        else:
            for name in ("q_input_norm", "k_input_norm", "v_input_norm"):
                data[f"{p}.{name}.weight"] = gain(hidden)
            for name in ("k_sconv", "v_sconv"):
                data[f"{a}.{name}.static_w"] = normal(taps, kv * dim, scale=0.4)
                data[f"{a}.{name}.weight_proj"] = normal(kv * taps, hidden, scale=0.2)
        data[f"{p}.attn_sconv.weight"] = normal(hidden, 1, taps, scale=0.4)
        data[f"{p}.mlp_sconv.weight"] = normal(hidden, 1, taps, scale=0.4)
        data[f"{p}.mlp_norm.weight"] = gain(hidden)
        data[f"{p}.mlp.gate.weight"] = normal(experts, hidden, scale=0.3)
        data[f"{p}.mlp.gate.bias"] = normal(experts, scale=0.2, dtype=torch.float32)
        nvfp4 = i < layers - 2
        data |= tiny_experts(f"{p}.mlp.experts.w13_weight", experts, 2 * inner, hidden, nvfp4, generator)
        data |= tiny_experts(f"{p}.mlp.experts.w2_weight", experts, hidden, inner, nvfp4, generator)
        if i in loop:
            for name in ("attn", "k", "v", "mlp"):
                c = f"{p}.controllers.{name}"
                data[f"{c}.down_weight"] = normal(rank, 2 * hidden, scale=0.2)
                data[f"{c}.forget_weight"] = normal(hidden, rank, scale=0.2)
                data[f"{c}.gain_weight"] = normal(hidden, rank, scale=0.2)
    return data


def quant_config(data):
    """The published `hf_quant_config.json` layout: each expert weight under its format."""
    groups = {"nvfp4": [], "mxfp8": []}
    for name, value in data.items():
        if name.endswith(("w13_weight", "w2_weight")):
            groups["nvfp4" if value.dtype == torch.uint8 else "mxfp8"].append(name)
    return {
        "model": {"parameter_groups": [
            {"compute_dtype": dtype, "tensors": sorted(names)} for dtype, names in groups.items()
        ]},
        "quant_method": "inkling2",
    }


def record_layers(model):
    """Hooks recording every decoder layer application's output, in call order."""
    outputs = []
    handles = [
        layer.register_forward_hook(lambda _m, _i, out: outputs.append(out.detach().float().squeeze(0).clone()))
        for layer in model.model.layers
    ]
    return outputs, handles


@torch.inference_mode()
def tiny_golden(directory, config, tokens):
    text = transformers_text_config(config["text_config"])
    with tempfile.TemporaryDirectory() as scratch:
        scratch = Path(scratch)
        shutil.copy(directory / "model.safetensors", scratch / "model.safetensors")
        (scratch / "config.json").write_text(text.to_json_string())
        model = Inkling2LoopForCausalLM.from_pretrained(scratch, dtype=torch.float32).eval()
    outputs, handles = record_layers(model)
    logits = model(input_ids=tokens[None], use_cache=False).logits.squeeze(0).contiguous()
    for handle in handles:
        handle.remove()
    result = {"tokens": tokens.to(torch.int64), "logits": logits}
    for index, value in enumerate(outputs):
        result[f"layer.{index}"] = value.contiguous()
    # Prefill, then decode the rest one token at a time through Transformers' own cache.
    step = model(input_ids=tokens[None, :PREFILL], use_cache=True)
    decoded, cache = [step.logits[0, -1]], step.cache
    for position in range(PREFILL, len(tokens) - 1):
        step = model(input_ids=tokens[None, position:position + 1], cache=cache, use_cache=True)
        decoded.append(step.logits[0, -1])
        cache = step.cache
    result["decode.logits"] = torch.stack(decoded).contiguous()
    return text, result


def tiny(output):
    output.mkdir(parents=True, exist_ok=True)
    config = tiny_config()
    generator = torch.Generator().manual_seed(23)
    data = tiny_weights(config, generator)
    tokens = torch.randint(2, config["text_config"]["unpadded_vocab_size"], (24,), generator=generator)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (output / "hf_quant_config.json").write_text(json.dumps(quant_config(data), indent=2) + "\n")
    save_file({k: v.contiguous() for k, v in data.items()}, output / "model.safetensors", metadata={"format": "pt"})
    text, result = tiny_golden(output, config, tokens)
    layered = layer_by_layer(text, Tensors(output), tokens[None])
    errors = {
        name: ((layered[name] - expected).abs().max() / expected.abs().max()).item()
        for name, expected in result.items()
        if name.startswith("layer.") or name == "logits"
    }
    assert max(errors.values()) < 2e-5, f"layer-at-a-time path differs: {errors}"
    decode = result["decode.logits"]
    error = (decode - result["logits"][PREFILL - 1:-1]).abs().max().item()
    assert error < 1e-4, f"cached decode differs from the full pass by {error}"
    save_file(result, output / "golden.safetensors", metadata={
        "transformers": transformers.__version__, "torch": torch.__version__, "prefill": str(PREFILL),
    })
    print(f"tiny fixture: {len(result)} tensors, decode matches full pass within {error:.2e}")


class Tensors:
    """Every tensor of a published checkpoint by name, read on demand."""

    def __init__(self, directory):
        index = directory / "model.safetensors.index.json"
        if index.exists():
            files = json.loads(index.read_text())["weight_map"]
            handles = {file: safe_open(directory / file, "pt") for file in set(files.values())}
            self.files = {name: handles[file] for name, file in files.items()}
        else:
            handle = safe_open(directory / "model.safetensors", "pt")
            self.files = {name: handle for name in handle.keys()}

    def get(self, name):
        return self.files[name].get_tensor(name)

    def has(self, name):
        return name in self.files


def interleaved_rows(tensor):
    """Transformers' `Interleave(dim=1)`: stored rows alternate gate and up; split them."""
    shape = list(tensor.shape)
    shape[1:2] = [shape[1] // 2, 2]
    return tensor.reshape(shape).transpose(1, 2).reshape(tensor.shape).contiguous()


def dequantized(tensors, prefix):
    parts = [f"{prefix}", f"{prefix}_scale", f"{prefix}_amax", f"{prefix}_outer_scale_denominator"]
    inputs = {part: tensors.get(part) for part in parts if tensors.has(part)}
    pattern = list(inputs)
    if len(pattern) == 2:
        pattern += [f"{prefix}_amax", f"{prefix}_outer_scale_denominator"]
    out = Inkling2LoopDequantizeExperts().convert(
        inputs, pattern, ["weight"], model=F32Parameters(), full_layer_name="weight"
    )
    return out["weight"]


class F32Parameters:
    """Stands in for the model the dequantizer asks for its parameter dtype; without one it
    rounds experts to BF16."""

    @staticmethod
    def get_parameter(_name):
        return torch.empty(0, dtype=torch.float32)


def layer_state(tensors, config, index):
    """Transformers' parameter names of decoder layer `index`, from the published tensors."""
    p, state = f"model.llm.layers.{index}", {}
    sliding = config.layer_types[index] == "hybrid_sliding"
    renames = {
        "attn.wq_du.weight": "self_attn.q_proj.linear.weight",
        "attn.wk_dv.weight": "self_attn.k_proj.linear.weight",
        "attn.wv_dv.weight": "self_attn.v_proj.linear.weight",
        "attn.wo_ud.weight": "self_attn.o_proj.linear.weight",
        "attn.gate_attn.weight": "self_attn.o_proj.gate_proj.weight",
        "attn.wr_du.weight": "self_attn.relative_position_bias.linear.weight",
        "attn.rel_logits_proj.proj": "self_attn.relative_position_bias.proj",
        "attn.q_l2_clip.scale": "self_attn.q_proj.norm.scale",
        "attn.rel_l2_clip.scale": "self_attn.relative_position_bias.norm.scale",
        "attn.attn_sink_proj.weight": "self_attn.sink_proj.weight",
        "attn_sconv.weight": "post_attention_convolution.conv1d.weight",
        "mlp_sconv.weight": "post_mlp_convolution.conv1d.weight",
        "mlp_norm.weight": "post_attention_layernorm.weight",
        "mlp.gate.weight": "mlp.gate.weight",
        "mlp.gate.bias": "mlp.gate.e_score_correction_bias",
    }
    if sliding:
        renames |= {
            "attn_norm.weight": "input_layernorm.weight",
            "attn.k_sconv.weight": "self_attn.k_proj.sconv.conv1d.weight",
            "attn.v_sconv.weight": "self_attn.v_proj.sconv.conv1d.weight",
        }
    else:
        renames |= {f"{x}_input_norm.weight": f"{x}_input_layernorm.weight" for x in "qkv"}
        for x in "kv":
            renames[f"attn.{x}_sconv.static_w"] = f"self_attn.{x}_proj.sconv.static_weight"
            renames[f"attn.{x}_sconv.weight_proj"] = f"self_attn.{x}_proj.sconv.weight_proj.weight"
    if config.loop_layer_index_start <= index < config.loop_layer_index_end:
        for source, target in (("attn", "q"), ("k", "k"), ("v", "v"), ("mlp", "mlp")):
            for part, name in (("down_weight", "down_proj"), ("gain_weight", "message_gate_proj"),
                               ("forget_weight", "forget_gate_proj")):
                renames[f"controllers.{source}.{part}"] = f"{target}_message_gate.{name}.weight"
    for source, target in renames.items():
        state[target] = tensors.get(f"{p}.{source}").float()
    if index > 0:
        state["attention_input.query"] = tensors.get(f"model.llm.attn_res.attn_queries.{index - 1}").float()
    state["mlp_input.query"] = tensors.get(f"model.llm.attn_res.mlp_queries.{index}").float()
    state["mlp.experts.gate_up_proj"] = interleaved_rows(dequantized(tensors, f"{p}.mlp.experts.w13_weight"))
    state["mlp.experts.down_proj"] = dequantized(tensors, f"{p}.mlp.experts.w2_weight")
    return state


def build_layer(config, index, tensors):
    sliding = config.layer_types[index] == "hybrid_sliding"
    looped = config.loop_layer_index_start <= index < config.loop_layer_index_end
    kind = {
        (True, False): Inkling2LoopSlidingDecoderLayer, (False, False): Inkling2LoopFullDecoderLayer,
        (True, True): Inkling2LoopSlidingLoopDecoderLayer, (False, True): Inkling2LoopFullLoopDecoderLayer,
    }[(sliding, looped)]
    layer = kind(config, index).float().eval()
    layer.load_state_dict(layer_state(tensors, config, index), strict=True)
    return layer


def masks(count, window):
    allowed = torch.ones(count, count).tril()
    distance = torch.arange(count)[:, None] - torch.arange(count)[None, :]
    full = torch.where(allowed > 0, 0.0, -torch.inf)[None, None]
    sliding = torch.where((allowed > 0) & (distance < window), 0.0, -torch.inf)[None, None]
    return {"full_attention": full, "sliding_attention": sliding}, distance


@torch.inference_mode()
def layer_by_layer(config, tensors, ids):
    """The text model's forward, one decoder layer resident at a time: the loop of
    `Inkling2LoopTextModel.forward` without a cache."""
    count = ids.shape[1]
    mask, distance = masks(count, config.sliding_window_size)
    positions = torch.arange(count)[None]
    embed = tensors.get("model.llm.embed.weight").float()
    norm = Inkling2LoopRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    norm.weight.data = tensors.get("model.llm.embed_norm.weight").float()
    hidden = norm(embed[ids])
    del embed
    blocks = hidden.new_zeros(*hidden.shape[:-1], 0, hidden.shape[-1])
    result, start, end = {"tokens": ids[0].to(torch.int64)}, config.loop_layer_index_start, config.loop_layer_index_end
    applications = 0

    def run(index, hidden, blocks, messages=None):
        nonlocal applications
        layer = build_layer(config, index, tensors)
        kwargs = dict(attention_mask=mask[layer.attention_type], relative_distance=distance, conv_mask=None,
                      position_ids=positions)
        if messages is not None:
            kwargs["messages"] = messages
        new = layer(hidden, blocks, **kwargs)
        if layer.is_block_boundary:
            blocks = torch.cat([blocks, hidden.unsqueeze(-2)], dim=-2)
        result[f"layer.{applications}"] = new.squeeze(0).contiguous().clone()
        applications += 1
        print(f"layer {index} ({applications} applied)", flush=True)
        return new, blocks

    for index in range(start):
        hidden, blocks = run(index, hidden, blocks)
    pre_hidden, pre_blocks = hidden, blocks
    for index in range(start, end):
        hidden, blocks = run(index, hidden, blocks)
    message = Inkling2LoopMessageAttentionResidual(config).float()
    queries = tensors.get("model.llm.middle_cycle_cross_pass_queries").float().unflatten(0, (-1, 2))
    kv_queries = tensors.get("model.llm.middle_cycle_cross_pass_kv_queries").float().unflatten(0, (-1, 2))
    message.query.data = torch.cat([queries[:, :1], kv_queries, queries[:, 1:]], dim=1)
    for _ in range(1, config.loop_num_passes):
        previous_blocks, previous_hidden = blocks[..., pre_blocks.shape[-2] + 1:, :], hidden
        hidden, blocks = pre_hidden, pre_blocks
        for offset, index in enumerate(range(start, end)):
            messages = message(previous_blocks, previous_hidden, offset)
            hidden, blocks = run(index, hidden, blocks, messages)
    for index in range(end, config.num_hidden_layers):
        hidden, blocks = run(index, hidden, blocks)
    final = Inkling2LoopBlockAttentionResidual(config).float()
    final.query.data = tensors.get("model.llm.attn_res.output_query").float()
    norm.weight.data = tensors.get("model.llm.norm.weight").float()
    hidden = norm(final(blocks, hidden)[0])
    head = tensors.get("model.llm.unembed.weight").float()[: config.unpadded_vocab_size]
    logits = hidden @ head.T
    cap = config.final_logit_softcapping
    result["logits"] = (torch.tanh(logits / cap) * cap).squeeze(0).contiguous()
    return result


def real(directory, output):
    published = json.loads((directory / "config.json").read_text())
    config = transformers_text_config(published["text_config"])
    ids = AutoTokenizer.from_pretrained(directory)(PROMPT, return_tensors="pt").input_ids
    result = layer_by_layer(config, Tensors(directory), ids)
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
