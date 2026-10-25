"""Pinned Gemma 4 text references: Transformers 5.20.2, torch 2.14.1, CPU F32.

Tiny fixtures are offline numerical oracles, not evidence that a real model works.
  python -I scripts/gemma4_reference.py --tiny crates/tur/tests/fixtures/gemma4
Real checkpoint (immutable google/gemma-4-E2B-it revision):
  python -I scripts/gemma4_reference.py --model <dir> --golden /tmp/gemma4.safetensors
"""
import argparse
import json
from pathlib import Path
import torch
import transformers
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import Gemma4TextConfig, Gemma4ForCausalLM, AutoTokenizer, Gemma4UnifiedTextConfig, Gemma4UnifiedForCausalLM

REVISION = "3e22461f65e89153144f8adb70e3b8c2cc9845a7"


def tiny_config(moe):
    return dict(
        architectures=["Gemma4ForCausalLM"], model_type="gemma4_text",
        hidden_size=64, intermediate_size=128, vocab_size=96,
        num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1,
        head_dim=64, global_head_dim=512, num_global_key_value_heads=1,
        max_position_embeddings=2048, rms_norm_eps=1e-6,
        hidden_activation="gelu_pytorch_tanh", attention_bias=False,
        attention_dropout=0.0, attention_k_eq_v=moe, tie_word_embeddings=True,
        layer_types=["sliding_attention", "full_attention"] * 2,
        rope_parameters={
            "sliding_attention": dict(rope_type="default", rope_theta=10000.0),
            "full_attention": dict(rope_type="proportional", rope_theta=1000000.0, partial_rotary_factor=0.25),
        }, sliding_window=4, num_kv_shared_layers=0 if moe else 2,
        hidden_size_per_layer_input=0 if moe else 32, vocab_size_per_layer_input=96,
        use_double_wide_mlp=not moe, final_logit_softcapping=30.0,
        use_bidirectional_attention=None, enable_moe_block=moe,
        num_experts=4 if moe else None, top_k_experts=2 if moe else None,
        moe_intermediate_size=64 if moe else None, eos_token_id=1,
        bos_token_id=2, pad_token_id=0, dtype="bfloat16",
    )


def trace(model, tokens, output, generate=0):
    data = {"tokens": tokens.to(torch.int64)}
    handles = []
    for index, layer in enumerate(model.model.layers):
        def record(_module, _inputs, result, index=index):
            data[f"layer.{index}"] = result.detach().float().squeeze(0).contiguous()
        handles.append(layer.register_forward_hook(record))
    with torch.inference_mode():
        result = model(input_ids=tokens, use_cache=False)
        data["logits"] = result.logits.squeeze(0).float().contiguous()
    for handle in handles:
        handle.remove()
    if generate:
        with torch.inference_mode():
            generated = model.generate(tokens, max_new_tokens=generate, do_sample=False, eos_token_id=[1, 106, 50], pad_token_id=0)
        data["greedy"] = generated[0, tokens.shape[-1]:].contiguous()
    save_file(data, output, metadata={"transformers": transformers.__version__, "torch": torch.__version__, "revision": REVISION})


def tiny(root):
    for kind in ["dense", "moe", "unified"]:
        moe = kind == "moe"
        directory = root / kind
        directory.mkdir(parents=True, exist_ok=True)
        config = tiny_config(moe)
        if kind == "unified":
            config.update(architectures=["Gemma4UnifiedForCausalLM"], model_type="gemma4_unified_text", hidden_size_per_layer_input=0, num_kv_shared_layers=0, use_double_wide_mlp=False, attention_k_eq_v=True)
            model = Gemma4UnifiedForCausalLM(Gemma4UnifiedTextConfig(**config)).eval()
        else:
            model = Gemma4ForCausalLM(Gemma4TextConfig(**config)).eval()
        with torch.no_grad():
            for index, (name, value) in enumerate(model.state_dict().items()):
                base = torch.arange(value.numel(), dtype=torch.float64).reshape(value.shape)
                scale = 0.04 if "norm" not in name else 0.08
                data = torch.sin(base * 0.37 + index * 1.7) * scale
                if "norm" in name or "scale" in name or "scalar" in name:
                    data += 1.0
                value.copy_(data.to(torch.bfloat16).float())
        weights = {name: value.to(torch.bfloat16).contiguous() for name, value in model.state_dict().items() if name != "lm_head.weight"}
        save_file(weights, directory / "model.safetensors")
        (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        trace(model, torch.tensor([[2, 7, 5, 9, 3, 12, 7, 17, 4]]), directory / "golden.safetensors")


class DiskEmbedding(torch.nn.Module):
    def __init__(self, path, key, scale):
        super().__init__()
        self.path, self.key, self.scale = path, key, scale

    def forward(self, ids):
        with safe_open(self.path, framework="pt", device="cpu") as file:
            table = file.get_slice(self.key)
            rows = [table[int(token):int(token)+1].float() for token in ids.reshape(-1)]
        return torch.cat(rows).reshape(*ids.shape, -1) * self.scale


def real(directory, output):
    raw = json.loads((directory / "config.json").read_text())
    config = Gemma4TextConfig(**raw["text_config"])
    config._attn_implementation = "eager"
    path = directory / "model.safetensors"
    with torch.device("meta"):
        model = Gemma4ForCausalLM(config).eval()
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding
    model.model.rotary_emb = Gemma4TextRotaryEmbedding(config)
    prefix = "model.language_model."
    # Stream one decoder layer at a time; the 2.3B PLE table is looked up on disk.
    def materialize(module, state):
        module.load_state_dict(state, assign=True, strict=True)
    with safe_open(path, framework="pt", device="cpu") as file:
        common = {name: file.get_tensor(prefix + name).float() for name in ["embed_tokens.weight", "norm.weight", "per_layer_model_projection.weight", "per_layer_projection_norm.weight"]}
    for name in ["embed_tokens", "norm", "per_layer_model_projection", "per_layer_projection_norm"]:
        materialize(getattr(model.model, name), {"weight": common[name + ".weight"]})
    model.lm_head.weight = model.model.embed_tokens.weight
    model.model.embed_tokens_per_layer = DiskEmbedding(path, prefix + "embed_tokens_per_layer.weight", config.hidden_size_per_layer_input**0.5)
    model.model.embed_tokens.embed_scale = torch.tensor(config.hidden_size**0.5)
    for index, layer in enumerate(model.model.layers):
        names = list(layer.state_dict())
        def load(module, _args, index=index, names=names):
            with safe_open(path, framework="pt", device="cpu") as file:
                state = {name: file.get_tensor(f"{prefix}layers.{index}.{name}").float() for name in names}
            materialize(module, state)
        layer.register_forward_pre_hook(load)
        layer.register_forward_hook(lambda module, _args, _output: module.to_empty(device="meta") and None)
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
    tokens = tokenizer.apply_chat_template([{"role": "user", "content": "What is the capital of France?"}], tokenize=True, add_generation_prompt=True, return_tensors="pt", enable_thinking=False, return_dict=True)["input_ids"]
    trace(model, tokens, output, generate=32)
    print(f"wrote {output}: {tokens.shape[-1]} tokens", flush=True)


def templates(directory, output):
    from transformers.utils.chat_template_utils import render_jinja_template
    source = (directory / "chat_template.jinja").read_text()
    tokens = json.loads((directory / "tokenizer_config.json").read_text())
    tools=[{'type':'function','function':{'name':'get_weather','description':'Weather — °C','parameters':{'type':'object','properties':{'city':{'type':'string'},'days':{'type':'integer'}},'required':['city']}}}]
    cases=[
     {'name':'default','messages':[{'role':'user','content':'Hi there!'}],'options':{}},
     {'name':'thinking_on','messages':[{'role':'user','content':'2+2?'}],'options':{'enable_thinking':True}},
     {'name':'thinking_off','messages':[{'role':'user','content':'2+2?'}],'options':{'enable_thinking':False}},
     {'name':'system_history','messages':[{'role':'system','content':'Be terse.'},{'role':'user','content':'Hi'},{'role':'assistant','content':'Hello'},{'role':'user','content':'Kyiv — ✓'}],'options':{}},
     {'name':'reasoning','messages':[{'role':'user','content':'Count'},{'role':'assistant','content':'1, 2','reasoning_content':'Simple.'}],'options':{'add_generation_prompt':False}},
     {'name':'tools','messages':[{'role':'user','content':'Weather in Kyiv?'}],'tools':tools,'options':{}},
     {'name':'tool_history','messages':[{'role':'user','content':'Weather?'},{'role':'assistant','content':'Checking.','tool_calls':[{'type':'function','function':{'name':'get_weather','arguments':{'city':'Kyiv','days':2}}}]},{'role':'tool','content':'12 °C'},{'role':'assistant','content':'12 °C'},{'role':'user','content':'Thanks'}],'tools':tools,'options':{'enable_thinking':False}},
    ]
    for case in cases:
        options = dict(case["options"])
        case["add_generation_prompt"] = options.pop("add_generation_prompt", True)
        rendered, _ = render_jinja_template([case["messages"]], tools=case.get("tools"), chat_template=source, add_generation_prompt=case["add_generation_prompt"], **tokens, **options)
        case["text"] = rendered[0]
    output.write_text(json.dumps(cases, ensure_ascii=False, indent=2) + "\n")


def main():
    if transformers.__version__ != "5.20.2" or torch.__version__.split("+")[0] != "2.14.1":
        raise RuntimeError("use pinned transformers 5.20.2 and torch 2.14.1")
    torch.set_num_threads(4)
    args = argparse.ArgumentParser()
    args.add_argument("--tiny", type=Path)
    args.add_argument("--model", type=Path)
    args.add_argument("--golden", type=Path)
    args.add_argument("--templates", type=Path)
    args.add_argument("--template-dir", type=Path, default=Path(__file__).resolve().parents[1] / "crates/tur/tests/fixtures/gemma4/e2b")
    options = args.parse_args()
    if options.templates:
        templates(options.template_dir, options.templates)
    elif options.tiny:
        tiny(options.tiny)
    elif options.model and options.golden:
        real(options.model, options.golden)
    else:
        args.error("provide --tiny or --model and --golden")

if __name__ == "__main__":
    main()
