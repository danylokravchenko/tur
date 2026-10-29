"""Pinned Gemma 4 assistant (MTP drafter) references: Transformers 5.20.2, torch 2.14.1, CPU F32.

The drafter attends over the target's last sliding and full K/V and joins the target's
scaled embedding of the last seen token with its final hidden state, at that token's
position. Each record teacher-forces one draft: target hidden at p, token at p + 1, and
chains a second one from the first's greedy token and post-projected hidden state at the
same position, as Transformers' assisted generation does.
Tiny fixture (offline numerical oracle over the dense target fixture):
  python -I scripts/gemma4_assistant_reference.py --tiny crates/tur/tests/fixtures/gemma4
Real checkpoints (immutable revisions):
  python -I scripts/gemma4_assistant_reference.py --model <E2B dir> --assistant <drafter dir> \
      --golden /tmp/gemma4-assistant.safetensors
"""
import argparse
import json
from pathlib import Path
import torch
import transformers
from safetensors.torch import save_file
from transformers import Gemma4AssistantConfig, Gemma4AssistantForCausalLM, Gemma4ForCausalLM

ASSISTANT_REVISION = "2d874ef7d29f9a30599a1e4b3c1cbc9595f005df"


def tiny_config():
    text = dict(
        model_type="gemma4_text", hidden_size=32, intermediate_size=64, vocab_size=96,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        head_dim=64, global_head_dim=512, max_position_embeddings=2048, rms_norm_eps=1e-6,
        hidden_activation="gelu_pytorch_tanh", attention_bias=False, attention_dropout=0.0,
        attention_k_eq_v=False, tie_word_embeddings=True,
        layer_types=["sliding_attention", "full_attention"],
        rope_parameters={
            "sliding_attention": dict(rope_type="default", rope_theta=10000.0),
            "full_attention": dict(rope_type="proportional", rope_theta=1000000.0, partial_rotary_factor=0.25),
        },
        sliding_window=4, num_kv_shared_layers=2, hidden_size_per_layer_input=0,
        vocab_size_per_layer_input=0, use_double_wide_mlp=False, final_logit_softcapping=None,
        use_bidirectional_attention=None, enable_moe_block=False, eos_token_id=1,
        bos_token_id=2, pad_token_id=0, dtype="bfloat16",
    )
    return dict(
        architectures=["Gemma4AssistantForCausalLM"], model_type="gemma4_assistant",
        backbone_hidden_size=64, use_ordered_embeddings=True, num_centroids=8,
        centroid_intermediate_top_k=3, tie_word_embeddings=True, text_config=text,
        dtype="bfloat16",
    )


def drafts(target, assistant, tokens, positions):
    """Teacher-forced drafts: for each p, the drafter's logits (unselected tokens at -inf)
    from the target's final hidden state at p and the token at p + 1, over keys 0..=p, and
    the logits of the draft chained from it."""
    with torch.inference_mode():
        out = target(input_ids=tokens, output_hidden_states=True, return_shared_kv_states=True, use_cache=False)
        hidden = out.hidden_states[-1]
        embed = target.get_input_embeddings()
        data = {"tokens": tokens[0].to(torch.int64), "positions": torch.tensor(positions, dtype=torch.int64)}
        logits, chained, second = [], [], []
        for p in positions:
            shared = {k: (v[0][:, :, : p + 1], v[1][:, :, : p + 1]) for k, v in out.shared_kv_states.items()}
            inputs = torch.cat([embed(tokens[:, p + 1 : p + 2]), hidden[:, p : p + 1]], dim=-1)
            row, projected = step(assistant, inputs, p, shared)
            logits.append(row)
            chained.append(projected[0, 0])
            # Transformers' generator chains greedily: the draft's embedding beside the
            # drafter's post-projected state, at the same position and keys.
            inputs = torch.cat([embed(row.argmax().view(1, 1)), projected], dim=-1)
            second.append(step(assistant, inputs, p, shared)[0])
        data["logits"] = torch.stack(logits).contiguous()
        data["chained"] = torch.stack(chained).contiguous()
        data["second"] = torch.stack(second).contiguous()
    return data


def step(assistant, inputs, p, shared):
    """One drafter step at position p + 1: masked logits and the post-projected state."""
    result = assistant(inputs_embeds=inputs, position_ids=torch.tensor([[p + 1]]), shared_kv_states=shared, use_cache=False)
    row = result.logits[0, 0].float()
    # Transformers fills unselected tokens with min - 1; Tur masks them with -inf.
    selected = torch.zeros_like(row, dtype=torch.bool)
    ordering = assistant.masked_embedding.token_ordering.long().view(assistant.config.num_centroids, -1)
    top = torch.topk(assistant.masked_embedding.centroids(result_hidden(assistant, inputs, p, shared)), assistant.config.centroid_intermediate_top_k).indices
    selected[ordering[top].reshape(-1)] = True
    return torch.where(selected, row, torch.tensor(float("-inf"))), result.last_hidden_state.float()


def result_hidden(assistant, inputs, p, shared):
    """The drafter's final-normed hidden state, which selects its centroids."""
    projected = assistant.pre_projection(inputs)
    masks = assistant.create_attention_masks(projected, None, shared)
    out = assistant.model(input_ids=None, inputs_embeds=projected, attention_mask=masks, position_ids=torch.tensor([[p + 1]]), shared_kv_states=shared, use_cache=False)
    return out.last_hidden_state[0, 0]


def tiny(root):
    directory = root / "assistant"
    directory.mkdir(parents=True, exist_ok=True)
    config = tiny_config()
    assistant = Gemma4AssistantForCausalLM(Gemma4AssistantConfig(**config)).eval()
    with torch.no_grad():
        for index, (name, value) in enumerate(assistant.state_dict().items()):
            if name == "masked_embedding.token_ordering":
                # A fixed permutation: canonical vocabulary order differs from cluster order.
                value.copy_(torch.tensor([(i * 37 + 11) % 96 for i in range(96)]))
                continue
            base = torch.arange(value.numel(), dtype=torch.float64).reshape(value.shape)
            data = torch.sin(base * 0.53 + index * 1.3) * 0.08
            if "norm" in name or "scalar" in name:
                data += 1.0
            value.copy_(data.to(torch.bfloat16).float())
    weights = {
        name: (value if name == "masked_embedding.token_ordering" else value.to(torch.bfloat16)).contiguous()
        for name, value in assistant.state_dict().items() if name != "lm_head.weight"
    }
    save_file(weights, directory / "model.safetensors")
    (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    target = Gemma4ForCausalLM.from_pretrained(root / "dense", dtype=torch.float32).eval()
    tokens = torch.tensor([[2, 7, 5, 9, 3, 12, 7, 17, 4, 11]])
    data = drafts(target, assistant.float(), tokens, list(range(tokens.shape[-1] - 1)))
    save_file(data, directory / "golden.safetensors", metadata={"transformers": transformers.__version__, "torch": torch.__version__, "revision": "synthetic-gemma4-assistant-v1"})


def real(model, assistant_dir, output):
    import importlib.util
    spec = importlib.util.spec_from_file_location("gemma4_reference", Path(__file__).with_name("gemma4_reference.py"))
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    # Past the 512-key window, so sliding drafts read a window of the target's keys.
    prose = " ".join(["Rivers carry water from mountains to the sea, shaping valleys, deltas and the towns that grow along their banks."] * 40)
    target, tokens = reference.streamed(model, f"Summarize the following text in one sentence.\n\n{prose}")
    assistant = Gemma4AssistantForCausalLM.from_pretrained(assistant_dir, dtype=torch.float32).eval()
    assistant.config.get_text_config()._attn_implementation = "eager"
    count = tokens.shape[-1]
    positions = sorted(set(range(0, count - 1, max(1, (count - 1) // 48))) | {count - 2})
    data = drafts(target, assistant, tokens, positions)
    save_file(data, output, metadata={"transformers": transformers.__version__, "torch": torch.__version__, "revision": reference.REVISION, "assistant": ASSISTANT_REVISION})
    print(f"wrote {output}: {count} tokens, {len(positions)} drafts", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiny", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--assistant", type=Path)
    parser.add_argument("--golden", type=Path)
    args = parser.parse_args()
    if args.tiny:
        tiny(args.tiny)
    else:
        real(args.model, args.assistant, args.golden)
