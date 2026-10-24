"""Golden reference for tur's native Qwen3.5 multi-token prediction (MTP) head.

Transformers drops the checkpoint's `mtp.*` tensors, so the head is assembled here from
Transformers' own Qwen3.5 modules: the backbone's hidden states and rotary embeddings, its
RMSNorm and a full-attention `Qwen3_5DecoderLayer`, plus the shared embedding and LM head:

    h'_p = norm(layer(fc([norm_e(embed(x_{p+1})), norm_h(h_p)])))   predicts x_{p+2}

The weights do not fix every convention, so each candidate (backbone hidden after or
before its final norm, MTP rotary position p or p + 1, concatenation order) is scored by
how often its argmax equals the true next-next token of a greedy continuation. The best
candidate's rows are saved for tur's conformance test; its accuracy is independent
evidence that the head is assembled correctly, since a wrong head cannot predict text.

Pinned environment as scripts/qwen3_5_reference.py (transformers 5.20.2, torch CPU F32):

    uv run --no-project --python 3.12 --with torch==2.14.1 --with transformers==5.20.2 --with safetensors \
        python -I scripts/qwen3_5_mtp_reference.py --model <snapshot dir> --golden <out>
"""

import argparse
import itertools
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5RMSNorm

REPO = Path(__file__).resolve().parents[1]
TEMPLATE = REPO / "crates/tur/tests/fixtures/checkpoints/small-chat_template.jinja"
PROMPTS = [
    "Explain how a write-ahead log lets a database recover after a crash. Be specific.",
    "Write a short Python function that merges two sorted lists, then explain it.",
]


def mtp_weights(model_dir):
    index = json.loads((Path(model_dir) / "model.safetensors.index.json").read_text())
    tensors = {}
    for name, file in index["weight_map"].items():
        if name.startswith("mtp."):
            with safe_open(Path(model_dir) / file, "pt") as f:
                tensors[name[len("mtp."):]] = f.get_tensor(name).float()
    return tensors


class Mtp(torch.nn.Module):
    def __init__(self, config, weights):
        super().__init__()
        hidden = config.hidden_size
        attention = config.layer_types.index("full_attention")
        self.fc = torch.nn.Linear(2 * hidden, hidden, bias=False)
        self.norm_e = Qwen3_5RMSNorm(hidden, eps=config.rms_norm_eps)
        self.norm_h = Qwen3_5RMSNorm(hidden, eps=config.rms_norm_eps)
        self.layer = Qwen3_5DecoderLayer(config, attention)
        self.norm = Qwen3_5RMSNorm(hidden, eps=config.rms_norm_eps)
        state = {
            "fc.weight": weights["fc.weight"],
            "norm_e.weight": weights["pre_fc_norm_embedding.weight"],
            "norm_h.weight": weights["pre_fc_norm_hidden.weight"],
            "norm.weight": weights["norm.weight"],
        }
        state.update({f"layer.{k[len('layers.0.'):]}": v for k, v in weights.items() if k.startswith("layers.0.")})
        self.load_state_dict(state, strict=True)
        self.eval()

    def forward(self, embeds, hidden, rotary, positions, embed_first=True):
        parts = [self.norm_e(embeds), self.norm_h(hidden)]
        x = self.fc(torch.cat(parts if embed_first else parts[::-1], dim=-1))
        length = x.shape[1]
        mask = torch.full((length, length), float("-inf")).triu(1)[None, None]
        cos_sin = rotary(x, positions[None, None, :].expand(3, 1, -1))
        out = self.layer(x, position_embeddings=cos_sin, attention_mask=mask, position_ids=positions[None])
        return self.norm(out)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--golden", required=True)
    parser.add_argument("--new-tokens", type=int, default=96)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(args.model, dtype=torch.float32).eval()
    text = model.model.language_model
    config = text.config
    mtp = Mtp(config, mtp_weights(args.model))
    embed, lm_head = text.embed_tokens, model.lm_head
    results, first = {}, {}
    for prompt_index, prompt in enumerate(PROMPTS):
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True,
            chat_template=TEMPLATE.read_text(), enable_thinking=False,
        )
        ids = torch.tensor([tokenizer(rendered, add_special_tokens=False)["input_ids"]])
        prompt_len = ids.shape[1]
        sequence = model.generate(ids, max_new_tokens=args.new_tokens, do_sample=False)
        captured = []
        hook = text.layers[-1].register_forward_hook(lambda _m, _i, o: captured.append(o))
        normed = text(input_ids=sequence).last_hidden_state
        hook.remove()
        raw = captured[0][0] if isinstance(captured[0], tuple) else captured[0]
        tokens = sequence[0]
        count = tokens.shape[0] - 1
        # Inputs at p = 0..count-1 pair h_p with x_{p+1}; the argmax predicts x_{p+2}.
        embeds = embed(tokens[1:])[None]
        targets = tokens[2:]
        scored = list(range(max(prompt_len - 2, 0), count - 1))
        for (kind, hidden), shift, embed_first in itertools.product(
            [("normed", normed), ("raw", raw)], [0, 1], [True, False]
        ):
            positions = torch.arange(count) + shift
            out = mtp(embeds, hidden[:, :count], text.rotary_emb, positions, embed_first)
            logits = lm_head(out)[0]
            hits = (logits[:-1].argmax(-1) == targets)[scored].float().mean().item()
            key = f"hidden={kind},position=p+{shift},concat={'embed,hidden' if embed_first else 'hidden,embed'}"
            results.setdefault(key, []).append(hits)
            if prompt_index == 0:
                first[key] = (tokens, prompt_len, logits)
    for key, hits in sorted(results.items(), key=lambda kv: -sum(kv[1])):
        print(f"{key}: next-next-token accuracy {', '.join(f'{h:.3f}' for h in hits)}")
    best = max(results, key=lambda k: sum(results[k]))
    tokens, prompt_len, logits = first[best]
    rows = sorted(set(list(range(0, logits.shape[0], 16)) + list(range(logits.shape[0] - 8, logits.shape[0]))))
    save_file({
        "tokens": tokens.to(torch.int64),
        "prompt_tokens": torch.tensor([prompt_len], dtype=torch.int64),
        "logit_rows": torch.tensor(rows, dtype=torch.int64),
        "logits": logits[rows].contiguous(),
        "argmax": logits.argmax(-1).to(torch.int64),
    }, args.golden, metadata={"convention": best})
    print(f"saved {best} for {tokens.shape[0]} tokens to {args.golden}")


if __name__ == "__main__":
    main()
