"""Golden references for tur's Qwen3.5/3.8 text implementation.

Pinned environment: transformers 5.20.2, torch CPU, float32.

    python -I scripts/qwen3_5_reference.py --model <snapshot dir> \
        --golden <out.safetensors> --templates crates/tur/tests/fixtures/qwen3_5_templates.json
"""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

REPO = Path(__file__).resolve().parents[1]
TEMPLATES = {
    "qwen3.5-0.8b": REPO / "crates/tur/tests/fixtures/checkpoints/small-chat_template.jinja",
    "qwen3.8-27b": REPO / "crates/tur/tests/fixtures/checkpoints/qwen3.8-chat_template.jinja",
}

LONG_PROMPT = (
    "Summarise the following notes in one paragraph. "
    + " ".join(
        f"Note {i}: the {w} sensor reported a stable reading after calibration."
        for i, w in enumerate(["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"])
    )
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Look up the weather — returns °C and conditions.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search notes",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        },
    },
]

CASES = [
    {"name": "single_user_default", "messages": [{"role": "user", "content": "Hi there!"}], "options": {}},
    {"name": "thinking_off", "messages": [{"role": "user", "content": "Hi there!"}], "options": {"enable_thinking": False}},
    {"name": "thinking_on", "messages": [{"role": "user", "content": "Hi there!"}], "options": {"enable_thinking": True}},
    {
        "name": "system_multi_turn_inline_reasoning",
        "messages": [
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "<think>\nadd them\n</think>\n\n4"},
            {"role": "user", "content": "And 3+3?"},
        ],
        "options": {"enable_thinking": False},
    },
    {
        "name": "reasoning_field_after_last_query",
        "messages": [
            {"role": "user", "content": "Count to two."},
            {"role": "assistant", "content": "1, 2", "reasoning_content": "Simple."},
        ],
        "options": {"add_generation_prompt": False},
    },
    {
        "name": "tools_multi_step",
        "tools": TOOLS,
        "messages": [
            {"role": "system", "content": "Use tools when useful."},
            {"role": "user", "content": "Weather in Kyiv and Paris? Ünïcode ✓"},
            {
                "role": "assistant",
                "content": "Checking both.",
                "tool_calls": [
                    {"type": "function", "function": {"name": "get_weather", "arguments": {"city": "Kyiv", "days": 2}}},
                    {"type": "function", "function": {"name": "search", "arguments": {"query": "Paris", "filters": {"tags": ["a", "b"]}, "limit": [1, 2]}}},
                ],
            },
            {"role": "tool", "content": "{\"temp\": 12}"},
            {"role": "tool", "content": "Paris: 15°C"},
            {"role": "assistant", "content": "Kyiv 12°C, Paris 15°C."},
            {"role": "user", "content": "Thanks"},
        ],
        "options": {},
    },
    {
        "name": "tool_call_without_content",
        "tools": TOOLS[:1],
        "messages": [
            {"role": "user", "content": "Weather?"},
            {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "get_weather", "arguments": {"city": "Lviv"}}}]},
            {"role": "tool", "content": "sunny"},
        ],
        "options": {"enable_thinking": True},
    },
    {"name": "effort_low", "messages": [{"role": "user", "content": "Hi"}], "options": {"reasoning_effort": "low"}, "only": "qwen3.8-27b"},
    {"name": "effort_medium_system", "messages": [{"role": "system", "content": "Be kind."}, {"role": "user", "content": "Hi"}], "options": {"reasoning_effort": "medium"}, "only": "qwen3.8-27b"},
    {
        "name": "preserve_thinking_false",
        "messages": [
            {"role": "user", "content": "A?"},
            {"role": "assistant", "content": "B", "reasoning_content": "because"},
            {"role": "user", "content": "C?"},
        ],
        "options": {"preserve_thinking": False},
        "only": "qwen3.8-27b",
    },
]


def render_templates(tokenizer, out):
    fixtures = []
    for template_name, path in TEMPLATES.items():
        template = path.read_text()
        for case in CASES:
            if case.get("only", template_name) != template_name:
                continue
            options = dict(case["options"])
            add_generation_prompt = options.pop("add_generation_prompt", True)
            text = tokenizer.apply_chat_template(
                case["messages"],
                tools=case.get("tools"),
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                chat_template=template,
                **options,
            )
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            fixtures.append({
                "template": template_name,
                "name": case["name"],
                "messages": case["messages"],
                "tools": case.get("tools"),
                "add_generation_prompt": add_generation_prompt,
                "options": options,
                "text": text,
                "ids": ids,
            })
    Path(out).write_text(json.dumps({"tokenizer": "Qwen/Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17", "cases": fixtures}, ensure_ascii=False, indent=1) + "\n")
    print(f"wrote {len(fixtures)} template cases to {out}")


@torch.no_grad()
def golden(model_dir, tokenizer, out):
    model = Qwen3_5ForConditionalGeneration.from_pretrained(model_dir, dtype=torch.float32).eval()
    text_model = model.model.language_model
    template = TEMPLATES["qwen3.5-0.8b"].read_text()
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": LONG_PROMPT}], tokenize=False, add_generation_prompt=True,
        chat_template=template, enable_thinking=False,
    )
    ids = torch.tensor([tokenizer(prompt, add_special_tokens=False)["input_ids"]])
    layers = []
    hooks = [layer.register_forward_hook(lambda _m, _i, o: layers.append(o[0] if isinstance(o, tuple) else o)) for layer in text_model.layers]
    logits = model(input_ids=ids).logits[0]
    for hook in hooks:
        hook.remove()
    hidden = torch.stack([h[0] for h in layers])
    count = ids.shape[1]
    rows = sorted(set(list(range(0, count, 16)) + list(range(max(0, count - 8), count))))
    generated = model.generate(ids, max_new_tokens=48, do_sample=False)[0, count:]
    tensors = {
        "input_ids": ids[0].to(torch.int64),
        "hidden": hidden.contiguous(),
        "logit_rows": torch.tensor(rows, dtype=torch.int64),
        "logits": logits[rows].contiguous(),
        "argmax": logits.argmax(-1).to(torch.int64),
        "greedy": generated.to(torch.int64),
    }
    save_file(tensors, out)
    print(f"prompt tokens {count}; hidden {tuple(hidden.shape)}; greedy {tokenizer.decode(generated)!r}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--golden")
    parser.add_argument("--templates")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if args.templates:
        render_templates(tokenizer, args.templates)
    if args.golden:
        golden(args.model, tokenizer, args.golden)


if __name__ == "__main__":
    main()
