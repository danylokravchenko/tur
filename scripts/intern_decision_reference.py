"""Golden references for tur's Intern-Decision readout, from the checkpoint's own `inference.py`.

Each case is an OpenAI `/v1/decisions` body. `to_intern` maps it to the checkpoint's request
format exactly as tur's route does; the official `DecisionEngine` then renders, tokenizes and
scores it. The golden file keeps the rendered prompt text, its token ids, and every field's
candidate probabilities before and after the checkpoint's temperature.

Pinned environment (CPU, float32):

    uv run --no-project --python 3.12 --with torch==2.14.1 --with transformers==5.20.2 \
        --with pillow python -I scripts/intern_decision_reference.py \
        --model <Intern-Decision-0.8B snapshot> --inference <its inference.py> \
        --golden crates/tur/tests/fixtures/intern_decision_golden.json
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

CASES = [
    {
        "input": "The package arrived with a broken screen. I want my money back.",
        "questions": [
            {"type": "predicate", "name": "damaged", "instructions": "Does the customer report a damaged item?"},
            {
                "type": "choice",
                "name": "queue",
                "instructions": "Which team should handle this ticket?",
                "choices": [
                    {"value": "billing", "description": "Refunds and payments"},
                    {"value": "shipping", "description": "Delivery and damaged items"},
                ],
            },
            {
                "type": "score",
                "name": "urgency",
                "instructions": "How urgent is this ticket?",
                "levels": [{"label": "low"}, {"label": "medium"}, {"label": "high", "description": "same day"}],
            },
        ],
    },
    {
        "input": [
            {
                "role": "user",
                "content": [{"type": "input_json", "data": {"order": 1182, "status": "delayed", "days_late": 9}}],
            }
        ],
        "questions": [
            {"type": "choice", "instructions": "Should we refund now?", "choices": [{"value": True}, {"value": False}]},
            {"type": "predicate", "instructions": "Is the order more than a week late?"},
        ],
    },
    {
        "input": [
            {"role": "user", "content": "Can I change the delivery address?"},
            {"role": "assistant", "content": "Yes, until the parcel ships."},
            {"role": "user", "content": "It already shipped. This is useless."},
        ],
        "questions": [
            {
                "type": "choice",
                "name": "sentiment",
                "instructions": "What is the customer's sentiment?",
                "choices": [{"value": "positive"}, {"value": "neutral"}, {"value": "negative"}],
            }
        ],
    },
]


def part_value(part):
    if part["type"] == "input_json":
        return part["data"]
    return part["text"]


def part_text(part):
    value = part_value(part)
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def state(input_):
    """A string stays a string, and a lone user message's lone part its own value. Anything
    else becomes a list of `{role, content}` turns."""
    if isinstance(input_, str):
        return input_
    if len(input_) == 1 and input_[0]["role"] == "user":
        content = input_[0]["content"]
        if isinstance(content, str):
            return content
        if len(content) == 1:
            return part_value(content[0])
    turns = []
    for message in input_:
        content = message["content"]
        text = content if isinstance(content, str) else "\n".join(part_text(p) for p in content)
        turns.append({"role": message["role"], "content": text})
    return turns


def value_key(value):
    return str(value).lower() if isinstance(value, bool) else value


def to_intern(body):
    """Unnamed questions are called q1, q2, ... by position."""
    questions = {}
    for index, question in enumerate(body["questions"]):
        name = question.get("name") or f"q{index + 1}"
        if question["type"] == "predicate":
            questions[name] = {"type": "noul", "instructions": question["instructions"]}
        elif question["type"] == "choice":
            criteria = {value_key(c["value"]): c.get("description") or "" for c in question["choices"]}
            questions[name] = {"type": "choice", "instructions": question["instructions"], "criteria": criteria}
        else:
            levels = [
                level["label"] if not level.get("description") else f"{level['label']}: {level['description']}"
                for level in question["levels"]
            ]
            questions[name] = {"type": "score", "instructions": question["instructions"], "criteria": levels}
    return {"state": state(body["input"]), "questions": questions}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--inference", required=True, type=Path)
    parser.add_argument("--golden", required=True, type=Path)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("intern_inference", args.inference)
    reference = importlib.util.module_from_spec(spec)
    # Its dataclasses look themselves up by module name.
    sys.modules[spec.name] = reference
    spec.loader.exec_module(reference)
    engine = reference.DecisionEngine(checkpoint=args.model, device="cpu", dtype="float32", temperature=1.0)
    golden = {"temperature": reference.DEFAULT_TEMPERATURE, "cases": []}
    for body in CASES:
        row = reference.validate_request(to_intern(body))
        compiled, batch, _ = engine.backend.encode(row)
        text = engine.tokenizer.decode(batch["input_ids"][0], skip_special_tokens=False)
        raw = engine.predict(to_intern(body))["answers"]
        calibrated = reference.scale_result({"answers": raw}, reference.DEFAULT_TEMPERATURE)["answers"]
        golden["cases"].append(
            {
                "request": body,
                "text": text,
                "input_ids": batch["input_ids"][0].tolist(),
                "fields": [
                    {
                        "name": field,
                        "probabilities": list(raw[field]["probabilities"].values()),
                        "calibrated": list(calibrated[field]["probabilities"].values()),
                    }
                    for field in compiled.fields
                ],
            }
        )
    args.golden.write_text(json.dumps(golden, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
