"""Golden references for tur's Intern-Decision readout, from the checkpoint's own `inference.py`.

Each case is an OpenAI `/v1/decisions` body. `to_intern` maps it to the checkpoint's request
format exactly as tur's route does; the official `DecisionEngine` then renders, tokenizes and
scores it. The golden file keeps the chat template's text (one placeholder per image), the
token ids after image expansion, and every field's candidate probabilities before and after
the checkpoint's temperature. Images are `data:` URLs of `tests/fixtures/vision/small.png`,
written to temporary files because the reference reads paths.

Pinned environment (CPU, float32):

    uv run --no-project --python 3.12 --with torch==2.14.1 --with transformers==5.20.2 \
        --with pillow python -I scripts/intern_decision_reference.py \
        --model <Intern-Decision-0.8B snapshot> --inference <its inference.py> \
        --golden crates/tur/tests/fixtures/intern_decision_golden.json
"""

import argparse
import base64
import copy
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SMALL_PNG = "data:image/png;base64," + base64.b64encode(
    (REPO / "crates/tur/tests/fixtures/vision/small.png").read_bytes()
).decode()

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
    {
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_image", "image_url": SMALL_PNG},
                    {"type": "input_text", "text": "Photo from the returns desk."},
                ],
            }
        ],
        "questions": [
            {"type": "predicate", "name": "colourful", "instructions": "Does the photo show bright colours?"},
            {
                "type": "choice",
                "name": "pattern",
                "instructions": "What does the photo mainly show?",
                "choices": [{"value": "stripes"}, {"value": "a face"}, {"value": "text"}],
            },
        ],
    },
]


def is_image(part):
    return part["type"] == "input_image"


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
        parts = [p for p in content if not is_image(p)]
        if len(parts) == 1:
            return part_value(parts[0])
        if not parts:
            return ""
    turns = []
    for message in input_:
        content = message["content"]
        parts = [] if isinstance(content, str) else [p for p in content if not is_image(p)]
        text = content if isinstance(content, str) else "\n".join(part_text(p) for p in parts)
        turns.append({"role": message["role"], "content": text})
    return turns


def value_key(value):
    return str(value).lower() if isinstance(value, bool) else value


def image_urls(input_):
    if isinstance(input_, str):
        return []
    return [
        p["image_url"]
        for m in input_
        if not isinstance(m["content"], str)
        for p in m["content"]
        if is_image(p)
    ]


def image_files(urls, directory):
    paths = []
    for index, url in enumerate(urls):
        path = Path(directory) / f"image-{index}.png"
        path.write_bytes(base64.b64decode(url.split(",", 1)[1]))
        paths.append(str(path))
    return paths


def to_intern(body, directory):
    """Unnamed questions are called q1, q2, ... by position. Images go to `images`."""
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
    row = {"state": state(body["input"]), "questions": questions}
    images = image_files(image_urls(body["input"]), directory)
    if images:
        row["images"] = images
    return row


def template_text(engine, reference, row):
    """The chat template's output before the processor expands each image placeholder."""
    messages = copy.deepcopy(reference.compile_row(row).messages)
    if row.get("images"):
        messages[1]["content"] = [
            {"type": "image"} if part["type"] == "image_url" else part for part in messages[1]["content"]
        ]
    template = engine.backend.processor if row.get("images") else engine.tokenizer
    return template.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, enable_thinking=False, add_vision_id=True
    )


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
    directory = tempfile.mkdtemp()
    for body in CASES:
        row = reference.validate_request(to_intern(body, directory))
        compiled, batch, _ = engine.backend.encode(row)
        text = template_text(engine, reference, row)
        raw = engine.predict(to_intern(body, directory))["answers"]
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
