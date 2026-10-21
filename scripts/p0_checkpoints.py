"""Materialize an isolated, revision-pinned Hugging Face cache from existing files."""

import argparse
import hashlib
import json
from pathlib import Path


CHECKPOINTS = [
    ("Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca",
     ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors"]),
    ("unsloth/Qwen3-0.6B-GGUF", "50968a4468ef4233ed78cd7c3de230dd1d61a56b",
     ["Qwen3-0.6B-Q4_K_M.gguf"]),
    ("ibm-granite/granite-4.1-3b", "c0650403e44e78ec0262dab1c90914c65b196c4e",
     ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]),
    ("unsloth/granite-4.1-3b-GGUF", "5b88826e4b80789548180f8faab39c5cf68772c9",
     ["granite-4.1-3b-Q4_K_M.gguf"]),
]


def file_hash(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def materialize(cache, output):
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for repo, revision, filenames in CHECKPOINTS:
        directory = "models--" + repo.replace("/", "--")
        snapshot = output / "hub" / directory / "snapshots" / revision
        snapshot.mkdir(parents=True)
        refs = output / "hub" / directory / "refs"
        refs.mkdir()
        (refs / "main").write_text(revision)
        files = []
        for filename in filenames:
            source = cache / directory / "snapshots" / revision / filename
            if not source.is_file():
                raise FileNotFoundError(f"Missing pinned checkpoint file: {source}")
            (snapshot / filename).symlink_to(source.resolve())
            files.append({"name": filename, "bytes": source.stat().st_size, "sha256": file_hash(source)})
        records.append({"repo": repo, "revision": revision, "files": files})
    template = cache / "models--Qwen--Qwen3-0.6B/snapshots" / CHECKPOINTS[0][1] / "tokenizer_config.json"
    template_text = json.loads(template.read_text())["chat_template"]
    manifest = {
        "schema_version": 1,
        "repositories": records,
        "qwen_chat_template_sha256": hashlib.sha256(template_text.encode()).hexdigest(),
        "quantization": "upstream GGUF Q4_K_M; imported unchanged; no local requantization",
        "hf_home": str(output.resolve()),
    }
    (output / "checkpoints.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path.home() / ".cache/huggingface/hub")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    materialize(args.cache, args.output)


if __name__ == "__main__":
    main()
