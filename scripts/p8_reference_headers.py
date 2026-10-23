"""Fetch only pinned Qwen3.6 tensor headers and Gemma 4 configuration, never weights."""
import concurrent.futures
import hashlib
import json
import struct
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "docs/references/p8"
QWEN = "995ad96eacd98c81ed38be0c5b274b04031597b0"


def read(url, size=None, start=0):
    headers = {} if size is None else {"Range": f"bytes={start}-{start + size - 1}"}
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=90) as response:
        if start and response.status != 206:
            raise ValueError("server did not honor a header-only range request")
        return response.read(size)


def header(filename):
    url = f"https://huggingface.co/Qwen/Qwen3.6-35B-A3B/resolve/{QWEN}/{filename}"
    length = struct.unpack("<Q", read(url + "?p8=length", 8))[0]
    if length > 256 << 20:
        raise ValueError("unreasonable header size")
    payload = read(url + "?p8=header", length, 8)
    return filename, json.loads(payload), hashlib.sha256(payload).hexdigest()


def main():
    index = json.loads((ROOT / "qwen3.6-model.safetensors.index.json").read_text())["weight_map"]
    files = sorted(set(index.values()))
    tensors, sources = {}, {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        for filename, entries, digest in executor.map(header, files):
            sources[filename] = {"header_sha256": digest}
            for name, entry in entries.items():
                if name.startswith(("model.language_model.", "mtp.")) or name == "lm_head.weight":
                    tensors[name] = {"shape": entry["shape"], "dtype": entry["dtype"], "file": filename}
    (ROOT / "qwen3.6-text-tensors.json").write_text(json.dumps({"revision": QWEN, "headers": sources, "tensors": dict(sorted(tensors.items()))}, indent=2) + "\n")
    info = json.loads(read("https://huggingface.co/api/models/google/gemma-4-26B-A4B"))
    revision = info["sha"]
    config = read(f"https://huggingface.co/google/gemma-4-26B-A4B/resolve/{revision}/config.json")
    (ROOT / "gemma4-config.json").write_bytes(config)
    (ROOT / "pins.json").write_text(json.dumps({"qwen3.6": QWEN, "transformers": "ea52711b4850c332be3f778396ce882ab84a9011", "gemma4": revision,
        "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ROOT.iterdir()) if p.is_file() and p.name != "pins.json"}}, indent=2) + "\n")


if __name__ == "__main__":
    main()
