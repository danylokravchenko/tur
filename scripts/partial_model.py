"""Export dense Qwen3.5 layer or MLP-neuron subsets without loading tensor arrays.

Python standard library only. Source tensors keep their stored precision and bytes.
The destination must not exist; a complete directory is published by atomic rename.
This is an unrecovered research candidate, not an assertion of retained capability.
"""

import argparse
import hashlib
import json
import math
import shutil
import struct
import tempfile
from pathlib import Path


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path):
    return json.loads(path.read_bytes(), object_pairs_hook=unique_object)


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def inventory(directory):
    index_path = directory / "model.safetensors.index.json"
    weight_map = read_json(index_path)["weight_map"] if index_path.exists() else None
    names = sorted(set(weight_map.values())) if weight_map is not None else ["model.safetensors"]
    entries = {}
    files = []
    for name in names:
        if not isinstance(name, str) or Path(name).name != name or name in (".", ".."):
            raise ValueError("shard filenames must stay inside the checkpoint directory")
        path = directory / name
        with path.open("rb") as source:
            prefix = source.read(8)
            if len(prefix) != 8:
                raise ValueError(f"truncated SafeTensors prefix: {name}")
            length = struct.unpack("<Q", prefix)[0]
            if length > 256 * 1024 * 1024 or length > path.stat().st_size - 8:
                raise ValueError(f"invalid SafeTensors header length: {name}")
            header = json.loads(source.read(length), object_pairs_hook=unique_object)
        cursor = 0
        tensors = [(key, info) for key, info in header.items() if key != "__metadata__"]
        for key, info in sorted(tensors, key=lambda item: item[1]["data_offsets"]):
            start, end = info["data_offsets"]
            shape = info["shape"]
            if (any(type(n) is not int or n < 0 for n in [start, end, *shape])
                    or start != cursor or end < start):
                raise ValueError(f"invalid shape/offsets: {key}")
            # Source checkpoints contain floating tensors only; prepacked input is excluded.
            size = {"BF16": 2, "F16": 2, "F32": 4}.get(info["dtype"])
            if size is None or end - start != math.prod(shape) * size:
                raise ValueError(f"unsupported dtype or payload size: {key}")
            if key in entries:
                raise ValueError(f"duplicate tensor: {key}")
            if weight_map is not None and weight_map.get(key) != name:
                raise ValueError(f"index disagrees with shard: {key}")
            entries[key] = (path, 8 + length + start, info)
            cursor = end
        if 8 + length + cursor != path.stat().st_size:
            raise ValueError(f"SafeTensors file length disagrees with header: {name}")
        files.append({"file": name, "bytes": path.stat().st_size, "sha256": digest(path)})
    if weight_map is not None and set(weight_map) != set(entries):
        raise ValueError("index contains missing tensors")
    return entries, files


def text_config(raw):
    architecture = (raw.get("model_type"), raw.get("architectures"))
    if architecture == ("qwen3_5", ["Qwen3_5ForConditionalGeneration"]):
        return raw["text_config"], "model.language_model"
    if architecture == ("qwen3_5_text", ["Qwen3_5ForCausalLM"]):
        return raw, "model"
    raise ValueError("partial-model pilot supports dense Qwen3.5 checkpoints only")


def choose_layers(config, raw, revision, drop_layers, scores, remove):
    count = config["num_hidden_layers"]
    kinds = config["layer_types"]
    if count != len(kinds) or count < 2 or set(kinds) - {"linear_attention", "full_attention"}:
        raise ValueError("invalid dense hybrid layer layout")
    if scores is not None:
        if (scores.get("schema_version") != 1 or scores["config"] != raw
                or scores["source_revision"] != revision):
            raise ValueError("calibration config/revision does not match source")
        rows = scores["scores"]
        if sorted(row["layer"] for row in rows) != list(range(1, count)):
            raise ValueError("calibration must score every layer after layer 0 exactly once")
        if any(not math.isfinite(row["influence"]) or not 0 <= row["influence"] <= 2 for row in rows):
            raise ValueError("invalid influence scores")
        ranked = sorted(
            (row for row in rows if row["layer"] < count - 1 and kinds[row["layer"]] == "linear_attention"),
            key=lambda row: (row["influence"], row["layer"]),
        )
        if remove is None or not 0 <= remove <= len(ranked):
            raise ValueError("remove count exceeds eligible recurrent layers")
        dropped = sorted(row["layer"] for row in ranked[:remove])
    else:
        dropped = sorted(drop_layers)
    if len(set(dropped)) != len(dropped) or any(type(i) is not int or not 0 <= i < count for i in dropped):
        raise ValueError("removed layer IDs must be unique and in range")
    if any(i in (0, count - 1) or kinds[i] != "linear_attention" for i in dropped):
        raise ValueError("first, last and full-attention layers are protected")
    return [i for i in range(count) if i not in dropped], dropped


def select_tensors(entries, prefix, config, retained):
    mapping = {old: new for new, old in enumerate(retained)}
    selected = {}
    text_parameters = 0
    seen_layers = set()
    for name, entry in sorted(entries.items()):
        info = entry[2]
        if name.startswith(prefix + ".layers."):
            suffix = name[len(prefix + ".layers."):]
            layer, suffix = suffix.split(".", 1)
            layer = int(layer)
            seen_layers.add(layer)
            text_parameters += math.prod(info["shape"])
            if layer in mapping:
                selected[f"model.layers.{mapping[layer]}.{suffix}"] = entry
        elif name in (prefix + ".embed_tokens.weight", prefix + ".norm.weight"):
            text_parameters += math.prod(info["shape"])
            selected["model" + name[len(prefix):]] = entry
        elif name == "lm_head.weight" and not config["tie_word_embeddings"]:
            text_parameters += math.prod(info["shape"])
            selected[name] = entry
        elif name.startswith(prefix + "."):
            raise ValueError(f"unexpected text tensor: {name}")
    if seen_layers != set(range(config["num_hidden_layers"])):
        raise ValueError("checkpoint layer inventory disagrees with config")
    required = {"model.embed_tokens.weight", "model.norm.weight"}
    if not config["tie_word_embeddings"]:
        required.add("lm_head.weight")
    if not required.issubset(selected):
        raise ValueError("missing embedding, final norm or LM head")
    return selected, text_parameters


def choose_neurons(config, raw, revision, scores, width):
    original = config["intermediate_size"]
    if type(width) is not int or not 0 < width <= original:
        raise ValueError("MLP width must be positive and no larger than the source")
    if (scores.get("schema_version") != 1 or scores.get("axis") != "mlp_width"
            or scores["config"] != raw or scores["source_revision"] != revision):
        raise ValueError("neuron calibration config/revision does not match source")
    rows = scores["scores"]
    if sorted(row["layer"] for row in rows) != list(range(config["num_hidden_layers"])):
        raise ValueError("neuron calibration must cover every original layer exactly once")
    neurons = {}
    for row in rows:
        importance = row["importance"]
        if len(importance) != original or any(not math.isfinite(n) or n < 0 for n in importance):
            raise ValueError("invalid neuron importance")
        ranked = sorted(range(original), key=lambda i: (-importance[i], i))
        neurons[str(row["layer"])] = sorted(ranked[:width])
    return neurons


def neuron_axes(selected, config, neurons):
    axes = {}
    for layer, ids in neurons.items():
        prefix = f"model.layers.{layer}.mlp."
        for projection, axis in [("gate_proj", 0), ("up_proj", 0), ("down_proj", 1)]:
            name = prefix + projection + ".weight"
            expected = [config["intermediate_size"], config["hidden_size"]]
            if axis == 1:
                expected.reverse()
            if name not in selected or selected[name][2]["shape"] != expected:
                raise ValueError(f"missing or malformed dense MLP tensor: {name}")
            axes[name] = (axis, ids)
    return axes


def contiguous_runs(ids):
    runs = []
    for value in ids:
        if runs and runs[-1][1] == value:
            runs[-1] = (runs[-1][0], value + 1)
        else:
            runs.append((value, value + 1))
    return runs


def copy_bytes(source, output, size):
    while size:
        data = source.read(min(size, 1024 * 1024))
        if not data:
            raise ValueError("truncated tensor payload")
        output.write(data)
        size -= len(data)


def copy_tensor(output, entry, selection):
    path, offset, info = entry
    with path.open("rb") as source:
        source.seek(offset)
        if selection is None:
            copy_bytes(source, output, info["data_offsets"][1] - info["data_offsets"][0])
            return
        axis, ids = selection
        rows, columns = info["shape"]
        size = {"BF16": 2, "F16": 2, "F32": 4}[info["dtype"]]
        runs = contiguous_runs(ids)
        if axis == 0:
            for start, end in runs:
                source.seek(offset + start * columns * size)
                copy_bytes(source, output, (end - start) * columns * size)
        else:
            for _row in range(rows):
                row = source.read(columns * size)
                if len(row) != columns * size:
                    raise ValueError("truncated down-projection row")
                output.write(b"".join(row[start * size:end * size] for start, end in runs))


def write_weights(path, selected, axes):
    header = {"__metadata__": {"format": "pt"}}
    cursor = 0
    parameters = 0
    for name, (_, _, info) in selected.items():
        shape = list(info["shape"])
        if name in axes:
            axis, ids = axes[name]
            shape[axis] = len(ids)
        elements = math.prod(shape)
        size = elements * {"BF16": 2, "F16": 2, "F32": 4}[info["dtype"]]
        parameters += elements
        header[name] = {**info, "shape": shape, "data_offsets": [cursor, cursor + size]}
        cursor += size
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    with path.open("xb") as output:
        output.write(struct.pack("<Q", len(encoded)))
        output.write(encoded)
        for name, entry in selected.items():
            copy_tensor(output, entry, axes.get(name))
    return cursor, parameters


def export_model(model, output, revision, drop_layers=(), scores=None, remove=None,
                 neuron_scores=None, width=None):
    model, output = Path(model).resolve(), Path(output).absolute()
    if not revision.strip():
        raise ValueError("an explicit source revision is required")
    if output.exists() or output.is_symlink():
        raise ValueError("output must not exist; source checkpoints are never overwritten")
    raw = read_json(model / "config.json")
    config, prefix = text_config(raw)
    if neuron_scores is not None and (drop_layers or scores is not None or remove is not None):
        raise ValueError("depth and width pruning must be evaluated separately")
    retained, dropped = choose_layers(config, raw, revision, drop_layers, scores, remove)
    neurons = choose_neurons(config, raw, revision, neuron_scores, width) if neuron_scores is not None else {}
    entries, files = inventory(model)
    selected, original_parameters = select_tensors(entries, prefix, config, retained)
    axes = neuron_axes(selected, config, neurons)
    exported_config = dict(config)
    exported_config.update(
        architectures=["Qwen3_5ForCausalLM"], num_hidden_layers=len(retained),
        layer_types=[config["layer_types"][i] for i in retained], mtp_num_hidden_layers=0,
    )
    # An interval no longer describes arbitrary explicit retained layer positions.
    exported_config.pop("full_attention_interval", None)
    if neurons:
        exported_config["intermediate_size"] = width
    copied = {}
    with tempfile.TemporaryDirectory(prefix=".tur-partial-", dir=output.parent) as temporary:
        staging = Path(temporary) / "checkpoint"
        staging.mkdir()
        for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                     "generation_config.json", "tokenizer.model", "vocab.json", "merges.txt", "LICENSE"):
            source = model / name
            if source.is_file():
                shutil.copyfile(source, staging / name)
                copied[name] = digest(staging / name)
        if "tokenizer.json" not in copied or "tokenizer_config.json" not in copied:
            raise ValueError("tokenizer.json and tokenizer_config.json are required")
        tokenizer_config = read_json(model / "tokenizer_config.json")
        if "chat_template.jinja" not in copied and not tokenizer_config.get("chat_template"):
            raise ValueError("source checkpoint has no chat template")
        write_json(staging / "config.json", exported_config)
        payload_bytes, parameters = write_weights(staging / "model.safetensors", selected, axes)
        manifest = {
            "schema_version": 1,
            "method": "coupled gate/up rows and down columns; no recovery" if neurons else "permanent dense decoder-layer removal; no recovery",
            "source_revision": revision, "source_config_sha256": digest(model / "config.json"),
            "source_files": files, "copied_files_sha256": copied,
            "source_text_parameters": original_parameters,
            "retained_text_parameters": parameters,
            "retained_layers": retained, "removed_layers": dropped,
            "layer_mapping": {str(new): old for new, old in enumerate(retained)},
            "protected": "all layers, mixers and residual width" if neurons else "first, last and every full-attention layer",
            "retained_neurons": neurons,
            "excluded": ["vision tower", "MTP head"], "precision": "unchanged source tensor bytes",
            "calibration": neuron_scores if neurons else scores, "payload_bytes": payload_bytes,
            "weights_sha256": digest(staging / "model.safetensors"),
            "config_sha256": digest(staging / "config.json"),
        }
        write_json(staging / "partial-model.json", manifest)
        staging.rename(output)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--drop-layers", help="comma-separated original recurrent layer IDs; empty keeps all")
    selection.add_argument("--scores", type=Path, help="layer_influence JSON; rank on calibration only")
    selection.add_argument("--neuron-scores", type=Path, help="neuron_influence JSON; keep every layer")
    parser.add_argument("--remove", type=int, help="number of recurrent layers to remove using --scores")
    parser.add_argument("--width", type=int, help="retained MLP intermediate width using --neuron-scores")
    args = parser.parse_args()
    try:
        if args.scores is None and args.remove is not None:
            raise ValueError("--remove requires --scores")
        if args.neuron_scores is None and args.width is not None:
            raise ValueError("--width requires --neuron-scores")
        dropped = [] if not args.drop_layers else [int(i) for i in args.drop_layers.split(",")]
        manifest = export_model(args.model, args.output, args.revision, dropped,
                                read_json(args.scores) if args.scores else None, args.remove,
                                read_json(args.neuron_scores) if args.neuron_scores else None, args.width)
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.error(str(error))
    print(json.dumps({key: manifest[key] for key in (
        "source_text_parameters", "retained_text_parameters", "retained_layers", "removed_layers", "payload_bytes")}, indent=2))


if __name__ == "__main__":
    main()
