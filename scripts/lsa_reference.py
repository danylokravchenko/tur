"""Latent sparse attention references for tur's Qwen3.5/3.8 full-attention layers.

Pinned environment: transformers 5.20.2, torch CPU, float32, and Prefrontal-AI/latent-sparse-attention at
5ebea732d639576d2a7579b0de40e6252f9001f5, whose `src` directory `--lsa` names. Only its pure-torch `lsa.core` is
imported (index space, Ward and the flat read), which is the oracle that repository tests its own kernels against.

`fit` takes a training-free map, each layer's top 64 principal directions of its normalized position-free keys, from
calibration text. `measure` prefills held-out text densely, then runs its last `--tail` tokens once densely and once
with each full-attention layer reading only the keys LSA selects, and reports how much dense attention weight the
selection holds and how far next-token logits move:

    python -I scripts/lsa_reference.py fit --model <snapshot dir> --lsa <repo>/src \
        --text a.txt b.txt --context 32768 --map map.safetensors
    python -I scripts/lsa_reference.py measure --model <snapshot dir> --lsa <repo>/src \
        --text c.txt d.txt --context 32768 --map map.safetensors > report.json

`golden` writes the reference's buckets and key lists for synthetic keys, queries and map that
`crates/tur/tests/integration/lsa.rs` rebuilds bit for bit from the same integer hash:

    python -I scripts/lsa_reference.py golden --lsa <repo>/src \
        --out crates/tur/tests/fixtures/lsa/golden.safetensors
"""

import argparse
import copy
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import AttentionInterface, AutoTokenizer, Qwen3_5ForConditionalGeneration
from transformers.masking_utils import AttentionMaskInterface, flash_attention_mask

# Qwen3.5 rotates the first quarter of each 256-dim head; the other 192 dims score independently of position
ROTARY_DIM = 64
INDEX_DIM = 64
POOL_SIZE, BUCKET_SIZE, BUCKET_CAP, CLUSTER_BLOCK = 4, 8, 16, 512
# Prefill chunk, as tur's model benchmark prefills
CHUNK = 2048
# Rows sampled evenly from each document for the PCA, the reference's `Space.INIT_ROWS`
PCA_ROWS = 8192
IMPLEMENTATION = "lsa_reference"


class Probe:
    """What the attention function does at each full-attention layer during the current forward."""

    def __init__(self, budget: argparse.Namespace, maps: dict[int, torch.Tensor]) -> None:
        self.mode = "dense"
        self.budget = budget
        self.maps = maps
        # Keys in the document, so the chunk that holds all of them is recognised
        self.total = 0
        self.samples: dict[int, list[torch.Tensor]] = {}
        self.selected: dict[int, torch.Tensor] = {}
        self.layers: dict[int, dict[str, float]] = {}


PROBE: Probe | None = None


def index_rows(query: torch.Tensor, key: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Unit index queries [Tq, Hkv * 192] and keys [Tk, Hkv * 192], before any map."""
    from lsa.core import RotaryLayout, position_free, regroup

    # Rotation leaves the dims after the prefix unchanged, so rotated heads give the same rows as pre-RoPE ones
    queries, keys = regroup(*position_free(query.float(), key.float(), RotaryLayout("prefix", ROTARY_DIM)))
    return queries[0], keys[0]


def select(layer: int, query: torch.Tensor, key: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """The reference's flat read: selected key ids [Tq, width], -1 padded."""
    from lsa.core import build_buckets, project, read

    budget = PROBE.budget
    queries, keys = index_rows(query, key)
    index_map = PROBE.maps[layer]
    buckets = build_buckets(project(keys, index_map), POOL_SIZE, BUCKET_SIZE, BUCKET_CAP, CLUSTER_BLOCK)
    selected, _ = read(
        project(queries, index_map), positions, buckets, budget.top_p, budget.top_k, budget.sink, budget.window
    )
    return selected


def dense_probs(query: torch.Tensor, key: torch.Tensor, scaling: float, positions: torch.Tensor) -> torch.Tensor:
    """Dense attention weights [Hq, Tq, Tk] in F32."""
    groups = query.shape[1] // key.shape[1]
    keys = key[0].float().repeat_interleave(groups, 0)
    scores = query[0].float() @ keys.transpose(1, 2) * scaling
    visible = torch.arange(key.shape[2])[None, :] <= positions[:, None]
    return scores.masked_fill(~visible, float("-inf")).softmax(-1)


def mask_of(ids: torch.Tensor, key_count: int) -> torch.Tensor:
    """Boolean mask [Tq, Tk] holding each row's ids, ignoring -1 padding."""
    mask = torch.zeros(ids.shape[0], key_count + 1, dtype=torch.bool)
    mask.scatter_(1, torch.where(ids >= 0, ids, key_count).long(), True)
    return mask[:, :key_count]


def record(layer: int, probs: torch.Tensor, selected: torch.Tensor, positions: torch.Tensor) -> None:
    """Attention weight held by LSA's keys, by the same count of most recent keys, and by the best shared list."""
    key_count = probs.shape[2]
    width = int((selected >= 0).sum(1).max())
    lsa = (probs * mask_of(selected, key_count)).sum(-1).mean(0)
    sink = PROBE.budget.sink
    recent = torch.full_like(selected, -1)
    for row, position in enumerate(positions.tolist()):
        ids = torch.cat([torch.arange(min(sink, position + 1)), torch.arange(max(sink, position + 1 - (width - sink)), position + 1)])
        recent[row, : ids.numel()] = ids
    held_recent = (probs * mask_of(recent, key_count)).sum(-1).mean(0)
    # The best list one set of keys can be for every head: the keys of highest head-mean weight
    best = probs.mean(0).topk(min(width, key_count), dim=-1).indices
    held_best = (probs * mask_of(best, key_count)).sum(-1).mean(0)
    PROBE.layers[layer] = {
        "lsa": float(lsa.mean()),
        "lsa_min": float(lsa.min()),
        "recent": float(held_recent.mean()),
        "best": float(held_best.mean()),
        "keys": width,
    }


def capture(layer: int, key: torch.Tensor) -> None:
    """Sample a document's index keys, evenly across it, for the PCA."""
    # Keys stand in for queries, whose rows are discarded
    _, keys = index_rows(key, key)
    stride = max(1, keys.shape[0] // PCA_ROWS)
    PROBE.samples.setdefault(layer, []).append(keys[::stride][:PCA_ROWS])


def attend(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, scaling: float, mask) -> torch.Tensor:
    """Grouped-query attention [B, Tq, Hq, dh], one K/V head and its query group at a time."""
    kv_heads = key.shape[1]
    groups = query.shape[1] // kv_heads
    outputs = [
        torch.nn.functional.scaled_dot_product_attention(
            query[:, head * groups : (head + 1) * groups],
            key[:, head : head + 1].expand(-1, groups, -1, -1),
            value[:, head : head + 1].expand(-1, groups, -1, -1),
            attn_mask=mask,
            scale=scaling,
        )
        for head in range(kv_heads)
    ]
    return torch.cat(outputs, dim=1).transpose(1, 2).contiguous()


def attention(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kwargs):
    """Causal attention aligned to the bottom right, dense or over LSA's selection, per the probe's mode."""
    from torch.nn.attention.bias import causal_lower_right

    if attention_mask is not None:
        raise ValueError("the reference attention takes no padding mask")
    layer = module.layer_idx
    query_count, key_count = query.shape[2], key.shape[2]
    positions = torch.arange(key_count - query_count, key_count)
    complete = key_count == PROBE.total
    if PROBE.mode == "capture" and complete:
        capture(layer, key)
    if PROBE.mode == "measure":
        selected = select(layer, query, key, positions)
        PROBE.selected[layer] = selected
        record(layer, dense_probs(query, key, scaling, positions), selected, positions)
    if PROBE.mode == "sparse":
        mask = mask_of(PROBE.selected[layer], key_count)
        return attend(query, key, value, scaling, mask), None
    return attend(query, key, value, scaling, causal_lower_right(query_count, key_count)), None


def load(model_dir: Path):
    """The checkpoint in F32 on the CPU, its vision tower dropped, and its tokenizer."""
    AttentionInterface.register(IMPLEMENTATION, attention)
    AttentionMaskInterface.register(IMPLEMENTATION, flash_attention_mask)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(model_dir, dtype=torch.float32)
    model.eval()
    model.model.visual = None
    model.model.language_model.config._attn_implementation = IMPLEMENTATION
    return model, AutoTokenizer.from_pretrained(model_dir)


def prefill(model, ids: list[int]):
    """Committed cache after prefilling `ids` in chunks."""
    cache = None
    for start in range(0, len(ids), CHUNK):
        chunk = torch.tensor([ids[start : start + CHUNK]])
        output = model(input_ids=chunk, past_key_values=cache, use_cache=True, logits_to_keep=1)
        cache = output.past_key_values
    return cache


def documents(paths: list[Path], tokenizer, context: int) -> list[tuple[str, list[int]]]:
    """Each text's first `context` tokens; a text shorter than that is refused."""
    docs = []
    for path in paths:
        ids = tokenizer(path.read_text(), add_special_tokens=False).input_ids
        if len(ids) < context:
            raise ValueError(f"{path} has {len(ids)} tokens, fewer than the {context} asked for")
        docs.append((path.name, ids[:context]))
    return docs


def fit(model, docs, out: Path) -> None:
    """Fit and save each full-attention layer's training-free map [Hkv * 192, 64]."""
    for name, ids in docs:
        print(f"capturing {name}", file=sys.stderr)
        PROBE.mode, PROBE.total = "capture", len(ids)
        with torch.no_grad():
            prefill(model, ids)
    maps = {}
    for layer, samples in sorted(PROBE.samples.items()):
        rows = torch.cat(samples)
        # The reference's `_leading_directions`: right singular vectors of the centred rows
        _, _, directions = torch.linalg.svd(rows - rows.mean(0), full_matrices=False)
        maps[f"layer.{layer}"] = directions[:INDEX_DIM].t().contiguous()
    save_file(maps, str(out), metadata={"rows": str(sum(s.shape[0] for s in samples)), "index_dim": str(INDEX_DIM)})


def logit_report(dense: torch.Tensor, sparse: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    """KL(dense || sparse), top-1 agreement and next-token NLL of both, over the tail's positions."""
    dense_log, sparse_log = dense.log_softmax(-1), sparse.log_softmax(-1)
    kl = (dense_log.exp() * (dense_log - sparse_log)).sum(-1)
    nll = lambda logs: float(-logs[:-1].gather(1, targets[:, None]).mean())
    return {
        "kl_mean": float(kl.mean()),
        "kl_max": float(kl.max()),
        "top1_agreement": float((dense.argmax(-1) == sparse.argmax(-1)).float().mean()),
        "nll_dense": nll(dense_log),
        "nll_sparse": nll(sparse_log),
    }


def measure(model, docs, tail: int) -> list[dict]:
    """Per document: attention weight held per layer, and how the tail's logits move under sparse reads."""
    reports = []
    for name, ids in docs:
        print(f"measuring {name}", file=sys.stderr)
        PROBE.mode, PROBE.total, PROBE.layers = "dense", len(ids), {}
        with torch.no_grad():
            cache = prefill(model, ids[:-tail])
            dense_cache = copy.deepcopy(cache)
            tail_ids = torch.tensor([ids[-tail:]])
            PROBE.mode = "measure"
            dense = model(input_ids=tail_ids, past_key_values=dense_cache, use_cache=True).logits[0]
            PROBE.mode = "sparse"
            sparse = model(input_ids=tail_ids, past_key_values=cache, use_cache=True).logits[0]
        report = {"document": name, "context": len(ids), "tail": tail, "layers": dict(PROBE.layers)}
        report.update(logit_report(dense, sparse, torch.tensor(ids[-tail + 1 :])))
        reports.append(report)
    return reports


# The golden case, mirrored by `crates/tur/tests/integration/lsa.rs`
GOLDEN_KV_HEADS, GOLDEN_HEADS, GOLDEN_DIM, GOLDEN_TOKENS = 2, 4, 256, 1700
GOLDEN_SEGMENT, GOLDEN_TOPICS = 23, 6
GOLDEN_POSITIONS = [511, 512, 1023, 1024, 1300, 1535, 1536, 1699]
# (top_k, top_p, window, sink): one budget cut short inside a bucket, one reading every bucket
GOLDEN_BUDGETS = {"cut": (200, 32, 512, 4), "all": (2048, 1024, 640, 4)}


def unit(seed: int, count: int) -> torch.Tensor:
    """Values in [-1, 1) on a 2^-23 grid, from a 32-bit integer hash of (seed, index)."""
    mask = 0xFFFFFFFF
    out = []
    for i in range(count):
        x = (seed * 0x9E3779B9 + i) & mask
        x ^= x >> 16
        x = (x * 0x7FEB352D) & mask
        x ^= x >> 15
        x = (x * 0x846CA68B) & mask
        x ^= x >> 16
        out.append((x >> 8) / float(1 << 23) - 1.0)
    return torch.tensor(out, dtype=torch.float32)


def golden_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Keys [kv_heads, tokens, dim], queries [heads, positions, dim] and the map [kv_heads * 192, 64]."""
    kv, heads, dim, tokens = GOLDEN_KV_HEADS, GOLDEN_HEADS, GOLDEN_DIM, GOLDEN_TOKENS
    topics = unit(11, GOLDEN_TOPICS * kv * dim).view(GOLDEN_TOPICS, kv, dim)
    segments = torch.tensor([(t // GOLDEN_SEGMENT) % GOLDEN_TOPICS for t in range(tokens)])
    keys = topics[segments] + unit(12, tokens * kv * dim).view(tokens, kv, dim) * 0.5
    count = len(GOLDEN_POSITIONS)
    chosen = torch.tensor([(t * 7) % GOLDEN_TOPICS for t in GOLDEN_POSITIONS])
    group = torch.arange(heads) // (heads // kv)
    queries = topics[chosen][:, group] + unit(14, count * heads * dim).view(count, heads, dim) * 0.5
    index_map = unit(13, kv * (dim - ROTARY_DIM) * INDEX_DIM).view(kv * (dim - ROTARY_DIM), INDEX_DIM)
    return keys.transpose(0, 1).contiguous(), queries.transpose(0, 1).contiguous(), index_map


def golden(out: Path) -> None:
    """Write the reference's buckets and selections for the golden case."""
    from lsa.core import RotaryLayout, build_buckets, position_free, project, read, regroup

    keys, queries, index_map = golden_inputs()
    rotary = RotaryLayout("prefix", ROTARY_DIM)
    _, index_keys = regroup(*position_free(keys[None], keys[None], rotary))
    index_queries, _ = regroup(*position_free(queries[None], keys[None], rotary))
    buckets = build_buckets(project(index_keys[0], index_map), POOL_SIZE, BUCKET_SIZE, BUCKET_CAP, CLUSTER_BLOCK)
    members = buckets.bucket_members.view(-1, 64, BUCKET_CAP)
    pools = torch.full((members.shape[0], 64, BUCKET_CAP // POOL_SIZE), -1.0)
    for block in range(members.shape[0]):
        for bucket in range(64):
            row = members[block, bucket]
            row = row[row >= 0] - block * CLUSTER_BLOCK
            if not torch.equal(row, (row[::POOL_SIZE, None] + torch.arange(POOL_SIZE)).flatten()):
                raise ValueError(f"bucket {bucket} of block {block} splits a pool")
            pools[block, bucket, : row.numel() // POOL_SIZE] = (row[::POOL_SIZE] // POOL_SIZE).float()
    tensors = {
        "centroids": buckets.bucket_centroids.view(-1, 64 * INDEX_DIM).contiguous(),
        "members": pools.view(members.shape[0], -1).contiguous(),
        "positions": torch.tensor(GOLDEN_POSITIONS, dtype=torch.int64),
    }
    positions = torch.tensor(GOLDEN_POSITIONS)
    for name, (top_k, top_p, window, sink) in GOLDEN_BUDGETS.items():
        selected, lengths = read(project(index_queries[0], index_map), positions, buckets, top_p, top_k, sink, window)
        tensors[f"keys_{name}"] = torch.cat([row[:n] for row, n in zip(selected, lengths.tolist())]).to(torch.int64)
        tensors[f"lengths_{name}"] = lengths.to(torch.int64)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out), metadata={"reference": "Prefrontal-AI/latent-sparse-attention@5ebea732d639576d2a7579b0de40e6252f9001f5"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["fit", "measure", "golden"])
    parser.add_argument("--model", type=Path)
    parser.add_argument("--lsa", type=Path, required=True, help="the reference repository's src directory")
    parser.add_argument("--text", type=Path, nargs="+")
    parser.add_argument("--context", type=int, default=32768)
    parser.add_argument("--map", type=Path)
    parser.add_argument("--out", type=Path, help="golden file to write")
    parser.add_argument("--tail", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=4096)
    parser.add_argument("--top-p", type=int, default=1024)
    parser.add_argument("--window", type=int, default=CLUSTER_BLOCK)
    parser.add_argument("--sink", type=int, default=4)
    args = parser.parse_args()
    sys.path.insert(0, str(args.lsa.resolve()))
    if args.command == "golden":
        if args.out is None:
            parser.error("golden needs --out")
        golden(args.out)
        return
    if args.model is None or args.text is None or args.map is None:
        parser.error(f"{args.command} needs --model, --text and --map")
    if args.tail < 2 or args.context - args.tail < CLUSTER_BLOCK or args.window < CLUSTER_BLOCK:
        parser.error("need a tail of at least 2, a complete cluster block before it, and a window of at least a block")

    torch.manual_seed(0)
    global PROBE
    model, tokenizer = load(args.model)
    docs = documents(args.text, tokenizer, args.context)
    if args.command == "fit":
        PROBE = Probe(args, {})
        fit(model, docs, args.map)
        return
    maps = {int(name.split(".")[1]): tensor for name, tensor in load_file(str(args.map)).items()}
    PROBE = Probe(args, maps)
    reports = measure(model, docs, args.tail)
    budget = vars(args) | {"model": str(args.model), "lsa": str(args.lsa), "text": [str(t) for t in args.text], "map": str(args.map)}
    print(json.dumps({"budget": budget, "documents": reports}, indent=2, default=str))


if __name__ == "__main__":
    main()
