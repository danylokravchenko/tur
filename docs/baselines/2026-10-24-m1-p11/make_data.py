"""Build P11 calibration and held-out texts from pinned local sources (see the P11 report)."""
import json, random, sys
R = sys.argv[1]  # cargo registry source directory
OUT = sys.argv[2]
TEMPLATES = sys.argv[3]

def read(path, limit=None):
    text = open(path, encoding="utf-8").read()
    return text[:limit] if limit else text

cases = json.load(open(TEMPLATES))["cases"]
qwen_readme = sys.argv[4]
# Calibration: model-card prose, Candle Rust source, URL-test JSON and half the chat/tool
# template renderings. Each source is cut to ~18K characters (~4K tokens).
calibration = "\n\n".join([
    read(qwen_readme, 18000),
    read(f"{R}/candle-core-0.11.0/src/tensor.rs", 18000),
    read(f"{R}/url-2.5.8/tests/urltestdata.json", 18000),
    "\n\n".join(c["text"] for c in cases[0::2]),
])
open(f"{OUT}/calibration.txt", "w").write(calibration)
# Held-out code and JSON/tools, disjoint from calibration sources.
open(f"{OUT}/heldout-code.txt", "w").write(
    read(f"{R}/tokio-1.52.3/src/runtime/scheduler/multi_thread/worker.rs")
    + "\n\n" + read(f"{R}/tokio-1.52.3/src/sync/mpsc/chan.rs"))
open(f"{OUT}/heldout-tools.txt", "w").write(
    "\n\n".join(c["text"] for c in cases[1::2]) + "\n\n"
    + read(f"{R}/rmcp-3.5.1/tests/test_message_schema/client_json_rpc_message_schema_current.json"))
# Long-context retrieval: codes listed first, unrelated prose, then questions whose answers
# (between markers) are only predictable by attending back to the list.
rng = random.Random(1011)
adjectives = "Amber Basalt Cobalt Dusky Ember Fallow Granite Hollow Indigo Jade Kestrel Linen Marble Nickel Onyx Pewter".split()
nouns = "Falcon Harbor Lantern Meadow Orchard Quarry Ridge Summit Thicket Vessel Willow Yarrow".split()
names = [f"{a} {n}" for a in adjectives for n in nouns]
rng.shuffle(names)
vaults = [(name, rng.randrange(10000, 100000)) for name in names[:120]]
lines = ["The registry below lists the access code of every storage vault."]
lines += [f"Vault {name}: access code {code}." for name, code in vaults]
filler = read(f"{R}/regex-1.13.1/src/lib.rs", 14000)
questions = rng.sample(vaults[:90], 60)
text = "\n".join(lines) + "\n\n" + filler + "\n\nAnswer each question from the registry above.\n"
text += "".join(f"Question: What is the access code of vault {n}? Answer: ⟦{c}⟧.\n" for n, c in questions)
open(f"{OUT}/heldout-retrieval.txt", "w").write(text)
