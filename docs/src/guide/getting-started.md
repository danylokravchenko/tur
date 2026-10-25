# Requirements and checkpoints

This page is the reference for what Tur needs. For a guided first run, follow the
[Quickstart](quickstart.md).

## Requirements

- A Mac with Apple Silicon and direct access to its Metal device. Virtual machines and
  sandboxes without GPU access fail at startup with "no accessible Metal device".
- Rust installed through rustup. The checkout's `rust-toolchain.toml` selects Rust
  1.99.0 (edition 2024), rustfmt and Clippy. The minimum supported Rust version is 1.99.
- Optional: [`just`](https://github.com/casey/just) for the task recipes,
  [`cargo-nextest`](https://nexte.st) for tests, and
  [`mdbook`](https://rust-lang.github.io/mdBook/) for this book.

## Build

Install the release binary from the checkout:

```sh
./install.sh                        # or: just install
./install.sh --root "$HOME/.local"   # installs into ~/.local/bin
```

The default destination is `~/.cargo/bin`; Cargo's `CARGO_INSTALL_ROOT`,
`CARGO_HOME` and install configuration overrides are respected. Add the destination's
`bin` directory to `PATH`. The script uses `Cargo.lock`, replaces an existing `tur`
installation, and forwards its arguments to `cargo install`.

To build without installing:

```sh
cargo build --release --bin tur      # or: just build
```

The binary is `target/release/tur`. Metal support is a default feature;
`--no-default-features` builds only the CPU reference, which exists for numerical tests,
not for serving.

## Get a checkpoint

Tur loads SafeTensors checkpoints, either from a local directory or from Hugging Face.
A checkpoint directory contains:

| File | Required | Purpose |
| --- | --- | --- |
| `config.json` | yes | Architecture and dimensions; decides whether Tur supports the model |
| `tokenizer.json` | yes | Hugging Face tokenizer |
| `tokenizer_config.json` | yes | Special tokens, EOS and the chat template |
| `model.safetensors` or `model.safetensors.index.json` plus shards | yes | Weights |
| `chat_template.jinja` | no | Overrides the template in `tokenizer_config.json` |
| `generation_config.json` | no | Default temperature, top-k, top-p and EOS tokens |

The architecture comes from the config's `model_type` and `architectures` fields, never
from the model's name. Unsupported architectures, settings or tensor layouts fail before
any weight is read.

For a remote checkpoint, give the repository as `organization/name` and pin an
immutable 40-character commit; branch names are rejected. Downloads use the standard
Hugging Face cache and honor `HF_HOME` and `HF_TOKEN`. For a local directory, the
revision is an identity you choose. It is recorded with prefix-cache entries and
prepacked artifacts.

## First generation

```sh
tur run \
  --model Qwen/Qwen3.5-0.8B \
  --revision 2fc06364715b967f1860aea9cf38778875588b17 \
  --prompt 'Write one sentence about the Moon.' \
  --temperature 0 --max-tokens 64
```

Text streams to stdout. Logs, including the memory plan and timing, go to stderr. Set
`RUST_LOG=debug` for more detail.

The `justfile` pins that checkpoint, so `just run --prompt 'Hello' -n 32` does the same.
Override it with `TUR_MODEL` and `TUR_REVISION`.

## First server

```sh
tur serve --model /path/to/checkpoint --revision my-checkpoint

curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":64}'
```

The server loads and validates the model before it binds the listener, so a listening
server is ready to answer. See [HTTP API](http-api.md) for the request format.

## Decisions

- Pin Rust 1.99.0 with rustfmt and Clippy: it matches the workspace's minimum Rust
  version and supplies the tools used by `just ci`.
- Install through Cargo with `--locked`: the CLI uses the checked-in dependency graph
  and Cargo's existing destination options.
