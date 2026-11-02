# Requirements and checkpoints

This page is the reference for what Tur needs. For a guided first run, follow the [Quickstart](quickstart.md).

## Requirements

- A Mac with Apple Silicon and direct access to its Metal device. Virtual machines and sandboxes without GPU access fail at startup with "no accessible Metal device".
- Rust installed through rustup. The checkout's `rust-toolchain.toml` selects Rust 1.99 (edition 2024) with rustfmt and Clippy.
- Optionally, [`just`](https://github.com/casey/just) for the task recipes, [`cargo-nextest`](https://nexte.st) for tests, and [`mdbook`](https://rust-lang.github.io/mdBook/) for this book.

## Download

Each [release](https://github.com/danylokravchenko/tur/releases) has a prebuilt binary for Apple Silicon, which needs no Rust toolchain, and an installer for it:

```sh
curl -fsSL https://github.com/danylokravchenko/tur/releases/latest/download/install.sh | sh
```

The installer checks the download against the release's SHA-256 checksum, signs the binary ad hoc, and moves it to `~/.local/bin`. If that directory isn't on your `PATH`, it adds it to `~/.zshrc` or `~/.bash_profile`. Running it again replaces the binary. Settings go after the pipe, for example `| TUR_VERSION=v0.3.0 sh`:

| Variable | Default | Meaning |
| --- | --- | --- |
| `TUR_VERSION` | latest | Release tag to install |
| `TUR_INSTALL_DIR` | `~/.local/bin` | Destination directory |
| `TUR_NO_MODIFY_PATH` | unset | Set to 1 to leave shell startup files alone |
| `TUR_RELEASE_URL` | GitHub | Directory holding the release assets, for mirrors |

To skip the installer, download the archive yourself:

```sh
curl -fsSL https://github.com/danylokravchenko/tur/releases/latest/download/tur-aarch64-apple-darwin.tar.gz | tar -xz
./tur --version
```

The binary isn't notarized, and an ad-hoc signature doesn't change that or identify who built it. `curl` doesn't mark its downloads for Gatekeeper, but a browser does; for a browser download, run `xattr -d com.apple.quarantine tur`.

## Build

Install the release binary from the checkout:

```sh
./install.sh                        # or: just install
./install.sh --root "$HOME/.local"   # installs into ~/.local/bin
```

The default destination is `~/.cargo/bin`, and the script respects Cargo's own `CARGO_INSTALL_ROOT`, `CARGO_HOME` and install settings. It builds from `Cargo.lock`, replaces an existing `tur`, and passes its arguments to `cargo install`. Add the destination's `bin` directory to `PATH`.

To build without installing:

```sh
cargo build --release --bin tur      # or: just build
```

The binary is `target/release/tur`. Metal support is on by default. `--no-default-features` builds only the CPU reference, which exists for numerical tests and is far too slow to serve.

## Get a checkpoint

Tur loads SafeTensors checkpoints from a local directory or from Hugging Face, and can take its weights from a [GGUF file](gguf.md) instead. A checkpoint directory contains:

| File | Required | Purpose |
| --- | --- | --- |
| `config.json` | yes | Architecture and dimensions; decides whether Tur supports the model |
| `tokenizer.json` | yes | Hugging Face tokenizer |
| `tokenizer_config.json` | yes | Special tokens, EOS and the chat template |
| `model.safetensors`, or `model.safetensors.index.json` plus shards | yes | Weights |
| `chat_template.jinja` | no | Overrides the template in `tokenizer_config.json` |
| `generation_config.json` | no | Default temperature, top-k, top-p and EOS tokens |

Tur identifies the architecture from the config's `model_type` and `architectures` fields, never from the model's name. An unsupported architecture, setting or tensor layout fails before any weight is read.

`tur run` and `tur serve` take one of three things:

| You write | Tur loads | Revision |
| --- | --- | --- |
| A name from `tur list`, such as `qwen` | That model, with settings measured for it | The commit the name is pinned to |
| `organization/name` | That Hugging Face repository | Its latest commit, printed so you can pin it with `--revision` |
| A directory | The checkpoint in it | The directory's name |

`--revision` overrides any of these. A remote revision must be a 40-character commit: branch names are rejected because they can move. Downloads use the standard Hugging Face cache and honor `HF_HOME` and `HF_TOKEN`. Starting a remote model needs the network even once it is cached, because Tur confirms the commit with Hugging Face, so run a directory for offline use. For a directory, the revision is only a label: Tur records it with prefix-cache entries and prepacked artifacts so they never mix with another model's.

A name wins over a directory of the same name, so write `./qwen` for a directory called `qwen`.

## First generation

```sh
tur run qwen 'Write one sentence about the Moon.' --temperature 0 --max-tokens 64
```

Text streams to stdout. Logs, including the memory plan and timing, go to stderr; set `RUST_LOG=debug` for more. Leave out the prompt to chat instead.

The `justfile` pins the same checkpoint, so `just run 'Hello' -n 32` does the same thing. `TUR_MODEL` and `TUR_REVISION` change which checkpoint its recipes pass; `tur` itself doesn't read them.

## First server

```sh
tur serve qwen

curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":64}'
```

The server loads and validates the model before it opens the port, so once it is listening it can answer. See [HTTP API](http-api.md) for the request format.

## Decisions

- The toolchain is pinned to Rust 1.99 with rustfmt and Clippy. That matches the workspace's minimum Rust version and supplies every tool `just ci` uses.
- Installation goes through `cargo install --locked`, so the CLI builds from the checked-in dependency graph and honors Cargo's usual destination options.
- Short names map to pinned commits rather than to a branch, so a name keeps the reproducibility that a typed commit gave. The list holds only checkpoints Tur has generated from on full weights at those commits, because a name implies that it works. Qwen3.8-27B and Qwen3.6-35B-A3B also ran, but only as prepacked artifacts, and they stay out until a command that needs no separate quantization step has run.
- A repository without `--revision` resolves to its latest commit once per start and prints it. Refusing would send people back to looking up a commit by hand, and the printed flag is one paste away from the old guarantee.
