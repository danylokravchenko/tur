# Tur development tasks. `just` lists them.

set positional-arguments
set shell := ["bash", "-cu"]

# Pinned development checkpoint; override with TUR_MODEL / TUR_REVISION.
model := env("TUR_MODEL", "Qwen/Qwen3.5-0.8B")
revision := env("TUR_REVISION", "2fc06364715b967f1860aea9cf38778875588b17")

_default:
    @just --list

# Type-check every target, with and without Metal.
check:
    cargo check --workspace --all-targets
    cargo check --workspace --all-targets --no-default-features

# Strict Clippy over every target.
lint:
    cargo clippy --workspace --all-targets -- -D warnings

fmt:
    cargo fmt --all

fmt-check:
    cargo fmt --all -- --check

# Offline test suite; extra arguments go to nextest (e.g. `just test -E 'test(prefix)'`).
test *args:
    cargo nextest run --workspace "$@"

# Checkpoint-dependent tests; needs TUR_QWEN35_0_8B and the golden files.
test-checkpoint *args:
    cargo nextest run --workspace --run-ignored only "$@"

# Everything CI should gate on.
ci: fmt-check lint check test

build:
    cargo build --release --bin tur

# Install the release binary; extra arguments go to cargo install.
install *args:
    ./install.sh "$@"

# One completion: `just run --prompt 'Hello' -n 64`.
run *args:
    cargo run --release --bin tur -- run --model {{model}} --revision {{revision}} "$@"

# HTTP server on 127.0.0.1:8080 unless --listen is given.
serve *args:
    cargo run --release --bin tur -- serve --model {{model}} --revision {{revision}} "$@"

# Prepack weights: `just quantize <checkpoint dir> <output dir> --method rtn`.
quantize dir output *args:
    cargo run --release --bin tur -- quantize --model "$1" --revision {{revision}} --output "$2" "${@:3}"

# Benchmarks: `just bench kernels`, `just bench serving`; no case lists them.
bench *args:
    cargo bench --offline -p tur --bench perf -- "$@"

# Serve the book at http://localhost:3000 with live reload.
docs:
    mdbook serve docs --open

# Build the book into docs/book.
docs-build:
    mdbook build docs
