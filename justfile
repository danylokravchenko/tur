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

# What hosted CI runs: its paravirtual GPU needs the reduced nextest `ci` profile.
ci-hosted: fmt-check lint check (test "--profile" "ci")

# Checkpoint-dependent tests; set the selected model and golden environment variables.
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

# One reading of a running server; `just top` watches it live.
status *args:
    cargo run --release --bin tur -- status "$@"

top *args:
    cargo run --release --bin tur -- top "$@"

stop *args:
    cargo run --release --bin tur -- stop "$@"

# Prepack weights: `just quantize <checkpoint dir> <output dir> --method rtn`.
quantize dir output *args:
    cargo run --release --bin tur -- quantize --model "$1" --revision {{revision}} --output "$2" "${@:3}"

# Benchmarks: `just bench kernels`, `just bench serving`; no case lists them.
bench *args:
    cargo bench --offline -p tur --bench perf -- "$@"

# Short Gemma 4 model benchmark (Q8_0, F16 K/V, 512-token prefill, one 4K context); knobs override.
bench-gemma *args:
    TUR_BENCH_MODEL="${TUR_BENCH_MODEL:-${TUR_GEMMA4_E2B:?set TUR_GEMMA4_E2B}}" \
    TUR_BENCH_FORMATS="${TUR_BENCH_FORMATS:-q8_0}" TUR_BENCH_KV="${TUR_BENCH_KV:-f16}" \
    TUR_BENCH_PREFILL="${TUR_BENCH_PREFILL:-512}" TUR_BENCH_CONTEXTS="${TUR_BENCH_CONTEXTS:-4096}" \
    TUR_BENCH_BATCHES="${TUR_BENCH_BATCHES:-1,4}" TUR_BENCH_CONTEXT_BATCHES="${TUR_BENCH_CONTEXT_BATCHES:-1}" \
    TUR_BENCH_ENGINE_TOKENS="${TUR_BENCH_ENGINE_TOKENS:-16}" \
    cargo bench --offline -p tur --bench perf -- model "$@"

# Serve the book at http://localhost:3000 with live reload.
docs:
    mdbook serve docs --open

# Build the book into docs/book.
docs-build:
    mdbook build docs
