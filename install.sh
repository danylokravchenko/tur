#!/bin/sh
set -eu

# Run from the checkout so rustup selects rust-toolchain.toml.
cd -- "$(dirname -- "$0")"

# Additional arguments are passed to cargo install, e.g. --root "$HOME/.local".
exec cargo install --locked --path crates/tur-cli --bin tur --force "$@"
