#!/bin/sh
# Installs the latest release binary. Published as the `install.sh` release asset:
#   curl -fsSL https://github.com/danylokravchenko/tur/releases/latest/download/install.sh | sh
#
# Settings go on the `sh` side of the pipe:
#   TUR_VERSION          release tag to install, e.g. v0.3.0 (default: latest)
#   TUR_INSTALL_DIR      destination directory (default: ~/.local/bin)
#   TUR_NO_MODIFY_PATH   set to 1 to leave shell startup files alone
#   TUR_RELEASE_URL      directory holding the release assets, for mirrors and tests
set -eu

ASSET=tur-aarch64-apple-darwin.tar.gz
REPOSITORY=https://github.com/danylokravchenko/tur/releases

fail() {
    echo "tur install: $1" >&2
    exit 1
}

release_url() {
    if [ -n "${TUR_RELEASE_URL:-}" ]; then
        echo "$TUR_RELEASE_URL"
    elif [ -n "${TUR_VERSION:-}" ]; then
        echo "$REPOSITORY/download/$TUR_VERSION"
    else
        echo "$REPOSITORY/latest/download"
    fi
}

# Prints the startup file that sets PATH for the user's login shell, if it is a known one.
profile_file() {
    case "${SHELL:-}" in
        */zsh) echo "${ZDOTDIR:-$HOME}/.zshrc" ;;
        */bash) echo "$HOME/.bash_profile" ;;
    esac
}

add_to_path() {
    case ":$PATH:" in
        *":$1:"*) return 0 ;;
    esac
    profile=$(profile_file)
    if [ "${TUR_NO_MODIFY_PATH:-0}" = 1 ] || [ -z "$profile" ]; then
        echo "Add $1 to your PATH to run tur."
        return 0
    fi
    line="export PATH=\"$1:\$PATH\""
    if ! grep -qsxF "$line" "$profile"; then
        printf '\n# Added by the tur installer\n%s\n' "$line" >>"$profile"
        echo "Added $1 to PATH in $profile."
    fi
    echo "Open a new terminal, or run: $line"
}

main() {
    [ "$(uname -s)" = Darwin ] && [ "$(uname -m)" = arm64 ] ||
        fail "release binaries are for Apple Silicon Macs; found $(uname -s) $(uname -m)"

    install_dir=${TUR_INSTALL_DIR:-$HOME/.local/bin}
    url=$(release_url)
    work=$(mktemp -d)
    trap 'rm -rf "$work"' EXIT

    echo "Downloading $url/$ASSET"
    curl -fsSL "$url/$ASSET" -o "$work/$ASSET" || fail "download failed: $url/$ASSET"
    curl -fsSL "$url/$ASSET.sha256" -o "$work/$ASSET.sha256" || fail "checksum download failed"
    (cd "$work" && shasum -a 256 -c "$ASSET.sha256" >/dev/null) || fail "checksum mismatch"

    tar -xzf "$work/$ASSET" -C "$work" tur
    # An ad-hoc signature only seals the bytes for this Mac's code-signing checks. It says
    # nothing about who built them; the checksum above is what ties them to the release.
    codesign --force --sign - "$work/tur" 2>/dev/null || fail "ad-hoc signing failed"
    version=$("$work/tur" --version) || fail "the downloaded binary does not run"

    mkdir -p "$install_dir"
    # Copy beside the destination first, so the final rename replaces any old binary at once.
    cp "$work/tur" "$install_dir/.tur.new"
    chmod 755 "$install_dir/.tur.new"
    mv -f "$install_dir/.tur.new" "$install_dir/tur"
    echo "Installed $version to $install_dir/tur"
    add_to_path "$install_dir"
}

# Everything runs from here, so a download cut off mid-script executes nothing.
main
