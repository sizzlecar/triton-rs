#!/usr/bin/env bash
# Fetch vendored Triton source at the pinned version.
#
# Why this script (and not a git submodule)? — Triton's repo is large
# (~350 MB with full history) and the user's proxy was unreliable for
# `git submodule add` in initial setup. A shallow clone of the tagged
# release lands the same source tree at ~21 MB, far more robust.
#
# This script is idempotent: if the vendored tree already exists at the
# right commit, it does nothing. Re-run after bumping `TRITON_TAG` /
# `TRITON_COMMIT` below.
#
# Called by:
#   - contributors manually before `cargo build --features compile-triton`
#   - CI Layer 2 (CUDA build job)
#   - eventually by triton-sys/build.rs as a pre-step (Phase 1B)

set -euo pipefail

TRITON_TAG="${TRITON_TAG:-v3.2.0}"
TRITON_COMMIT="${TRITON_COMMIT:-9641643}"        # truncated SHA — informational
TRITON_REPO="${TRITON_REPO:-https://github.com/triton-lang/triton}"

# Resolve the vendor dir relative to this script's location so the script
# is callable from anywhere.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENDOR_DIR="${SCRIPT_DIR}/../vendor/triton"

if [[ -d "$VENDOR_DIR/.git" ]]; then
    cd "$VENDOR_DIR"
    HEAD_SHA="$(git rev-parse HEAD 2>/dev/null || echo "")"
    if [[ "$HEAD_SHA" == "${TRITON_COMMIT}"* ]]; then
        echo "vendor/triton already at ${TRITON_TAG} (${HEAD_SHA:0:7}); nothing to do"
        exit 0
    fi
    echo "vendor/triton exists but at ${HEAD_SHA:0:7}, expected ${TRITON_COMMIT}*; re-cloning"
    cd - >/dev/null
    rm -rf "$VENDOR_DIR"
fi

mkdir -p "$(dirname "$VENDOR_DIR")"

# Shallow clone at the tag — ~21 MB on disk, single round-trip.
# `git clone -b TAG --depth=1` is supported for both branches and tags
# (unlike `git submodule add -b` which only accepts branches).
echo "fetching Triton ${TRITON_TAG} into $VENDOR_DIR ..."
git clone --depth=1 --branch "${TRITON_TAG}" "${TRITON_REPO}" "$VENDOR_DIR"

ACTUAL_SHA="$(git -C "$VENDOR_DIR" rev-parse HEAD)"
if [[ "$ACTUAL_SHA" != "${TRITON_COMMIT}"* ]]; then
    echo "WARNING: cloned HEAD ${ACTUAL_SHA:0:7} doesn't match expected ${TRITON_COMMIT}" >&2
    echo "         (the tag ${TRITON_TAG} may have moved upstream — verify and update TRITON_COMMIT in this script)" >&2
fi

echo "OK — vendor/triton ready at ${TRITON_TAG} (${ACTUAL_SHA:0:7})"
