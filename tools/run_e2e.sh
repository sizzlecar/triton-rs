#!/bin/bash
# Phase-1-lite end-to-end smoke test: DSL -> MLIR -> Triton -> PTX -> cudarc -> GPU.
# Run on a host with CUDA + Triton + Rust installed (i.e. NOT macOS).
#
# Usage:  ./tools/run_e2e.sh [--arch 89]
set -euo pipefail

ARCH=89
if [[ "${1:-}" == "--arch" ]]; then
    ARCH="$2"
fi

OUT=/tmp/triton_rs_e2e
rm -rf "$OUT"
mkdir -p "$OUT"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "== [1/3] generate MLIR via the DSL =="
cargo run --quiet --example dump_vec_add_generic -p triton-dsl > "$OUT/vec_add.mlir"
wc -l "$OUT/vec_add.mlir"

echo "== [2/3] compile MLIR -> PTX/cubin via Triton (Python) =="
python3 tools/mlir_to_cubin.py "$OUT/vec_add.mlir" "$OUT" --arch "$ARCH"

echo "== [3/3] launch on GPU via cudarc =="
cargo run --quiet --example run_vec_add -p triton-runtime --features cuda --release -- \
    "$OUT/kernel.ptx" "$OUT/kernel.json"

echo "== DONE =="
