#!/bin/bash
# Phase-1-lite end-to-end smoke runner for any 3-ptr-+-i32 element-wise kernel:
#   DSL example  ->  MLIR  ->  Triton  ->  PTX/cubin  ->  cudarc  ->  GPU.
#
# Usage:
#   ./tools/run_e2e.sh                                # vec_add (default)
#   ./tools/run_e2e.sh ferrum_residual_add f32        # residual_add f32
#   ./tools/run_e2e.sh ferrum_residual_add f32 --arch 89
#
# Run on a host with CUDA + Triton + Rust (i.e. NOT macOS).
set -euo pipefail

EXAMPLE="${1:-dump_vec_add_generic}"
DTYPE="${2:-}"
ARCH=89
if [[ "${3:-}" == "--arch" ]]; then
    ARCH="$4"
fi

OUT="/tmp/triton_rs_e2e_${EXAMPLE}"
rm -rf "$OUT"
mkdir -p "$OUT"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "== [1/3] generate MLIR via the DSL ($EXAMPLE${DTYPE:+ $DTYPE}) =="
if [[ -n "$DTYPE" ]]; then
    cargo run --quiet --example "$EXAMPLE" -p triton-dsl -- "$DTYPE" > "$OUT/kernel.mlir"
else
    cargo run --quiet --example "$EXAMPLE" -p triton-dsl > "$OUT/kernel.mlir"
fi
wc -l "$OUT/kernel.mlir"

echo "== [2/3] compile MLIR -> PTX/cubin via Triton (Python) =="
python3 tools/mlir_to_cubin.py "$OUT/kernel.mlir" "$OUT" --arch "$ARCH"

echo "== [3/3] launch on GPU via cudarc =="
cargo run --quiet --example run_vec_add -p triton-runtime --features cuda --release -- \
    "$OUT/kernel.ptx" "$OUT/kernel.json"

echo "== DONE =="
