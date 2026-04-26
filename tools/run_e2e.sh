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
DTYPE=""
OP="add"
ARCH=89
shift || true
while (( $# > 0 )); do
    case "$1" in
        --op) OP="$2"; shift 2 ;;
        --arch) ARCH="$2"; shift 2 ;;
        --dtype) DTYPE="$2"; shift 2 ;;
        *)
            # First positional after example is treated as dtype when no flag
            # was used (back-compat with the old `run_e2e.sh ferrum_residual_add f32` form).
            if [[ -z "$DTYPE" && "$1" != --* ]]; then
                DTYPE="$1"; shift
            else
                echo "unknown arg: $1" >&2; exit 2
            fi
            ;;
    esac
done

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
# Pick a runner that matches the kernel signature.
case "${RUNNER:-auto}" in
    auto)
        case "$EXAMPLE" in
            ferrum_gelu)                 RUNNER=run_gelu ;;
            ferrum_residual_add_inplace) RUNNER=run_residual_add_inplace ;;
            ferrum_add_bias)             RUNNER=run_add_bias ;;
            ferrum_rms_norm)             RUNNER=run_rms_norm ;;
            ferrum_softmax)              RUNNER=run_softmax ;;
            ferrum_layer_norm)           RUNNER=run_layer_norm ;;
            ferrum_fused_add_rms_norm)   RUNNER=run_fused_add_rms_norm ;;
            *)                           RUNNER=run_vec_add ;;
        esac
        ;;
esac
cargo run --quiet --example "$RUNNER" -p triton-runtime --features cuda --release -- \
    "$OUT/kernel.ptx" "$OUT/kernel.json" --op "$OP"

echo "== DONE =="
