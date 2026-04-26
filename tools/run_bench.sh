#!/bin/bash
# End-to-end head-to-head benchmark for residual_add_f32:
#   triton-rs DSL  vs  ferrum hand-written .cu  vs  Python @triton.jit.
#
# All three compile through different frontends but the same GPU runs them
# under identical launch shapes, with cudarc as the launcher and CUDA
# events as the timer.
set -euo pipefail

ARCH=89
if [[ "${1:-}" == "--arch" ]]; then
    ARCH="$2"
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

BENCH=/tmp/triton_rs_bench/residual_add
rm -rf "$BENCH"
mkdir -p "$BENCH/dsl" "$BENCH/cu" "$BENCH/py"

echo "== [1/3] triton-rs DSL  ->  MLIR  ->  Triton (Rust shim)  ->  PTX =="
cargo run --quiet --example ferrum_residual_add -p triton-dsl -- f32 \
    > "$BENCH/dsl/kernel.mlir"
if [[ "${USE_PYTHON_COMPILE:-0}" == "1" ]]; then
    python3 tools/mlir_to_cubin.py "$BENCH/dsl/kernel.mlir" "$BENCH/dsl" --arch "$ARCH"
else
    cargo run --quiet --release -p triton-sys --features compile-triton \
        --example compile_mlir -- "$BENCH/dsl/kernel.mlir" "$BENCH/dsl" --arch "$ARCH"
fi

echo "== [2/3] ferrum hand-written .cu  ->  nvcc  ->  PTX =="
# Use the in-tree fixture under tools/reference_cu/ so the bench works on
# any host without ferrum-infer-rs checked out alongside.
FERRUM_CU="${FERRUM_CU:-${REPO_ROOT}/tools/reference_cu/residual_add.cu}"
if [[ ! -f "$FERRUM_CU" ]]; then
    echo "ERROR: ferrum reference .cu not found at $FERRUM_CU" >&2
    exit 2
fi
KERNEL_NAME=residual_add_f32 bash tools/compile_cu.sh "$FERRUM_CU" "$BENCH/cu" --arch "$ARCH"

echo "== [3/3] Python @triton.jit  ->  Triton  ->  PTX =="
python3 tools/compile_python_kernel.py \
    tools/python_kernels/residual_add.py residual_add_f32 \
    "$BENCH/py" \
    --signature 'a_ptr=*fp32,b_ptr=*fp32,out_ptr=*fp32,n=i32' \
    --constants 'BLOCK=1024' \
    --arch "$ARCH"

echo
echo "== Sizes =="
ls -la "$BENCH/dsl/kernel.ptx" "$BENCH/cu/kernel.ptx" "$BENCH/py/kernel.ptx" \
    | awk '{printf "  %s  %s\n", $5, $9}'

echo
echo "== Bench =="
cargo run --quiet --example bench_residual_add -p triton-runtime --features cuda --release -- \
    "$BENCH"
