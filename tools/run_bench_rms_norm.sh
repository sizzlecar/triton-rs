#!/bin/bash
# Head-to-head benchmark for rms_norm_f32: triton-rs DSL vs ferrum .cu vs
# Python @triton.jit. Same orchestration shape as run_bench.sh (which
# does residual_add) but for the row-block reduction kernel.
set -euo pipefail

ARCH=89
if [[ "${1:-}" == "--arch" ]]; then
    ARCH="$2"
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

BENCH=/tmp/triton_rs_bench/rms_norm
rm -rf "$BENCH"
mkdir -p "$BENCH/dsl" "$BENCH/cu" "$BENCH/py"

echo "== [1/3] triton-rs DSL  ->  MLIR  ->  Triton (Rust shim)  ->  PTX =="
cargo run --quiet --example ferrum_rms_norm -p triton-dsl > "$BENCH/dsl/kernel.mlir"
if [[ "${USE_PYTHON_COMPILE:-0}" == "1" ]]; then
    python3 tools/mlir_to_cubin.py "$BENCH/dsl/kernel.mlir" "$BENCH/dsl" --arch "$ARCH"
else
    cargo run --quiet --release -p triton-sys --features compile-triton \
        --example compile_mlir -- "$BENCH/dsl/kernel.mlir" "$BENCH/dsl" --arch "$ARCH"
fi

echo "== [2/3] ferrum hand-written .cu  ->  nvcc  ->  PTX =="
FERRUM_CU="${FERRUM_CU:-${REPO_ROOT}/tools/reference_cu/rms_norm.cu}"
KERNEL_NAME=rms_norm_f32 bash tools/compile_cu.sh "$FERRUM_CU" "$BENCH/cu" --arch "$ARCH"

echo "== [3/3] Python @triton.jit  ->  Triton  ->  PTX =="
python3 tools/compile_python_kernel.py \
    tools/python_kernels/rms_norm.py rms_norm_f32 \
    "$BENCH/py" \
    --signature 'input_ptr=*fp32,weight_ptr=*fp32,output_ptr=*fp32,row_size=i32,inv_n=fp32,eps=fp32' \
    --constants 'BLOCK=1024' \
    --arch "$ARCH"

echo
echo "== Sizes =="
ls -la "$BENCH/dsl/kernel.ptx" "$BENCH/cu/kernel.ptx" "$BENCH/py/kernel.ptx" \
    | awk '{printf "  %s  %s\n", $5, $9}'

echo
echo "== Bench =="
cargo run --quiet --example bench_rms_norm -p triton-runtime --features cuda --release -- \
    "$BENCH"
