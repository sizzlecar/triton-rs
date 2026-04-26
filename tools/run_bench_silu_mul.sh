#!/bin/bash
# Head-to-head benchmark for fused_silu_mul_f32: triton-rs DSL vs
# ferrum .cu vs Python @triton.jit.
set -euo pipefail

ARCH=89
if [[ "${1:-}" == "--arch" ]]; then
    ARCH="$2"
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

BENCH=/tmp/triton_rs_bench/silu_mul
rm -rf "$BENCH"
mkdir -p "$BENCH/dsl" "$BENCH/cu" "$BENCH/py"

echo "== [1/3] triton-rs DSL  ->  MLIR  ->  Triton (Rust shim)  ->  PTX =="
cargo run --quiet --example ferrum_fused_silu_mul -p triton-dsl > "$BENCH/dsl/kernel.mlir"
if [[ "${USE_PYTHON_COMPILE:-0}" == "1" ]]; then
    python3 tools/mlir_to_cubin.py "$BENCH/dsl/kernel.mlir" "$BENCH/dsl" --arch "$ARCH"
else
    cargo run --quiet --release -p triton-sys --features compile-triton \
        --example compile_mlir -- "$BENCH/dsl/kernel.mlir" "$BENCH/dsl" --arch "$ARCH"
fi

echo "== [2/3] ferrum hand-written .cu  ->  nvcc  ->  PTX =="
KERNEL_NAME=fused_silu_mul_f32 bash tools/compile_cu.sh \
    "${REPO_ROOT}/tools/reference_cu/fused_silu_mul.cu" "$BENCH/cu" --arch "$ARCH"

echo "== [3/3] Python @triton.jit  ->  Triton  ->  PTX =="
python3 tools/compile_python_kernel.py \
    tools/python_kernels/fused_silu_mul.py fused_silu_mul_f32 \
    "$BENCH/py" \
    --signature 'gate_ptr=*fp32,up_ptr=*fp32,out_ptr=*fp32,n=i32' \
    --constants 'BLOCK=1024' \
    --arch "$ARCH"

echo
echo "== Sizes =="
ls -la "$BENCH/dsl/kernel.ptx" "$BENCH/cu/kernel.ptx" "$BENCH/py/kernel.ptx" \
    | awk '{printf "  %s  %s\n", $5, $9}'

echo
echo "== Bench =="
cargo run --quiet --example bench_silu_mul -p triton-runtime --features cuda --release -- \
    "$BENCH"
