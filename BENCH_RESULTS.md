# Bench Results — triton-rs DSL vs Python @triton.jit vs ferrum hand-.cu

**Hardware:** RTX 5070 Ti (Blackwell consumer, sm_120 native, PTX targets sm_89), Vast.ai
**Toolchain:** Triton v3.2.0 vendored; LLVM-86b69c31 prebuilt; ptxas from CUDA 12.8
**Status:** Phase 1 done — Rust C ABI shim drives Triton's MLIR pass pipeline directly. Zero Python in compile or runtime path.
**Date:** 2026-04-26

All three sources go through identical launchers (`cudarc` + CUDA events as the timer) and identical buffer sizes. Rust shim path = `cargo run -p triton-sys --features compile-triton --example compile_mlir`; the Python fallback (`tools/mlir_to_cubin.py`) stays available via `USE_PYTHON_COMPILE=1` for differential testing.

## residual_add_f32 — `out[i] = a[i] + b[i]`

```
N = 33554432 elements (134 MB per buffer, 402.7 MB touched per call)
iters = 200 (after 20 warmup)

source                  us / call   GB/s eff.     % peak  grid×block
triton-rs DSL               510.2       789.1      88.1%  32768×128
ferrum .cu (nvcc)           509.7       789.9      88.2%  131072×256
Python @triton.jit          510.3       789.0      88.1%  32768×128
```

**Result:** Rust shim ≡ Python ≡ ferrum within timer noise. All hit 88% of theoretical DRAM peak (~896 GB/s on this card). Memory-bound kernel; compiler version effectively doesn't matter.

## fused_silu_mul_f32 — `out[i] = silu(gate[i]) * up[i]`

```
N = 33554432 elements, 402.7 MB / call
iters = 200 (after 20 warmup)

source                  us / call   GB/s eff.     % peak  grid×block
triton-rs DSL               511.0       788.0      87.9%  32768×128
ferrum .cu (nvcc)           509.9       789.7      88.1%  131072×256
Python @triton.jit          510.5       788.8      88.0%  32768×128
```

**Result:** Rust shim ≡ Python ≡ ferrum within timer noise. All hit 88% of theoretical DRAM peak. Sanity err = 2.3e-10 (float rounding).

## rms_norm_f32 — row-block reduction + reciprocal sqrt

```
ROWS x ROW_SIZE = 32768 x 1024 = 268.4 MB input + output (no weight)
iters = 100 (after 10 warmup)

source                  us / call   GB/s eff.     % peak  grid×block
triton-rs DSL               349.0       769.2      85.8%  32768×128
ferrum .cu (nvcc)           550.4       487.7      54.4%  32768×1024
Python @triton.jit          349.4       768.3      85.7%  32768×128
```

**Result:** Rust shim ≡ Python (slightly faster on this run — within noise) and **1.58x faster than ferrum's hand-written .cu**. Triton's autovectorization wins; the thin Rust DSL inherits all of it. Sanity err = 0.

## Takeaways

1. **Memory-bound kernels: parity** with Python @triton.jit and ferrum hand-CUDA. The compiler step contributes nothing measurable here.
2. **Math-heavy kernels: parity** with Python @triton.jit. The earlier 13-14% gap turned out to be the LLVM NVPTX backend emitting `.target sm_89, debug` whenever DI metadata is present (we run `createLLVMDIScopePass` for source-line info). ptxas treats `, debug` as a build-time hint to disable optimization. Triton's Python `compiler.py` strips it before handing PTX to ptxas; we now do the same in `strip_debug_target`. Source-line info still works via the `-lineinfo` ptxas flag.
3. **All paths beat or match ferrum's hand-written .cu** — and beat it 1.58x on rms_norm. Confirms the original thesis: Triton's optimization pipeline (autovectorization, layout selection) makes a thin DSL competitive with — sometimes better than — hand-tuned CUDA.

## Attention coverage

| Kernel | Status | MLIR (B) | Cubin (B) | Notes |
|---|---|---|---|---|
| `decode_attention_f32` | ✅ | 6848 | 81504 | Standard seq-major KV cache, GQA |
| `decode_attention_hm_f32` | ✅ | 6864 | 81760 | Head-major KV cache (`[nkv, cap, hd]`) |
| `batched_decode_attention_f32` | ✅ | 7336 | 83552 | Continuous batching (Z×Hq×D) |
| `paged_decode_attention_f32` | ✅ | 7771 | 97888 | Block-table indirection (vLLM pattern) |
| `flash_decode_attention` | 🚧 | — | — | Split-K + reduce phase (2 kernels) |
| `batched_flash_decode_attention` | 🚧 | — | — | Batched + split-K |
| `flash_attn_full` | 🚧 | — | — | Prefill, multi-token Q |

All 4 done variants ship via `triton_kernels::prelude::*` and compile end-to-end through the Rust shim. Online-softmax pattern uses `scf_for` with 3 iter_args (m_i, l_i, acc); Q·K via broadcast-mul-reduce instead of `tt.dot` (avoids the transposed-K load and works for HEAD_DIM ≤ 256). Paged variant adds a `block_table[logical]` gather load per KV tile — `tt.load(block_table_ptr + logical_blocks_tile)` works out of the box.

DSL extension added for these: 2D singleton-broadcast (`[m,1]+[1,n]→[m,n]` auto-emits `tt.broadcast` on each side; see `crates/triton-ir/src/module.rs::coerce_elemwise`).

## Reproduce

```bash
ssh -p 16495 root@ssh3.vast.ai
export TRITON_LLVM_SYSPATH=$HOME/.cache/triton-rs/llvm/llvm-86b69c31-ubuntu-x64
export CUDA_HOME=/usr/local/cuda
export TRITON_LIBDEVICE_PATH=$HOME/triton-rs/crates/triton-sys/vendor/triton/third_party/nvidia/backend/lib/libdevice.10.bc
cd ~/triton-rs
bash tools/run_bench.sh             # residual_add
bash tools/run_bench_silu_mul.sh    # silu_mul
bash tools/run_bench_rms_norm.sh    # rms_norm
```

For an A/B vs the Python compile path: `USE_PYTHON_COMPILE=1 bash tools/run_bench.sh`.
