# triton-rs

> Rust-native GPU DSL on top of the Triton MLIR backend.
> **Zero Python at build- or run-time.**

---

## What is this?

triton-rs is a Rust frontend, runtime, and toolchain for writing high-performance
GPU kernels. It reuses the [Triton](https://github.com/triton-lang/triton) MLIR
compiler backend (PTX/AMDGPU/SPIR-V codegen) but replaces the Python frontend
with a Rust DSL designed to fix the known ergonomics, type-safety, and tooling
shortcomings of `@triton.jit`.

## Status (this is a moving target — see `git log`)

| Layer | Status |
|---|---|
| `triton-ir` — pure-Rust MLIR text builder, `arith` / `tt` / `scf` dialect ops | ✅ working, validated against Triton 3.2.0 |
| `triton-dsl` — `#[triton_kernel]` proc-macro, body translation, operator overloading, control flow, const generics | ✅ working |
| `triton-runtime` — `DeviceRuntime` trait + cudarc backend | ✅ working (cudarc launchers under `examples/run_*.rs`) |
| `triton-sys` — C ABI shim over vendored Triton C++ libs (zero-Python build path) | **✅ Phase 1 done** — `Context::compile()` drives Triton's MLIR pass pipeline directly from C++; cubin loads via cudarc; **no Python in build or runtime path**. See `crates/triton-sys/SPIKE.md` for the orchestrator map and `BENCH_RESULTS.md` for parity numbers vs Python @triton.jit. |
| `triton-kernels` — ready-to-use kernel library (rms_norm, residual_add, fused_silu_mul, gelu, softmax, layer_norm, embedding, RoPE, kv_cache_append, decode_attention, …) | ✅ 22 kernels in 8 modules; `use triton_kernels::prelude::*;` |
| `triton-core` — user-facing facade re-exporting `triton_kernel`, `sys`, `runtime`, `ir` | ✅ working; `examples/ferrum_integration_demo.rs` is the worked end-to-end example |

### One-shot integration (downstream like ferrum-infer-rs)

```toml
[dependencies]
triton-core = { git = "https://github.com/sizzlecar/triton-rs", features = ["cuda", "compile-triton"] }
```

```rust
use triton_core::{triton_kernel, sys::{Context, CompileOptions}};

#[triton_kernel]
fn vec_add<const BLOCK: usize>(x: Ptr<f32>, y: Ptr<f32>, out: Ptr<f32>, n: i32) { /* ... */ }

let mlir = vec_add::<1024>::mlir();
let cubin = Context::new()?.compile(&mlir, &CompileOptions::default())?;
// load via cudarc, launch on GPU.
```

Build `compile-triton` only on Linux + CUDA boxes; the rest of the workspace `cargo check`s on Mac without it. See `crates/triton-core/examples/ferrum_integration_demo.rs` for the full demo.

## What works today

A user writes natural Rust syntax:

```rust
use triton_dsl::triton_kernel;

#[triton_kernel]
fn vec_add<const BLOCK: usize>(
    x: Ptr<f32>,
    y: Ptr<f32>,
    out: Ptr<f32>,
    n: i32,
) {
    let pid = program_id(0);
    let off = splat_1d(pid * const_i32(BLOCK as i32), BLOCK as i64)
            + make_range(0, BLOCK as i32);
    let mask = off < splat_1d(n, BLOCK as i64);
    let xv = load(splat_1d(x, BLOCK as i64) + off, mask);
    let yv = load(splat_1d(y, BLOCK as i64) + off, mask);
    store(splat_1d(out, BLOCK as i64) + off, xv + yv, mask);
}

fn main() {
    println!("{}", vec_add::<1024>::mlir());
}
```

The proc-macro generates Triton-compatible MLIR text. The output for
`BLOCK = 1024` looks like this (Triton's pretty-printer round-trip shown):

```mlir
module {
  tt.func @vec_add(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: i32) {
    %0 = tt.get_program_id x : i32
    %c1024_i32 = arith.constant 1024 : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.splat %1 : i32 -> tensor<1024xi32>
    %3 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %4 = arith.addi %2, %3 : tensor<1024xi32>
    %5 = tt.splat %arg3 : i32 -> tensor<1024xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<1024xi32>
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %8 = tt.addptr %7, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    %9 = tt.load %8, %6 : tensor<1024x!tt.ptr<f32>>
    %10 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %11 = tt.addptr %10, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    %12 = tt.load %11, %6 : tensor<1024x!tt.ptr<f32>>
    %13 = arith.addf %9, %12 : tensor<1024xf32>
    %14 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %15 = tt.addptr %14, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    tt.store %15, %13, %6 {operandSegmentSizes = array<i32: 1, 1, 1>} : tensor<1024x!tt.ptr<f32>>
    tt.return
  }
}
```

Each instantiation (`vec_add::<1024>`, `vec_add::<256>`) produces its own
module — the const generic flows through to MLIR as a literal.

## DSL features

### Type system

`Ptr<T>`, `i1`/`i8`/`i16`/`i32`/`i64`, `f16`/`f32`/`bf16`, plus tensor types
inferred from operations. Const generics for tile sizes. Casts (`BLOCK as i32`)
pass through transparently.

### Operator overloading with runtime type dispatch

```rust
let off = splat_1d(x, 1024) + range;  // tensor<i32> + tensor<i32> → arith.addi
let xp  = splat_1d(x_ptr, 1024) + off; // tensor<!tt.ptr<f32>> + tensor<i32> → tt.addptr
let sum = xv + yv;                     // tensor<f32> + tensor<f32> → arith.addf
let mask = off < splat_1d(n, 1024);    // tensor<i32> < tensor<i32> → arith.cmpi slt
```

Supported ops: `+ - * / < <= > >= == !=`. Float comparisons use the ordered
predicates by default.

### Control flow: `scf_for`

```rust
#[triton_kernel]
fn sum_squares(out: Ptr<i32>) {
    let lb   = const_i32(0);
    let ub   = const_i32(10);
    let step = const_i32(1);
    let init = const_i32(0);

    let result = scf_for(lb, ub, step, init, |i, acc| {
        let i_sq = i * i;
        acc + i_sq
    });
    store(out, result);
}
```

Closure-bound `i` and `acc` map to the `scf.for` induction variable and
iter_arg. The trailing expression of the closure body becomes the
`scf.yield`.

### Reductions: `reduce`

```rust
let total = reduce(tile, 0, |a, b| a + b);
```

Translates into a `tt.reduce` with a region body terminated by
`tt.reduce.return`. The result type drops the reduced axis.

### Built-in vocabulary

| Category | Names |
|---|---|
| Tile primitives | `program_id`, `make_range`, `splat_1d`, `addptr`, `load`, `store` |
| Constants | `const_i32`, `const_i64`, `const_f32` |
| Arithmetic | `add_i32` / `add_f32`, `sub_i32`, `mul_i32` / `mul_f32`, `lt_i32`, `le_i32`, `eq_i32` (most of these also reachable through operator overloading) |
| Tile ops | `dot`, `broadcast_2d`, `expand_dims`, `reshape_2d` |
| Special forms | `scf_for(lb, ub, step, init, |i, acc| body)`, `reduce(input, axis, |a, b| body)`, `return_()` |

## How the architecture stays stable

See `ARCHITECTURE.md` for the full design. Two decisions that keep us
insulated from upstream Triton churn:

1. **MLIR text is the ABI boundary.** The Rust frontend emits MLIR
   textual format strings and never binds Triton's C++ dialect builders.
   Triton's parser is round-trip-stable on the text it produces (their own
   `test/Triton/*.mlir` regression tests are written in this format), so
   our generated output keeps working across upstream releases.

2. **Triton C++ libs vendored at a pinned tag** (currently `v3.6.0`). The
   future C ABI shim links against this fixed copy through ~5 entry points,
   not against Triton's internal C++ namespace. Upgrading is a deliberate
   per-release decision.

## Build & test

```bash
# Pure CPU validation — no GPU / CUDA / Triton install required.
cargo test --workspace

# Round-trip a generated MLIR module through Triton 3.2.0 (uses Docker
# because Triton has no macOS wheel — see tools/README.md).
cargo run --example dump_vec_add_generic -p triton-dsl --quiet > tools/vec_add_generic.mlir
docker run --rm --platform linux/amd64 \
  -v "$(pwd)/tools":/work -w /work \
  python:3.11-slim \
  bash -c "pip install --quiet triton==3.2.0 && python validate_mlir.py vec_add_generic.mlir"
```

## Roadmap

- ✅ **Phase 0** — workspace bootstrap, decision lock-in
- ✅ **Phase 2** — IR builder + arith/tt/scf dialects, all validated against Triton 3.2.0
- ✅ **Phase 3** — `#[triton_kernel]` proc-macro covering signatures, type mapping,
   body translation, operator overloading, control flow, reductions, const generics
- 🚧 **Phase 1** — C ABI shim + cudarc runtime: get a real cubin onto a real GPU
- **Phase 4** — Autotuning (`#[autotune(BLOCK = 64..=1024 step 64, NUM_WARPS = [4,8])]`)
- **Phase 5** — AMD ROCm / Intel XPU backends through Triton's existing dispatch
- **Phase 6** — composable algebra, borrow-checked SMEM (`SharedMem<'_, f32, 1024>`)

## License

Apache-2.0
