# triton-rs

> Rust-native GPU DSL on top of the Triton MLIR backend.
> **Zero Python at build- or run-time.**

## What is this?

triton-rs is a Rust frontend, runtime, and toolchain for writing high-performance GPU kernels. It reuses the [Triton](https://github.com/triton-lang/triton) MLIR compiler backend (PTX/AMDGPU/SPIR-V codegen) but replaces the Python frontend with a Rust DSL designed to fix the known ergonomics, type-safety, and tooling shortcomings of `@triton.jit`.

```rust
#[triton_kernel]
fn vector_add<const BLOCK: usize>(
    x: Ptr<f32>,
    y: Ptr<f32>,
    out: Ptr<f32>,
    n: i32,
) {
    let pid = program_id(0);
    let offsets = pid * BLOCK + arange::<BLOCK>();
    let mask = offsets.lt(n);
    let xv = load(x + offsets, mask);
    let yv = load(y + offsets, mask);
    store(out + offsets, xv + yv, mask);
}

fn main() -> Result<()> {
    let stream = CudaRuntime::default_stream()?;
    let mut out = DeviceBuffer::<f32>::zeros(N, &stream)?;
    vector_add::<1024>::launch(&stream, &x, &y, &mut out, N as i32)?;
    Ok(())
}
```

## Design highlights

- **MLIR text as the ABI boundary** — Rust frontend emits Triton MLIR text strings; never binds Triton's C++ dialect builders. Triton C++ libs are vendored and pinned (currently `v3.6.0`).
- **Const-generic shape & constexpr** — `Tensor<f32, BLOCK>` shape mismatches caught by `rustc`, not at GPU runtime.
- **CPU emulation backend** — `cargo test` validates kernel semantics on GitHub free runners; GPU CI only for end-to-end runs.
- **Single binary deployment** — no Python, no `libtriton`, no LLVM at runtime. Just your binary + cubin.
- **Multi-arch via `target_arch`** — NVIDIA today (`sm_80/89/90a`), AMD (`gfx*`) and Intel XPU planned, all through Triton's existing backends.

## Status

**Phase 0 — scaffolding.** See `ARCHITECTURE.md` for the full design and roadmap.

| Phase | What | When |
|---|---|---|
| 0 | Workspace skeleton + decisions locked in | now |
| 1 | C ABI shim + cudarc runtime, `vector_add` end-to-end | next |
| 2 | Rust IR builder (manual MLIR text emission) | |
| 3 | `#[triton_kernel]` proc-macro DSL (v0.1) | |
| 4 | Autotune + ergonomics (v0.2) | |
| 5 | AMD / Intel backends | |
| 6 | v1.0: composable algebra, borrow-checked SMEM | |

## License

Apache-2.0
