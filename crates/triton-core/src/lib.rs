//! triton-rs — Rust-native GPU DSL on top of the Triton MLIR backend.
//!
//! Single dependency for downstream projects (ferrum-infer-rs et al). Re-exports the
//! `#[triton_kernel]` macro, the IR builder, the cudarc-based runtime, and the C ABI
//! shim's safe wrapper. End users typically just need:
//!
//! ```ignore
//! use triton_core::{triton_kernel, sys::{Context, CompileOptions}};
//! ```
//!
//! # Build features
//!
//! - `cpu-emulation` (default): pure-Rust runtime stub, no CUDA needed. Lets the
//!   workspace `cargo check` on Mac.
//! - `cuda`: cudarc-based device runtime. Requires CUDA toolkit + driver.
//! - `compile-triton`: forwards to `triton-sys/compile-triton`. Builds the C ABI
//!   shim against vendored Triton + LLVM (Linux only — see SPIKE.md). Without it,
//!   `Context::new()` returns `Err(TritonError::NotCompiled)` and you'll need
//!   to compile MLIR via the legacy `tools/mlir_to_cubin.py` script.
//!
//! See `BENCH_RESULTS.md` for head-to-head benchmark numbers (Rust shim ≡ Python
//! @triton.jit ≡ ferrum hand-written .cu on memory-bound kernels; up to 1.58x
//! faster than ferrum on rms_norm).

#![deny(missing_docs)]

pub use triton_dsl::triton_kernel;
pub use triton_ir as ir;
pub use triton_runtime as runtime;
pub use triton_sys as sys;

/// The version of the vendored Triton backend this build targets.
pub const TRITON_BACKEND_VERSION: &str = sys::TARGETED_TRITON_VERSION;

/// One-stop helpers for the most common pattern: take a `kernel::mlir()` string
/// and produce a cubin + metadata. Equivalent to constructing a [`sys::Context`]
/// manually and calling `compile()`, but keeps the imports tidy for the common
/// case (single kernel per process).
///
/// For high-throughput compile loops (many kernels per process), construct a
/// long-lived [`sys::Context`] yourself and reuse it — context creation
/// allocates an MLIRContext + dialect registry which is non-trivial.
pub fn compile_kernel(
    mlir_text: &str,
    opts: &sys::CompileOptions,
) -> Result<sys::Compiled, sys::TritonError> {
    let ctx = sys::Context::new()?;
    ctx.compile(mlir_text, opts)
}
