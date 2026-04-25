//! triton-rs — Rust-native GPU DSL on top of the Triton MLIR backend.
//!
//! This crate is the user-facing facade. Most projects should depend on this
//! crate alone and ignore the underlying `triton-{ir,runtime,dsl,sys}` split.

#![deny(missing_docs)]

pub use triton_dsl::triton_kernel;
pub use triton_ir as ir;
pub use triton_runtime as runtime;

/// The version of the vendored Triton backend this build targets.
pub const TRITON_BACKEND_VERSION: &str = "v3.6.0";
