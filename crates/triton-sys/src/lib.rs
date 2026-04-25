//! Low-level FFI bindings for the Triton MLIR backend.
//!
//! The ABI surface of this crate is intentionally tiny — see `ARCHITECTURE.md`
//! §2.3. Higher-level safe wrappers live in `triton-runtime` and `triton-ir`.
//!
//! # Status
//!
//! Phase 0: skeleton only. No Triton C++ libs are linked yet.
//! Phase 1 will populate this with bindgen-generated bindings to the
//! C shim defined in `shim/triton_c.h`.

#![deny(missing_docs)]

use thiserror::Error;

/// Errors crossing the C ABI boundary.
#[derive(Debug, Error)]
pub enum TritonError {
    /// The Triton C++ libs were not compiled into this build.
    #[error("triton-sys was built without the `compile-triton` feature")]
    NotCompiled,

    /// The MLIR text failed to parse or the pass pipeline failed.
    #[error("triton compile failed: {0}")]
    CompileFailed(String),

    /// FFI returned a null pointer where a value was expected.
    #[error("ffi null pointer (kind: {0})")]
    NullPointer(&'static str),
}

/// The version of the vendored Triton we target.
///
/// Locked in `ARCHITECTURE.md` §2.2. Bumped only on planned upgrades.
pub const TARGETED_TRITON_VERSION: &str = "v3.6.0";

// Phase 1: include the bindgen output here.
// #[allow(non_upper_case_globals, non_camel_case_types, non_snake_case)]
// mod ffi {
//     include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
// }
// pub use ffi::*;
