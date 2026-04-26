//! Low-level FFI bindings for the Triton MLIR backend.
//!
//! The ABI surface of this crate is intentionally tiny — see `ARCHITECTURE.md`
//! §2.3. Five C entry points (`triton_context_*`, `triton_compile_mlir`,
//! `triton_result_destroy`, `triton_get_version`); higher-level safe wrappers
//! provided here.
//!
//! # Feature flags
//!
//! By default this crate ships only the Rust skeleton — `TritonError`,
//! `TARGETED_TRITON_VERSION`, the `Context::new()` constructor returns
//! `Err(TritonError::NotCompiled)`. This keeps `cargo check --workspace`
//! fast on Mac / no-CUDA boxes.
//!
//! Enable `compile-triton` (Linux + CUDA only) to actually build the C++
//! shim against the vendored Triton.

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
/// Pinned in `crates/triton-sys/tools/fetch_vendor.sh`. Bumped to `v3.6.0`
/// in Phase 1F per ARCHITECTURE.md §2.2.
pub const TARGETED_TRITON_VERSION: &str = "v3.6.0";

/// Compile options forwarded to Triton's pass pipeline. Mirrors the
/// `TritonCompileOptions` C struct in `shim/triton_c.h`.
#[derive(Debug, Clone)]
pub struct CompileOptions {
    /// Target architecture string. Examples: `"sm_80"`, `"sm_89"`, `"sm_90a"`.
    pub target_arch: String,
    /// Number of warps per program. `0` means "backend default" (typically 4).
    pub num_warps: u32,
    /// Number of pipeline stages for `add_pipeline`. `0` = default.
    pub num_stages: u32,
    /// Number of cooperative thread arrays per cluster (Hopper+). `0` = default (1).
    pub num_ctas: u32,
}

impl Default for CompileOptions {
    fn default() -> Self {
        Self {
            target_arch: "sm_89".into(), // matches Vast.ai test box (RTX 5070 Ti, treated as sm_89 PTX)
            num_warps: 4,
            num_stages: 3,
            num_ctas: 1,
        }
    }
}

/// Successful compile output: cubin bytes + PTX text + metadata JSON.
pub struct Compiled {
    cubin: Vec<u8>,
    ptx_text: String,
    metadata_json: String,
}

impl Compiled {
    /// Cubin (CUDA binary) bytes — load via `cuModuleLoadData` /
    /// equivalent. Empty when the backend is non-NVIDIA.
    pub fn cubin(&self) -> &[u8] {
        &self.cubin
    }

    /// PTX text — emitted by LLVM's NVPTX backend before ptxas. Use
    /// this with cudarc 0.13's `Ptx::from(...)` (which only accepts
    /// PTX text, not raw cubin). Empty when backend is non-NVIDIA.
    pub fn ptx_text(&self) -> &str {
        &self.ptx_text
    }

    /// JSON-encoded kernel metadata (name, num_warps, shared_mem,
    /// target_arch, cluster_dims, ...). Parse with serde_json upstream.
    pub fn metadata_json(&self) -> &str {
        &self.metadata_json
    }
}

#[cfg(feature = "compile-triton")]
mod ffi {
    #![allow(non_upper_case_globals, non_camel_case_types, non_snake_case)]
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

/// Owned Triton compilation context.
///
/// Wraps a `TritonContext*` from the C ABI with a `Drop` that calls
/// `triton_context_destroy`. One per process is typically enough; the
/// MLIRContext + dialect registry inside is reusable across compiles.
pub struct Context {
    #[cfg(feature = "compile-triton")]
    raw: *mut ffi::TritonContext,
}

impl Context {
    /// Create a new context. With the `compile-triton` feature off, returns
    /// `Err(TritonError::NotCompiled)`.
    #[cfg(not(feature = "compile-triton"))]
    pub fn new() -> Result<Self, TritonError> {
        Err(TritonError::NotCompiled)
    }

    /// Create a new context. Calls `triton_context_create` under the hood.
    #[cfg(feature = "compile-triton")]
    pub fn new() -> Result<Self, TritonError> {
        let raw = unsafe { ffi::triton_context_create() };
        if raw.is_null() {
            return Err(TritonError::NullPointer("triton_context_create"));
        }
        Ok(Self { raw })
    }

    /// Compile MLIR text → cubin + metadata.
    #[cfg(not(feature = "compile-triton"))]
    pub fn compile(&self, _mlir_text: &str, _opts: &CompileOptions) -> Result<Compiled, TritonError> {
        Err(TritonError::NotCompiled)
    }

    /// Compile MLIR text → cubin + metadata.
    #[cfg(feature = "compile-triton")]
    pub fn compile(&self, mlir_text: &str, opts: &CompileOptions) -> Result<Compiled, TritonError> {
        use std::ffi::CString;
        let mlir_c = CString::new(mlir_text)
            .map_err(|_| TritonError::CompileFailed("mlir_text contains NUL".into()))?;
        let arch_c = CString::new(opts.target_arch.as_str())
            .map_err(|_| TritonError::CompileFailed("target_arch contains NUL".into()))?;

        let c_opts = ffi::TritonCompileOptions {
            target_arch: arch_c.as_ptr(),
            num_warps: opts.num_warps as i32,
            num_stages: opts.num_stages as i32,
            num_ctas: opts.num_ctas as i32,
            reserved: [0u64; 8],
        };

        let result_ptr = unsafe { ffi::triton_compile_mlir(self.raw, mlir_c.as_ptr(), &c_opts) };
        if result_ptr.is_null() {
            return Err(TritonError::NullPointer("triton_compile_mlir"));
        }

        // Take ownership so it gets freed regardless of branch below.
        let result = ResultGuard { raw: result_ptr };
        let r = unsafe { &*result_ptr };

        if r.status != 0 {
            let msg = if r.error_message.is_null() {
                "<no error message>".to_string()
            } else {
                unsafe { std::ffi::CStr::from_ptr(r.error_message).to_string_lossy().into_owned() }
            };
            return Err(TritonError::CompileFailed(msg));
        }

        let cubin = if r.binary_data.is_null() || r.binary_size == 0 {
            Vec::new()
        } else {
            unsafe { std::slice::from_raw_parts(r.binary_data, r.binary_size).to_vec() }
        };
        let metadata_json = if r.metadata_json.is_null() {
            String::new()
        } else {
            unsafe { std::ffi::CStr::from_ptr(r.metadata_json).to_string_lossy().into_owned() }
        };
        let ptx_text = if r.ptx_text.is_null() {
            String::new()
        } else {
            unsafe { std::ffi::CStr::from_ptr(r.ptx_text).to_string_lossy().into_owned() }
        };

        drop(result); // explicit free of the C-side TritonResult
        Ok(Compiled { cubin, ptx_text, metadata_json })
    }
}

#[cfg(feature = "compile-triton")]
impl Drop for Context {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { ffi::triton_context_destroy(self.raw) };
            self.raw = std::ptr::null_mut();
        }
    }
}

#[cfg(feature = "compile-triton")]
struct ResultGuard {
    raw: *mut ffi::TritonResult,
}

#[cfg(feature = "compile-triton")]
impl Drop for ResultGuard {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { ffi::triton_result_destroy(self.raw) };
            self.raw = std::ptr::null_mut();
        }
    }
}

/// Returns the version of Triton this build was compiled against.
/// With the `compile-triton` feature off, returns the hardcoded
/// [`TARGETED_TRITON_VERSION`] without calling FFI.
pub fn triton_version() -> &'static str {
    #[cfg(feature = "compile-triton")]
    unsafe {
        let p = ffi::triton_get_version();
        if p.is_null() {
            return TARGETED_TRITON_VERSION;
        }
        std::ffi::CStr::from_ptr(p).to_str().unwrap_or(TARGETED_TRITON_VERSION)
    }
    #[cfg(not(feature = "compile-triton"))]
    TARGETED_TRITON_VERSION
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_string_present() {
        assert!(!triton_version().is_empty());
    }

    #[test]
    #[cfg(not(feature = "compile-triton"))]
    fn context_returns_not_compiled() {
        assert!(matches!(Context::new(), Err(TritonError::NotCompiled)));
    }
}
