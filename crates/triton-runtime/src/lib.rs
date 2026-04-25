//! Device runtime for triton-rs.
//!
//! Loads compiled kernel binaries (cubin/hsaco/spirv) and launches them on
//! the target device. The `DeviceRuntime` trait abstracts over CUDA / HIP /
//! Level Zero / CPU-emulation, all selectable through Cargo features.
//!
//! # Status
//!
//! Phase 0: trait skeleton only, no real backend implementations.
//! Phase 1 will fill in [`cuda::CudaRuntime`] (cudarc-based) and the CPU
//! emulation backend used by the test suite on GPU-less CI runners.

#![deny(missing_docs)]

use thiserror::Error;

/// Errors from the runtime layer.
#[derive(Debug, Error)]
pub enum RuntimeError {
    /// Failed to load a compiled kernel binary onto the device.
    #[error("module load failed: {0}")]
    ModuleLoad(String),

    /// `launch` rejected the grid/block configuration.
    #[error("invalid launch config: {0}")]
    InvalidLaunchConfig(String),

    /// Underlying device API returned an error.
    #[error("device error: {0}")]
    Device(String),

    /// Requested feature is not available in this build (missing feature flag).
    #[error("backend `{0}` not enabled in this build")]
    BackendDisabled(&'static str),
}

/// 3D launch grid (block count along x/y/z).
pub type Grid = [u32; 3];

/// 3D thread block shape (threads along x/y/z).
pub type Block = [u32; 3];

/// Type-erased kernel argument passed to a device launch.
///
/// `triton-dsl` generates strongly-typed wrappers; this is the FFI layer.
#[derive(Debug, Clone, Copy)]
pub enum KernelArg<'a> {
    /// 32-bit signed integer.
    I32(i32),
    /// 32-bit unsigned integer.
    U32(u32),
    /// 64-bit signed integer.
    I64(i64),
    /// 32-bit float.
    F32(f32),
    /// Opaque device pointer (`*const T` / `*mut T`).
    DevicePtr(u64),
    /// Borrowed byte slice copied into the kernel param buffer.
    Bytes(&'a [u8]),
}

/// Abstract GPU-like runtime. Implementations: CUDA (Phase 1), CPU emulation
/// (Phase 1 for CI), HIP (Phase 5), Level Zero (Phase 6).
pub trait DeviceRuntime: Send + Sync {
    /// Loaded module handle (typically wraps a `CUmodule` / equivalent).
    type Module: Send + Sync;
    /// Stream / queue handle.
    type Stream: Send + Sync;

    /// Load a compiled kernel binary (cubin / hsaco / spirv) onto the device.
    fn load_module(&self, binary: &[u8]) -> Result<Self::Module, RuntimeError>;

    /// Launch a kernel by name from a previously-loaded module.
    #[allow(clippy::too_many_arguments)]
    fn launch(
        &self,
        module: &Self::Module,
        kernel_name: &str,
        grid: Grid,
        block: Block,
        shared_mem_bytes: u32,
        stream: &Self::Stream,
        args: &[KernelArg<'_>],
    ) -> Result<(), RuntimeError>;
}

/// CUDA runtime (cudarc-based). Stub in Phase 0.
#[cfg(feature = "cuda")]
pub mod cuda {
    //! CUDA backend — populated in Phase 1.
}

/// CPU emulation backend used by `cargo test` on GitHub free runners.
/// Populated in Phase 3 alongside the proc-macro.
#[cfg(feature = "cpu-emulation")]
pub mod cpu {}
