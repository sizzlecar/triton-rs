//! Pure-Rust MLIR text builder for the Triton dialect.
//!
//! This crate emits **MLIR textual format** for the dialects Triton consumes
//! (`builtin`, `func`, `arith`, `scf`, `tt`, `triton_gpu`). It deliberately
//! does *not* link against any C/C++ MLIR libraries — the output is a string
//! handed to `triton-sys::triton_compile_mlir` over the FFI boundary.
//!
//! See `ARCHITECTURE.md` §2.1 for why MLIR text is the chosen ABI surface.
//!
//! # Quick taste
//!
//! ```
//! use triton_ir::prelude::*;
//!
//! let mut m = Module::new();
//! let mut f = m.func("hello");
//! let _x = f.arg("x", Type::ptr(Type::f32()));
//! let pid = f.op_one(tt::get_program_id(0));
//! let _ = f.op_one(arith::constant_i32(42));
//! let _ = pid; // suppress unused warning
//! f.op_void(tt::return_());
//! f.finish();
//!
//! let mlir_text = m.to_string();
//! assert!(mlir_text.contains("tt.func"));
//! ```

#![deny(missing_docs)]

pub mod attr;
pub mod dialect;
pub mod module;
pub mod op;
pub mod printer;
pub mod ty;
pub mod value;

/// Re-exports for typical authoring use.
pub mod prelude {
    pub use crate::attr::Attr;
    pub use crate::dialect::{arith, scf, tt};
    pub use crate::module::{FuncBuilder, Module, Visibility};
    pub use crate::op::{Block, Op, OpSpec, Region};
    pub use crate::ty::Type;
    pub use crate::value::Value;
}
