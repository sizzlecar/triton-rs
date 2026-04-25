//! `#[triton_kernel]` proc-macro — the user-facing DSL.
//!
//! See `ARCHITECTURE.md` §3 for the design goals — especially how this
//! crate intentionally improves on the ergonomics, type safety, and IDE
//! experience of Python's `@triton.jit`.
//!
//! # Status
//!
//! Phase 3.1: signature parsing + MLIR func skeleton emission. The
//! generated code does *not* yet translate the function body — that
//! arrives in Phase 3.3. For now the body is replaced by `tt.return`
//! so the emitted MLIR is well-formed.
//!
//! # Generated API
//!
//! Given:
//! ```ignore
//! #[triton_kernel]
//! fn vec_add(x: Ptr<f32>, y: Ptr<f32>, out: Ptr<f32>, n: i32) { /* body */ }
//! ```
//!
//! The macro replaces the function with a marker struct carrying compile-time
//! generators:
//! ```ignore
//! pub struct vec_add;
//! impl vec_add {
//!     pub fn module() -> ::triton_ir::module::Module { /* ... */ }
//!     pub fn mlir()   -> String { Self::module().to_string() }
//! }
//! ```

use proc_macro::TokenStream;

mod codegen;

/// Mark a function as a Triton GPU kernel.
///
/// Phase 3.1: the function body is ignored; only the signature drives
/// MLIR generation. Phase 3.3 will translate the body into IR builder
/// calls.
#[proc_macro_attribute]
pub fn triton_kernel(_args: TokenStream, item: TokenStream) -> TokenStream {
    let parsed = match syn::parse::<syn::ItemFn>(item) {
        Ok(f) => f,
        Err(e) => return e.to_compile_error().into(),
    };
    match codegen::expand(&parsed) {
        Ok(ts) => ts.into(),
        Err(e) => e.to_compile_error().into(),
    }
}
