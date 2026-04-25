//! `#[triton_kernel]` proc-macro — the user-facing DSL.
//!
//! Parses a Rust function body, validates it against the DSL grammar (a
//! restricted subset of Rust mappable to Triton's tile-level semantics),
//! type-checks shapes via const generics, and emits `triton-ir` builder calls
//! that produce an MLIR text module at compile time.
//!
//! See `ARCHITECTURE.md` §3 for the DSL design goals — especially how this
//! crate intentionally improves on the ergonomics, type safety, and IDE
//! experience of Python's `@triton.jit`.
//!
//! # Status
//!
//! Phase 0: stub. Phase 3 implements parsing, type inference, and codegen.

use proc_macro::TokenStream;

/// Mark a function as a Triton GPU kernel.
///
/// Phase 0: macro is recognized but currently a no-op pass-through so the
/// workspace compiles end-to-end.
#[proc_macro_attribute]
pub fn triton_kernel(_args: TokenStream, item: TokenStream) -> TokenStream {
    // TODO(phase-3): full parse → typecheck → MLIR text emission pipeline.
    item
}
