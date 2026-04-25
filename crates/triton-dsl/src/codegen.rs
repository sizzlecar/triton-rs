//! Codegen for `#[triton_kernel]`. Splits responsibility into two pieces:
//!   1. Map a Rust function signature → a list of (arg name, IR type expr).
//!   2. Wrap the result in a `pub struct <name>;` + `impl` providing the
//!      `module()` / `mlir()` accessors.

use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use syn::{spanned::Spanned, FnArg, ItemFn, Pat, PathArguments, Type};

/// Top-level expansion entry point.
pub fn expand(input: &ItemFn) -> Result<TokenStream2, syn::Error> {
    let fn_name = &input.sig.ident;
    let kernel_name = fn_name.to_string();

    let arg_decls = collect_arg_decls(input)?;

    let vis = &input.vis;
    Ok(quote! {
        #[allow(non_camel_case_types)]
        #vis struct #fn_name;

        impl #fn_name {
            /// Build the MLIR `Module` for this kernel.
            ///
            /// Phase 3.1: signature only — the body is a single `tt.return`
            /// terminator. Phase 3.3 will translate the user-written body
            /// into IR builder calls here.
            pub fn module() -> ::triton_ir::module::Module {
                use ::triton_ir::prelude::*;

                let mut m = Module::new();
                let mut f = m.func(#kernel_name);
                #(#arg_decls)*
                f.op_void(tt::return_());
                f.finish();
                m
            }

            /// Pretty-printed MLIR text for this kernel.
            pub fn mlir() -> ::std::string::String {
                Self::module().to_string()
            }
        }
    })
}

/// Walk the function's arguments, mapping each to a `let _argN = f.arg(...)`
/// statement that calls into the IR builder.
fn collect_arg_decls(input: &ItemFn) -> Result<Vec<TokenStream2>, syn::Error> {
    let mut out = Vec::with_capacity(input.sig.inputs.len());
    for (i, fn_arg) in input.sig.inputs.iter().enumerate() {
        let pat_type = match fn_arg {
            FnArg::Typed(p) => p,
            FnArg::Receiver(r) => {
                return Err(syn::Error::new(
                    r.span(),
                    "#[triton_kernel] cannot have a receiver (no `self` argument)",
                ));
            }
        };
        let arg_name = match &*pat_type.pat {
            Pat::Ident(p) => p.ident.to_string(),
            other => {
                return Err(syn::Error::new(
                    other.span(),
                    "kernel argument names must be plain identifiers",
                ));
            }
        };
        let ty_expr = rust_type_to_ir_type(&pat_type.ty)?;
        let var_ident = format_ident!("_arg{}", i);
        out.push(quote! { let #var_ident = f.arg(#arg_name, #ty_expr); });
    }
    Ok(out)
}

/// Map a Rust type token tree into a `triton_ir::ty::Type` constructor
/// expression.
///
/// Supported in Phase 3.1: `i1`, `i32`, `i64`, `f16`, `f32`, `bf16`,
/// `Ptr<T>` (recursively). Tensor / const-generic shapes arrive in 3.2.
fn rust_type_to_ir_type(ty: &Type) -> Result<TokenStream2, syn::Error> {
    let path = match ty {
        Type::Path(tp) if tp.qself.is_none() => &tp.path,
        other => {
            return Err(syn::Error::new(
                other.span(),
                "unsupported type form in #[triton_kernel] signature \
                 (use bare type names: i32 / f32 / Ptr<f32> / ...)",
            ));
        }
    };

    let seg = path
        .segments
        .last()
        .ok_or_else(|| syn::Error::new(path.span(), "empty type path"))?;

    match seg.ident.to_string().as_str() {
        "i1" => Ok(quote! { ::triton_ir::ty::Type::i1() }),
        "i32" => Ok(quote! { ::triton_ir::ty::Type::i32() }),
        "i64" => Ok(quote! { ::triton_ir::ty::Type::i64() }),
        "f16" => Ok(quote! { ::triton_ir::ty::Type::f16() }),
        "f32" => Ok(quote! { ::triton_ir::ty::Type::f32() }),
        "bf16" => Ok(quote! { ::triton_ir::ty::Type::bf16() }),
        "Ptr" => {
            let args = match &seg.arguments {
                PathArguments::AngleBracketed(a) => a,
                _ => {
                    return Err(syn::Error::new(
                        seg.span(),
                        "Ptr requires a `<T>` type argument",
                    ));
                }
            };
            let inner_ty = args
                .args
                .iter()
                .find_map(|a| match a {
                    syn::GenericArgument::Type(t) => Some(t),
                    _ => None,
                })
                .ok_or_else(|| {
                    syn::Error::new(args.span(), "Ptr<T>: T must be a type argument")
                })?;
            let inner_expr = rust_type_to_ir_type(inner_ty)?;
            Ok(quote! { ::triton_ir::ty::Type::ptr(#inner_expr) })
        }
        other => Err(syn::Error::new(
            seg.span(),
            format!(
                "unsupported type `{}` in #[triton_kernel] signature \
                 (Phase 3.1 supports i1/i32/i64/f16/f32/bf16 and Ptr<T>)",
                other
            ),
        )),
    }
}
