//! Codegen for `#[triton_kernel]`.
//!
//! Splits responsibility into:
//!   1. Map a Rust function signature → IR type expressions for each arg.
//!   2. Translate the function body's statements/exprs into IR builder calls
//!      against the `__triton_f: FuncBuilder` we hold in scope.
//!   3. Wrap the result in a `pub struct <name>;` + `impl` providing the
//!      `module()` / `mlir()` accessors.
//!
//! ## Internal naming
//!
//! All locals introduced by the macro are prefixed `__triton_` so they
//! cannot collide with a kernel argument the user happens to name `m` /
//! `f` / etc.

use proc_macro2::{Span, TokenStream as TokenStream2};
use quote::{format_ident, quote};
use syn::{
    spanned::Spanned, BinOp, Block, Expr, ExprBinary, ExprCall, FnArg, ItemFn, Local, Pat,
    PathArguments, Stmt, Type,
};

/// Top-level expansion entry point.
pub fn expand(input: &ItemFn) -> Result<TokenStream2, syn::Error> {
    let fn_name = &input.sig.ident;
    let kernel_name = fn_name.to_string();

    let arg_decls = collect_arg_decls(input)?;

    // Counter for auto-generated temporary names introduced by call-arg
    // hoisting. Lives only during expansion; reset per kernel.
    let mut tmp_counter: u32 = 0;
    let body_stmts = translate_block(&input.block, &mut tmp_counter)?;

    let vis = &input.vis;
    Ok(quote! {
        #[allow(non_camel_case_types)]
        #vis struct #fn_name;

        impl #fn_name {
            /// Build the MLIR `Module` for this kernel.
            // Args bound for the user's reference may legitimately go unused
            // (e.g. signature-only kernels in tests); auto-hoisted temps in
            // nested calls produce identifiers we never re-read by design.
            #[allow(unused_variables)]
            pub fn module() -> ::triton_ir::module::Module {
                let mut __triton_module = ::triton_ir::module::Module::new();
                let mut __triton_f = __triton_module.func(#kernel_name);
                #(#arg_decls)*
                #body_stmts
                __triton_f.op_void(::triton_ir::dialect::tt::return_());
                __triton_f.finish();
                __triton_module
            }

            /// Pretty-printed MLIR text for this kernel.
            pub fn mlir() -> ::std::string::String {
                Self::module().to_string()
            }
        }
    })
}

// ── Signature handling ──────────────────────────────────────────────────

/// Map each function arg to a `let <name> = __triton_f.arg(...)` statement.
/// The Rust binding uses the user's actual argument identifier so the body
/// can refer to args by name (`store(out, v)` references the `out` arg
/// directly).
fn collect_arg_decls(input: &ItemFn) -> Result<Vec<TokenStream2>, syn::Error> {
    let mut out = Vec::with_capacity(input.sig.inputs.len());
    for fn_arg in input.sig.inputs.iter() {
        let pat_type = match fn_arg {
            FnArg::Typed(p) => p,
            FnArg::Receiver(r) => {
                return Err(syn::Error::new(
                    r.span(),
                    "#[triton_kernel] cannot have a receiver (no `self` argument)",
                ));
            }
        };
        let arg_ident = match &*pat_type.pat {
            Pat::Ident(p) => p.ident.clone(),
            other => {
                return Err(syn::Error::new(
                    other.span(),
                    "kernel argument names must be plain identifiers",
                ));
            }
        };
        let arg_name = arg_ident.to_string();
        let ty_expr = rust_type_to_ir_type(&pat_type.ty)?;
        out.push(quote! { let #arg_ident = __triton_f.arg(#arg_name, #ty_expr); });
    }
    Ok(out)
}

/// Map a Rust type token tree into a `triton_ir::ty::Type` constructor expr.
///
/// Supported in Phase 3.1: `i1`, `i32`, `i64`, `f16`, `f32`, `bf16`,
/// `Ptr<T>` (recursively).
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

// ── Body translation (Phase 3.3) ────────────────────────────────────────
//
// Recognised user syntax:
//   - `let x = <expr>;`            — pass through, RHS translated
//   - `<expr>;` and trailing `<expr>` — translated, side-effecting
//   - call expressions (see `op_spec_for` for the supported names)
//   - path expressions (variable references) and integer literals
//     (used as numeric arguments to known calls)
//
// Nested calls are auto-hoisted into named temporaries so the borrow
// checker doesn't see two simultaneous mut borrows of `__triton_f`.

#[derive(Debug, Clone, Copy)]
enum CallKind {
    /// Produces an SSA `Value` — emitted via `__triton_f.op_one(spec)`.
    Value,
    /// Side-effect only — emitted via `__triton_f.op_void(spec)`.
    Void,
}

/// Result of translating an expression: zero or more setup statements,
/// followed by a single Rust expression that evaluates to the desired
/// value (or `()` for void calls).
struct TranslatedExpr {
    setup: Vec<TokenStream2>,
    expr: TokenStream2,
}

fn translate_block(block: &Block, counter: &mut u32) -> Result<TokenStream2, syn::Error> {
    let mut out = Vec::with_capacity(block.stmts.len());
    for stmt in &block.stmts {
        out.push(translate_stmt(stmt, counter)?);
    }
    Ok(quote! { #(#out)* })
}

fn translate_stmt(stmt: &Stmt, counter: &mut u32) -> Result<TokenStream2, syn::Error> {
    match stmt {
        Stmt::Local(local) => translate_local(local, counter),
        Stmt::Expr(expr, maybe_semi) => {
            // Top-level statement: no need to hoist a top-level call (the
            // statement boundary already releases the borrow).
            let t = translate_expr(expr, /*hoist=*/ false, counter)?;
            let setup = &t.setup;
            let e = &t.expr;
            if let Some(semi) = maybe_semi {
                Ok(quote! { #(#setup)* #e #semi })
            } else {
                // Trailing expr — discard its value, kernel funcs return ().
                Ok(quote! { #(#setup)* let _ = #e; })
            }
        }
        Stmt::Item(item) => Err(syn::Error::new(
            item.span(),
            "nested items are not supported inside #[triton_kernel] bodies",
        )),
        Stmt::Macro(m) => Err(syn::Error::new(
            m.span(),
            "macro invocations are not supported inside #[triton_kernel] bodies (Phase 3.3 step 1)",
        )),
    }
}

fn translate_local(local: &Local, counter: &mut u32) -> Result<TokenStream2, syn::Error> {
    let init = local.init.as_ref().ok_or_else(|| {
        syn::Error::new(
            local.span(),
            "`let` bindings inside #[triton_kernel] must have an initializer",
        )
    })?;
    if init.diverge.is_some() {
        return Err(syn::Error::new(
            init.diverge.as_ref().unwrap().0.span,
            "`let ... else { ... }` is not supported inside #[triton_kernel] bodies",
        ));
    }
    let pat = &local.pat;
    // The let binding itself acts as the hoist for the top-level expression.
    let t = translate_expr(&init.expr, /*hoist=*/ false, counter)?;
    let setup = &t.setup;
    let rhs = &t.expr;
    Ok(quote! {
        #(#setup)*
        let #pat = #rhs;
    })
}

fn translate_expr(
    expr: &Expr,
    hoist: bool,
    counter: &mut u32,
) -> Result<TranslatedExpr, syn::Error> {
    match expr {
        Expr::Call(call) => translate_call(call, hoist, counter),
        Expr::Binary(b) => translate_binop(b, hoist, counter),
        // Variable references appear at every use site. `triton_ir::Value`
        // is `Clone` but not `Copy`, so emit an explicit `.clone()` at every
        // reference. Cloning a Value is cheap (one Type box for tensors,
        // pure Copy for scalar types) and lets users write
        // `load(p, mask); store(p, v, mask);` naturally without `mask` being
        // moved into the first call.
        Expr::Path(_) => Ok(TranslatedExpr {
            setup: vec![],
            expr: quote! { ::std::clone::Clone::clone(&#expr) },
        }),
        // Integer / float literals pass through. They're `Copy`.
        Expr::Lit(_) => Ok(TranslatedExpr {
            setup: vec![],
            expr: quote! { #expr },
        }),
        Expr::Paren(p) => translate_expr(&p.expr, hoist, counter),
        other => Err(syn::Error::new(
            other.span(),
            "this expression form is not yet supported inside #[triton_kernel] bodies \
             (supported: function calls, binary ops `+ - * < <= > >= == !=`, \
             variable refs, integer literals)",
        )),
    }
}

/// Wrap an emitted value-producing expression in a hoisting `let __tmp = ...;`
/// if the caller wants the result as a sub-expression of an outer call (which
/// would otherwise mut-borrow `__triton_f` simultaneously). Top-level callers
/// pass `hoist = false` because the surrounding `let pat = ...;` (or
/// `expr;` statement) already releases the borrow at the semicolon.
fn maybe_hoist(
    mut setup: Vec<TokenStream2>,
    expr: TokenStream2,
    kind: CallKind,
    hoist: bool,
    counter: &mut u32,
) -> TranslatedExpr {
    if hoist && matches!(kind, CallKind::Value) {
        let tmp = format_ident!("__triton_tmp_{}", *counter);
        *counter += 1;
        setup.push(quote! { let #tmp = #expr; });
        TranslatedExpr {
            setup,
            expr: quote! { #tmp },
        }
    } else {
        TranslatedExpr { setup, expr }
    }
}

/// Translate `a OP b` into a call into `triton_ir::ops::*`, which dispatches
/// on the runtime type of `a` to pick the right MLIR op (e.g. `+` becomes
/// `tt.addptr` for pointer-typed lhs, `arith.addf` for float, otherwise
/// `arith.addi`). Operands are recursively translated and force-hoisted so
/// nested binary expressions don't trigger borrow-checker conflicts.
fn translate_binop(
    node: &ExprBinary,
    hoist: bool,
    counter: &mut u32,
) -> Result<TranslatedExpr, syn::Error> {
    let lt = translate_expr(&node.left, /*hoist=*/ true, counter)?;
    let rt = translate_expr(&node.right, /*hoist=*/ true, counter)?;

    let mut setup = lt.setup;
    setup.extend(rt.setup);
    let l = lt.expr;
    let r = rt.expr;

    let dispatch_fn = match node.op {
        BinOp::Add(_) => quote! { ::triton_ir::ops::add },
        BinOp::Sub(_) => quote! { ::triton_ir::ops::sub },
        BinOp::Mul(_) => quote! { ::triton_ir::ops::mul },
        BinOp::Lt(_) => quote! { ::triton_ir::ops::lt },
        BinOp::Le(_) => quote! { ::triton_ir::ops::le },
        BinOp::Gt(_) => quote! { ::triton_ir::ops::gt },
        BinOp::Ge(_) => quote! { ::triton_ir::ops::ge },
        BinOp::Eq(_) => quote! { ::triton_ir::ops::eq },
        BinOp::Ne(_) => quote! { ::triton_ir::ops::ne },
        other => {
            return Err(syn::Error::new(
                node.op.span(),
                format!(
                    "binary operator `{:?}` is not supported in #[triton_kernel] body \
                     (supported: `+ - * < <= > >= == !=`)",
                    other
                ),
            ));
        }
    };

    let call_expr = quote! { #dispatch_fn(&mut __triton_f, #l, #r) };
    Ok(maybe_hoist(setup, call_expr, CallKind::Value, hoist, counter))
}

fn translate_call(
    call: &ExprCall,
    hoist: bool,
    counter: &mut u32,
) -> Result<TranslatedExpr, syn::Error> {
    // Args to a call are always evaluated to a temporary first to avoid
    // multiple simultaneous mut borrows of __triton_f.
    let mut setup = Vec::new();
    let mut translated_args = Vec::new();
    for arg in &call.args {
        let t = translate_expr(arg, /*hoist=*/ true, counter)?;
        setup.extend(t.setup);
        translated_args.push(t.expr);
    }

    let name = call_name(call)?;
    let (kind, spec_expr) = op_spec_for(&name, &translated_args, call.span())?;

    let call_expr = match kind {
        CallKind::Value => quote! { __triton_f.op_one(#spec_expr) },
        CallKind::Void => quote! { __triton_f.op_void(#spec_expr) },
    };

    Ok(maybe_hoist(setup, call_expr, kind, hoist, counter))
}

fn call_name(call: &ExprCall) -> Result<String, syn::Error> {
    match &*call.func {
        Expr::Path(p) if p.qself.is_none() => p
            .path
            .segments
            .last()
            .map(|s| s.ident.to_string())
            .ok_or_else(|| syn::Error::new(p.span(), "empty function path")),
        other => Err(syn::Error::new(
            other.span(),
            "expected a bare function name (e.g. `program_id`, `const_i32`, ...)",
        )),
    }
}

/// Vocabulary of recognised function names → IR builder helpers.
/// Anything not listed here errors out at compile time.
fn op_spec_for(
    name: &str,
    args: &[TokenStream2],
    call_span: Span,
) -> Result<(CallKind, TokenStream2), syn::Error> {
    let n = args.len();
    let arity_err = |needed: &str| -> syn::Error {
        syn::Error::new(
            call_span,
            format!("`{}` expects {} arguments, got {}", name, needed, n),
        )
    };

    let result = match name {
        // ── tt dialect ──
        "program_id" if n == 1 => (
            CallKind::Value,
            quote! { ::triton_ir::dialect::tt::get_program_id(#(#args),*) },
        ),
        "program_id" => return Err(arity_err("1")),

        // load(ptrs)        : no mask
        // load(ptrs, mask)  : with mask
        "load" if n == 1 => (
            CallKind::Value,
            quote! { ::triton_ir::dialect::tt::load(#(#args),*, ::std::option::Option::None) },
        ),
        "load" if n == 2 => {
            let p = &args[0];
            let m = &args[1];
            (
                CallKind::Value,
                quote! {
                    ::triton_ir::dialect::tt::load(
                        #p, ::std::option::Option::Some(#m),
                    )
                },
            )
        }
        "load" => return Err(arity_err("1 or 2 (ptrs[, mask])")),

        // store(ptrs, vals)        : no mask
        // store(ptrs, vals, mask)  : with mask
        "store" if n == 2 => (
            CallKind::Void,
            quote! {
                ::triton_ir::dialect::tt::store(#(#args),*, ::std::option::Option::None)
            },
        ),
        "store" if n == 3 => {
            let p = &args[0];
            let v = &args[1];
            let m = &args[2];
            (
                CallKind::Void,
                quote! {
                    ::triton_ir::dialect::tt::store(
                        #p, #v, ::std::option::Option::Some(#m),
                    )
                },
            )
        }
        "store" => return Err(arity_err("2 or 3 (ptrs, vals[, mask])")),

        // make_range(start, end) : tensor<NxLENxi32>
        "make_range" if n == 2 => (
            CallKind::Value,
            quote! { ::triton_ir::dialect::tt::make_range(#(#args),*) },
        ),
        "make_range" => return Err(arity_err("2 (start, end)")),

        // splat_1d(scalar, len) : tensor<LENx<scalar_type>>
        "splat_1d" if n == 2 => {
            let scalar = &args[0];
            let len = &args[1];
            (
                CallKind::Value,
                quote! {
                    ::triton_ir::dialect::tt::splat(
                        #scalar,
                        ::std::vec![(#len) as i64],
                    )
                },
            )
        }
        "splat_1d" => return Err(arity_err("2 (scalar, len)")),

        // addptr(ptrs, offsets)
        "addptr" if n == 2 => (
            CallKind::Value,
            quote! { ::triton_ir::dialect::tt::addptr(#(#args),*) },
        ),
        "addptr" => return Err(arity_err("2 (ptrs, offsets)")),

        // dot(a, b, c) — block matmul a @ b + c, dispatched by IR builder.
        "dot" if n == 3 => (
            CallKind::Value,
            quote! { ::triton_ir::dialect::tt::dot(#(#args),*) },
        ),
        "dot" => return Err(arity_err("3 (a, b, c_init)")),

        // broadcast_2d(input, m, n)
        "broadcast_2d" if n == 3 => {
            let input = &args[0];
            let m_dim = &args[1];
            let n_dim = &args[2];
            (
                CallKind::Value,
                quote! {
                    ::triton_ir::dialect::tt::broadcast(
                        #input,
                        ::std::vec![(#m_dim) as i64, (#n_dim) as i64],
                    )
                },
            )
        }
        "broadcast_2d" => return Err(arity_err("3 (input, m, n)")),

        // expand_dims(input, axis)
        "expand_dims" if n == 2 => (
            CallKind::Value,
            quote! { ::triton_ir::dialect::tt::expand_dims(#(#args),*) },
        ),
        "expand_dims" => return Err(arity_err("2 (input, axis)")),

        // reshape_2d(input, m, n)
        "reshape_2d" if n == 3 => {
            let input = &args[0];
            let m_dim = &args[1];
            let n_dim = &args[2];
            (
                CallKind::Value,
                quote! {
                    ::triton_ir::dialect::tt::reshape(
                        #input,
                        ::std::vec![(#m_dim) as i64, (#n_dim) as i64],
                    )
                },
            )
        }
        "reshape_2d" => return Err(arity_err("3 (input, m, n)")),

        "return_" if n == 0 => (
            CallKind::Void,
            quote! { ::triton_ir::dialect::tt::return_() },
        ),
        "return_" => return Err(arity_err("0")),

        // ── arith dialect (explicit type suffixes) ──
        "const_i32" if n == 1 => (
            CallKind::Value,
            quote! { ::triton_ir::dialect::arith::constant_i32(#(#args),*) },
        ),
        "const_i64" if n == 1 => (
            CallKind::Value,
            quote! { ::triton_ir::dialect::arith::constant_i64(#(#args),*) },
        ),
        "const_f32" if n == 1 => (
            CallKind::Value,
            quote! { ::triton_ir::dialect::arith::constant_f32(#(#args),*) },
        ),

        "add_i32" if n == 2 => (
            CallKind::Value,
            quote! { ::triton_ir::dialect::arith::addi(#(#args),*) },
        ),
        "sub_i32" if n == 2 => (
            CallKind::Value,
            quote! { ::triton_ir::dialect::arith::subi(#(#args),*) },
        ),
        "mul_i32" if n == 2 => (
            CallKind::Value,
            quote! { ::triton_ir::dialect::arith::muli(#(#args),*) },
        ),
        "add_f32" if n == 2 => (
            CallKind::Value,
            quote! { ::triton_ir::dialect::arith::addf(#(#args),*) },
        ),
        "mul_f32" if n == 2 => (
            CallKind::Value,
            quote! { ::triton_ir::dialect::arith::mulf(#(#args),*) },
        ),
        "add_i32" | "sub_i32" | "mul_i32" | "add_f32" | "mul_f32" => return Err(arity_err("2")),

        // arith.cmpi with explicit predicates. Result type is i1 for scalar,
        // tensor<...xi1> for tensor inputs (handled by the IR builder).
        "lt_i32" if n == 2 => {
            let a = &args[0];
            let b = &args[1];
            (
                CallKind::Value,
                quote! {
                    ::triton_ir::dialect::arith::cmpi(
                        ::triton_ir::dialect::arith::CmpiPred::Slt, #a, #b,
                    )
                },
            )
        }
        "le_i32" if n == 2 => {
            let a = &args[0];
            let b = &args[1];
            (
                CallKind::Value,
                quote! {
                    ::triton_ir::dialect::arith::cmpi(
                        ::triton_ir::dialect::arith::CmpiPred::Sle, #a, #b,
                    )
                },
            )
        }
        "eq_i32" if n == 2 => {
            let a = &args[0];
            let b = &args[1];
            (
                CallKind::Value,
                quote! {
                    ::triton_ir::dialect::arith::cmpi(
                        ::triton_ir::dialect::arith::CmpiPred::Eq, #a, #b,
                    )
                },
            )
        }
        "lt_i32" | "le_i32" | "eq_i32" => return Err(arity_err("2")),

        "const_i32" | "const_i64" | "const_f32" => return Err(arity_err("1")),

        other => {
            return Err(syn::Error::new(
                call_span,
                format!(
                    "unknown function `{}` inside #[triton_kernel] body. \
                     Supported: \
                     program_id, load (1-2 args), store (2-3 args), return_, \
                     make_range, splat_1d, addptr, \
                     dot, broadcast_2d, expand_dims, reshape_2d, \
                     const_i32, const_i64, const_f32, \
                     add_i32, sub_i32, mul_i32, add_f32, mul_f32, \
                     lt_i32, le_i32, eq_i32",
                    other
                ),
            ));
        }
    };
    Ok(result)
}
