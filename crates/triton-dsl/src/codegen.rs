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
    spanned::Spanned, BinOp, Block, Expr, ExprBinary, ExprCall, ExprClosure, FnArg, ItemFn, Local,
    Pat, PathArguments, Stmt, Type,
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

    // Preserve the function's generics so callers can write
    // `kernel_name::<1024>::mlir()`. Const generic params (e.g.
    // `<const BLOCK: usize>`) are visible inside the body since the
    // proc-macro emits the body unchanged where it references them.
    let (impl_gen, ty_gen, where_clause) = input.sig.generics.split_for_impl();
    let has_generics = !input.sig.generics.params.is_empty();

    // Collect type-param idents for the PhantomData tuple. Const generics
    // need no phantom (the value is already on the type). Lifetimes do.
    // For type/lifetime params, build a tuple `(PhantomData<T1>, ...)`
    // — the `fn() <ty_gen>` trick fails when const generics are mixed in
    // (cannot apply <...> to a fn() type).
    let mut phantom_fields: Vec<TokenStream2> = Vec::new();
    for p in input.sig.generics.params.iter() {
        match p {
            syn::GenericParam::Type(t) => {
                let ident = &t.ident;
                phantom_fields.push(quote! { ::std::marker::PhantomData<#ident> });
            }
            syn::GenericParam::Lifetime(lt) => {
                let lifetime = &lt.lifetime;
                phantom_fields.push(quote! { ::std::marker::PhantomData<& #lifetime ()> });
            }
            syn::GenericParam::Const(_) => {}
        }
    }

    // Struct definitions need the *declaration*-style generics
    // (`<const BLOCK: usize>`), which ImplGenerics emits — TypeGenerics
    // would emit `<BLOCK>` alone, dropping the kind/bound and triggering
    // E0392 / E0747 at the use site.
    let struct_def = if has_generics {
        if !phantom_fields.is_empty() {
            quote! {
                #[allow(non_camel_case_types)]
                #vis struct #fn_name #impl_gen #where_clause (
                    #(#phantom_fields),*,
                );
            }
        } else {
            quote! {
                #[allow(non_camel_case_types)]
                #vis struct #fn_name #impl_gen #where_clause;
            }
        }
    } else {
        quote! {
            #[allow(non_camel_case_types)]
            #vis struct #fn_name;
        }
    };

    Ok(quote! {
        #struct_def

        impl #impl_gen #fn_name #ty_gen #where_clause {
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
        // Generic type parameter (e.g. `T` in `vec_add<T: TritonElem, ...>`).
        // Single-segment uppercase-starting identifiers are treated as type
        // generics — emit a TritonElem trait dispatch to resolve at
        // monomorphization time. `vec_add::<f32, ...>` then picks
        // `<f32 as TritonElem>::ir_type() = Type::F32`.
        other if path.segments.len() == 1
            && other.starts_with(|c: char| c.is_ascii_uppercase()) =>
        {
            let ident = &seg.ident;
            Ok(quote! { <#ident as ::triton_ir::ty::TritonElem>::ir_type() })
        }
        other => Err(syn::Error::new(
            seg.span(),
            format!(
                "unsupported type `{}` in #[triton_kernel] signature \
                 (Phase 3.1 supports i1/i32/i64/f16/f32/bf16, Ptr<T>, and uppercase generic type params bound by TritonElem)",
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
        // `BLOCK as i32` (cast) is a runtime Rust expression evaluated at
        // module()-build time. The whole thing passes through unchanged —
        // it's ordinary Rust, not a kernel-IR construct.
        Expr::Cast(_) => Ok(TranslatedExpr {
            setup: vec![],
            expr: quote! { (#expr) },
        }),
        // Method calls on plain values (e.g. `args[0].clone()` if the user
        // ever writes one) are also pass-through. Don't attempt to
        // translate them as kernel ops.
        Expr::MethodCall(_) | Expr::Index(_) | Expr::Field(_) => Ok(TranslatedExpr {
            setup: vec![],
            expr: quote! { (#expr) },
        }),
        other => Err(syn::Error::new(
            other.span(),
            "this expression form is not yet supported inside #[triton_kernel] bodies \
             (supported: function calls, binary ops `+ - * / < <= > >= == !=`, \
             casts `x as T`, variable refs, integer literals)",
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

/// What gets passed to FuncBuilder::lit_i64 / lit_f64 at runtime: a
/// TokenStream representing the lift value as `i64` or `f64`. Three
/// cases produce one:
///
///   - integer literal:        `pid * 1024`           → `1024i64 as i64`
///   - float literal:          `xv * 0.5`             → `0.5_f64`
///   - cast to int / float:    `pid * (BLOCK as i32)` → `(BLOCK as i32) as i64`
///
/// The cast case lets users use const-generic params (or any other
/// integer-typed Rust expression) directly in binary ops without
/// `const_i32(...)` wrappers — the cast's runtime value gets lifted
/// against the other operand's element type at IR-build time.
fn lit_int_lift(expr: &Expr) -> Option<proc_macro2::TokenStream> {
    match expr {
        // Unwrap parentheses so `(BLOCK as i32)` etc. resolve.
        Expr::Paren(p) => lit_int_lift(&p.expr),
        Expr::Lit(syn::ExprLit { lit: syn::Lit::Int(li), .. }) => {
            let v: i64 = li.base10_parse().ok()?;
            Some(quote! { (#v) as i64 })
        }
        Expr::Unary(syn::ExprUnary { op: syn::UnOp::Neg(_), expr, .. }) => {
            // Recurse — wraps `-(...)` so negative literals work.
            let inner = lit_int_lift(expr)?;
            Some(quote! { -((#inner) as i64) })
        }
        Expr::Cast(syn::ExprCast { expr, ty, .. }) => {
            // Pass through any cast whose target is a primitive integer.
            let target_ident = primitive_type_name(ty)?;
            match target_ident.as_str() {
                "i8" | "i16" | "i32" | "i64" | "isize" | "u8" | "u16" | "u32" | "u64"
                | "usize" => {
                    Some(quote! { ((#expr) as #ty) as i64 })
                }
                _ => None,
            }
        }
        _ => None,
    }
}

fn lit_float_lift(expr: &Expr) -> Option<proc_macro2::TokenStream> {
    match expr {
        Expr::Paren(p) => lit_float_lift(&p.expr),
        Expr::Lit(syn::ExprLit { lit: syn::Lit::Float(lf), .. }) => {
            let v: f64 = lf.base10_parse().ok()?;
            Some(quote! { (#v) as f64 })
        }
        Expr::Unary(syn::ExprUnary { op: syn::UnOp::Neg(_), expr, .. }) => {
            let inner = lit_float_lift(expr)?;
            Some(quote! { -((#inner) as f64) })
        }
        Expr::Cast(syn::ExprCast { expr, ty, .. }) => {
            let target_ident = primitive_type_name(ty)?;
            match target_ident.as_str() {
                "f32" | "f64" => Some(quote! { ((#expr) as #ty) as f64 }),
                _ => None,
            }
        }
        _ => None,
    }
}

/// Pull a primitive type name out of `syn::Type::Path`. Returns the last
/// segment's identifier as a String (e.g. `i32`, `f64`).
fn primitive_type_name(ty: &syn::Type) -> Option<String> {
    if let syn::Type::Path(tp) = ty {
        if tp.qself.is_none() {
            return tp.path.segments.last().map(|s| s.ident.to_string());
        }
    }
    None
}

/// Translate `a OP b` into a call into `triton_ir::ops::*`, which dispatches
/// on the runtime type of `a` to pick the right MLIR op (e.g. `+` becomes
/// `tt.addptr` for pointer-typed lhs, `arith.addf` for float, otherwise
/// `arith.addi`). Operands are recursively translated and force-hoisted so
/// nested binary expressions don't trigger borrow-checker conflicts.
///
/// Auto-promotion: if exactly one operand is a Rust integer/float literal,
/// emit a `__triton_f.lit_i64(&other, lit)` call to lift it to a Value
/// matching the other operand's type. Lets users write `pid * 1024` and
/// `xv + 0.5` without sprinkling `const_i32(...)` everywhere.
fn translate_binop(
    node: &ExprBinary,
    hoist: bool,
    counter: &mut u32,
) -> Result<TranslatedExpr, syn::Error> {
    // Auto-promotion path: at most one side is a numeric literal /
    // cast-to-primitive that we can lift to a Value via lit_i64 /
    // lit_f64 at runtime.
    enum LitSide<T> {
        Left(T),
        Right(T),
    }
    enum LitKind {
        Int(LitSide<proc_macro2::TokenStream>),
        Float(LitSide<proc_macro2::TokenStream>),
        None,
    }
    // Probe right side first so e.g. `pid * 1024` and `xv + 0.5` (lit on
    // right, the common case) pick the cheaper lift.
    let lit = match (lit_int_lift(&node.right), lit_float_lift(&node.right)) {
        (Some(t), _) => LitKind::Int(LitSide::Right(t)),
        (_, Some(t)) => LitKind::Float(LitSide::Right(t)),
        _ => match (lit_int_lift(&node.left), lit_float_lift(&node.left)) {
            (Some(t), _) => LitKind::Int(LitSide::Left(t)),
            (_, Some(t)) => LitKind::Float(LitSide::Left(t)),
            _ => LitKind::None,
        },
    };

    if !matches!(lit, LitKind::None) {
        let (other_expr, on_left, lit_method, lit_value): (
            &Expr,
            bool,
            proc_macro2::TokenStream,
            proc_macro2::TokenStream,
        ) = match lit {
            LitKind::Int(LitSide::Right(v)) => (&node.left, false, quote!(lit_i64), v),
            LitKind::Int(LitSide::Left(v)) => (&node.right, true, quote!(lit_i64), v),
            LitKind::Float(LitSide::Right(v)) => (&node.left, false, quote!(lit_f64), v),
            LitKind::Float(LitSide::Left(v)) => (&node.right, true, quote!(lit_f64), v),
            LitKind::None => unreachable!(),
        };

        let other_t = translate_expr(other_expr, /*hoist=*/ true, counter)?;
        let setup = other_t.setup;
        let other = other_t.expr;
        let method = binop_method(&node.op, &node.span())?;

        let body = if on_left {
            quote! {
                {
                    let __triton_other = #other;
                    let __triton_lit = __triton_f.#lit_method(&__triton_other, #lit_value);
                    __triton_f.#method(__triton_lit, __triton_other)
                }
            }
        } else {
            quote! {
                {
                    let __triton_other = #other;
                    let __triton_lit = __triton_f.#lit_method(&__triton_other, #lit_value);
                    __triton_f.#method(__triton_other, __triton_lit)
                }
            }
        };

        return Ok(maybe_hoist(setup, body, CallKind::Value, hoist, counter));
    }

    // Standard path: neither side is a literal / liftable cast.
    let lt = translate_expr(&node.left, /*hoist=*/ true, counter)?;
    let rt = translate_expr(&node.right, /*hoist=*/ true, counter)?;

    let mut setup = lt.setup;
    setup.extend(rt.setup);
    let l = lt.expr;
    let r = rt.expr;

    let method = binop_method(&node.op, &node.op.span())?;
    let call_expr = quote! { __triton_f.#method(#l, #r) };
    Ok(maybe_hoist(setup, call_expr, CallKind::Value, hoist, counter))
}

/// Map a syn BinOp to the corresponding FuncBuilder method name. Returns
/// an error for unsupported operators with a span pointing at the offender.
fn binop_method(op: &BinOp, span: &Span) -> Result<proc_macro2::Ident, syn::Error> {
    let name = match op {
        BinOp::Add(_) => "add",
        BinOp::Sub(_) => "sub",
        BinOp::Mul(_) => "mul",
        BinOp::Div(_) => "div",
        BinOp::Rem(_) => "rem",
        BinOp::Lt(_) => "lt",
        BinOp::Le(_) => "le",
        BinOp::Gt(_) => "gt",
        BinOp::Ge(_) => "ge",
        BinOp::Eq(_) => "eq",
        BinOp::Ne(_) => "ne",
        other => {
            return Err(syn::Error::new(
                *span,
                format!(
                    "binary operator `{:?}` is not supported in #[triton_kernel] body \
                     (supported: `+ - * / % < <= > >= == !=`)",
                    other
                ),
            ));
        }
    };
    Ok(format_ident!("{}", name))
}

fn translate_call(
    call: &ExprCall,
    hoist: bool,
    counter: &mut u32,
) -> Result<TranslatedExpr, syn::Error> {
    // Special forms with arity > 1 and a closure tail get handled before the
    // generic translation path, since their final arg can't be evaluated
    // eagerly into a temp like a plain Value.
    let early_name = call_name(call).ok();
    if let Some(n) = early_name.as_deref() {
        if n == "scf_for" {
            return translate_scf_for(call, hoist, counter);
        }
        if n == "reduce" {
            return translate_reduce(call, hoist, counter);
        }
        // `max` / `min` go through inherent FuncBuilder methods (same path
        // as the binary operators) so the final emit is a direct method
        // call returning Value — NOT wrapped in `op_one(...)`. The generic
        // op_spec_for path would double-wrap and produce
        // `op_one(__triton_f.max(...))`, which is a type error.
        if matches!(n, "max" | "min") {
            return translate_method_call_2arg(call, n, hoist, counter);
        }
        // Casts: same reason as max/min — these route through inherent
        // FuncBuilder methods that already return Value, so they must
        // bypass the generic `op_one(spec)` wrapper.
        if matches!(n, "to_f32" | "to_f16" | "to_i32") {
            return translate_method_call_1arg(call, n, hoist, counter);
        }
    }

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

/// Translate `to_f32(x)` / `to_f16(x)` / `to_i32(x)` into a direct
/// FuncBuilder method call. Lives in the special-form list so it doesn't
/// get double-wrapped by op_spec_for's `op_one(...)` envelope.
fn translate_method_call_1arg(
    call: &ExprCall,
    method: &str,
    hoist: bool,
    counter: &mut u32,
) -> Result<TranslatedExpr, syn::Error> {
    if call.args.len() != 1 {
        return Err(syn::Error::new(
            call.span(),
            format!("`{}` expects 1 argument, got {}", method, call.args.len()),
        ));
    }
    let a_t = translate_expr(&call.args[0], /*hoist=*/ true, counter)?;
    let setup = a_t.setup;
    let a = a_t.expr;
    let method_ident = format_ident!("{}", method);
    let call_expr = quote! { __triton_f.#method_ident(#a) };
    Ok(maybe_hoist(setup, call_expr, CallKind::Value, hoist, counter))
}

/// Translate `max(a, b)` / `min(a, b)` into a direct FuncBuilder method
/// call. Lives in the special-form list so it doesn't get double-wrapped
/// by op_spec_for's `op_one(...)` envelope.
fn translate_method_call_2arg(
    call: &ExprCall,
    method: &str,
    hoist: bool,
    counter: &mut u32,
) -> Result<TranslatedExpr, syn::Error> {
    if call.args.len() != 2 {
        return Err(syn::Error::new(
            call.span(),
            format!("`{}` expects 2 arguments, got {}", method, call.args.len()),
        ));
    }
    let a_t = translate_expr(&call.args[0], /*hoist=*/ true, counter)?;
    let b_t = translate_expr(&call.args[1], /*hoist=*/ true, counter)?;
    let mut setup = a_t.setup;
    setup.extend(b_t.setup);
    let a = a_t.expr;
    let b = b_t.expr;
    let method_ident = format_ident!("{}", method);
    let call_expr = quote! { __triton_f.#method_ident(#a, #b) };
    Ok(maybe_hoist(setup, call_expr, CallKind::Value, hoist, counter))
}

/// Translate `scf_for(lb, ub, step, init, |i, acc| body)` into a call to
/// `__triton_f.for_loop_with(...)`. The closure body is translated with the
/// same machinery as the outer body — ops emitted via `__triton_f.op_one`
/// land in the loop region thanks to FuncBuilder's region_stack routing.
fn translate_scf_for(
    call: &ExprCall,
    hoist: bool,
    counter: &mut u32,
) -> Result<TranslatedExpr, syn::Error> {
    if call.args.len() != 5 {
        return Err(syn::Error::new(
            call.span(),
            "`scf_for` expects 5 arguments: lb, ub, step, init, |i, acc| body",
        ));
    }

    // Translate the three leading scalar/Value args normally — force-hoist
    // each into a temp so they evaluate before the closure runs. The init
    // arg is handled below alongside multi-iter_arg tuple unpacking.
    let lb_t = translate_expr(&call.args[0], /*hoist=*/ true, counter)?;
    let ub_t = translate_expr(&call.args[1], /*hoist=*/ true, counter)?;
    let step_t = translate_expr(&call.args[2], /*hoist=*/ true, counter)?;
    let mut setup = Vec::new();
    setup.extend(lb_t.setup);
    setup.extend(ub_t.setup);
    setup.extend(step_t.setup);
    let lb = lb_t.expr;
    let ub = ub_t.expr;
    let step = step_t.expr;

    // Final arg must be a closure with exactly two parameters.
    let closure: &ExprClosure = match &call.args[4] {
        Expr::Closure(c) => c,
        other => {
            return Err(syn::Error::new(
                other.span(),
                "scf_for: 5th argument must be a closure `|i, acc| { ... }`",
            ));
        }
    };
    if closure.inputs.len() != 2 {
        return Err(syn::Error::new(
            closure.span(),
            "scf_for closure must take exactly 2 parameters: |induction, iter_arg(s)|",
        ));
    }
    let i_ident = closure_param_ident(&closure.inputs[0])?;

    // Iter-arg side: either a single `Pat::Ident` (one iter_arg, the
    // common case) OR a `Pat::Tuple` of identifiers (multi iter_args
    // for kernels that thread several state values through the loop —
    // e.g. flash-attention's (m_i, l_i, acc) running stats).
    let acc_idents = pat_to_ident_list(&closure.inputs[1])?;
    let n_iter = acc_idents.len();
    if n_iter == 0 {
        return Err(syn::Error::new(
            closure.inputs[1].span(),
            "scf_for: iter_arg slot needs at least one identifier",
        ));
    }

    // The user-supplied init value can match: a single value when n_iter == 1,
    // or an `Expr::Tuple` of n_iter values when there are multiple iter_args.
    let init_exprs = expr_to_value_list(&call.args[3], n_iter)?;
    let mut init_value_tokens = Vec::new();
    for e in &init_exprs {
        let t = translate_expr(e, /*hoist=*/ true, counter)?;
        setup.extend(t.setup);
        init_value_tokens.push(t.expr);
    }

    // The closure body can be a block `{ stmts; trailing_expr }` or a bare
    // expression (when the user writes `|i, acc| acc + i`). Treat the bare-
    // expression form as a body with no preceding statements, just a yield.
    let (init_stmts, yield_t_collection) = match &*closure.body {
        Expr::Block(eb) => {
            let (init_st, yield_expr) = split_block_for_yield(&eb.block, counter)?;
            // Yield must match n_iter values too — single value or n-tuple.
            let yield_exprs = expr_to_value_list(yield_expr, n_iter)?;
            let mut yield_translated = Vec::with_capacity(n_iter);
            for e in &yield_exprs {
                yield_translated.push(translate_expr(e, /*hoist=*/ false, counter)?);
            }
            (init_st, yield_translated)
        }
        other => {
            let yield_exprs = expr_to_value_list(other, n_iter)?;
            let mut yield_translated = Vec::with_capacity(n_iter);
            for e in &yield_exprs {
                yield_translated.push(translate_expr(e, /*hoist=*/ false, counter)?);
            }
            (Vec::new(), yield_translated)
        }
    };

    // Bind each iter_arg to its body-visible name.
    let acc_bindings = acc_idents.iter().enumerate().map(|(i, name)| {
        quote! { let #name = __triton_iter_args[#i].clone(); }
    });

    // Per-yield setup + value extraction.
    let mut yield_setups = Vec::new();
    let mut yield_values = Vec::new();
    for (i, ye) in yield_t_collection.iter().enumerate() {
        yield_setups.extend(ye.setup.clone());
        let v = &ye.expr;
        let tmp = format_ident!("__triton_yield_{}", i);
        yield_setups.push(quote! { let #tmp = #v; });
        yield_values.push(quote! { #tmp });
    }

    let closure_body = quote! {
        let #i_ident = __triton_i;
        #(#acc_bindings)*
        #(#init_stmts)*
        #(#yield_setups)*
        ::std::vec![#(#yield_values),*]
    };

    let init_vec = quote! { ::std::vec![#(#init_value_tokens),*] };
    let call_expr = if n_iter == 1 {
        quote! {
            {
                let __triton_results = __triton_f.for_loop_with(
                    #lb, #ub, #step,
                    #init_vec,
                    |__triton_f: &mut ::triton_ir::module::FuncBuilder<'_>,
                     __triton_i: ::triton_ir::value::Value,
                     __triton_iter_args: ::std::vec::Vec<::triton_ir::value::Value>|
                     -> ::std::vec::Vec<::triton_ir::value::Value> {
                        #closure_body
                    },
                );
                __triton_results.into_iter().next()
                    .expect("scf_for must yield exactly one iter_arg result")
            }
        }
    } else {
        // Returns a tuple of n_iter Values.
        let result_idents: Vec<_> = (0..n_iter)
            .map(|i| format_ident!("__triton_res_{}", i))
            .collect();
        let extracts = result_idents.iter().enumerate().map(|(i, id)| {
            quote! { let #id = __triton_results[#i].clone(); }
        });
        let result_tuple = quote! { (#(#result_idents),*) };
        quote! {
            {
                let __triton_results = __triton_f.for_loop_with(
                    #lb, #ub, #step,
                    #init_vec,
                    |__triton_f: &mut ::triton_ir::module::FuncBuilder<'_>,
                     __triton_i: ::triton_ir::value::Value,
                     __triton_iter_args: ::std::vec::Vec<::triton_ir::value::Value>|
                     -> ::std::vec::Vec<::triton_ir::value::Value> {
                        #closure_body
                    },
                );
                #(#extracts)*
                #result_tuple
            }
        }
    };

    Ok(maybe_hoist(setup, call_expr, CallKind::Value, hoist, counter))
}

/// Extract a flat list of identifiers from a closure parameter pattern.
/// Single `Pat::Ident` → one entry. `Pat::Tuple` of idents → that many.
fn pat_to_ident_list(pat: &Pat) -> Result<Vec<proc_macro2::Ident>, syn::Error> {
    match pat {
        Pat::Ident(p) => Ok(vec![p.ident.clone()]),
        Pat::Tuple(t) => {
            let mut out = Vec::with_capacity(t.elems.len());
            for elem in &t.elems {
                match elem {
                    Pat::Ident(pi) => out.push(pi.ident.clone()),
                    Pat::Type(pt) => match &*pt.pat {
                        Pat::Ident(pi) => out.push(pi.ident.clone()),
                        other => return Err(syn::Error::new(
                            other.span(),
                            "tuple iter_arg elements must be plain identifiers",
                        )),
                    },
                    other => return Err(syn::Error::new(
                        other.span(),
                        "tuple iter_arg elements must be plain identifiers",
                    )),
                }
            }
            Ok(out)
        }
        Pat::Type(pt) => pat_to_ident_list(&pt.pat),
        other => Err(syn::Error::new(
            other.span(),
            "iter_arg pattern must be a plain identifier or a tuple of identifiers",
        )),
    }
}

/// Extract a list of expressions matching a target arity. Single value
/// when `expected == 1`, `Expr::Tuple` when `expected > 1`.
fn expr_to_value_list(expr: &Expr, expected: usize) -> Result<Vec<Expr>, syn::Error> {
    match expr {
        Expr::Tuple(t) if expected > 1 => {
            if t.elems.len() != expected {
                return Err(syn::Error::new(
                    t.span(),
                    format!("expected tuple of {} values, got {}", expected, t.elems.len()),
                ));
            }
            Ok(t.elems.iter().cloned().collect())
        }
        Expr::Paren(p) if expected == 1 => Ok(vec![(*p.expr).clone()]),
        other if expected == 1 => Ok(vec![other.clone()]),
        other => Err(syn::Error::new(
            other.span(),
            format!(
                "expected a tuple of {} values for multi-iter_arg scf_for",
                expected
            ),
        )),
    }
}

/// Translate `reduce(input, axis, |a, b| body)` into a call to
/// `__triton_f.reduce_with(...)`. Closure body has the same shape as
/// scf_for's: optional setup statements followed by a trailing yield expr.
fn translate_reduce(
    call: &ExprCall,
    hoist: bool,
    counter: &mut u32,
) -> Result<TranslatedExpr, syn::Error> {
    if call.args.len() != 3 {
        return Err(syn::Error::new(
            call.span(),
            "`reduce` expects 3 arguments: input, axis, |a, b| body",
        ));
    }

    let input_t = translate_expr(&call.args[0], /*hoist=*/ true, counter)?;
    let axis_t = translate_expr(&call.args[1], /*hoist=*/ false, counter)?;
    let mut setup = Vec::new();
    setup.extend(input_t.setup);
    setup.extend(axis_t.setup);
    let input_e = input_t.expr;
    let axis_e = axis_t.expr;

    let closure: &ExprClosure = match &call.args[2] {
        Expr::Closure(c) => c,
        other => {
            return Err(syn::Error::new(
                other.span(),
                "reduce: 3rd argument must be a closure `|a, b| { ... }`",
            ));
        }
    };
    if closure.inputs.len() != 2 {
        return Err(syn::Error::new(
            closure.span(),
            "reduce closure must take exactly 2 parameters: |lhs, rhs|",
        ));
    }
    let lhs_ident = closure_param_ident(&closure.inputs[0])?;
    let rhs_ident = closure_param_ident(&closure.inputs[1])?;

    let (init_stmts, yield_t) = match &*closure.body {
        Expr::Block(eb) => {
            let (init, yield_expr) = split_block_for_yield(&eb.block, counter)?;
            let y = translate_expr(yield_expr, /*hoist=*/ false, counter)?;
            (init, y)
        }
        other => (Vec::new(), translate_expr(other, /*hoist=*/ false, counter)?),
    };
    let yield_setup = yield_t.setup;
    let yield_expr_ts = yield_t.expr;

    let closure_body = quote! {
        let #lhs_ident = __triton_lhs;
        let #rhs_ident = __triton_rhs;
        #(#init_stmts)*
        #(#yield_setup)*
        #yield_expr_ts
    };

    let call_expr = quote! {
        __triton_f.reduce_with(
            #input_e,
            #axis_e,
            |__triton_f: &mut ::triton_ir::module::FuncBuilder<'_>,
             __triton_lhs: ::triton_ir::value::Value,
             __triton_rhs: ::triton_ir::value::Value|
             -> ::triton_ir::value::Value {
                #closure_body
            },
        )
    };

    Ok(maybe_hoist(setup, call_expr, CallKind::Value, hoist, counter))
}

/// Pull a plain identifier out of a closure parameter pattern. The DSL only
/// supports `|name|` style — destructuring patterns get a clearer error
/// than the generic translator would emit.
fn closure_param_ident(pat: &Pat) -> Result<proc_macro2::Ident, syn::Error> {
    match pat {
        Pat::Ident(p) => Ok(p.ident.clone()),
        Pat::Type(pt) => closure_param_ident(&pt.pat),
        other => Err(syn::Error::new(
            other.span(),
            "scf_for closure parameters must be plain identifiers (no destructuring)",
        )),
    }
}

/// Split a closure body into (statements before yield, yield expression).
/// Insists on a trailing expression with no semicolon — that's the value
/// flowing through `scf.yield`.
fn split_block_for_yield<'b>(
    block: &'b Block,
    counter: &mut u32,
) -> Result<(Vec<TokenStream2>, &'b Expr), syn::Error> {
    if block.stmts.is_empty() {
        return Err(syn::Error::new(
            block.span(),
            "scf_for body cannot be empty — it must yield a value as its trailing expression",
        ));
    }
    let last_idx = block.stmts.len() - 1;
    let yield_expr = match &block.stmts[last_idx] {
        Stmt::Expr(e, None) => e,
        other => {
            return Err(syn::Error::new(
                other.span(),
                "scf_for body must end with a trailing expression (no semicolon) \
                 that yields the next iter_arg",
            ));
        }
    };

    let mut init = Vec::with_capacity(last_idx);
    for stmt in &block.stmts[..last_idx] {
        init.push(translate_stmt(stmt, counter)?);
    }
    Ok((init, yield_expr))
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

        // ── element-wise math intrinsics ──
        // Each takes 1 operand, returns same shape & element type.
        "exp" if n == 1 => (
            CallKind::Value,
            quote! { ::triton_ir::dialect::tt::exp(#(#args),*) },
        ),
        "exp2" if n == 1 => (
            CallKind::Value,
            quote! { ::triton_ir::dialect::tt::exp2(#(#args),*) },
        ),
        "log" if n == 1 => (
            CallKind::Value,
            quote! { ::triton_ir::dialect::tt::log(#(#args),*) },
        ),
        "log2" if n == 1 => (
            CallKind::Value,
            quote! { ::triton_ir::dialect::tt::log2(#(#args),*) },
        ),
        "sqrt" if n == 1 => (
            CallKind::Value,
            quote! { ::triton_ir::dialect::tt::sqrt(#(#args),*) },
        ),
        "sin" if n == 1 => (
            CallKind::Value,
            quote! { ::triton_ir::dialect::tt::sin(#(#args),*) },
        ),
        "cos" if n == 1 => (
            CallKind::Value,
            quote! { ::triton_ir::dialect::tt::cos(#(#args),*) },
        ),
        "abs" if n == 1 => (
            CallKind::Value,
            quote! { ::triton_ir::dialect::tt::abs(#(#args),*) },
        ),
        "rsqrt" if n == 1 => (
            CallKind::Value,
            quote! { ::triton_ir::dialect::tt::rsqrt(#(#args),*) },
        ),

        // (Casts `to_f32` / `to_f16` / `to_i32` are handled in the
        // special-form branch above, so they never reach this match.)
        "tanh" if n == 1 => (
            CallKind::Value,
            quote! { ::triton_ir::dialect::tt::tanh(#(#args),*) },
        ),
        "erf" if n == 1 => (
            CallKind::Value,
            quote! { ::triton_ir::dialect::tt::erf(#(#args),*) },
        ),

        // max / min are handled before this match by `translate_method_call_2arg`.
        // If we somehow get here with them, that's a logic bug — surface it
        // clearly rather than silently double-wrapping.
        "max" | "min" => unreachable!("max/min should be handled by translate_method_call_2arg"),
        "exp" | "exp2" | "log" | "log2" | "sqrt" | "rsqrt" | "sin" | "cos" | "abs"
        | "tanh" | "erf" => {
            return Err(arity_err("1"));
        }

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
