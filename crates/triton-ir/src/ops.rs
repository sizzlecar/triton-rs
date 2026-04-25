//! Type-dispatched op helpers for the user-facing DSL.
//!
//! These wrap the bare `arith` / `tt` op constructors with one extra
//! capability: they look at the input [`Value`]'s type at runtime and pick
//! the right MLIR op variant. This is what lets the `#[triton_kernel]`
//! proc-macro translate `a + b` without knowing whether `a` is a scalar
//! `i32`, a `tensor<128xf16>`, or an `!tt.ptr<f32>`.
//!
//! The proc-macro emits calls of the shape:
//! ```ignore
//! let __tmp = ::triton_ir::ops::add(&mut __triton_f, a.clone(), b.clone());
//! ```
//!
//! Tensor inputs dispatch on their *element* type, mirroring MLIR's
//! "elementwise" semantics for arith ops.

use crate::dialect::{arith, tt};
use crate::dialect::arith::CmpiPred;
use crate::module::FuncBuilder;
use crate::ty::Type;
use crate::value::Value;

/// Pick the element type out of a Value's type. For tensors this is the
/// element; for scalars it's the type itself.
fn elem(t: &Type) -> &Type {
    match t {
        Type::Tensor { elem, .. } => elem,
        other => other,
    }
}

fn is_float(t: &Type) -> bool {
    matches!(
        elem(t),
        Type::F16 | Type::F32 | Type::F64 | Type::BF16
    )
}

fn is_ptr_like(t: &Type) -> bool {
    matches!(elem(t), Type::Ptr(_))
}

// ── arithmetic ──────────────────────────────────────────────────────────

/// `a + b`. Dispatches on `a`'s type:
///   - `!tt.ptr<T>` or `tensor<...x!tt.ptr<T>>` → `tt.addptr`
///   - float (scalar or tensor element) → `arith.addf`
///   - integer (scalar or tensor element) → `arith.addi`
pub fn add(f: &mut FuncBuilder<'_>, a: Value, b: Value) -> Value {
    if is_ptr_like(a.ty()) {
        f.op_one(tt::addptr(a, b))
    } else if is_float(a.ty()) {
        f.op_one(arith::addf(a, b))
    } else {
        f.op_one(arith::addi(a, b))
    }
}

/// `a - b`. Dispatches between `arith.subi` (integer) and... well, we don't
/// have `arith.subf` exposed yet, so floats currently error out clearly
/// rather than silently emit the wrong op.
pub fn sub(f: &mut FuncBuilder<'_>, a: Value, b: Value) -> Value {
    if is_float(a.ty()) {
        panic!(
            "ops::sub on float types not yet supported (subf TBD); got {}",
            a.ty()
        );
    }
    if is_ptr_like(a.ty()) {
        panic!("ops::sub on pointer types is not meaningful; use addptr with a negated offset");
    }
    f.op_one(arith::subi(a, b))
}

/// `a * b`. Dispatches between `arith.muli` and `arith.mulf`.
pub fn mul(f: &mut FuncBuilder<'_>, a: Value, b: Value) -> Value {
    if is_ptr_like(a.ty()) {
        panic!("ops::mul on pointer types is not meaningful; got {}", a.ty());
    }
    if is_float(a.ty()) {
        f.op_one(arith::mulf(a, b))
    } else {
        f.op_one(arith::muli(a, b))
    }
}

// ── comparison (integer only for now; cmpf TBD) ─────────────────────────

fn cmpi(f: &mut FuncBuilder<'_>, pred: CmpiPred, a: Value, b: Value) -> Value {
    if is_float(a.ty()) {
        panic!(
            "ops::cmp on float types requires arith.cmpf which is not yet wrapped \
             (got {})",
            a.ty()
        );
    }
    if is_ptr_like(a.ty()) {
        panic!("ops::cmp on pointer types is not meaningful; got {}", a.ty());
    }
    f.op_one(arith::cmpi(pred, a, b))
}

/// `a < b` (signed less-than).
pub fn lt(f: &mut FuncBuilder<'_>, a: Value, b: Value) -> Value {
    cmpi(f, CmpiPred::Slt, a, b)
}
/// `a <= b` (signed less-or-equal).
pub fn le(f: &mut FuncBuilder<'_>, a: Value, b: Value) -> Value {
    cmpi(f, CmpiPred::Sle, a, b)
}
/// `a > b` (signed greater-than).
pub fn gt(f: &mut FuncBuilder<'_>, a: Value, b: Value) -> Value {
    cmpi(f, CmpiPred::Sgt, a, b)
}
/// `a >= b` (signed greater-or-equal).
pub fn ge(f: &mut FuncBuilder<'_>, a: Value, b: Value) -> Value {
    cmpi(f, CmpiPred::Sge, a, b)
}
/// `a == b`.
pub fn eq(f: &mut FuncBuilder<'_>, a: Value, b: Value) -> Value {
    cmpi(f, CmpiPred::Eq, a, b)
}
/// `a != b`.
pub fn ne(f: &mut FuncBuilder<'_>, a: Value, b: Value) -> Value {
    cmpi(f, CmpiPred::Ne, a, b)
}
