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
use crate::dialect::arith::{CmpfPred, CmpiPred};
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

/// `a - b`. Dispatches between `arith.subi` (integer) and `arith.subf`.
pub fn sub(f: &mut FuncBuilder<'_>, a: Value, b: Value) -> Value {
    if is_ptr_like(a.ty()) {
        panic!("ops::sub on pointer types is not meaningful; use addptr with a negated offset");
    }
    if is_float(a.ty()) {
        f.op_one(arith::subf(a, b))
    } else {
        f.op_one(arith::subi(a, b))
    }
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

/// `a / b`. Dispatches between `arith.divsi` (signed integer) and
/// `arith.divf` (float).
pub fn div(f: &mut FuncBuilder<'_>, a: Value, b: Value) -> Value {
    if is_ptr_like(a.ty()) {
        panic!("ops::div on pointer types is not meaningful; got {}", a.ty());
    }
    if is_float(a.ty()) {
        f.op_one(arith::divf(a, b))
    } else {
        f.op_one(arith::divsi(a, b))
    }
}

/// `a % b` — signed integer remainder. Floats and pointers panic.
pub fn rem(f: &mut FuncBuilder<'_>, a: Value, b: Value) -> Value {
    if is_ptr_like(a.ty()) {
        panic!("ops::rem on pointer types is not meaningful; got {}", a.ty());
    }
    if is_float(a.ty()) {
        panic!("ops::rem on float types is not yet supported; got {}", a.ty());
    }
    f.op_one(arith::remsi(a, b))
}

/// `max(a, b)`. Dispatches between `arith.maximumf` (float) and
/// `arith.maxsi` (signed integer).
pub fn max(f: &mut FuncBuilder<'_>, a: Value, b: Value) -> Value {
    if is_ptr_like(a.ty()) {
        panic!("ops::max on pointer types is not meaningful; got {}", a.ty());
    }
    if is_float(a.ty()) {
        f.op_one(arith::maximumf(a, b))
    } else {
        f.op_one(arith::maxsi(a, b))
    }
}

/// `min(a, b)`. Dispatches between `arith.minimumf` and `arith.minsi`.
pub fn min(f: &mut FuncBuilder<'_>, a: Value, b: Value) -> Value {
    if is_ptr_like(a.ty()) {
        panic!("ops::min on pointer types is not meaningful; got {}", a.ty());
    }
    if is_float(a.ty()) {
        f.op_one(arith::minimumf(a, b))
    } else {
        f.op_one(arith::minsi(a, b))
    }
}

// ── comparison (int + float) ────────────────────────────────────────────

fn cmp(
    f: &mut FuncBuilder<'_>,
    int_pred: CmpiPred,
    flt_pred: CmpfPred,
    a: Value,
    b: Value,
) -> Value {
    if is_ptr_like(a.ty()) {
        panic!("ops::cmp on pointer types is not meaningful; got {}", a.ty());
    }
    if is_float(a.ty()) {
        f.op_one(arith::cmpf(flt_pred, a, b))
    } else {
        f.op_one(arith::cmpi(int_pred, a, b))
    }
}

/// `a < b`. Ordered float compare for floats; signed less-than for ints.
pub fn lt(f: &mut FuncBuilder<'_>, a: Value, b: Value) -> Value {
    cmp(f, CmpiPred::Slt, CmpfPred::Olt, a, b)
}
/// `a <= b`.
pub fn le(f: &mut FuncBuilder<'_>, a: Value, b: Value) -> Value {
    cmp(f, CmpiPred::Sle, CmpfPred::Ole, a, b)
}
/// `a > b`.
pub fn gt(f: &mut FuncBuilder<'_>, a: Value, b: Value) -> Value {
    cmp(f, CmpiPred::Sgt, CmpfPred::Ogt, a, b)
}
/// `a >= b`.
pub fn ge(f: &mut FuncBuilder<'_>, a: Value, b: Value) -> Value {
    cmp(f, CmpiPred::Sge, CmpfPred::Oge, a, b)
}
/// `a == b`.
pub fn eq(f: &mut FuncBuilder<'_>, a: Value, b: Value) -> Value {
    cmp(f, CmpiPred::Eq, CmpfPred::Oeq, a, b)
}
/// `a != b`.
pub fn ne(f: &mut FuncBuilder<'_>, a: Value, b: Value) -> Value {
    cmp(f, CmpiPred::Ne, CmpfPred::One, a, b)
}
