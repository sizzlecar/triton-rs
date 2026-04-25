//! `arith` dialect — integer/float arithmetic and constants.
//!
//! Op set is the minimum required by the Phase 2 vector_add demo plus a few
//! near-mandatory companions (cmpi, divsi). Extend as new kernels need ops.

use crate::attr::Attr;
use crate::op::OpSpec;
use crate::ty::Type;
use crate::value::Value;

/// `arith.constant` producing an `i32`.
pub fn constant_i32(value: i32) -> OpSpec {
    OpSpec::new("arith.constant")
        .with_result(Type::I32)
        .with_attr("value", Attr::i32(value))
}

/// `arith.constant` producing an `i64`.
pub fn constant_i64(value: i64) -> OpSpec {
    OpSpec::new("arith.constant")
        .with_result(Type::I64)
        .with_attr("value", Attr::i64(value))
}

/// `arith.constant` producing an `f32`.
pub fn constant_f32(value: f32) -> OpSpec {
    OpSpec::new("arith.constant")
        .with_result(Type::F32)
        .with_attr("value", Attr::f32(value))
}

/// `arith.addi` — integer add, infers result type from `lhs`.
pub fn addi(lhs: Value, rhs: Value) -> OpSpec {
    let ty = lhs.ty().clone();
    OpSpec::new("arith.addi")
        .with_operand(lhs)
        .with_operand(rhs)
        .with_result(ty)
}

/// `arith.subi` — integer subtract.
pub fn subi(lhs: Value, rhs: Value) -> OpSpec {
    let ty = lhs.ty().clone();
    OpSpec::new("arith.subi")
        .with_operand(lhs)
        .with_operand(rhs)
        .with_result(ty)
}

/// `arith.muli` — integer multiply.
pub fn muli(lhs: Value, rhs: Value) -> OpSpec {
    let ty = lhs.ty().clone();
    OpSpec::new("arith.muli")
        .with_operand(lhs)
        .with_operand(rhs)
        .with_result(ty)
}

/// `arith.divsi` — signed integer divide.
pub fn divsi(lhs: Value, rhs: Value) -> OpSpec {
    let ty = lhs.ty().clone();
    OpSpec::new("arith.divsi")
        .with_operand(lhs)
        .with_operand(rhs)
        .with_result(ty)
}

/// `arith.addf` — float add.
pub fn addf(lhs: Value, rhs: Value) -> OpSpec {
    let ty = lhs.ty().clone();
    OpSpec::new("arith.addf")
        .with_operand(lhs)
        .with_operand(rhs)
        .with_result(ty)
}

/// `arith.mulf` — float multiply.
pub fn mulf(lhs: Value, rhs: Value) -> OpSpec {
    let ty = lhs.ty().clone();
    OpSpec::new("arith.mulf")
        .with_operand(lhs)
        .with_operand(rhs)
        .with_result(ty)
}

/// Integer comparison predicates accepted by `arith.cmpi`.
#[derive(Debug, Clone, Copy)]
pub enum CmpiPred {
    /// equal
    Eq,
    /// not equal
    Ne,
    /// signed less than
    Slt,
    /// signed less or equal
    Sle,
    /// signed greater than
    Sgt,
    /// signed greater or equal
    Sge,
    /// unsigned less than
    Ult,
    /// unsigned less or equal
    Ule,
    /// unsigned greater than
    Ugt,
    /// unsigned greater or equal
    Uge,
}

impl CmpiPred {
    /// MLIR enum integer (matches `arith::CmpIPredicate` order).
    pub fn as_i64(&self) -> i64 {
        match self {
            CmpiPred::Eq => 0,
            CmpiPred::Ne => 1,
            CmpiPred::Slt => 2,
            CmpiPred::Sle => 3,
            CmpiPred::Sgt => 4,
            CmpiPred::Sge => 5,
            CmpiPred::Ult => 6,
            CmpiPred::Ule => 7,
            CmpiPred::Ugt => 8,
            CmpiPred::Uge => 9,
        }
    }
}

/// `arith.cmpi predicate, %lhs, %rhs : T` — produces `i1` or `tensor<...xi1>`.
pub fn cmpi(pred: CmpiPred, lhs: Value, rhs: Value) -> OpSpec {
    let result_ty = match lhs.ty() {
        Type::Tensor { shape, .. } => Type::tensor(shape.clone(), Type::I1),
        _ => Type::I1,
    };
    OpSpec::new("arith.cmpi")
        .with_operand(lhs)
        .with_operand(rhs)
        .with_result(result_ty)
        .with_attr("predicate", Attr::Int(pred.as_i64(), Type::I64))
}

// ── float arith ─────────────────────────────────────────────────────────

/// `arith.subf` — float subtract.
pub fn subf(lhs: Value, rhs: Value) -> OpSpec {
    let ty = lhs.ty().clone();
    OpSpec::new("arith.subf")
        .with_operand(lhs)
        .with_operand(rhs)
        .with_result(ty)
}

/// `arith.divf` — float divide.
pub fn divf(lhs: Value, rhs: Value) -> OpSpec {
    let ty = lhs.ty().clone();
    OpSpec::new("arith.divf")
        .with_operand(lhs)
        .with_operand(rhs)
        .with_result(ty)
}

/// Float comparison predicates accepted by `arith.cmpf`. Codes match
/// MLIR's `arith::CmpFPredicate` enum. Prefer the ordered variants
/// (`Oeq`, `Olt`, ...) for kernel comparisons unless you specifically
/// need NaN-tolerant semantics.
#[derive(Debug, Clone, Copy)]
pub enum CmpfPred {
    /// always false
    AlwaysFalse,
    /// ordered equal
    Oeq,
    /// ordered greater than
    Ogt,
    /// ordered greater or equal
    Oge,
    /// ordered less than
    Olt,
    /// ordered less or equal
    Ole,
    /// ordered not equal
    One,
    /// ordered (no NaN)
    Ord,
    /// unordered equal
    Ueq,
    /// unordered greater than
    Ugt,
    /// unordered greater or equal
    Uge,
    /// unordered less than
    Ult,
    /// unordered less or equal
    Ule,
    /// unordered not equal
    Une,
    /// unordered (any NaN)
    Uno,
    /// always true
    AlwaysTrue,
}

impl CmpfPred {
    /// MLIR enum integer (matches `arith::CmpFPredicate` order).
    pub fn as_i64(&self) -> i64 {
        match self {
            CmpfPred::AlwaysFalse => 0,
            CmpfPred::Oeq => 1,
            CmpfPred::Ogt => 2,
            CmpfPred::Oge => 3,
            CmpfPred::Olt => 4,
            CmpfPred::Ole => 5,
            CmpfPred::One => 6,
            CmpfPred::Ord => 7,
            CmpfPred::Ueq => 8,
            CmpfPred::Ugt => 9,
            CmpfPred::Uge => 10,
            CmpfPred::Ult => 11,
            CmpfPred::Ule => 12,
            CmpfPred::Une => 13,
            CmpfPred::Uno => 14,
            CmpfPred::AlwaysTrue => 15,
        }
    }
}

/// `arith.cmpf predicate, %lhs, %rhs : T` — element-wise float comparison.
/// Result type is `i1` for scalar inputs, `tensor<...xi1>` for tensor inputs.
pub fn cmpf(pred: CmpfPred, lhs: Value, rhs: Value) -> OpSpec {
    let result_ty = match lhs.ty() {
        Type::Tensor { shape, .. } => Type::tensor(shape.clone(), Type::I1),
        _ => Type::I1,
    };
    OpSpec::new("arith.cmpf")
        .with_operand(lhs)
        .with_operand(rhs)
        .with_result(result_ty)
        .with_attr("predicate", Attr::Int(pred.as_i64(), Type::I64))
}
