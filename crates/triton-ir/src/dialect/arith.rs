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
