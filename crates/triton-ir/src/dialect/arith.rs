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

/// `arith.remsi` — signed integer remainder (`lhs % rhs`). Heavily used
/// in attention/RoPE/KV-cache kernels where one program_id encodes a
/// flattened (batch, head) tuple.
pub fn remsi(lhs: Value, rhs: Value) -> OpSpec {
    let ty = lhs.ty().clone();
    OpSpec::new("arith.remsi")
        .with_operand(lhs)
        .with_operand(rhs)
        .with_result(ty)
}

/// `arith.remui` — unsigned integer remainder.
pub fn remui(lhs: Value, rhs: Value) -> OpSpec {
    let ty = lhs.ty().clone();
    OpSpec::new("arith.remui")
        .with_operand(lhs)
        .with_operand(rhs)
        .with_result(ty)
}

/// `arith.divui` — unsigned integer divide. Used when group counts /
/// strides are guaranteed non-negative (e.g. quant unpack indexing).
pub fn divui(lhs: Value, rhs: Value) -> OpSpec {
    let ty = lhs.ty().clone();
    OpSpec::new("arith.divui")
        .with_operand(lhs)
        .with_operand(rhs)
        .with_result(ty)
}

// ── bitwise / shift (integer-only) ──────────────────────────────────────
//
// These unblock packed-int unpack patterns common to quantization kernels:
// GPTQ `(packed >> 4*i) & 0xF`, AWQ `(packed >> shift) & 0xFFFF`, FP8
// pack/unpack, mask synthesis, etc. Float operands are nonsensical for
// these ops; the IR builder should reject them at the dispatcher layer.

/// `arith.andi` — bitwise AND. Result type follows `lhs`.
pub fn andi(lhs: Value, rhs: Value) -> OpSpec {
    let ty = lhs.ty().clone();
    OpSpec::new("arith.andi")
        .with_operand(lhs)
        .with_operand(rhs)
        .with_result(ty)
}

/// `arith.ori` — bitwise OR.
pub fn ori(lhs: Value, rhs: Value) -> OpSpec {
    let ty = lhs.ty().clone();
    OpSpec::new("arith.ori")
        .with_operand(lhs)
        .with_operand(rhs)
        .with_result(ty)
}

/// `arith.xori` — bitwise XOR.
pub fn xori(lhs: Value, rhs: Value) -> OpSpec {
    let ty = lhs.ty().clone();
    OpSpec::new("arith.xori")
        .with_operand(lhs)
        .with_operand(rhs)
        .with_result(ty)
}

/// `arith.shli` — shift left (integer; shift amount in `rhs`).
pub fn shli(lhs: Value, rhs: Value) -> OpSpec {
    let ty = lhs.ty().clone();
    OpSpec::new("arith.shli")
        .with_operand(lhs)
        .with_operand(rhs)
        .with_result(ty)
}

/// `arith.shrsi` — arithmetic (sign-extending) shift right.
/// Matches Rust's `>>` on signed integers.
pub fn shrsi(lhs: Value, rhs: Value) -> OpSpec {
    let ty = lhs.ty().clone();
    OpSpec::new("arith.shrsi")
        .with_operand(lhs)
        .with_operand(rhs)
        .with_result(ty)
}

/// `arith.shrui` — logical (zero-extending) shift right.
/// Use for unsigned bit-field extraction such as quant nibble unpack.
pub fn shrui(lhs: Value, rhs: Value) -> OpSpec {
    let ty = lhs.ty().clone();
    OpSpec::new("arith.shrui")
        .with_operand(lhs)
        .with_operand(rhs)
        .with_result(ty)
}

/// `arith.extui` — zero-extend a smaller integer to a wider one (i32 → i64
/// without sign extension). Companion to [`extsi`] for unsigned widening,
/// which packed-int unpacking often needs after the masking step.
pub fn extui(x: Value, target: Type) -> OpSpec {
    OpSpec::new("arith.extui")
        .with_operand(x)
        .with_result(target)
}

/// `arith.maximumf` — element-wise float max (NaN-propagating). Use this
/// for softmax's row-max pass and similar reductions. The `_num` variant
/// (`maxnumf`) treats NaN as a missing value and prefers non-NaN; we
/// expose the propagating one here since most kernels assume finite inputs.
pub fn maximumf(lhs: Value, rhs: Value) -> OpSpec {
    let ty = lhs.ty().clone();
    OpSpec::new("arith.maximumf")
        .with_operand(lhs)
        .with_operand(rhs)
        .with_result(ty)
}

/// `arith.minimumf` — element-wise float min (NaN-propagating).
pub fn minimumf(lhs: Value, rhs: Value) -> OpSpec {
    let ty = lhs.ty().clone();
    OpSpec::new("arith.minimumf")
        .with_operand(lhs)
        .with_operand(rhs)
        .with_result(ty)
}

/// `arith.maxsi` — signed integer max.
pub fn maxsi(lhs: Value, rhs: Value) -> OpSpec {
    let ty = lhs.ty().clone();
    OpSpec::new("arith.maxsi")
        .with_operand(lhs)
        .with_operand(rhs)
        .with_result(ty)
}

/// `arith.minsi` — signed integer min.
pub fn minsi(lhs: Value, rhs: Value) -> OpSpec {
    let ty = lhs.ty().clone();
    OpSpec::new("arith.minsi")
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

// ── casts ───────────────────────────────────────────────────────────────
//
// Each cast op preserves shape (tensor stays a tensor, scalar stays scalar);
// only the element type changes. The caller is responsible for passing the
// correct full `target` type — the high-level `FuncBuilder::to_f32`/`to_i32`
// helpers compute it from the input shape, so DSL users rarely call these
// directly.

/// `arith.sitofp` — signed integer → float (e.g. i32 → f32).
pub fn sitofp(x: Value, target: Type) -> OpSpec {
    OpSpec::new("arith.sitofp")
        .with_operand(x)
        .with_result(target)
}

/// `arith.fptosi` — float → signed integer (truncating, e.g. f32 → i32).
pub fn fptosi(x: Value, target: Type) -> OpSpec {
    OpSpec::new("arith.fptosi")
        .with_operand(x)
        .with_result(target)
}

/// `arith.extf` — widen a float (f16/bf16 → f32, f32 → f64).
pub fn extf(x: Value, target: Type) -> OpSpec {
    OpSpec::new("arith.extf").with_operand(x).with_result(target)
}

/// `arith.truncf` — narrow a float (f32 → f16/bf16, f64 → f32).
pub fn truncf(x: Value, target: Type) -> OpSpec {
    OpSpec::new("arith.truncf").with_operand(x).with_result(target)
}

/// `arith.extsi` — sign-extend a smaller integer to a wider one (i32 → i64).
pub fn extsi(x: Value, target: Type) -> OpSpec {
    OpSpec::new("arith.extsi").with_operand(x).with_result(target)
}

/// `arith.trunci` — narrow a wider integer to a smaller one (i64 → i32).
pub fn trunci(x: Value, target: Type) -> OpSpec {
    OpSpec::new("arith.trunci").with_operand(x).with_result(target)
}
