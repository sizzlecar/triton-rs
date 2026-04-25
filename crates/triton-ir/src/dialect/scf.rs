//! `scf` dialect — structured control flow (`scf.for`, `scf.if`, `scf.yield`).
//!
//! Region-having ops are built in two steps:
//!   1. Call [`crate::module::FuncBuilder::new_region`] to obtain a fresh
//!      empty region and the SSA values bound to its entry-block args
//!      (loop induction var + iter_args for `scf.for`, branch params for
//!      `scf.if`).
//!   2. Append ops to the region with `append_to_region*`, ending with a
//!      terminator op produced by [`yield_`].
//!   3. Hand the region to [`for_loop`] / [`if_then_else`], which packages
//!      it into an `OpSpec` ready for `FuncBuilder::op` / `op_one`.

use crate::op::{OpSpec, Region};
use crate::ty::Type;
use crate::value::Value;

/// `scf.yield` — terminator that returns values from a region back to the
/// enclosing region-having op (`scf.for`, `scf.if`, ...).
pub fn yield_(values: Vec<Value>) -> OpSpec {
    let mut spec = OpSpec::new("scf.yield");
    for v in values {
        spec = spec.with_operand(v);
    }
    spec
}

/// `scf.for %i = %lb to %ub step %step iter_args(%a = %init) -> T { body }`.
///
/// Triton-style: `lb`, `ub`, `step` use the same integer type (typically
/// `i32`, sometimes `index`). `iter_args` is the list of values threaded
/// through the loop; the body region must take `(induction, iter_args...)`
/// as block args and end with `scf.yield <new_iter_args>`.
///
/// The op produces one result per `iter_args` entry, with matching types.
pub fn for_loop(
    lb: Value,
    ub: Value,
    step: Value,
    iter_args: Vec<Value>,
    body: Region,
) -> OpSpec {
    let result_types: Vec<Type> = iter_args.iter().map(|v| v.ty().clone()).collect();
    let mut spec = OpSpec::new("scf.for")
        .with_operand(lb)
        .with_operand(ub)
        .with_operand(step);
    for arg in iter_args {
        spec = spec.with_operand(arg);
    }
    for t in result_types {
        spec = spec.with_result(t);
    }
    spec.with_region(body)
}

/// `scf.if %cond -> (T1, T2) { then_region } else { else_region }`.
///
/// Both regions yield the same number/types of values, exposed as the op's
/// results. For an `if` with no result types, pass `result_types = vec![]`.
pub fn if_then_else(
    cond: Value,
    result_types: Vec<Type>,
    then_region: Region,
    else_region: Region,
) -> OpSpec {
    let mut spec = OpSpec::new("scf.if").with_operand(cond);
    for t in result_types {
        spec = spec.with_result(t);
    }
    spec.with_region(then_region).with_region(else_region)
}
