//! End-to-end test for `scf.if` IR construction via the closure API.
//!
//! Two shapes:
//!   1. **Statement form (no result)** — early-return / guarded-store
//!      pattern. `if cond { store(...) }` with an empty else.
//!   2. **Expression form (single result)** — `let v = if cond { a } else { b };`.

use triton_ir::prelude::*;

/// Statement form: guarded store. No iter-arg threading, no yielded value.
fn build_guarded_store() -> Module {
    let mut m = Module::new();
    let mut f = m.func("guarded_store");

    let out_ptr = f.arg("out", Type::ptr(Type::i32()));
    let n = f.arg("n", Type::i32());

    let pid = f.op_one(tt::get_program_id(0));
    // cond: pid < n  → i1
    let cond = f.op_one(arith::cmpi(arith::CmpiPred::Slt, pid.clone(), n));

    f.if_then_with(cond, |fb| {
        // Inside the then region: store(out, pid + pid).
        let v = fb.op_one(arith::addi(pid.clone(), pid.clone()));
        fb.op_void(tt::store(out_ptr.clone(), v, None));
    });

    f.op_void(tt::return_());
    f.finish();
    m
}

#[test]
fn scf_if_statement_form_emits_guarded_region() {
    let m = build_guarded_store();
    let text = m.to_string();
    eprintln!("===== guarded_store MLIR =====\n{text}\n==============================");

    assert!(text.contains("\"scf.if\""), "missing scf.if op:\n{text}");
    // Two scf.yield terminators — one per region (then + empty else).
    assert!(
        text.matches("\"scf.yield\"").count() >= 2,
        "expected at least two scf.yield terminators (then + else):\n{text}"
    );
    // No-result form: scf.if's signature should print `(i1) -> ()`.
    assert!(
        text.contains("\"scf.if\"") && text.contains("(i1) -> ()"),
        "scf.if statement form should have `(i1) -> ()` signature:\n{text}"
    );
}

/// Expression form: `let v = if cond { a } else { b }`. Single i32 result.
fn build_if_yields_value() -> Module {
    let mut m = Module::new();
    let mut f = m.func("if_yields");
    let out_ptr = f.arg("out", Type::ptr(Type::i32()));

    let pid = f.op_one(tt::get_program_id(0));
    let zero = f.op_one(arith::constant_i32(0));
    let cond = f.op_one(arith::cmpi(arith::CmpiPred::Eq, pid.clone(), zero.clone()));

    let then_v = f.op_one(arith::constant_i32(1));
    let else_v = f.op_one(arith::constant_i32(2));

    let results = f.if_then_else_with(
        cond,
        |_fb| vec![then_v.clone()],
        |_fb| vec![else_v.clone()],
    );
    let v = results.into_iter().next().unwrap();

    f.op_void(tt::store(out_ptr, v, None));
    f.op_void(tt::return_());
    f.finish();
    m
}

#[test]
fn scf_if_expression_form_yields_typed_result() {
    let m = build_if_yields_value();
    let text = m.to_string();
    eprintln!("===== if_yields MLIR =====\n{text}\n==========================");

    assert!(text.contains("\"scf.if\""), "missing scf.if op:\n{text}");
    // Both branches yield exactly one i32 — scf.if signature should be `(i1) -> i32`.
    assert!(
        text.contains("\"scf.if\"") && text.contains("(i1) -> i32"),
        "scf.if expression form should have `(i1) -> i32` signature:\n{text}"
    );
    // Yield ops carry their operand.
    assert!(
        text.contains("\"scf.yield\""),
        "missing scf.yield in branches:\n{text}"
    );
}
