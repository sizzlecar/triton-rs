//! End-to-end test for `scf.for` IR construction.
//!
//! Builds a tiny kernel that sums `0 + 1 + ... + 9` via an `scf.for` loop
//! and stores the result. Verifies the printed MLIR has the expected
//! structure (entry block label, iter_args, yield terminator, result type).

use triton_ir::prelude::*;

fn build_sum_loop() -> Module {
    let mut m = Module::new();
    let mut f = m.func("sum_0_to_10");

    let out_ptr = f.arg("out", Type::ptr(Type::i32()));

    // Loop bounds and initial accumulator.
    let c0 = f.op_one(arith::constant_i32(0));
    let c1 = f.op_one(arith::constant_i32(1));
    let c10 = f.op_one(arith::constant_i32(10));

    // Build the loop body region. Its entry block carries:
    //   %i   : i32   — induction variable
    //   %acc : i32   — current accumulator (the sole iter_arg)
    let (mut body, body_args) = f.new_region(vec![Type::i32(), Type::i32()]);
    let i = body_args[0].clone();
    let acc = body_args[1].clone();

    let new_acc = f.append_to_region_one(&mut body, arith::addi(acc, i));
    f.append_to_region_void(&mut body, scf::yield_(vec![new_acc]));

    let result = f.op_one(scf::for_loop(c0.clone(), c10, c1, vec![c0], body));

    // Store and return.
    f.op_void(tt::store(out_ptr, result, None));
    f.op_void(tt::return_());
    f.finish();

    m
}

#[test]
fn scf_for_prints_block_args_and_yield() {
    let m = build_sum_loop();
    let text = m.to_string();

    eprintln!("===== sum_0_to_10 MLIR =====");
    eprintln!("{text}");
    eprintln!("============================");

    // Structural pieces we expect.
    assert!(text.contains("\"scf.for\""), "missing scf.for op:\n{text}");
    assert!(text.contains("\"scf.yield\""), "missing scf.yield terminator:\n{text}");
    // Block args inside the for body must be printed as `^bb0(%i: i32, %acc: i32):`.
    assert!(text.contains("^bb0("), "missing entry block label for scf.for body:\n{text}");
    // The for op result type must reflect the sole iter_arg (i32).
    assert!(
        text.contains("(i32, i32, i32, i32) -> i32"),
        "scf.for type signature should be (lb, ub, step, init) -> acc_result:\n{text}"
    );
}

/// Same kernel built with the closure-based `for_loop_with` API. Should
/// produce structurally equivalent MLIR (same op set, same types).
fn build_sum_loop_via_closure() -> Module {
    let mut m = Module::new();
    let mut f = m.func("sum_via_closure");
    let out_ptr = f.arg("out", Type::ptr(Type::i32()));

    let c0 = f.op_one(arith::constant_i32(0));
    let c1 = f.op_one(arith::constant_i32(1));
    let c10 = f.op_one(arith::constant_i32(10));

    let results = f.for_loop_with(c0.clone(), c10, c1, vec![c0], |fb, i, accs| {
        let acc = accs[0].clone();
        // op_one on `fb` here lands in the loop body region thanks to
        // region_stack — no special API required.
        let new_acc = fb.op_one(arith::addi(acc, i));
        vec![new_acc]
    });
    let result = results.into_iter().next().unwrap();

    f.op_void(tt::store(out_ptr, result, None));
    f.op_void(tt::return_());
    f.finish();
    m
}

#[test]
fn for_loop_with_closure_yields_same_structure() {
    let m = build_sum_loop_via_closure();
    let text = m.to_string();
    eprintln!("===== sum_via_closure MLIR =====\n{text}\n================================");
    assert!(text.contains("\"scf.for\""));
    assert!(text.contains("\"scf.yield\""));
    assert!(text.contains("^bb0("));
    assert!(text.contains("(i32, i32, i32, i32) -> i32"));
}
