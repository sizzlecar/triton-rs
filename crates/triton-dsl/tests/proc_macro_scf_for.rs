//! Phase 3.4 step 1: control flow in the DSL via `scf_for(lb, ub, step,
//! init, |i, acc| body)`. The closure body uses the same DSL syntax as
//! the outer kernel — ops emitted inside it land in the loop region
//! thanks to FuncBuilder's region_stack routing.

use triton_dsl::triton_kernel;

#[allow(dead_code)]
struct Ptr<T>(::std::marker::PhantomData<T>);

// Simplest possible loop: sum of 0..10 stored to *out.
#[triton_kernel]
fn sum_to_ten(out: Ptr<i32>) {
    let lb = const_i32(0);
    let ub = const_i32(10);
    let step = const_i32(1);
    let init = const_i32(0);

    let result = scf_for(lb, ub, step, init, |i, acc| acc + i);
    store(out, result);
}

// Same kernel, but the loop body is multi-statement (let binding before
// the trailing yield) — exercises split_block_for_yield's "all stmts
// except the last" path.
#[triton_kernel]
fn sum_squares(out: Ptr<i32>) {
    let lb = const_i32(0);
    let ub = const_i32(10);
    let step = const_i32(1);
    let init = const_i32(0);

    let result = scf_for(lb, ub, step, init, |i, acc| {
        let i_sq = i * i;
        acc + i_sq
    });
    store(out, result);
}

#[test]
fn scf_for_emits_loop_with_yield() {
    let text = sum_to_ten::mlir();
    eprintln!("===== sum_to_ten MLIR =====\n{text}\n===========================");

    assert!(text.contains("\"scf.for\""), "missing scf.for op:\n{text}");
    assert!(text.contains("\"scf.yield\""), "missing scf.yield:\n{text}");
    // Loop body has its own entry block label since it carries args.
    assert!(text.contains("^bb0("), "missing nested-region block label:\n{text}");
    // The body should call addi (acc + i) inside the for loop.
    assert!(text.contains("\"arith.addi\""), "missing addi inside loop:\n{text}");
}

#[test]
fn scf_for_handles_multi_statement_body() {
    let text = sum_squares::mlir();
    eprintln!("===== sum_squares MLIR =====\n{text}\n============================");

    // `i * i` inside the body should produce a muli before the addi.
    assert!(text.contains("\"arith.muli\""), "missing muli (i*i) inside loop:\n{text}");
    assert!(text.contains("\"arith.addi\""), "missing addi (acc + i_sq) inside loop:\n{text}");
}

#[test]
fn scf_for_result_threads_back_into_outer_scope() {
    let text = sum_to_ten::mlir();
    // The store at the end of sum_to_ten should consume the result that
    // came out of scf.for. Check that there's exactly one scf.for and
    // that a store appears AFTER it.
    let for_pos = text.find("scf.for").expect("scf.for not found");
    let store_pos = text.find("tt.store").expect("tt.store not found");
    assert!(
        store_pos > for_pos,
        "tt.store should follow scf.for in textual order:\n{text}"
    );
}
