//! Tests for the operator-overloading layer (Phase 3.3 step 3): the
//! proc-macro should translate `+`, `-`, `*`, `<`, `<=`, `>`, `>=`, `==`,
//! `!=` into calls into `triton_ir::ops::*`, which dispatches at runtime
//! on the operand type to pick the right MLIR op variant.

use triton_dsl::triton_kernel;

#[allow(dead_code)]
struct Ptr<T>(::std::marker::PhantomData<T>);

#[triton_kernel]
fn add_chain(out: Ptr<i32>) {
    let pid = program_id(0);
    let c = const_i32(5);
    let v = pid + c;
    store(out, v);
}

#[triton_kernel]
fn mixed_ops(out: Ptr<i32>) {
    let pid = program_id(0);
    let two = const_i32(2);
    let one = const_i32(1);
    // (pid * 2) + 1 — exercises both * and +
    let v = pid * two + one;
    store(out, v);
}

#[triton_kernel]
fn comparison(x: Ptr<f32>, out: Ptr<f32>, n: i32) {
    let pid = program_id(0);
    let block = const_i32(1024);
    let base = pid * block;
    let range = make_range(0, 1024);
    let off = splat_1d(base, 1024) + range;
    let mask = off < splat_1d(n, 1024); // tensor cmpi.slt
    let xv = load(splat_1d(x, 1024) + off, mask);
    store(splat_1d(out, 1024) + off, xv, mask);
}

// The full vec_add kernel re-expressed using natural Rust operators.
// Should produce the same MLIR (up to function name) as the explicit
// `add_i32`/`mul_i32`/`addptr`/`lt_i32` version in proc_macro_vec_add.rs.
#[triton_kernel]
fn vec_add_op(x: Ptr<f32>, y: Ptr<f32>, out: Ptr<f32>, n: i32) {
    let pid = program_id(0);
    let base = pid * const_i32(1024);
    let off = splat_1d(base, 1024) + make_range(0, 1024);
    let mask = off < splat_1d(n, 1024);
    let xv = load(splat_1d(x, 1024) + off, mask);
    let yv = load(splat_1d(y, 1024) + off, mask);
    let sum = xv + yv;
    store(splat_1d(out, 1024) + off, sum, mask);
}

#[test]
fn add_chain_uses_addi() {
    let text = add_chain::mlir();
    eprintln!("===== add_chain =====\n{text}");
    assert!(text.contains("\"arith.addi\""), "missing addi:\n{text}");
}

#[test]
fn mixed_ops_emits_both_muli_and_addi() {
    let text = mixed_ops::mlir();
    eprintln!("===== mixed_ops =====\n{text}");
    assert!(text.contains("\"arith.muli\""), "missing muli:\n{text}");
    assert!(text.contains("\"arith.addi\""), "missing addi:\n{text}");
}

#[test]
fn comparison_emits_cmpi_slt() {
    let text = comparison::mlir();
    eprintln!("===== comparison =====\n{text}");
    assert!(text.contains("\"arith.cmpi\""), "missing cmpi:\n{text}");
    assert!(text.contains("predicate = 2 : i64"), "missing slt predicate:\n{text}");
    // Tensor pointer arithmetic via `+` should dispatch into addptr.
    assert!(text.contains("\"tt.addptr\""), "missing addptr from `+`:\n{text}");
    // Float load → float result element type.
    assert!(text.contains("tensor<1024xf32>"), "missing tensor<1024xf32>:\n{text}");
}

#[test]
fn vec_add_op_matches_explicit_version() {
    let text = vec_add_op::mlir();
    eprintln!("===== vec_add_op =====\n{text}");

    // Same op presence as the explicit-helper version, just authored with
    // natural Rust operator syntax.
    assert!(text.contains("\"tt.get_program_id\""));
    assert!(text.contains("\"arith.muli\""));
    assert!(text.contains("\"tt.make_range\""));
    assert!(text.contains("\"arith.addi\""));
    assert!(text.contains("\"arith.cmpi\""));
    assert_eq!(text.matches("\"tt.splat\"").count(), 5);
    assert_eq!(text.matches("\"tt.addptr\"").count(), 3);
    assert_eq!(text.matches("\"tt.load\"").count(), 2);
    assert_eq!(text.matches("\"tt.store\"").count(), 1);
    assert!(text.contains("\"arith.addf\""), "missing addf for tensor<f32> sum:\n{text}");
}
