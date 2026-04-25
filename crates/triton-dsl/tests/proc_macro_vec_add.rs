//! End-to-end DSL test: write the canonical vector-add kernel using
//! `#[triton_kernel]` only (no IR builder calls in user code), and check
//! the emitted MLIR matches what the hand-rolled `triton-ir` example
//! produces.

use triton_dsl::triton_kernel;

#[allow(dead_code)]
struct Ptr<T>(::std::marker::PhantomData<T>);

#[triton_kernel]
fn vec_add(x: Ptr<f32>, y: Ptr<f32>, out: Ptr<f32>, n: i32) {
    let pid = program_id(0);
    let block = const_i32(1024);
    let base = mul_i32(pid, block);
    let range = make_range(0, 1024);
    let base_v = splat_1d(base, 1024);
    let off = add_i32(base_v, range);

    let n_v = splat_1d(n, 1024);
    let mask = lt_i32(off, n_v);

    let xp = splat_1d(x, 1024);
    let xp_off = addptr(xp, off);
    let xv = load(xp_off, mask);

    let yp = splat_1d(y, 1024);
    let yp_off = addptr(yp, off);
    let yv = load(yp_off, mask);

    let sum = add_f32(xv, yv);

    let outp = splat_1d(out, 1024);
    let outp_off = addptr(outp, off);
    store(outp_off, sum, mask);
}

#[test]
fn dsl_vec_add_matches_canonical_structure() {
    let text = vec_add::mlir();
    eprintln!("===== vec_add (DSL) MLIR =====\n{text}\n==============================");

    // Function header: 3 pointer params + 1 i32 (signature line only — the
    // pointer type also appears inside tensor types and op signatures).
    assert!(text.contains(
        "tt.func @vec_add(%0: !tt.ptr<f32>, %1: !tt.ptr<f32>, %2: !tt.ptr<f32>, %3: i32)"
    ));

    // Op presence mirroring the hand-rolled IR builder version.
    assert!(text.contains("\"tt.get_program_id\""));
    assert!(text.contains("\"arith.constant\"() {value = 1024 : i32}"));
    assert!(text.contains("\"arith.muli\""));
    assert!(text.contains("\"tt.make_range\""));
    assert!(text.contains("\"arith.addi\""));
}

#[test]
fn dsl_vec_add_uses_correct_splat_count() {
    let text = vec_add::mlir();
    // 5 splat operations: base, n, x ptr, y ptr, out ptr
    assert_eq!(
        text.matches("\"tt.splat\"").count(),
        5,
        "expected 5 tt.splat operations:\n{text}"
    );
}

#[test]
fn dsl_vec_add_uses_three_addptrs_and_two_loads_one_store() {
    let text = vec_add::mlir();
    assert_eq!(text.matches("\"tt.addptr\"").count(), 3);
    assert_eq!(text.matches("\"tt.load\"").count(), 2);
    assert_eq!(text.matches("\"tt.store\"").count(), 1);
}

#[test]
fn dsl_vec_add_has_one_cmpi_for_mask() {
    let text = vec_add::mlir();
    assert_eq!(text.matches("\"arith.cmpi\"").count(), 1);
    // The mask predicate is signed less-than = 2 (per arith::CmpiPred::Slt).
    assert!(text.contains("predicate = 2 : i64"));
}

#[test]
fn dsl_vec_add_emits_tensor_types_throughout() {
    let text = vec_add::mlir();
    // Every relevant tensor stays at width 1024.
    assert!(text.contains("tensor<1024xi32>"));
    assert!(text.contains("tensor<1024xi1>"));
    assert!(text.contains("tensor<1024xf32>"));
    assert!(text.contains("tensor<1024x!tt.ptr<f32>>"));
}
