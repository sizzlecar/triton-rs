//! Phase 3.4 step 4: integer / float literal auto-promotion.
//!
//! Users can write `pid * 1024`, `xv + 0.5`, `off < 4096` directly. The
//! proc-macro detects the literal operand and emits
//! `__triton_f.lit_i64(&other, lit)` to lift it to a Value matching the
//! other operand's element type at IR-build time.

use triton_dsl::triton_kernel;

#[allow(dead_code)]
struct Ptr<T>(::std::marker::PhantomData<T>);

#[triton_kernel]
fn lit_int_kernel(out: Ptr<i32>) {
    let pid = program_id(0);
    let v = pid * 1024 + 7;
    store(out, v);
}

#[triton_kernel]
fn lit_float_kernel(x: Ptr<f32>, out: Ptr<f32>, n: i32) {
    let pid = program_id(0);
    let off = pid * 1024 + make_range(0, 1024);
    let mask = off < n;
    let xv = load(x + off, mask);
    // Float literal 0.5 should auto-lift to a scalar f32 then auto-broadcast.
    let scaled = xv * 0.5 + 1.0;
    store(out + off, scaled, mask);
}

#[test]
fn int_literal_auto_promotes() {
    let text = lit_int_kernel::mlir();
    eprintln!("===== lit_int_kernel =====\n{text}");
    assert!(text.contains("\"arith.constant\"() {value = 1024 : i32}"),
            "missing auto-emitted constant 1024:\n{text}");
    assert!(text.contains("\"arith.constant\"() {value = 7 : i32}"),
            "missing auto-emitted constant 7:\n{text}");
    assert!(text.contains("\"arith.muli\""), "missing muli for `pid * 1024`:\n{text}");
    assert!(text.contains("\"arith.addi\""), "missing addi for `... + 7`:\n{text}");
}

#[test]
fn float_literal_auto_promotes_through_broadcast() {
    let text = lit_float_kernel::mlir();
    eprintln!("===== lit_float_kernel =====\n{text}");
    // `xv * 0.5` → constant 0.5 then mulf on tensor<1024xf32>
    assert!(text.contains("\"arith.constant\"() {value = 0.5 : f32}"),
            "missing 0.5 constant:\n{text}");
    assert!(text.contains("\"arith.constant\"() {value = 1.0 : f32}"),
            "missing 1.0 constant:\n{text}");
    assert!(text.contains("\"arith.mulf\""),
            "missing mulf for tensor * 0.5:\n{text}");
    assert!(text.contains("\"arith.addf\""),
            "missing addf for tensor + 1.0:\n{text}");
    // The 0.5 / 1.0 constants are scalars; broadcast splats them to
    // tensor<1024xf32> via tt.splat before the binary op.
    assert!(text.contains("\"tt.splat\""),
            "missing tt.splat for scalar→tensor broadcast:\n{text}");
}

// And the show-the-API kernel — vec_add written in maximally clean DSL
// using literal auto-promote AND scalar/tensor auto-broadcast. Compare
// to the verbose original now ~3 lines longer.
#[triton_kernel]
fn vec_add_lit(x: Ptr<f32>, y: Ptr<f32>, out: Ptr<f32>, n: i32) {
    let pid = program_id(0);
    let off = pid * 1024 + make_range(0, 1024);
    let mask = off < n;
    let xv = load(x + off, mask);
    let yv = load(y + off, mask);
    store(out + off, xv + yv, mask);
}

#[test]
fn vec_add_with_literals_emits_clean_mlir() {
    let text = vec_add_lit::mlir();
    eprintln!("===== vec_add_lit =====\n{text}");
    assert!(text.contains("tt.func @vec_add_lit"));
    assert!(text.contains("\"arith.constant\"() {value = 1024 : i32}"));
    assert!(text.contains("tensor<1024xi32>"));
    assert!(text.contains("\"tt.addptr\""));
    assert!(text.contains("\"arith.addf\""));
    assert!(text.contains("\"tt.store\""));
}
