//! Structural assertions for the element-wise kernels — every kernel
//! must emit a well-formed `tt.func` of the right shape, with the
//! expected ops actually present.

use triton_kernels::prelude::*;

#[test]
fn vec_add_emits_addf_load_store() {
    let text = vec_add_f32::<1024>::mlir();
    assert!(text.contains("tt.func @vec_add_f32("));
    assert!(text.contains("\"tt.get_program_id\""));
    assert!(text.contains("\"tt.load\""));
    assert!(text.contains("\"tt.store\""));
    assert!(text.contains("\"arith.addf\""));
    assert!(text.contains("tensor<1024xf32>"));
}

#[test]
fn residual_add_matches_vec_add_modulo_name() {
    let v = vec_add_f32::<1024>::mlir().replace("vec_add_f32", "X");
    let r = residual_add_f32::<1024>::mlir().replace("residual_add_f32", "X");
    assert_eq!(v, r, "vec_add and residual_add should differ only in the function name");
}

#[test]
fn residual_add_inplace_uses_a_for_both_load_and_store() {
    let text = residual_add_inplace_f32::<1024>::mlir();
    assert!(text.contains("tt.func @residual_add_inplace_f32("));
    // 2 args (a, b) + n => 2 ptr arg patterns appear in the func header.
    let header = &text[..text.find(") {").unwrap()];
    assert_eq!(header.matches("!tt.ptr<f32>").count(), 2,
               "header should declare 2 f32 pointers (a, b):\n{header}");
    // One load on a, one load on b, one store on a → 2 loads + 1 store
    assert_eq!(text.matches("\"tt.load\"").count(), 2);
    assert_eq!(text.matches("\"tt.store\"").count(), 1);
}

#[test]
fn gelu_uses_erf_and_constants() {
    let text = gelu_f32::<1024>::mlir();
    assert!(text.contains("\"math.erf\""), "gelu should use math.erf:\n{text}");
    // Two float constants: 0.5 and 1.0 (1/sqrt(2) is also a constant)
    assert!(text.contains("\"arith.constant\"() {value = 0.5 : f32}"));
    assert!(text.contains("\"arith.constant\"() {value = 1.0 : f32}"));
}

#[test]
fn fused_silu_mul_uses_exp_div_mul() {
    let text = fused_silu_mul_f32::<1024>::mlir();
    assert!(text.contains("\"math.exp\""));
    assert!(text.contains("\"arith.divf\""), "missing divf for silu denom:\n{text}");
    assert!(text.contains("\"arith.mulf\""), "missing mulf for silu*up:\n{text}");
    // 2 inputs (gate, up) + 1 output → 2 loads + 1 store
    assert_eq!(text.matches("\"tt.load\"").count(), 2);
    assert_eq!(text.matches("\"tt.store\"").count(), 1);
}

#[test]
fn fused_silu_mul_interleaved_decodes_b_and_i_indices() {
    let text = fused_silu_mul_interleaved_f32::<1024>::mlir();
    assert!(text.contains("tt.func @fused_silu_mul_interleaved_f32("));
    // (b, i) decomposition needs divsi + remsi against `inter`.
    assert!(text.contains("\"arith.divsi\""), "missing divsi for b decode:\n{text}");
    assert!(text.contains("\"arith.remsi\""), "missing remsi for i decode:\n{text}");
    // 2 loads (gate, up from same buffer) + 1 store.
    assert_eq!(text.matches("\"tt.load\"").count(), 2);
    assert_eq!(text.matches("\"tt.store\"").count(), 1);
    // Same silu math as fused_silu_mul_f32.
    assert!(text.contains("\"math.exp\""));
    assert!(text.contains("\"arith.divf\""));
}

#[test]
fn add_bias_two_ptrs_two_i32_args() {
    let text = add_bias_f32::<1024>::mlir();
    assert!(text.contains("tt.func @add_bias_f32("));
    let header = &text[..text.find(") {").unwrap()];
    assert_eq!(header.matches("!tt.ptr<f32>").count(), 2,
               "header should declare 2 f32 pointers (data, bias):\n{header}");
    assert!(text.contains("\"tt.addptr\""), "missing addptr for row offset:\n{text}");
    assert!(text.contains("\"arith.addf\""), "missing addf for bias add:\n{text}");
}
