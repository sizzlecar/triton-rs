//! Structural assertions for the normalisation kernels.

use triton_kernels::prelude::*;

#[test]
fn rms_norm_uses_reduce_and_rsqrt() {
    let text = rms_norm_f32::<1024>::mlir();
    assert!(text.contains("tt.func @rms_norm_f32("));
    assert!(text.contains("\"tt.reduce\""), "missing tt.reduce:\n{text}");
    assert!(text.contains("\"tt.reduce.return\""));
    assert!(text.contains("\"math.rsqrt\""), "missing math.rsqrt:\n{text}");
    // 3 ptrs + (row_size: i32, inv_n: f32, eps: f32) — count in header.
    let header = &text[..text.find(") {").unwrap()];
    assert_eq!(header.matches("!tt.ptr<f32>").count(), 3);
}

#[test]
fn layer_norm_emits_two_reduces_and_rsqrt() {
    let text = layer_norm_f32::<1024>::mlir();
    assert!(text.contains("tt.func @layer_norm_f32("));
    assert_eq!(text.matches("\"tt.reduce\"").count(), 2,
               "layer_norm should compute mean AND variance via reduce:\n{text}");
    assert!(text.contains("\"math.rsqrt\""));
    assert!(text.contains("\"arith.addf\""));
}

#[test]
fn fused_add_rms_norm_writes_both_outputs() {
    let text = fused_add_rms_norm_f32::<1024>::mlir();
    assert!(text.contains("tt.func @fused_add_rms_norm_f32("));
    // Two stores: one to output, one to residual_out.
    assert_eq!(text.matches("\"tt.store\"").count(), 2,
               "fused_add_rms_norm writes output + residual_out:\n{text}");
    assert!(text.contains("\"tt.reduce\""));
    assert!(text.contains("\"math.rsqrt\""));
}

#[test]
fn rms_norm_signature_uses_inv_n_arg() {
    // The kernel should take row_size: i32, inv_n: f32, eps: f32
    // (matches the precomputed-inv_n convention).
    let text = rms_norm_f32::<512>::mlir();
    // 3 i32-ish or f32 args after the 3 pointers.
    assert!(text.contains(": i32"), "missing i32 row_size:\n{text}");
    // f32 args from the arg list (not the constants emitted in the body).
    let header_end = text.find(") {").expect("no func body open");
    let header = &text[..header_end];
    assert!(header.contains(": f32"), "header should declare an f32 arg:\n{header}");
}
