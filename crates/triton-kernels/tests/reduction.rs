//! Structural assertions for softmax + cross_entropy.

use triton_kernels::prelude::*;

#[test]
fn softmax_uses_two_reduces_one_exp_one_div() {
    let text = softmax_f32::<1024>::mlir();
    assert!(text.contains("tt.func @softmax_f32("));
    // Phase 1 row-max, phase 2 sum-of-exp.
    assert_eq!(text.matches("\"tt.reduce\"").count(), 2);
    // arith.maximumf inside the row-max reducer body.
    assert!(text.contains("\"arith.maximumf\""), "missing maximumf for row-max:\n{text}");
    // exp + div for the normalisation pass.
    assert!(text.contains("\"math.exp\""));
    assert!(text.contains("\"arith.divf\""));
}

#[test]
fn cross_entropy_uses_max_exp_log_and_label_gather() {
    let text = cross_entropy_forward_f32::<1024>::mlir();
    assert!(text.contains("tt.func @cross_entropy_forward_f32("));
    assert!(text.contains("\"arith.maximumf\""), "missing row-max:\n{text}");
    assert!(text.contains("\"math.exp\""));
    assert!(text.contains("\"math.log\""));
    // Three loads in the body: logits row, label, label_logit (scalar gather).
    assert_eq!(text.matches("\"tt.load\"").count(), 3,
               "expected 3 loads (logits, label, label_logit):\n{text}");
    assert_eq!(text.matches("\"tt.store\"").count(), 1);
}
