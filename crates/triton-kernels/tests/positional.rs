//! Structural assertions for the rotary position embedding kernels.

use triton_kernels::prelude::*;

#[test]
fn rope_q_emits_pair_rotation() {
    let text = rope_q_f32::<64>::mlir();
    assert!(text.contains("tt.func @rope_q_f32("));
    // 2 input loads (cos, sin) + 2 data loads (x0, x1) = 4 loads
    assert_eq!(text.matches("\"tt.load\"").count(), 4,
               "rope_q should load cos, sin, x0, x1:\n{text}");
    // 2 stores (lo half, hi half)
    assert_eq!(text.matches("\"tt.store\"").count(), 2);
    // The rotation needs both addf and subf (out0 = x0*c - x1*s; out1 = x1*c + x0*s)
    assert!(text.contains("\"arith.addf\""));
    assert!(text.contains("\"arith.subf\""));
    assert!(text.contains("\"arith.mulf\""));
    // Half-dim tile shape:
    assert!(text.contains("tensor<64xf32>"));
}

#[test]
fn rope_k_has_same_shape_as_rope_q() {
    let q = rope_q_f32::<64>::mlir().replace("rope_q_f32", "X");
    let k = rope_k_f32::<64>::mlir().replace("rope_k_f32", "X");
    assert_eq!(q, k,
               "rope_q and rope_k differ only in the function name and the \
                tensor name `q`/`k` — same body shape");
}
