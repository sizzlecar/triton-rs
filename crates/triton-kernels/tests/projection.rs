//! Structural assertions for split_qkv + transpose.

use triton_kernels::prelude::*;

#[test]
fn split_qkv_emits_3_loads_3_stores() {
    let text = split_qkv_f32::<128>::mlir();
    assert!(text.contains("tt.func @split_qkv_f32("));
    assert!(text.contains("\"arith.divsi\""));
    assert!(text.contains("\"arith.remsi\""));
    // 3 input gathers (q/k/v slice), 3 output stores.
    assert_eq!(text.matches("\"tt.load\"").count(), 3);
    assert_eq!(text.matches("\"tt.store\"").count(), 3);
    // 4 f32 pointer args: qkv input + q_out + k_out + v_out.
    let header = &text[..text.find(") {").unwrap()];
    assert_eq!(header.matches("!tt.ptr<f32>").count(), 4);
}

#[test]
fn transpose_uses_pid_decode_and_one_load_one_store() {
    let text = transpose_head_to_token_f32::<128>::mlir();
    assert!(text.contains("tt.func @transpose_head_to_token_f32("));
    assert!(text.contains("\"arith.divsi\""), "missing divsi for head decode:\n{text}");
    assert!(text.contains("\"arith.remsi\""), "missing remsi for tok decode:\n{text}");
    assert_eq!(text.matches("\"tt.load\"").count(), 1);
    assert_eq!(text.matches("\"tt.store\"").count(), 1);
}

#[test]
fn transpose_token_to_head_inverse_layout() {
    let text = transpose_token_to_head_f32::<128>::mlir();
    assert!(text.contains("tt.func @transpose_token_to_head_f32("));
    assert!(text.contains("\"arith.divsi\""));
    assert!(text.contains("\"arith.remsi\""));
}

#[test]
fn qk_norm_rope_transpose_combines_norm_rope_transpose() {
    let text = qk_norm_rope_transpose_f32::<128, 64>::mlir();
    assert!(text.contains("tt.func @qk_norm_rope_transpose_f32("));
    // Norm pass: reduce + rsqrt.
    assert!(text.contains("\"tt.reduce\""), "missing tt.reduce for norm:\n{text}");
    assert!(text.contains("\"math.rsqrt\""));
    // RoPE pass: 4 element-wise float ops (mul/sub/mul/add).
    assert!(text.contains("\"arith.subf\""), "missing subf for rope:\n{text}");
    // Norm load + 2 cos/sin + norm_w lo/hi + x0/x1 = 7 loads;
    // 2 stores (lo + hi half).
    let loads = text.matches("\"tt.load\"").count();
    assert!(loads >= 6, "expected at least 6 loads (norm tile + cos + sin + nw lo/hi + x0/x1), got {loads}:\n{text}");
    assert_eq!(text.matches("\"tt.store\"").count(), 2);
}

#[test]
fn rope_transpose_skips_norm_path() {
    let text = rope_transpose_f32::<64>::mlir();
    assert!(text.contains("tt.func @rope_transpose_f32("));
    // No reduce / rsqrt — this kernel skips the QK-norm.
    assert!(!text.contains("\"tt.reduce\""), "rope_transpose should NOT have reduce:\n{text}");
    assert!(!text.contains("\"math.rsqrt\""), "rope_transpose should NOT have rsqrt:\n{text}");
    // 4 loads (cos, sin, x0, x1), 2 stores.
    assert_eq!(text.matches("\"tt.load\"").count(), 4);
    assert_eq!(text.matches("\"tt.store\"").count(), 2);
}
