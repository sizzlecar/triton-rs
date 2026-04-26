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
