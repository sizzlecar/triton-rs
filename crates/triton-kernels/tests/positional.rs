//! Structural assertions for the rotary position embedding kernels.

use triton_kernels::prelude::*;

#[test]
fn rope_q_emits_pair_rotation() {
    let text = rope_q_f32::<64>::mlir();
    assert!(text.contains("tt.func @rope_q_f32("));
    assert_eq!(text.matches("\"tt.load\"").count(), 4,
               "rope_q should load cos, sin, x0, x1:\n{text}");
    assert_eq!(text.matches("\"tt.store\"").count(), 2);
    assert!(text.contains("\"arith.addf\""));
    assert!(text.contains("\"arith.subf\""));
    assert!(text.contains("\"arith.mulf\""));
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

#[test]
fn rope_full_decodes_pid_then_loads_position() {
    let text = rope_full_f32::<64>::mlir();
    assert!(text.contains("tt.func @rope_full_f32("));
    // Decode `(tok, h)` from a flat program_id via div + rem.
    assert!(text.contains("\"arith.divsi\""), "missing divsi for tok decode:\n{text}");
    assert!(text.contains("\"arith.remsi\""), "missing remsi for h decode:\n{text}");
    // 5 loads: positions, cos, sin, x0, x1; 2 stores: out_lo, out_hi.
    assert_eq!(text.matches("\"tt.load\"").count(), 5);
    assert_eq!(text.matches("\"tt.store\"").count(), 2);
    // Positions table is i32-typed.
    let header = &text[..text.find(") {").unwrap()];
    assert!(header.contains("!tt.ptr<i32>"), "missing i32 positions ptr:\n{header}");
}
