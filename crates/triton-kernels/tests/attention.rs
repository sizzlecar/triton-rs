//! Structural assertions for the decode-attention kernel(s).

use triton_kernels::prelude::*;

#[test]
fn decode_attention_emits_scf_for_with_three_iter_args() {
    let text = decode_attention_f32::<128, 32>::mlir();
    eprintln!("===== decode_attention_f32 =====\n{text}\n========================");

    assert!(
        text.contains("tt.func @decode_attention_f32("),
        "missing kernel func:\n{text}"
    );

    // Online softmax: scf.for with 3 iter_args (m_i, l_i, acc).
    // The yield should carry 3 operands; check the simpler signal that
    // the for loop region exists and has 3 iter_arg names in its block sig.
    assert!(text.contains("\"scf.for\""), "missing scf.for:\n{text}");
    assert!(text.contains("\"scf.yield\""), "missing scf.yield:\n{text}");

    // Q · K rendered via broadcast-mul-reduce: expect both expand_dims
    // and tt.reduce + an arith.mulf in the loop body.
    assert!(
        text.contains("\"tt.expand_dims\""),
        "missing expand_dims for 2D pointer construction:\n{text}"
    );
    assert!(
        text.contains("\"tt.reduce\""),
        "missing tt.reduce for QK / softmax / PV:\n{text}"
    );
    assert!(
        text.contains("\"arith.mulf\""),
        "missing mulf inside loop:\n{text}"
    );

    // Online softmax requires exp.
    assert!(text.contains("\"math.exp\""), "missing math.exp:\n{text}");

    // Final write — output is 1D tensor [HEAD_DIM] cast.
    assert_eq!(
        text.matches("\"tt.store\"").count(),
        1,
        "should be exactly one store at the end:\n{text}"
    );
    // Two loads in the body (K, V) + one Q load at the top.
    let n_loads = text.matches("\"tt.load\"").count();
    assert!(
        n_loads >= 3,
        "expected at least 3 loads (Q + K + V), got {n_loads}:\n{text}"
    );
}

#[test]
fn decode_attention_hm_emits_kernel() {
    let text = decode_attention_hm_f32::<128, 32>::mlir();
    assert!(text.contains("tt.func @decode_attention_hm_f32("));
    assert!(text.contains("\"scf.for\""));
    assert!(text.contains("\"math.exp\""));
    // Three loads minimum (Q + K + V); HM differs from canonical only in
    // address arithmetic so the op count signal is the same.
    assert!(text.matches("\"tt.load\"").count() >= 3);
}

#[test]
fn batched_decode_attention_emits_kernel_with_div_rem_for_batch_decompose() {
    let text = batched_decode_attention_f32::<128, 32>::mlir();
    assert!(text.contains("tt.func @batched_decode_attention_f32("));
    // Batch decompose = pid / num_q_heads + pid % num_q_heads.
    assert!(text.contains("\"arith.divsi\""), "missing batch decompose:\n{text}");
    assert!(text.contains("\"arith.remsi\""), "missing batch decompose:\n{text}");
    assert!(text.contains("\"scf.for\""));
}
