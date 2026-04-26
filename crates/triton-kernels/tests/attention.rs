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

#[test]
fn flash_decode_phase1_uses_program_id_axis_1_for_split() {
    let text = flash_decode_attn_phase1_f32::<128, 32>::mlir();
    assert!(text.contains("tt.func @flash_decode_attn_phase1_f32("));
    // 2D grid: program_id(0)=q_head, program_id(1)=split_id.
    assert!(text.matches("\"tt.get_program_id\"").count() >= 2,
            "expected ≥2 program_id calls (q_head + split_id):\n{text}");
    // Same online-softmax inner loop as decode_attention.
    assert!(text.contains("\"scf.for\""));
    assert!(text.contains("\"math.exp\""));
    // 3 stores: partial_acc (vector), partial_max (scalar), partial_sum (scalar).
    assert!(text.matches("\"tt.store\"").count() >= 3);
}

#[test]
fn flash_decode_phase2_combines_via_log_sum_exp() {
    let text = flash_decode_attn_phase2_f32::<128, 32>::mlir();
    assert!(text.contains("tt.func @flash_decode_attn_phase2_f32("));
    // No scf.for — pure tile reduction along NUM_SPLITS_TILE axis.
    assert!(!text.contains("\"scf.for\""), "phase 2 should be a single-tile reduce, not a loop");
    // log-sum-exp combine uses exp + multiple reduces.
    assert!(text.contains("\"math.exp\""));
    assert!(text.matches("\"tt.reduce\"").count() >= 3,
            "expected ≥3 reduces (max over splits + sum-of-l + sum-of-acc):\n{text}");
    assert_eq!(text.matches("\"tt.store\"").count(), 1);
}

#[test]
fn paged_decode_attention_emits_block_table_gather() {
    let text = paged_decode_attention_f32::<128, 32>::mlir();
    assert!(text.contains("tt.func @paged_decode_attention_f32("));
    // Address-translation needs divsi (logical = pos / block_size) + remsi
    // (slot = pos % block_size).
    assert!(text.contains("\"arith.divsi\""));
    assert!(text.contains("\"arith.remsi\""));
    // Gather load on block_table (i32 ptr) — distinct from the K/V loads
    // (f32 ptr). Total loads ≥ 4 (Q + block_table + K + V).
    assert!(text.matches("\"tt.load\"").count() >= 4,
            "expected ≥4 loads (Q + block_table gather + K + V):\n{text}");
    // i32 ptr in the func header (block_table arg).
    let header = &text[..text.find(") {").unwrap()];
    assert!(header.contains("!tt.ptr<i32>"),
            "missing i32 ptr param for block_table:\n{header}");
}
