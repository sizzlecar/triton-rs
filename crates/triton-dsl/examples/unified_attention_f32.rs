//! Unified prefill + decode attention with paged KV — DSL dump.
//!
//! Stripped-down port of vLLM's `kernel_unified_attention`. The whole
//! point of the unified kernel is that prefill and decode go through the
//! SAME body — there is no `if prefill else decode` branch. Instead, all
//! dispatching falls out of three masks combined into the score:
//!   - query_mask: rows where `q_pos_in_seq < cur_batch_query_len`
//!   - kv_bounds_mask: cols where `kv_pos < seq_len`
//!   - causal_mask: cols where `kv_pos <= context_len + q_pos_in_seq`
//! `final_mask = causal & kv_bounds & query`, then
//! `scores += (mask_f - 1) * 1e30`. Standard `flash_attn_full` trick —
//! no `arith.select` / `tt.where` needed.
//!
//! Block-table indirection follows `paged_decode_attention_f32`:
//!   physical = block_table[seq_idx, kv_pos / block_size]
//!   slot     = kv_pos % block_size
//!
//! See `crates/triton-kernels/src/attention.rs::unified_attention_f32`
//! for the canonical version (this example exists so the requested
//! `cargo run --example unified_attention_f32 -p triton-dsl` invocation
//! works without depending on triton-kernels).
//!
//! FIXME(scope): explicitly skipped vs vLLM main kernel:
//!   ALiBi, soft-cap, sinks, sliding window, FP8/INT8 dequant, qq_bias,
//!   batch-invariant mode, 3D segm-output split-softmax, output FP8 clip,
//!   `out_scale` / `k_scale` / `v_scale`. All are constexpr-disabled in
//!   vLLM and stack on top; adding any one is mechanical once the core
//!   compiles.

use triton_dsl::triton_kernel;

#[allow(dead_code)]
struct Ptr<T>(::std::marker::PhantomData<T>);

#[triton_kernel]
fn unified_attention_f32<const HEAD_DIM: usize, const BLOCK_Q: usize, const BLOCK_KV: usize>(
    q_ptr: Ptr<f32>,
    k_cache_ptr: Ptr<f32>,
    v_cache_ptr: Ptr<f32>,
    out_ptr: Ptr<f32>,
    block_table_ptr: Ptr<i32>,
    seq_lens_ptr: Ptr<i32>,
    query_start_loc_ptr: Ptr<i32>,
    num_q_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    block_size: i32,
    max_blocks_per_seq: i32,
    sm_scale: f32,
) {
    let _ = num_q_heads;
    let seq_idx = program_id(0);
    let q_head = program_id(1);
    let q_tile_id = program_id(2);

    let num_kv_groups = num_q_heads / num_kv_heads;
    let kv_head = q_head / num_kv_groups;

    // ── per-seq metadata ──
    let q_start = load(query_start_loc_ptr + seq_idx);
    let q_stop = load(query_start_loc_ptr + (seq_idx + 1));
    let cur_batch_query_len = q_stop - q_start;
    let seq_len = load(seq_lens_ptr + seq_idx);
    // Query tokens append at the tail of KV: kv_pos for query token q is
    // (seq_len - cur_batch_query_len) + q. context_len = seq_len-1 in
    // pure decode (q_len==1), 0 in first prefill.
    let context_len = seq_len - cur_batch_query_len;

    // ── Q-tile coordinates ──
    let dim_range = make_range(0, HEAD_DIM as i32);
    let _ = head_dim;
    let q_pos_in_tile = make_range(0, BLOCK_Q as i32);
    let q_pos_in_seq = q_pos_in_tile + q_tile_id * (BLOCK_Q as i32);
    let query_mask = q_pos_in_seq < cur_batch_query_len; // [BLOCK_Q] i1

    let token_idx = q_pos_in_seq + q_start;
    let q_row_stride = num_q_heads * head_dim;
    let q_row_base = token_idx * q_row_stride + q_head * head_dim;
    let q_row_2d = expand_dims(q_row_base, 1);
    let dim_2d = expand_dims(dim_range, 0);
    let q_off_2d = q_row_2d + dim_2d;
    let q_tile = load(q_ptr + q_off_2d); // [BLOCK_Q, HEAD_DIM]

    // ── online-softmax init ──
    let zero_2d = q_tile * 0.0_f32;
    let zero_q = reduce(zero_2d, 1, |a, b| a + b); // [BLOCK_Q] f32
    let m_i_init = zero_q - 1.0e30_f32;
    let l_i_init = zero_q;
    let acc_init = q_tile * 0.0_f32; // [BLOCK_Q, HEAD_DIM]

    // ── KV loop ──
    let kv_blocks = seq_len / (BLOCK_KV as i32);
    let kv_inner_stride = num_kv_heads * head_dim;
    let block_stride = block_size * kv_inner_stride;
    let bt_seq_base = seq_idx * max_blocks_per_seq;

    let (_, l_i, acc) = scf_for(
        const_i32(0),
        kv_blocks,
        const_i32(1),
        (m_i_init, l_i_init, acc_init),
        |kb, (m_i, l_i, acc)| {
            let kv_pos_base = kb * (BLOCK_KV as i32);
            let kv_pos_range = make_range(0, BLOCK_KV as i32) + kv_pos_base;

            // Block-table indirection.
            let logical_blocks = kv_pos_range / block_size;
            let slots = kv_pos_range % block_size;
            let physical_blocks = load(block_table_ptr + (bt_seq_base + logical_blocks));

            let row_base =
                physical_blocks * block_stride + slots * kv_inner_stride + kv_head * head_dim;
            let row_base_2d = expand_dims(row_base, 1);
            let kv_off_2d = row_base_2d + dim_2d;

            let k_tile = load(k_cache_ptr + kv_off_2d);
            let v_tile = load(v_cache_ptr + kv_off_2d);

            // scores = Q @ Kᵀ via broadcast-mul-reduce over HEAD_DIM.
            let q_3d = expand_dims(q_tile, 1);
            let k_3d = expand_dims(k_tile, 0);
            let qk = q_3d * k_3d;
            let scores_raw = reduce(qk, 2, |a, b| a + b);
            let scores_scaled = scores_raw * sm_scale;

            // ── three-mask combination ──
            let q_abs_2d = expand_dims(q_pos_in_seq + context_len, 1);
            let kv_pos_2d = expand_dims(kv_pos_range, 0);
            let causal_mask = kv_pos_2d <= q_abs_2d;
            let kv_in_bounds_1d = kv_pos_range < seq_len;
            let kv_in_bounds_2d = expand_dims(kv_in_bounds_1d, 0);
            let query_mask_2d = expand_dims(query_mask, 1);

            let combined_mask = causal_mask & kv_in_bounds_2d & query_mask_2d;
            let mask_f = to_f32(combined_mask);
            let masked_scores = scores_scaled + (mask_f - 1.0_f32) * 1.0e30_f32;

            // online softmax (per-row)
            let row_max = reduce(masked_scores, 1, |a, b| max(a, b));
            let m_ij = max(m_i, row_max);
            let alpha = exp(m_i - m_ij);
            let p = exp(masked_scores - expand_dims(m_ij, 1));
            let l_ij = reduce(p, 1, |a, b| a + b);

            let p_3d = expand_dims(p, 2);
            let v_3d = expand_dims(v_tile, 0);
            let pv = p_3d * v_3d;
            let pv_sum = reduce(pv, 1, |a, b| a + b);

            let alpha_2d = expand_dims(alpha, 1);
            let new_acc = acc * alpha_2d + pv_sum;
            let new_l_i = l_i * alpha + l_ij;
            (m_ij, new_l_i, new_acc)
        },
    );

    let l_i_2d = expand_dims(l_i, 1);
    let out_v = acc / l_i_2d;
    let out_off_2d = q_off_2d;
    let _ = query_mask; // unused as a 2D store mask in v0 (same caveat as flash_attn_full)
    store(out_ptr + out_off_2d, out_v);
}

fn main() {
    // BLOCK_Q=16 works for both decode (q_len=1, only row 0 unmasked) and
    // small prefill tiles. Production may use BLOCK_Q=1 for pure decode.
    print!("{}", unified_attention_f32::<128, 16, 32>::mlir());
}
