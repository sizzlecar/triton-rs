//! Attention kernels — single-query decode and (later) flash variants.
//!
//! These are the largest kernels in the LLM inference path; ferrum-infer-rs
//! currently relies on hand-written .cu equivalents. The Rust DSL ports
//! follow the canonical online-softmax + tile-loop pattern from Triton's
//! tutorial 06-fused-attention.py, simplified for the decode case (one
//! query token).

use triton_dsl::triton_kernel;

#[allow(dead_code)]
struct Ptr<T>(::std::marker::PhantomData<T>);

/// Single-query decode attention with grouped-query attention (GQA) support.
///
/// Shapes:
/// - `q`         : `[num_q_heads, head_dim]`
/// - `k_cache`   : `[seq_len, num_kv_heads, head_dim]` (candle layout)
/// - `v_cache`   : `[seq_len, num_kv_heads, head_dim]`
/// - `output`    : `[num_q_heads, head_dim]`
///
/// Launch: grid = (num_q_heads,). One block per Q head; the block streams
/// over KV positions in tiles of `BLOCK_KV`. Online softmax keeps three
/// running state values (m_i, l_i, acc) threaded through `scf_for`.
///
/// **Constraints (v0 — to be relaxed):**
/// - `head_dim <= HEAD_DIM` const-generic.
/// - `valid_kv_len` must be an exact multiple of `BLOCK_KV` — caller pads
///   the cache and clears the tail. Non-multiple lengths will read past
///   `valid_kv_len` and silently mix tail garbage into the softmax.
///   (Phase-2 fix: add `arith.select` / `tt.where` to the DSL and mask
///   per-position scores with `-inf`.)
///
/// **Why broadcast-mul-reduce instead of `tt.dot` for `Q·Kᵀ`:**
/// `tt.dot` needs K shaped `[HEAD_DIM, BLOCK_KV]`, requiring a transposed
/// load. The broadcast-mul-reduce path lets us load K naturally as
/// `[BLOCK_KV, HEAD_DIM]` — slower for huge head dims but matches ferrum's
/// access pattern and is easier to verify. Switch to `tt.dot` once we've
/// validated correctness.
#[triton_kernel]
pub fn decode_attention_f32<const HEAD_DIM: usize, const BLOCK_KV: usize>(
    q: Ptr<f32>,
    k_cache: Ptr<f32>,
    v_cache: Ptr<f32>,
    output: Ptr<f32>,
    num_q_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    valid_kv_len: i32,
    scale: f32,
) {
    let _ = num_q_heads;
    let q_head = program_id(0);
    let num_kv_groups = num_q_heads / num_kv_heads;
    let kv_head = q_head / num_kv_groups;

    // ── load Q row [HEAD_DIM] ──
    let dim_range = make_range(0, HEAD_DIM as i32);
    let dim_mask = dim_range < head_dim;
    let q_off = q_head * head_dim + dim_range;
    let q_v = load(q + q_off, dim_mask);

    // ── online-softmax state ──
    // Negative-float literals aren't accepted directly by the DSL macro;
    // build -1e30 from `0 - 1e30`.
    let m_init = const_f32(0.0_f32) - const_f32(1.0e30_f32);
    let l_init = const_f32(0.0_f32);
    // acc starts at zero across HEAD_DIM. Use Q*0 so we get a tensor of the
    // right shape without a dedicated zeros-tensor builder.
    let acc_init = q_v * 0.0_f32;

    // ── tile loop over KV ──
    let kv_stride = num_kv_heads * head_dim;
    let kv_blocks = valid_kv_len / (BLOCK_KV as i32);

    let (_, l_i, acc) = scf_for(
        const_i32(0),
        kv_blocks,
        const_i32(1),
        (m_init, l_init, acc_init),
        |kb, (m_i, l_i, acc)| {
            let pos_base = kb * (BLOCK_KV as i32);
            let pos_range = make_range(0, BLOCK_KV as i32) + pos_base;

            // K block [BLOCK_KV, HEAD_DIM]: 2D pointer construction via
            // outer-add of `pos[i] * kv_stride + kv_head * head_dim` (rows)
            // and `dim_range[d]` (cols). Auto-broadcast handles the 1D→2D
            // expansion.
            let row_base = pos_range * kv_stride + kv_head * head_dim;
            let row_base_2d = expand_dims(row_base, 1); // [BLOCK_KV, 1]
            let col_2d = expand_dims(dim_range, 0); // [1, HEAD_DIM]
            let kv_off_2d = row_base_2d + col_2d; // [BLOCK_KV, HEAD_DIM]

            // v0 simplification: caller pads valid_kv_len to a BLOCK_KV
            // multiple AND head_dim==HEAD_DIM, so neither row nor col load
            // can go OOB and we don't need a mask. (Real masking needs
            // bitwise AND of two i1 tensors — DSL extension TODO.)
            let k_block = load(k_cache + kv_off_2d);
            let v_block = load(v_cache + kv_off_2d);

            // scores[BLOCK_KV] = sum_d Q[d] * K[i, d]
            let q_2d = expand_dims(q_v, 0); // [1, HEAD_DIM]
            let qk = q_2d * k_block; // [BLOCK_KV, HEAD_DIM]
            let scores_raw = reduce(qk, 1, |a, b| a + b); // [BLOCK_KV]
            let scores = scores_raw * scale;

            // online softmax
            let scores_max = reduce(scores, 0, |a, b| max(a, b)); // scalar
            let m_ij = max(m_i, scores_max);
            let alpha = exp(m_i - m_ij);
            let p = exp(scores - m_ij); // [BLOCK_KV]
            let l_ij = reduce(p, 0, |a, b| a + b); // scalar

            // pv_sum[HEAD_DIM] = sum_i p[i] * V[i, d]
            let p_2d = expand_dims(p, 1); // [BLOCK_KV, 1]
            let pv = p_2d * v_block; // [BLOCK_KV, HEAD_DIM]
            let pv_sum = reduce(pv, 0, |a, b| a + b); // [HEAD_DIM]

            let new_acc = acc * alpha + pv_sum;
            let new_l_i = l_i * alpha + l_ij;
            (m_ij, new_l_i, new_acc)
        },
    );

    // ── normalize + write back ──
    let out_v = acc / l_i;
    let out_off = q_head * head_dim + dim_range;
    store(output + out_off, out_v, dim_mask);
}
