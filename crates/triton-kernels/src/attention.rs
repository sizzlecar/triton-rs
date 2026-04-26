//! Attention kernels — single-query decode and (later) flash variants.
//!
//! These are the largest kernels in the LLM inference path; ferrum-infer-rs
//! currently relies on hand-written .cu equivalents. The Rust DSL ports
//! follow the canonical online-softmax + tile-loop pattern from Triton's
//! tutorial 06-fused-attention.py, simplified for the decode case (one
//! query token).

use triton_dsl::triton_kernel;
#[allow(unused_imports)]
pub use triton_ir::ty::{f16, TritonElem};

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

/// Same online-softmax decode attention as [`decode_attention_f32`] but
/// for the **head-major** KV-cache layout `[num_kv_heads, capacity, head_dim]`
/// (used by ferrum's LlamaFamilyModel decode path after
/// `kv_cache_append_head_major_*`). Q and output stay
/// `[num_q_heads, head_dim]`.
///
/// Only the K/V address arithmetic changes:
///   - seq-major:  `kv_off = pos * (num_kv_heads*head_dim) + kv_head*head_dim + d`
///   - head-major: `kv_off = kv_head * (capacity*head_dim) + pos * head_dim + d`
///
/// `capacity` is the allocated slot count (allocated_kv_slots), which may
/// be > `valid_kv_len`; the kernel only reads `valid_kv_len` positions.
#[triton_kernel]
pub fn decode_attention_hm_f32<const HEAD_DIM: usize, const BLOCK_KV: usize>(
    q: Ptr<f32>,
    k_cache: Ptr<f32>,
    v_cache: Ptr<f32>,
    output: Ptr<f32>,
    num_q_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    capacity: i32,
    valid_kv_len: i32,
    scale: f32,
) {
    let _ = num_q_heads;
    let q_head = program_id(0);
    let num_kv_groups = num_q_heads / num_kv_heads;
    let kv_head = q_head / num_kv_groups;

    let dim_range = make_range(0, HEAD_DIM as i32);
    let dim_mask = dim_range < head_dim;
    let q_off = q_head * head_dim + dim_range;
    let q_v = load(q + q_off, dim_mask);

    let m_init = const_f32(0.0_f32) - const_f32(1.0e30_f32);
    let l_init = const_f32(0.0_f32);
    let acc_init = q_v * 0.0_f32;

    let kv_blocks = valid_kv_len / (BLOCK_KV as i32);
    // Head-major: each KV head's slab starts at kv_head * capacity * head_dim.
    let kv_head_base = kv_head * capacity * head_dim;

    let (_, l_i, acc) = scf_for(
        const_i32(0),
        kv_blocks,
        const_i32(1),
        (m_init, l_init, acc_init),
        |kb, (m_i, l_i, acc)| {
            let pos_base = kb * (BLOCK_KV as i32);
            let pos_range = make_range(0, BLOCK_KV as i32) + pos_base;

            // Head-major K/V address: kv_head_base + pos[i] * head_dim + d
            let row_base = pos_range * head_dim + kv_head_base;
            let row_base_2d = expand_dims(row_base, 1); // [BLOCK_KV, 1]
            let col_2d = expand_dims(dim_range, 0); // [1, HEAD_DIM]
            let kv_off_2d = row_base_2d + col_2d; // [BLOCK_KV, HEAD_DIM]

            let k_block = load(k_cache + kv_off_2d);
            let v_block = load(v_cache + kv_off_2d);

            let q_2d = expand_dims(q_v, 0);
            let qk = q_2d * k_block;
            let scores_raw = reduce(qk, 1, |a, b| a + b);
            let scores = scores_raw * scale;

            let scores_max = reduce(scores, 0, |a, b| max(a, b));
            let m_ij = max(m_i, scores_max);
            let alpha = exp(m_i - m_ij);
            let p = exp(scores - m_ij);
            let l_ij = reduce(p, 0, |a, b| a + b);

            let p_2d = expand_dims(p, 1);
            let pv = p_2d * v_block;
            let pv_sum = reduce(pv, 0, |a, b| a + b);

            let new_acc = acc * alpha + pv_sum;
            let new_l_i = l_i * alpha + l_ij;
            (m_ij, new_l_i, new_acc)
        },
    );

    let out_v = acc / l_i;
    let out_off = q_head * head_dim + dim_range;
    store(output + out_off, out_v, dim_mask);
}

/// **Batched** single-query decode attention. Same online-softmax + GQA as
/// [`decode_attention_f32`] but with an outer batch dimension for
/// continuous-batching decode (multiple sequences in flight).
///
/// Shapes:
/// - `q`         : `[batch, num_q_heads, head_dim]`
/// - `k_cache`   : `[batch, seq_len, num_kv_heads, head_dim]` (per-sequence
///                  cache, seq-major layout — ferrum's batched path)
/// - `v_cache`   : `[batch, seq_len, num_kv_heads, head_dim]`
/// - `output`    : `[batch, num_q_heads, head_dim]`
///
/// Launch: grid = (batch * num_q_heads,). One block per (batch, q_head).
/// `valid_kv_len` is per-sequence — for now assume it's the same across
/// batch (caller pads). Per-batch lengths is a follow-up that needs an
/// `int* per_batch_lens` parameter.
#[triton_kernel]
pub fn batched_decode_attention_f32<const HEAD_DIM: usize, const BLOCK_KV: usize>(
    q: Ptr<f32>,
    k_cache: Ptr<f32>,
    v_cache: Ptr<f32>,
    output: Ptr<f32>,
    batch_size: i32,
    num_q_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    seq_len: i32,
    valid_kv_len: i32,
    scale: f32,
) {
    let _ = batch_size;
    let pid = program_id(0);
    let b = pid / num_q_heads;
    let q_head = pid % num_q_heads;
    let num_kv_groups = num_q_heads / num_kv_heads;
    let kv_head = q_head / num_kv_groups;

    let dim_range = make_range(0, HEAD_DIM as i32);
    let dim_mask = dim_range < head_dim;
    // Q row for this batch + head: b * (num_q_heads * head_dim) + q_head * head_dim
    let q_row_base = b * num_q_heads * head_dim + q_head * head_dim;
    let q_v = load(q + q_row_base + dim_range, dim_mask);

    let m_init = const_f32(0.0_f32) - const_f32(1.0e30_f32);
    let l_init = const_f32(0.0_f32);
    let acc_init = q_v * 0.0_f32;

    // Per-batch KV cache base: b * (seq_len * num_kv_heads * head_dim).
    let kv_batch_base = b * seq_len * num_kv_heads * head_dim;
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

            // Per-batch + per-pos K/V address.
            let row_base = pos_range * kv_stride + kv_head * head_dim + kv_batch_base;
            let row_base_2d = expand_dims(row_base, 1);
            let col_2d = expand_dims(dim_range, 0);
            let kv_off_2d = row_base_2d + col_2d;

            let k_block = load(k_cache + kv_off_2d);
            let v_block = load(v_cache + kv_off_2d);

            let q_2d = expand_dims(q_v, 0);
            let qk = q_2d * k_block;
            let scores_raw = reduce(qk, 1, |a, b| a + b);
            let scores = scores_raw * scale;

            let scores_max = reduce(scores, 0, |a, b| max(a, b));
            let m_ij = max(m_i, scores_max);
            let alpha = exp(m_i - m_ij);
            let p = exp(scores - m_ij);
            let l_ij = reduce(p, 0, |a, b| a + b);

            let p_2d = expand_dims(p, 1);
            let pv = p_2d * v_block;
            let pv_sum = reduce(pv, 0, |a, b| a + b);

            let new_acc = acc * alpha + pv_sum;
            let new_l_i = l_i * alpha + l_ij;
            (m_ij, new_l_i, new_acc)
        },
    );

    let out_v = acc / l_i;
    let out_off = q_row_base + dim_range;
    store(output + out_off, out_v, dim_mask);
}

/// **Paged** single-query decode attention with block-table indirection.
///
/// K/V live in a paged block pool instead of a contiguous tensor:
///   `k_block_pool` shape `[max_blocks, block_size, num_kv_heads, head_dim]`
///   `block_table[logical_block] = physical_block_id`
///
/// For each `kv_pos in [0, valid_kv_len)`:
///   logical = kv_pos / block_size
///   slot    = kv_pos % block_size
///   physical = block_table[logical]
///   addr = physical * (block_size * num_kv_heads * head_dim)
///        + slot     * (num_kv_heads * head_dim)
///        + kv_head  * head_dim
///        + d
///
/// Compared to the contiguous variants this kernel does an extra gather
/// load on `block_table` per KV tile. The gather is straightforward in
/// Triton DSL: `tt.load(block_table + logical_blocks_tile)` produces a
/// `tensor<BLOCK_KVxi32>` that we use as the per-position physical-block
/// index in subsequent address arithmetic.
///
/// Used by ferrum's PagedAttention path. v0 still assumes
/// `valid_kv_len % BLOCK_KV == 0` and `head_dim == HEAD_DIM`.
#[triton_kernel]
pub fn paged_decode_attention_f32<const HEAD_DIM: usize, const BLOCK_KV: usize>(
    q: Ptr<f32>,
    k_block_pool: Ptr<f32>,
    v_block_pool: Ptr<f32>,
    block_table: Ptr<i32>,
    output: Ptr<f32>,
    num_q_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    valid_kv_len: i32,
    block_size: i32,
    scale: f32,
) {
    let _ = num_q_heads;
    let q_head = program_id(0);
    let num_kv_groups = num_q_heads / num_kv_heads;
    let kv_head = q_head / num_kv_groups;

    let kv_stride = num_kv_heads * head_dim;
    let block_stride = block_size * kv_stride;

    let dim_range = make_range(0, HEAD_DIM as i32);
    let dim_mask = dim_range < head_dim;
    let q_off = q_head * head_dim + dim_range;
    let q_v = load(q + q_off, dim_mask);

    let m_init = const_f32(0.0_f32) - const_f32(1.0e30_f32);
    let l_init = const_f32(0.0_f32);
    let acc_init = q_v * 0.0_f32;

    let kv_blocks = valid_kv_len / (BLOCK_KV as i32);

    let (_, l_i, acc) = scf_for(
        const_i32(0),
        kv_blocks,
        const_i32(1),
        (m_init, l_init, acc_init),
        |kb, (m_i, l_i, acc)| {
            let pos_base = kb * (BLOCK_KV as i32);
            let pos_range = make_range(0, BLOCK_KV as i32) + pos_base;

            // Address-translation gather: for each pos, look up
            // physical_block via block_table[pos / block_size]. The
            // logical block can repeat within a tile when block_size
            // is small — this still works (we just re-read the same
            // physical_block id for adjacent slots).
            let logical_blocks = pos_range / block_size; // [BLOCK_KV]
            let slots = pos_range % block_size;          // [BLOCK_KV]
            let physical_blocks = load(block_table + logical_blocks); // GATHER

            // Per-position row base: physical * block_stride + slot * kv_stride + kv_head*head_dim
            let row_base = physical_blocks * block_stride + slots * kv_stride + kv_head * head_dim;
            let row_base_2d = expand_dims(row_base, 1);
            let col_2d = expand_dims(dim_range, 0);
            let kv_off_2d = row_base_2d + col_2d;

            let k_block = load(k_block_pool + kv_off_2d);
            let v_block = load(v_block_pool + kv_off_2d);

            let q_2d = expand_dims(q_v, 0);
            let qk = q_2d * k_block;
            let scores_raw = reduce(qk, 1, |a, b| a + b);
            let scores = scores_raw * scale;

            let scores_max = reduce(scores, 0, |a, b| max(a, b));
            let m_ij = max(m_i, scores_max);
            let alpha = exp(m_i - m_ij);
            let p = exp(scores - m_ij);
            let l_ij = reduce(p, 0, |a, b| a + b);

            let p_2d = expand_dims(p, 1);
            let pv = p_2d * v_block;
            let pv_sum = reduce(pv, 0, |a, b| a + b);

            let new_acc = acc * alpha + pv_sum;
            let new_l_i = l_i * alpha + l_ij;
            (m_ij, new_l_i, new_acc)
        },
    );

    let out_v = acc / l_i;
    let out_off = q_head * head_dim + dim_range;
    store(output + out_off, out_v, dim_mask);
}

/// **Flash decode attention — Phase 1 (split-K)**.
///
/// Splits the KV stream across `num_splits` blocks per Q head. Each block
/// runs the same online-softmax over its slice and writes UNNORMALIZED
/// state (`partial_acc`, `partial_max`, `partial_sum`) to global memory.
/// Phase 2 then combines the per-split partials via log-sum-exp.
///
/// Shapes (caller pads `valid_kv_len` to `num_splits * BLOCK_KV`):
/// - `q`               : `[num_q_heads, head_dim]`
/// - `k_cache, v_cache`: `[seq_len, num_kv_heads, head_dim]`
/// - `partial_acc`     : `[num_q_heads, num_splits, head_dim]` f32
/// - `partial_max`     : `[num_q_heads, num_splits]`           f32
/// - `partial_sum`     : `[num_q_heads, num_splits]`           f32
///
/// Launch: grid = (num_q_heads, num_splits, 1).
#[triton_kernel]
pub fn flash_decode_attn_phase1_f32<const HEAD_DIM: usize, const BLOCK_KV: usize>(
    q: Ptr<f32>,
    k_cache: Ptr<f32>,
    v_cache: Ptr<f32>,
    partial_acc: Ptr<f32>,
    partial_max: Ptr<f32>,
    partial_sum: Ptr<f32>,
    num_q_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    valid_kv_len: i32,
    num_splits: i32,
    scale: f32,
) {
    let _ = num_q_heads;
    let q_head = program_id(0);
    let split_id = program_id(1);
    let num_kv_groups = num_q_heads / num_kv_heads;
    let kv_head = q_head / num_kv_groups;

    let dim_range = make_range(0, HEAD_DIM as i32);
    let dim_mask = dim_range < head_dim;
    let q_off = q_head * head_dim + dim_range;
    let q_v = load(q + q_off, dim_mask);

    let m_init = const_f32(0.0_f32) - const_f32(1.0e30_f32);
    let l_init = const_f32(0.0_f32);
    let acc_init = q_v * 0.0_f32;

    let kv_stride = num_kv_heads * head_dim;
    let chunk_blocks = (valid_kv_len / num_splits) / (BLOCK_KV as i32);
    let split_start_block = split_id * chunk_blocks;

    let (m_final, l_final, acc_final) = scf_for(
        const_i32(0),
        chunk_blocks,
        const_i32(1),
        (m_init, l_init, acc_init),
        |kb, (m_i, l_i, acc)| {
            let pos_base = (split_start_block + kb) * (BLOCK_KV as i32);
            let pos_range = make_range(0, BLOCK_KV as i32) + pos_base;

            let row_base = pos_range * kv_stride + kv_head * head_dim;
            let row_base_2d = expand_dims(row_base, 1);
            let col_2d = expand_dims(dim_range, 0);
            let kv_off_2d = row_base_2d + col_2d;

            let k_block = load(k_cache + kv_off_2d);
            let v_block = load(v_cache + kv_off_2d);

            let q_2d = expand_dims(q_v, 0);
            let qk = q_2d * k_block;
            let scores_raw = reduce(qk, 1, |a, b| a + b);
            let scores = scores_raw * scale;

            let scores_max = reduce(scores, 0, |a, b| max(a, b));
            let m_ij = max(m_i, scores_max);
            let alpha = exp(m_i - m_ij);
            let p = exp(scores - m_ij);
            let l_ij = reduce(p, 0, |a, b| a + b);

            let p_2d = expand_dims(p, 1);
            let pv = p_2d * v_block;
            let pv_sum = reduce(pv, 0, |a, b| a + b);

            let new_acc = acc * alpha + pv_sum;
            let new_l_i = l_i * alpha + l_ij;
            (m_ij, new_l_i, new_acc)
        },
    );

    // Write partials. partial_acc[(q_head, split_id)] is a head_dim row.
    let out_idx = q_head * num_splits + split_id;
    let partial_acc_off = out_idx * head_dim + dim_range;
    store(partial_acc + partial_acc_off, acc_final, dim_mask);
    store(partial_max + out_idx, m_final);
    store(partial_sum + out_idx, l_final);
}

/// **Flash decode attention — Phase 2 (combine splits)**.
///
/// Reduces the per-split partial state from Phase 1 into the final output.
/// Combines via log-sum-exp:
///   `global_m = max_s(m_s)`
///   `global_l = sum_s(l_s * exp(m_s - global_m))`
///   `output[d] = sum_s(acc_s[d] * exp(m_s - global_m)) / global_l`
///
/// Launch: grid = (num_q_heads,). One block per Q head; the block
/// streams over `num_splits` partials. `num_splits <= NUM_SPLITS_TILE`
/// is required (a constant tile width that bounds shared/register use).
#[triton_kernel]
pub fn flash_decode_attn_phase2_f32<
    const HEAD_DIM: usize,
    const NUM_SPLITS_TILE: usize,
>(
    partial_acc: Ptr<f32>,
    partial_max: Ptr<f32>,
    partial_sum: Ptr<f32>,
    output: Ptr<f32>,
    num_q_heads: i32,
    head_dim: i32,
    num_splits: i32,
) {
    let _ = num_q_heads;
    let q_head = program_id(0);

    let split_range = make_range(0, NUM_SPLITS_TILE as i32);
    let split_mask = split_range < num_splits;

    let base = q_head * num_splits + split_range;
    let m_s = load(partial_max + base, split_mask);
    let l_s = load(partial_sum + base, split_mask);

    let global_m = reduce(m_s, 0, |a, b| max(a, b));
    let alpha = exp(m_s - global_m);
    let weighted_l = l_s * alpha;
    let global_l = reduce(weighted_l, 0, |a, b| a + b);

    let weight = alpha / global_l;

    let dim_range = make_range(0, HEAD_DIM as i32);
    let dim_mask = dim_range < head_dim;
    let row_base = base * head_dim;
    let row_base_2d = expand_dims(row_base, 1);
    let col_2d = expand_dims(dim_range, 0);
    let acc_off_2d = row_base_2d + col_2d;
    let acc_block = load(partial_acc + acc_off_2d);

    let weight_2d = expand_dims(weight, 1);
    let weighted_acc = weight_2d * acc_block;
    let out_v = reduce(weighted_acc, 0, |a, b| a + b);

    let out_off = q_head * head_dim + dim_range;
    store(output + out_off, out_v, dim_mask);
}

/// **Batched flash decode attention — Phase 1 (batched split-K)**.
///
/// Combines [`batched_decode_attention_f32`]'s outer batch dimension
/// with [`flash_decode_attn_phase1_f32`]'s split-K. Used by ferrum's
/// long-context continuous-batch decode: many sequences in flight,
/// each with KV cache long enough to benefit from split-K.
///
/// Shapes:
/// - `q`               : `[batch, num_q_heads, head_dim]`
/// - `k_cache, v_cache`: `[batch, seq_len, num_kv_heads, head_dim]`
/// - `partial_acc`     : `[batch, num_q_heads, num_splits, head_dim]` f32
/// - `partial_max`     : `[batch, num_q_heads, num_splits]`           f32
/// - `partial_sum`     : `[batch, num_q_heads, num_splits]`           f32
///
/// Launch: grid = (batch * num_q_heads, num_splits, 1).
/// Phase 2 reuses [`flash_decode_attn_phase2_f32`] applied per
/// (batch, q_head); caller flattens the batch dim into num_q_heads
/// when invoking phase 2.
#[triton_kernel]
pub fn batched_flash_decode_attn_phase1_f32<const HEAD_DIM: usize, const BLOCK_KV: usize>(
    q: Ptr<f32>,
    k_cache: Ptr<f32>,
    v_cache: Ptr<f32>,
    partial_acc: Ptr<f32>,
    partial_max: Ptr<f32>,
    partial_sum: Ptr<f32>,
    batch_size: i32,
    num_q_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    seq_len: i32,
    valid_kv_len: i32,
    num_splits: i32,
    scale: f32,
) {
    let _ = batch_size;
    let pid = program_id(0);
    let split_id = program_id(1);
    let b = pid / num_q_heads;
    let q_head = pid % num_q_heads;
    let num_kv_groups = num_q_heads / num_kv_heads;
    let kv_head = q_head / num_kv_groups;

    let dim_range = make_range(0, HEAD_DIM as i32);
    let dim_mask = dim_range < head_dim;
    // Q row for this batch + head.
    let q_row_base = b * num_q_heads * head_dim + q_head * head_dim;
    let q_v = load(q + q_row_base + dim_range, dim_mask);

    let m_init = const_f32(0.0_f32) - const_f32(1.0e30_f32);
    let l_init = const_f32(0.0_f32);
    let acc_init = q_v * 0.0_f32;

    let kv_batch_base = b * seq_len * num_kv_heads * head_dim;
    let kv_stride = num_kv_heads * head_dim;
    let chunk_blocks = (valid_kv_len / num_splits) / (BLOCK_KV as i32);
    let split_start_block = split_id * chunk_blocks;

    let (m_final, l_final, acc_final) = scf_for(
        const_i32(0),
        chunk_blocks,
        const_i32(1),
        (m_init, l_init, acc_init),
        |kb, (m_i, l_i, acc)| {
            let pos_base = (split_start_block + kb) * (BLOCK_KV as i32);
            let pos_range = make_range(0, BLOCK_KV as i32) + pos_base;

            let row_base = pos_range * kv_stride + kv_head * head_dim + kv_batch_base;
            let row_base_2d = expand_dims(row_base, 1);
            let col_2d = expand_dims(dim_range, 0);
            let kv_off_2d = row_base_2d + col_2d;

            let k_block = load(k_cache + kv_off_2d);
            let v_block = load(v_cache + kv_off_2d);

            let q_2d = expand_dims(q_v, 0);
            let qk = q_2d * k_block;
            let scores_raw = reduce(qk, 1, |a, b| a + b);
            let scores = scores_raw * scale;

            let scores_max = reduce(scores, 0, |a, b| max(a, b));
            let m_ij = max(m_i, scores_max);
            let alpha = exp(m_i - m_ij);
            let p = exp(scores - m_ij);
            let l_ij = reduce(p, 0, |a, b| a + b);

            let p_2d = expand_dims(p, 1);
            let pv = p_2d * v_block;
            let pv_sum = reduce(pv, 0, |a, b| a + b);

            let new_acc = acc * alpha + pv_sum;
            let new_l_i = l_i * alpha + l_ij;
            (m_ij, new_l_i, new_acc)
        },
    );

    // out_idx in flattened (batch, q_head, split) layout.
    let bh = b * num_q_heads + q_head;
    let out_idx = bh * num_splits + split_id;
    let partial_acc_off = out_idx * head_dim + dim_range;
    store(partial_acc + partial_acc_off, acc_final, dim_mask);
    store(partial_max + out_idx, m_final);
    store(partial_sum + out_idx, l_final);
}

/// **Full flash attention — prefill / autoregressive with causal mask**.
///
/// Multi-token Q version of decode_attention. Used for prefill
/// (initial sequence pass) and full-sequence autoregressive. Each block
/// handles a `BLOCK_Q`-wide tile of query positions; within the block,
/// the canonical online-softmax loop sweeps the KV stream in
/// `BLOCK_KV` tiles.
///
/// Causal mask: position `i` can only attend to positions `≤ pos_offset + i`.
/// Implemented via score-arithmetic (no `arith.select` in DSL yet):
///   `scores += (mask_f32 - 1) * 1e30` where `mask_f32` is 1 for allowed,
///   0 for blocked. Blocked positions get -1e30, which softmax flushes
///   to ~0. (Once DSL has `arith.select`, swap to a cleaner `where`.)
///
/// Shapes (one block per (batch, head, q_tile)):
/// - `Q`: `[batch, num_heads, q_len, head_dim]`
/// - `K, V`: `[batch, num_kv_heads, kv_len, head_dim]`
/// - `O`: `[batch, num_heads, q_len, head_dim]`
///
/// Launch: grid = (ceil(q_len / BLOCK_Q), num_heads, batch).
/// v0 assumes `q_len % BLOCK_Q == 0`, `kv_len % BLOCK_KV == 0`,
/// `head_dim == HEAD_DIM`.
///
/// **Internal compute is f32 regardless of T.** Q/K/V are loaded at T's
/// element type then upcast via `to_f32`; m_i / l_i / acc / scores all
/// run in f32 throughout the loop; the final accumulator is downcast
/// back to T via `as_t::<T>(out_v)` at the store boundary. This is
/// required for f16 because NVPTX has no native f16 division or
/// `math.exp` instructions, and matches Python @triton.jit's strategy
/// for mixed-precision attention.
#[triton_kernel]
pub fn flash_attn_full<
    T: TritonElem,
    const HEAD_DIM: usize,
    const BLOCK_Q: usize,
    const BLOCK_KV: usize,
>(
    q: Ptr<T>,
    k: Ptr<T>,
    v: Ptr<T>,
    output: Ptr<T>,
    num_heads: i32,
    num_kv_heads: i32,
    q_len: i32,
    kv_len: i32,
    head_dim: i32,
    pos_offset: i32,
    scale: f32,
) {
    let _ = num_heads;
    let q_tile_id = program_id(0);
    let head = program_id(1);
    let batch = program_id(2);

    let num_kv_groups = num_heads / num_kv_heads;
    let kv_head = head / num_kv_groups;

    let dim_range = make_range(0, HEAD_DIM as i32);
    let dim_mask = dim_range < head_dim;
    let q_pos_range = make_range(0, BLOCK_Q as i32) + q_tile_id * (BLOCK_Q as i32);

    // Per-batch, per-head Q row base: b*(num_heads*q_len*head_dim) + h*(q_len*head_dim)
    let q_batch_base = batch * num_heads * q_len * head_dim + head * q_len * head_dim;
    // Q tile [BLOCK_Q, HEAD_DIM] — load at T then immediately upcast to
    // f32 so all downstream math stays in f32 (see kernel doc-comment).
    let q_row_base = q_pos_range * head_dim + q_batch_base;
    let q_row_2d = expand_dims(q_row_base, 1);
    let dim_2d = expand_dims(dim_range, 0);
    let q_off_2d = q_row_2d + dim_2d;
    let q_tile_t = load(q + q_off_2d);
    let q_tile = to_f32(q_tile_t);

    // KV tile pointer base (per-batch, per-kv-head).
    let kv_batch_base = batch * num_kv_heads * kv_len * head_dim
        + kv_head * kv_len * head_dim;
    let kv_blocks = kv_len / (BLOCK_KV as i32);

    // Online softmax state: m_i [BLOCK_Q], l_i [BLOCK_Q], acc [BLOCK_Q, HEAD_DIM].
    // Build init tensors by multiplying q_tile by 0 — q_tile is f32 here
    // so the init state inherits f32 element type, regardless of T.
    let q_col0_t = load(q + q_pos_range * head_dim + q_batch_base);
    let q_col0 = to_f32(q_col0_t); // [BLOCK_Q] f32
    let m_i_init = q_col0 * 0.0_f32 - 1.0e30_f32;
    let l_i_init = q_col0 * 0.0_f32;
    let acc_init = q_tile * 0.0_f32; // [BLOCK_Q, HEAD_DIM] f32

    let (_, l_i, acc) = scf_for(
        const_i32(0),
        kv_blocks,
        const_i32(1),
        (m_i_init, l_i_init, acc_init),
        |kb, (m_i, l_i, acc)| {
            let kv_pos_base = kb * (BLOCK_KV as i32);
            let kv_pos_range = make_range(0, BLOCK_KV as i32) + kv_pos_base;

            // K tile [BLOCK_KV, HEAD_DIM] — load at T then upcast.
            let k_row_base = kv_pos_range * head_dim + kv_batch_base;
            let k_row_2d = expand_dims(k_row_base, 1);
            let k_off_2d = k_row_2d + dim_2d;
            let k_tile_t = load(k + k_off_2d);
            let v_tile_t = load(v + k_off_2d);
            let k_tile = to_f32(k_tile_t);
            let v_tile = to_f32(v_tile_t);

            // scores[BLOCK_Q, BLOCK_KV] = Q_tile[BLOCK_Q, HD] · K_tile[BLOCK_KV, HD]^T
            // Computed as broadcast-mul-reduce on a 3D intermediate:
            //   q_3d: [BLOCK_Q, 1, HD] ; k_3d: [1, BLOCK_KV, HD]
            //   qk_3d = q_3d * k_3d → [BLOCK_Q, BLOCK_KV, HD]
            //   scores = reduce(qk_3d, dim=2)  → [BLOCK_Q, BLOCK_KV]
            // Real production may swap to tt.dot for HEAD_DIM ≥ 64.
            let q_3d = expand_dims(q_tile, 1); // [BLOCK_Q, 1, HEAD_DIM]
            let k_3d = expand_dims(k_tile, 0); // [1, BLOCK_KV, HEAD_DIM]
            let qk = q_3d * k_3d; // [BLOCK_Q, BLOCK_KV, HEAD_DIM]
            let scores_raw = reduce(qk, 2, |a, b| a + b); // [BLOCK_Q, BLOCK_KV]
            let scores_scaled = scores_raw * scale;

            // ── causal mask ──
            // For each (q, k) in tile: allow if (kv_pos_base + k) <= (pos_offset + q_pos_range[q])
            let q_pos_2d = expand_dims(q_pos_range + pos_offset, 1); // [BLOCK_Q, 1]
            let k_pos_2d = expand_dims(kv_pos_range, 0);              // [1, BLOCK_KV]
            let causal_mask = k_pos_2d <= q_pos_2d;                   // [BLOCK_Q, BLOCK_KV] i1
            let mask_f = to_f32(causal_mask);
            let masked_scores = scores_scaled + (mask_f - 1.0_f32) * 1.0e30_f32;

            // ── online softmax (per-row) ──
            let row_max = reduce(masked_scores, 1, |a, b| max(a, b)); // [BLOCK_Q]
            let m_ij = max(m_i, row_max);
            let alpha = exp(m_i - m_ij);
            let p = exp(masked_scores - expand_dims(m_ij, 1)); // [BLOCK_Q, BLOCK_KV]
            let l_ij = reduce(p, 1, |a, b| a + b);              // [BLOCK_Q]

            // ── pv accumulation: acc[q, d] += p[q, k] * V[k, d] summed over k ──
            // p_3d: [BLOCK_Q, BLOCK_KV, 1] ; v_3d: [1, BLOCK_KV, HEAD_DIM]
            let p_3d = expand_dims(p, 2);
            let v_3d = expand_dims(v_tile, 0);
            let pv = p_3d * v_3d;                                // [BLOCK_Q, BLOCK_KV, HEAD_DIM]
            let pv_sum = reduce(pv, 1, |a, b| a + b);           // [BLOCK_Q, HEAD_DIM]

            let alpha_2d = expand_dims(alpha, 1);
            let new_acc = acc * alpha_2d + pv_sum;
            let new_l_i = l_i * alpha + l_ij;
            (m_ij, new_l_i, new_acc)
        },
    );

    // ── normalize + write ──
    // out_v_f32 is f32 (acc and l_i_2d are both f32). Downcast to T via
    // as_t::<T>(...) so the store matches the output pointer's element
    // type. For T == f32 the cast is a no-op.
    let l_i_2d = expand_dims(l_i, 1);
    let out_v_f32 = acc / l_i_2d;
    let out_v = as_t::<T>(out_v_f32);
    let out_row_base = q_pos_range * head_dim + q_batch_base;
    let out_row_2d = expand_dims(out_row_base, 1);
    let out_off_2d = out_row_2d + dim_2d;
    // v0: assume head_dim==HEAD_DIM so the per-column mask is unused;
    // emitting a 1D mask on a 2D ptr trips Triton's verifier. Real
    // partial-head masking needs a 2D broadcast of dim_mask.
    let _ = dim_mask;
    store(output + out_off_2d, out_v);
}
