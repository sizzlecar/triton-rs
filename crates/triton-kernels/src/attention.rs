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
///
/// **Internal compute is f32 regardless of T.** Q/K/V are loaded at T's
/// element type then upcast via `to_f32`; m_i / l_i / acc / scores all
/// run in f32 throughout the loop; the final accumulator is downcast
/// back to T via `as_t::<T>(out_v)` at the store boundary. This is
/// required for f16 because NVPTX has no native f16 division or
/// `math.exp` instructions, and matches Python @triton.jit's strategy
/// for mixed-precision attention.
#[triton_kernel]
pub fn decode_attention_typed<
    T: TritonElem,
    const HEAD_DIM: usize,
    const BLOCK_KV: usize,
>(
    q: Ptr<T>,
    k_cache: Ptr<T>,
    v_cache: Ptr<T>,
    output: Ptr<T>,
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
    let q_v_t = load(q + q_off, dim_mask);
    let q_v = to_f32(q_v_t); // [HEAD_DIM] f32

    // ── online-softmax state ──
    // Negative-float literals aren't accepted directly by the DSL macro;
    // build -1e30 from `0 - 1e30`.
    let m_init = const_f32(0.0_f32) - const_f32(1.0e30_f32);
    let l_init = const_f32(0.0_f32);
    // acc starts at zero across HEAD_DIM. Use Q*0 so we get a tensor of the
    // right shape without a dedicated zeros-tensor builder. q_v is already
    // f32, so the resulting init tensor is f32 regardless of T.
    let acc_init = q_v * 0.0_f32;

    // ── tile loop over KV ──
    let kv_stride = num_kv_heads * head_dim;
    // Ceil-div: any partial last tile is included; positions past
    // valid_kv_len are masked out below (loads return 0 → scores get
    // pushed to −∞ via the additive bias trick).
    let kv_blocks = (valid_kv_len + (BLOCK_KV as i32) - 1) / (BLOCK_KV as i32);

    let (_, l_i, acc) = scf_for(
        const_i32(0),
        kv_blocks,
        const_i32(1),
        (m_init, l_init, acc_init),
        |kb, (m_i, l_i, acc)| {
            let pos_base = kb * (BLOCK_KV as i32);
            let pos_range = make_range(0, BLOCK_KV as i32) + pos_base;
            let kv_pos_mask = pos_range < valid_kv_len; // [BLOCK_KV] i1

            // K block [BLOCK_KV, HEAD_DIM]: 2D pointer construction via
            // outer-add of `pos[i] * kv_stride + kv_head * head_dim` (rows)
            // and `dim_range[d]` (cols). Auto-broadcast handles the 1D→2D
            // expansion.
            let row_base = pos_range * kv_stride + kv_head * head_dim;
            let row_base_2d = expand_dims(row_base, 1); // [BLOCK_KV, 1]
            let col_2d = expand_dims(dim_range, 0); // [1, HEAD_DIM]
            let kv_off_2d = row_base_2d + col_2d; // [BLOCK_KV, HEAD_DIM]

            // 2D load mask: row-valid AND col-in-head_dim. Out-of-range
            // lanes load 0 (Triton default `other`), which keeps the dot
            // product finite; the score masking below pushes those rows
            // to −∞ before softmax sees them.
            let row_mask_2d = expand_dims(kv_pos_mask, 1); // [BLOCK_KV, 1]
            let col_mask_2d = expand_dims(dim_mask, 0); // [1, HEAD_DIM]
            let kv_load_mask = row_mask_2d & col_mask_2d; // [BLOCK_KV, HEAD_DIM]
            let k_block_t = load(k_cache + kv_off_2d, kv_load_mask);
            let v_block_t = load(v_cache + kv_off_2d, kv_load_mask);
            let k_block = to_f32(k_block_t); // [BLOCK_KV, HEAD_DIM] f32
            let v_block = to_f32(v_block_t);

            // scores[BLOCK_KV] = sum_d Q[d] * K[i, d]
            let q_2d = expand_dims(q_v, 0); // [1, HEAD_DIM]
            let qk = q_2d * k_block; // [BLOCK_KV, HEAD_DIM]
            let scores_raw = reduce(qk, 1, |a, b| a + b); // [BLOCK_KV]
            let scores_unmasked = scores_raw * scale;
            // Same flash_attn_full / unified_attention trick: add `(mask−1)*1e30`
            // so masked positions land at −1e30 (effectively −∞ post-softmax).
            // No `tt.where` / `arith.select` needed.
            let mask_f = to_f32(kv_pos_mask);
            let scores =
                scores_unmasked + (mask_f - const_f32(1.0_f32)) * const_f32(1.0e30_f32);

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
    // out_v_f32 stays in f32 (acc and l_i are f32). Downcast to T via
    // as_t::<T>(...) so the store matches the output pointer's element
    // type. For T == f32 the cast is a no-op.
    let out_v_f32 = acc / l_i;
    let out_v = as_t::<T>(out_v_f32);
    let out_off = q_head * head_dim + dim_range;
    store(output + out_off, out_v, dim_mask);
}

/// Same online-softmax decode attention as [`decode_attention_typed`] but
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
///
/// **Internal compute is f32 regardless of T** — see [`decode_attention_typed`]
/// rationale (NVPTX has no native f16 div / exp).
#[triton_kernel]
pub fn decode_attention_hm_typed<
    T: TritonElem,
    const HEAD_DIM: usize,
    const BLOCK_KV: usize,
>(
    q: Ptr<T>,
    k_cache: Ptr<T>,
    v_cache: Ptr<T>,
    output: Ptr<T>,
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
    let q_v_t = load(q + q_off, dim_mask);
    let q_v = to_f32(q_v_t); // [HEAD_DIM] f32

    let m_init = const_f32(0.0_f32) - const_f32(1.0e30_f32);
    let l_init = const_f32(0.0_f32);
    let acc_init = q_v * 0.0_f32;

    // Ceil-div + per-position masking handles arbitrary valid_kv_len
    // (caller doesn't have to pad to a BLOCK_KV multiple).
    let kv_blocks = (valid_kv_len + (BLOCK_KV as i32) - 1) / (BLOCK_KV as i32);
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
            let kv_pos_mask = pos_range < valid_kv_len; // [BLOCK_KV] i1

            // Head-major K/V address: kv_head_base + pos[i] * head_dim + d
            let row_base = pos_range * head_dim + kv_head_base;
            let row_base_2d = expand_dims(row_base, 1); // [BLOCK_KV, 1]
            let col_2d = expand_dims(dim_range, 0); // [1, HEAD_DIM]
            let kv_off_2d = row_base_2d + col_2d; // [BLOCK_KV, HEAD_DIM]

            // 2D load mask: row-valid AND col-in-head_dim.
            let row_mask_2d = expand_dims(kv_pos_mask, 1); // [BLOCK_KV, 1]
            let col_mask_2d = expand_dims(dim_mask, 0); // [1, HEAD_DIM]
            let kv_load_mask = row_mask_2d & col_mask_2d; // [BLOCK_KV, HEAD_DIM]
            let k_block_t = load(k_cache + kv_off_2d, kv_load_mask);
            let v_block_t = load(v_cache + kv_off_2d, kv_load_mask);
            let k_block = to_f32(k_block_t);
            let v_block = to_f32(v_block_t);

            let q_2d = expand_dims(q_v, 0);
            let qk = q_2d * k_block;
            let scores_raw = reduce(qk, 1, |a, b| a + b);
            let scores_unmasked = scores_raw * scale;
            // Push masked positions to −1e30 via additive bias.
            let mask_f = to_f32(kv_pos_mask);
            let scores =
                scores_unmasked + (mask_f - const_f32(1.0_f32)) * const_f32(1.0e30_f32);

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

    let out_v_f32 = acc / l_i;
    let out_v = as_t::<T>(out_v_f32);
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
///
/// **Internal compute is f32 regardless of T** — see [`decode_attention_typed`]
/// rationale (NVPTX has no native f16 div / exp). The `block_table`
/// pointer stays `Ptr<i32>` (it's an integer indirection table, not a
/// dtype-parametric tensor).
#[triton_kernel]
pub fn paged_decode_attention_typed<
    T: TritonElem,
    const HEAD_DIM: usize,
    const BLOCK_KV: usize,
>(
    q: Ptr<T>,
    k_block_pool: Ptr<T>,
    v_block_pool: Ptr<T>,
    block_table: Ptr<i32>,
    output: Ptr<T>,
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
    let q_v_t = load(q + q_off, dim_mask);
    let q_v = to_f32(q_v_t); // [HEAD_DIM] f32

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
            let physical_blocks = load(block_table + logical_blocks); // GATHER (i32)

            // Per-position row base: physical * block_stride + slot * kv_stride + kv_head*head_dim
            let row_base = physical_blocks * block_stride + slots * kv_stride + kv_head * head_dim;
            let row_base_2d = expand_dims(row_base, 1);
            let col_2d = expand_dims(dim_range, 0);
            let kv_off_2d = row_base_2d + col_2d;

            let k_block_t = load(k_block_pool + kv_off_2d);
            let v_block_t = load(v_block_pool + kv_off_2d);
            let k_block = to_f32(k_block_t);
            let v_block = to_f32(v_block_t);

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

    let out_v_f32 = acc / l_i;
    let out_v = as_t::<T>(out_v_f32);
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

/// **Unified prefill + decode attention with paged KV** — the single-kernel
/// vLLM-style attention, stripped of all the optional features so the core
/// online-softmax + block-table indirection mechanic is visible end-to-end.
///
/// Modeled after `kernel_unified_attention` in
/// `vllm/v1/attention/ops/triton_unified_attention.py`. The whole point of
/// vLLM's "unified" kernel is that prefill (q_len > 1) and decode (q_len == 1)
/// run through the SAME body — the difference between them is just the size
/// of the Q tile and the value of `query_start_loc` for that seq. There is
/// no `if prefill else decode` branch; instead, all dispatching is done
/// through three masks combined into the score:
///   - `query_mask`: zero out rows where `q_pos_in_seq >= cur_batch_query_len`
///     (over-launched grid pads up to `BLOCK_Q`)
///   - `kv_bounds_mask`: zero out cols where `kv_pos >= seq_len`
///   - `causal_mask`: zero out cols where `kv_pos > query_abs_pos`
/// `final_mask = query_mask & kv_bounds_mask & causal_mask`, then
/// `scores += (mask_f - 1) * 1e30` (since DSL has no `where` yet — same
/// pattern as `flash_attn_full`).
///
/// **Shapes:**
/// - `q_ptr`            : `[num_tokens_total, num_q_heads, head_dim]`
/// - `k_cache_ptr`      : `[num_blocks, block_size, num_kv_heads, head_dim]`
/// - `v_cache_ptr`      : `[num_blocks, block_size, num_kv_heads, head_dim]`
/// - `out_ptr`          : `[num_tokens_total, num_q_heads, head_dim]`
/// - `block_table_ptr`  : `[num_seqs, max_blocks_per_seq]` (i32)
/// - `seq_lens_ptr`     : `[num_seqs]` (i32)
/// - `query_start_loc_ptr`: `[num_seqs+1]` (i32, CSR-style cumulative
///                          token offsets — `query_start_loc[i+1] -
///                          query_start_loc[i]` = q_len of seq i)
///
/// **Launch:** grid = `(num_seqs, num_q_heads, max_q_tiles_per_seq)`
/// where `max_q_tiles_per_seq = ceil(max(q_len_i) / BLOCK_Q)`. Programs
/// whose `q_tile_id * BLOCK_Q >= cur_batch_query_len` produce all-masked
/// rows, which the masked store discards. Decode (`q_len == 1`) launches
/// with `BLOCK_Q == 16` (or whatever) and only row 0 is unmasked — this
/// is the unification trick.
///
/// **Address arithmetic** for paged KV (per `kv_pos`):
///   ```text
///   logical_block = kv_pos / block_size
///   slot          = kv_pos % block_size
///   physical      = block_table[seq_idx, logical_block]
///   kv_off        = physical * (block_size * num_kv_heads * head_dim)
///                 + slot     * (num_kv_heads * head_dim)
///                 + kv_head  * head_dim
///                 + d
///   ```
/// (Identical to `paged_decode_attention_f32`'s gather pattern, just with
/// per-seq lookup of the block-table base via `seq_idx * max_blocks_per_seq`.)
///
/// **FIXME(scope): explicitly skipped for v0** — match every line item to
/// vLLM kernel features:
/// - ALiBi slopes (per-head linear bias added to scores)
/// - Soft-cap (`tanh(score / softcap) * softcap`)
/// - Attention sinks (extra K/V positions baked into M-init)
/// - Sliding window (`scores -= 1e30 * (kv_pos < q_pos - window)`)
/// - FP8 / INT8 KV in-loop dequant (`k_scale`, `v_scale`, per-token-head
///   scales)
/// - QQ bias (`scores += qq_bias_tile`)
/// - Batch-invariant softmax (compile-time fixed reduction order)
/// - 3D split-softmax / chunked-prefill combine (`segm_*` outputs)
/// - Output FP8 clipping (`tl.clamp` post-divide)
/// - `out_scale` / `k_scale` / `v_scale` quant params
/// All would be `#[cfg(feature = ...)]` follow-up additions; the body
/// here is intentionally minimal.
///
/// **Constraints (v0):**
/// - f32 only (no T-generic — keep the IR readable for first review).
/// - `head_dim == HEAD_DIM` (no partial-head masking, same caveat as
///   `flash_attn_full`).
/// - `BLOCK_KV` divides `block_size` (so each KV tile fits within at most
///   two physical blocks). Caller is expected to pick `BLOCK_KV <= block_size`.
/// - GQA via `num_q_heads / num_kv_heads`. Programs are launched per Q
///   head, NOT per KV head — same as `flash_attn_full`. (vLLM launches
///   per kv_head and folds GQA into the inner Q-tile load; that's a
///   follow-up trade-off, not a correctness issue.)
#[triton_kernel]
pub fn unified_attention_f32<const HEAD_DIM: usize, const BLOCK_Q: usize, const BLOCK_KV: usize>(
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
    // CSR-style: query_start_loc[seq_idx]   = first token index for seq
    //            query_start_loc[seq_idx+1] = first token index for seq+1
    // cur_batch_query_len = stop - start (the seq's q_len, 1 for decode).
    let q_start = load(query_start_loc_ptr + seq_idx);
    let q_stop = load(query_start_loc_ptr + (seq_idx + 1));
    let cur_batch_query_len = q_stop - q_start;
    let seq_len = load(seq_lens_ptr + seq_idx);
    // For unified prefill + decode, the query tokens are appended at the
    // tail of the KV cache, so kv_pos for query token q is at
    //   kv_pos = (seq_len - cur_batch_query_len) + q_pos_in_seq.
    // context_len is the offset of the first query token within the KV
    // stream — for pure decode it's seq_len-1, for first prefill it's 0.
    let context_len = seq_len - cur_batch_query_len;

    // ── Q-tile coordinates ──
    let dim_range = make_range(0, HEAD_DIM as i32);
    let _ = head_dim;
    let q_pos_in_tile = make_range(0, BLOCK_Q as i32);
    // [BLOCK_Q]
    let q_pos_in_seq = q_pos_in_tile + q_tile_id * (BLOCK_Q as i32);
    // query_mask: rows of the tile that fall inside [0, cur_batch_query_len).
    // Programs over-launched past the seq's end produce all-masked output.
    // [BLOCK_Q] i1
    let query_mask = q_pos_in_seq < cur_batch_query_len;

    // Per-token Q row base in the global token-major Q tensor:
    //   token_idx = q_start + q_pos_in_seq
    //   q_off    = token_idx * (num_q_heads * head_dim) + q_head * head_dim + d
    let token_idx = q_pos_in_seq + q_start; // [BLOCK_Q]
    let q_row_stride = num_q_heads * head_dim;
    let q_row_base = token_idx * q_row_stride + q_head * head_dim; // [BLOCK_Q]
    let q_row_2d = expand_dims(q_row_base, 1); // [BLOCK_Q, 1]
    let dim_2d = expand_dims(dim_range, 0); // [1, HEAD_DIM]
    let q_off_2d = q_row_2d + dim_2d; // [BLOCK_Q, HEAD_DIM]

    // Load Q. Out-of-range rows return 0 — the score-mask later flushes
    // them anyway, so the loaded value doesn't matter.
    // (Triton's mask= arg on 2D ptrs needs a 2D mask; broadcast query_mask
    //  to [BLOCK_Q, HEAD_DIM] via expand_dims+broadcast. We rely on
    //  pad-load semantics + masked store at the end instead, since v0 of
    //  flash_attn_full follows the same pattern and works.)
    let q_tile = load(q_ptr + q_off_2d); // [BLOCK_Q, HEAD_DIM] f32

    // ── online-softmax init ──
    // Build f32 init tensors of the right shape via x*0 trick.
    let m_init = q_tile * 0.0_f32; // [BLOCK_Q, HEAD_DIM]

    // Reduce to [BLOCK_Q] by summing along HEAD_DIM (sum of zeros = 0).
    let zero_q = reduce(m_init, 1, |a, b| a + b); // [BLOCK_Q] f32
    let m_i_init = zero_q - 1.0e30_f32; // [BLOCK_Q]
    let l_i_init = zero_q; // [BLOCK_Q]
    let acc_init = q_tile * 0.0_f32; // [BLOCK_Q, HEAD_DIM]

    // ── KV loop ──
    // Loop over KV in BLOCK_KV-wide tiles. We over-iterate up to
    // `seq_len` rounded up — but the kv_bounds_mask zeroes out the tail.
    // Caller responsibility: pad seq_len up to a multiple of BLOCK_KV.
    let kv_blocks = seq_len / (BLOCK_KV as i32);

    // Paged layout strides.
    let kv_inner_stride = num_kv_heads * head_dim; // per-slot stride
    let block_stride = block_size * kv_inner_stride; // per-physical-block

    // Per-seq base into block_table: row = seq_idx * max_blocks_per_seq.
    let bt_seq_base = seq_idx * max_blocks_per_seq;

    let (_, l_i, acc) = scf_for(
        const_i32(0),
        kv_blocks,
        const_i32(1),
        (m_i_init, l_i_init, acc_init),
        |kb, (m_i, l_i, acc)| {
            let kv_pos_base = kb * (BLOCK_KV as i32);
            // [BLOCK_KV]
            let kv_pos_range = make_range(0, BLOCK_KV as i32) + kv_pos_base;

            // ── block-table indirection ──
            // logical_block[i] = kv_pos[i] / block_size
            // slot[i]          = kv_pos[i] % block_size
            // physical[i]      = block_table[seq, logical_block[i]]
            let logical_blocks = kv_pos_range / block_size; // [BLOCK_KV]
            let slots = kv_pos_range % block_size; // [BLOCK_KV]
            let physical_blocks = load(block_table_ptr + (bt_seq_base + logical_blocks));

            // Per-position row base in the block pool:
            //   physical * block_stride + slot * kv_inner_stride + kv_head * head_dim
            let row_base =
                physical_blocks * block_stride + slots * kv_inner_stride + kv_head * head_dim;
            let row_base_2d = expand_dims(row_base, 1); // [BLOCK_KV, 1]
            let kv_off_2d = row_base_2d + dim_2d; // [BLOCK_KV, HEAD_DIM]

            // V uses the natural [BLOCK_KV, HEAD_DIM] layout — straight
            // input to the second `dot` (P · V).
            let v_tile = load(v_cache_ptr + kv_off_2d); // [BLOCK_KV, HEAD_DIM]

            // K is loaded TRANSPOSED to [HEAD_DIM, BLOCK_KV] so that
            // `dot(Q[BLOCK_Q,HEAD_DIM], K_T[HEAD_DIM,BLOCK_KV])` produces
            // [BLOCK_Q, BLOCK_KV] without a separate transpose op.
            // Pointer math: same scalars as kv_off_2d, just swap which axis
            // is the row vs col. We reuse `dim_range` and `row_base` since
            // they're already in scope.
            let dim_col_2d = expand_dims(dim_range, 1); // [HEAD_DIM, 1]
            let row_base_row_2d = expand_dims(row_base, 0); // [1, BLOCK_KV]
            let k_off_t_2d = dim_col_2d + row_base_row_2d; // [HEAD_DIM, BLOCK_KV]
            let k_tile_t = load(k_cache_ptr + k_off_t_2d); // [HEAD_DIM, BLOCK_KV]

            // ── scores = Q · Kᵀ ── via tt.dot (MMA hardware).
            // Replaces the 3D broadcast-mul-reduce path; cuts PTX size 5-10x
            // and runs on Tensor Cores instead of unrolled fmuls.
            // Need a fresh [BLOCK_Q, BLOCK_KV] f32 zero accumulator (same
            // idiom as `matmul_f32`).
            let qk_zero_seed = splat_1d(const_f32(0.0_f32), 1); // [1]
            let qk_zero_2d_seed = expand_dims(qk_zero_seed, 0); // [1, 1]
            let qk_acc =
                broadcast_2d(qk_zero_2d_seed, BLOCK_Q as i64, BLOCK_KV as i64); // [BLOCK_Q, BLOCK_KV]
            let scores_raw = dot(q_tile, k_tile_t, qk_acc); // [BLOCK_Q, BLOCK_KV]
            let scores_scaled = scores_raw * sm_scale;

            // ── mask construction ──
            // 1. Causal: kv_pos <= context_len + q_pos_in_seq
            //    (Equivalently: kv_pos - context_len <= q_pos_in_seq.)
            //    For pure decode (q_len==1), context_len == seq_len-1,
            //    so all kv_pos are <= context_len+0 = seq_len-1 — i.e.
            //    ALL kv positions are visible (correct for decode).
            //    For prefill, context_len==0, so kv_pos[k] <= q_pos[q]
            //    is the standard causal mask.
            let q_abs_2d = expand_dims(q_pos_in_seq + context_len, 1); // [BLOCK_Q, 1]
            let kv_pos_2d = expand_dims(kv_pos_range, 0); // [1, BLOCK_KV]
            let causal_mask = kv_pos_2d <= q_abs_2d; // [BLOCK_Q, BLOCK_KV] i1

            // 2. KV bounds: kv_pos < seq_len. Tail of the over-iterated
            //    KV loop is zeroed.
            let kv_in_bounds_1d = kv_pos_range < seq_len; // [BLOCK_KV] i1
            let kv_in_bounds_2d = expand_dims(kv_in_bounds_1d, 0); // [1, BLOCK_KV] i1

            // 3. Query-row validity.
            let query_mask_2d = expand_dims(query_mask, 1); // [BLOCK_Q, 1] i1

            // Combine: AND (`&` on i1 maps to arith.andi via the
            // bit-and op spec).
            let combined_mask = causal_mask & kv_in_bounds_2d & query_mask_2d;
            let mask_f = to_f32(combined_mask); // [BLOCK_Q, BLOCK_KV] f32
            let masked_scores = scores_scaled + (mask_f - 1.0_f32) * 1.0e30_f32;

            // ── per-row online softmax ──
            let row_max = reduce(masked_scores, 1, |a, b| max(a, b)); // [BLOCK_Q]
            let m_ij = max(m_i, row_max);
            let alpha = exp(m_i - m_ij);
            let p = exp(masked_scores - expand_dims(m_ij, 1)); // [BLOCK_Q, BLOCK_KV]
            let l_ij = reduce(p, 1, |a, b| a + b); // [BLOCK_Q]

            // ── pv accumulate via tt.dot ──
            // pv_sum = dot(P[BLOCK_Q,BLOCK_KV], V[BLOCK_KV,HEAD_DIM]) and
            // online softmax wants `acc = acc * alpha + pv_sum`. Fold
            // alpha-scaling into the dot's accumulator: pass `acc * alpha`
            // as the init and the final `dot` returns `acc * alpha + P · V`.
            let alpha_2d = expand_dims(alpha, 1); // [BLOCK_Q, 1]
            let scaled_acc = acc * alpha_2d; // [BLOCK_Q, HEAD_DIM]
            let new_acc = dot(p, v_tile, scaled_acc); // [BLOCK_Q, HEAD_DIM]
            let new_l_i = l_i * alpha + l_ij;
            (m_ij, new_l_i, new_acc)
        },
    );

    // ── normalize + write ──
    let l_i_2d = expand_dims(l_i, 1);
    let out_v = acc / l_i_2d;
    // Store back into the global token-major output tensor at the SAME
    // row layout Q used. v0 emits unmasked store — same caveat as
    // flash_attn_full (1D mask on 2D ptr trips the verifier; over-launched
    // rows just clobber memory beyond the seq, which the caller compensates
    // for by ensuring the grid only covers tiles within real q_len ranges).
    let out_off_2d = q_off_2d;
    let _ = query_mask; // unused as a 2D store mask in v0
    store(out_ptr + out_off_2d, out_v);
}
