//! Projection helpers — splitting combined QKV outputs and reshape /
//! transpose between layout conventions, plus the LLM-style "fused
//! norm + RoPE + transpose" kernels that ferrum uses on the Q / K
//! pre-attention path.

use triton_dsl::triton_kernel;

#[allow(dead_code)]
struct Ptr<T>(::std::marker::PhantomData<T>);

/// Split a combined QKV tensor into separate Q, K, V outputs.
/// Input layout: `[batch, 3, num_heads, head_dim]` (interleaved Q/K/V
/// per head). Output: each is `[batch, num_heads, head_dim]`.
///
/// One block per `(batch * num_heads + head)` slot. Requires
/// `head_dim <= BLOCK`. Replaces 3 `gather`/`view` style launches with 1.
#[triton_kernel]
pub fn split_qkv_f32<const BLOCK: usize>(
    qkv: Ptr<f32>,
    q_out: Ptr<f32>,
    k_out: Ptr<f32>,
    v_out: Ptr<f32>,
    num_heads: i32,
    head_dim: i32,
) {
    let bh = program_id(0);
    let b = bh / num_heads;
    let h = bh % num_heads;

    let cols = make_range(0, BLOCK as i32);
    let mask = cols < head_dim;

    let head_block = num_heads * head_dim;
    let batch_block = 3 * head_block;
    let head_off = h * head_dim + cols;

    let q_in_off = b * batch_block + 0 * head_block + head_off;
    let k_in_off = b * batch_block + 1 * head_block + head_off;
    let v_in_off = b * batch_block + 2 * head_block + head_off;
    let out_off = b * head_block + h * head_dim + cols;

    let q = load(qkv + q_in_off, mask);
    let k = load(qkv + k_in_off, mask);
    let v = load(qkv + v_in_off, mask);

    store(q_out + out_off, q, mask);
    store(k_out + out_off, k, mask);
    store(v_out + out_off, v, mask);
}

/// Transpose `[heads, tokens, head_dim]` -> `[tokens, heads, head_dim]`.
/// Inverse of the qk_norm_rope-style head-major layout — runs after
/// flash attention to restore token-major for the O-projection GEMM.
///
/// Mirrors ferrum's `transpose.cu` but tile-based: one block per
/// `(head, tok)` pair, all threads in the block move that whole row.
/// Requires `head_dim <= BLOCK`.
#[triton_kernel]
pub fn transpose_head_to_token_f32<const BLOCK: usize>(
    input: Ptr<f32>,
    output: Ptr<f32>,
    tokens: i32,
    heads: i32,
    head_dim: i32,
) {
    let pid = program_id(0);
    let head = pid / tokens;
    let tok = pid % tokens;

    let cols = make_range(0, BLOCK as i32);
    let mask = cols < head_dim;

    let src_off = head * tokens * head_dim + tok * head_dim + cols;
    let dst_off = tok * heads * head_dim + head * head_dim + cols;

    let v = load(input + src_off, mask);
    store(output + dst_off, v, mask);
}

/// Token-major -> head-major transpose, no compute. Equivalent to
/// `qk_norm_rope.cu mode=0` (V path).
///
/// `input[tok, head, :head_dim]` -> `output[head, tok, :head_dim]`.
/// Launch: grid = (tokens * heads, 1, 1).
#[triton_kernel]
pub fn transpose_token_to_head_f32<const BLOCK: usize>(
    input: Ptr<f32>,
    output: Ptr<f32>,
    tokens: i32,
    heads: i32,
    head_dim: i32,
) {
    let pid = program_id(0);
    let tok = pid / heads;
    let head = pid % heads;

    let cols = make_range(0, BLOCK as i32);
    let mask = cols < head_dim;

    let src_off = (tok * heads + head) * head_dim + cols;
    let dst_off = (head * tokens + tok) * head_dim + cols;

    let v = load(input + src_off, mask);
    store(output + dst_off, v, mask);
}

/// **Fused QK-norm + RoPE + transpose** (Qwen3 / "QK-norm" attention path).
///
/// Equivalent to `qk_norm_rope.cu mode=1`. In one launch, for each
/// `(tok, head)`:
///   1. RMS-normalise `input[tok, head, :]` with `norm_w[head_dim]`,
///   2. apply RoPE rotation on the (lo, hi) halves using
///      `cos/sin_tab[pos_offset + tok, :half_d]`,
///   3. write into the head-major `output[head, tok, :head_dim]`.
///
/// Caller pre-computes `inv_n = 1.0 / head_dim`. Requires
/// `HALF_DIM = HEAD_DIM / 2` and `head_dim <= HEAD_DIM`.
#[triton_kernel]
pub fn qk_norm_rope_transpose_f32<const HEAD_DIM: usize, const HALF_DIM: usize>(
    input: Ptr<f32>,
    norm_w: Ptr<f32>,
    cos_tab: Ptr<f32>,
    sin_tab: Ptr<f32>,
    output: Ptr<f32>,
    tokens: i32,
    heads: i32,
    head_dim: i32,
    pos_offset: i32,
    inv_n: f32,
    eps: f32,
) {
    let pid = program_id(0);
    let tok = pid / heads;
    let head = pid % heads;

    // ── Phase 1: RMS-norm scale (full head_dim tile) ──
    let full_cols = make_range(0, HEAD_DIM as i32);
    let full_mask = full_cols < head_dim;
    let src_base = (tok * heads + head) * head_dim;

    let src_full = load(input + src_base + full_cols, full_mask);
    let sq = src_full * src_full;
    let sum_sq = reduce(sq, 0, |a, b| a + b);
    let mean = sum_sq * inv_n;
    let inv_rms = rsqrt(mean + eps);

    // ── Phase 2: RoPE on (low, high) halves with norm scaling ──
    let pair = make_range(0, HALF_DIM as i32);
    let half_d = head_dim / 2;
    let pair_mask = pair < half_d;

    let pos = pos_offset + tok;
    let cs_row = pos * (HALF_DIM as i32) + pair;
    let c = load(cos_tab + cs_row, pair_mask);
    let s = load(sin_tab + cs_row, pair_mask);

    let nw_lo = load(norm_w + pair, pair_mask);
    let nw_hi = load(norm_w + pair + (HALF_DIM as i32), pair_mask);

    let lo_in_off = src_base + pair;
    let hi_in_off = src_base + pair + (HALF_DIM as i32);
    let x0 = load(input + lo_in_off, pair_mask);
    let x1 = load(input + hi_in_off, pair_mask);

    let scaled_x0 = x0 * inv_rms * nw_lo;
    let scaled_x1 = x1 * inv_rms * nw_hi;

    let out_lo = scaled_x0 * c - scaled_x1 * s;
    let out_hi = scaled_x1 * c + scaled_x0 * s;

    // ── Phase 3: write transposed (head-major) ──
    let dst_base = (head * tokens + tok) * head_dim;
    store(output + dst_base + pair, out_lo, pair_mask);
    store(output + dst_base + pair + (HALF_DIM as i32), out_hi, pair_mask);
}

/// **Fused RoPE + transpose** (Llama / Mistral attention path,
/// no QK-norm). Equivalent to `qk_norm_rope.cu mode=2`. Same shape as
/// `qk_norm_rope_transpose_f32` minus the RMS-norm pass — saves the
/// reduce + rsqrt + weight load when the model doesn't use QK-norm.
#[triton_kernel]
pub fn rope_transpose_f32<const HALF_DIM: usize>(
    input: Ptr<f32>,
    cos_tab: Ptr<f32>,
    sin_tab: Ptr<f32>,
    output: Ptr<f32>,
    tokens: i32,
    heads: i32,
    head_dim: i32,
    pos_offset: i32,
) {
    let pid = program_id(0);
    let tok = pid / heads;
    let head = pid % heads;

    let pair = make_range(0, HALF_DIM as i32);
    let half_d = head_dim / 2;
    let pair_mask = pair < half_d;

    let pos = pos_offset + tok;
    let cs_row = pos * (HALF_DIM as i32) + pair;
    let c = load(cos_tab + cs_row, pair_mask);
    let s = load(sin_tab + cs_row, pair_mask);

    let src_base = (tok * heads + head) * head_dim;
    let x0 = load(input + src_base + pair, pair_mask);
    let x1 = load(input + src_base + pair + (HALF_DIM as i32), pair_mask);

    let out_lo = x0 * c - x1 * s;
    let out_hi = x1 * c + x0 * s;

    let dst_base = (head * tokens + tok) * head_dim;
    store(output + dst_base + pair, out_lo, pair_mask);
    store(output + dst_base + pair + (HALF_DIM as i32), out_hi, pair_mask);
}
