//! Projection helpers — splitting combined QKV outputs and reshape /
//! transpose between layout conventions.

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

    // Input strides: batch * (3 * num_heads * head_dim) + qkv_idx * (num_heads
    // * head_dim) + head * head_dim + col, where qkv_idx ∈ {0, 1, 2}.
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

    // Source: input[head, tok, :head_dim]
    let src_off = head * tokens * head_dim + tok * head_dim + cols;
    // Destination: output[tok, head, :head_dim]
    let dst_off = tok * heads * head_dim + head * head_dim + cols;

    let v = load(input + src_off, mask);
    store(output + dst_off, v, mask);
}
