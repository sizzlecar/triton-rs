//! Rotary position embedding (RoPE) — applied to query and key
//! projections inside attention. Splits Q and K into separate kernels
//! so the body is branch-free; callers launch each once per sequence.
//!
//! For each rotation pair (low, high) where `high = low + head_dim/2`:
//!   x0 = data[low]
//!   x1 = data[high]
//!   data[low]  = x0 * cos - x1 * sin
//!   data[high] = x1 * cos + x0 * sin
//!
//! `cos_table` / `sin_table` are precomputed for the current position
//! and indexed by the pair index in [0, head_dim/2).

use triton_dsl::triton_kernel;

#[allow(dead_code)]
struct Ptr<T>(::std::marker::PhantomData<T>);

/// RoPE applied to the query tensor. `q` is `[num_q_heads, head_dim]`;
/// one block per head; tile width `HALF_DIM = head_dim / 2`.
#[triton_kernel]
pub fn rope_q_f32<const HALF_DIM: usize>(
    q: Ptr<f32>,
    cos_table: Ptr<f32>,
    sin_table: Ptr<f32>,
    q_out: Ptr<f32>,
    head_dim: i32,
) {
    let head = program_id(0);
    let pair = make_range(0, HALF_DIM as i32);
    let mask = pair < (HALF_DIM as i32);

    let c = load(cos_table + pair, mask);
    let s = load(sin_table + pair, mask);

    let head_off = head * head_dim;
    let lo_off = head_off + pair;
    let hi_off = head_off + pair + (HALF_DIM as i32);

    let x0 = load(q + lo_off, mask);
    let x1 = load(q + hi_off, mask);

    let out_lo = x0 * c - x1 * s;
    let out_hi = x1 * c + x0 * s;

    store(q_out + lo_off, out_lo, mask);
    store(q_out + hi_off, out_hi, mask);
}

/// RoPE applied to the key tensor. Same shape as `rope_q_f32` but
/// dimensioned for `[num_k_heads, head_dim]`. Separate kernel so the
/// inner loop has no `if head_idx < num_q_heads` branch.
#[triton_kernel]
pub fn rope_k_f32<const HALF_DIM: usize>(
    k: Ptr<f32>,
    cos_table: Ptr<f32>,
    sin_table: Ptr<f32>,
    k_out: Ptr<f32>,
    head_dim: i32,
) {
    let head = program_id(0);
    let pair = make_range(0, HALF_DIM as i32);
    let mask = pair < (HALF_DIM as i32);

    let c = load(cos_table + pair, mask);
    let s = load(sin_table + pair, mask);

    let head_off = head * head_dim;
    let lo_off = head_off + pair;
    let hi_off = head_off + pair + (HALF_DIM as i32);

    let x0 = load(k + lo_off, mask);
    let x1 = load(k + hi_off, mask);

    let out_lo = x0 * c - x1 * s;
    let out_hi = x1 * c + x0 * s;

    store(k_out + lo_off, out_lo, mask);
    store(k_out + hi_off, out_hi, mask);
}
