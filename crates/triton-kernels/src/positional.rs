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

/// Single-position RoPE applied to the query tensor. `q` is
/// `[num_q_heads, head_dim]`; one block per head; tile width
/// `HALF_DIM = head_dim / 2`. Caller has already sliced
/// `cos_table` / `sin_table` to the current position.
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

/// Single-position RoPE applied to the key tensor.
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

/// Multi-token RoPE with explicit per-token position lookup.
/// Layout: `q` is `[tokens, num_heads, head_dim]`; one block per
/// `(token * num_heads + head)`. Per-token position read from
/// `positions[tok]` selects the row of the precomputed
/// `cos_table` / `sin_table` (each `[max_pos, HALF_DIM]`).
///
/// Mirrors the prefill / batched-decode RoPE pattern:
/// `out[tok, h, lo]  = q[tok, h, lo] * cos[pos[tok], lo]`
/// `                 - q[tok, h, hi] * sin[pos[tok], lo]`
/// `out[tok, h, hi]  = q[tok, h, hi] * cos[pos[tok], lo]`
/// `                 + q[tok, h, lo] * sin[pos[tok], lo]`
#[triton_kernel]
pub fn rope_full_f32<const HALF_DIM: usize>(
    q: Ptr<f32>,
    cos_table: Ptr<f32>,
    sin_table: Ptr<f32>,
    positions: Ptr<i32>,
    q_out: Ptr<f32>,
    num_heads: i32,
    head_dim: i32,
) {
    let th = program_id(0);
    let tok = th / num_heads;
    let h = th % num_heads;
    let pos = load(positions + tok);

    let pair = make_range(0, HALF_DIM as i32);
    let mask = pair < (HALF_DIM as i32);

    // Row of cos/sin table for this token's position.
    let cs_off = pos * (HALF_DIM as i32) + pair;
    let c = load(cos_table + cs_off, mask);
    let s = load(sin_table + cs_off, mask);

    // Element offset within the [tokens, heads, head_dim] tensor.
    let head_off = (tok * num_heads + h) * head_dim;
    let lo_off = head_off + pair;
    let hi_off = head_off + pair + (HALF_DIM as i32);

    let x0 = load(q + lo_off, mask);
    let x1 = load(q + hi_off, mask);

    let out_lo = x0 * c - x1 * s;
    let out_hi = x1 * c + x0 * s;

    store(q_out + lo_off, out_lo, mask);
    store(q_out + hi_off, out_hi, mask);
}
