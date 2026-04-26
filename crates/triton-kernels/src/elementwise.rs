//! Element-wise kernels — one tile per block, no inter-thread comms.

use triton_dsl::triton_kernel;

#[allow(dead_code)]
struct Ptr<T>(::std::marker::PhantomData<T>);

// ── pure vector add (vec_add / residual_add are semantically identical) ──

/// `out[i] = a[i] + b[i]` — element-wise add, out-of-place.
#[triton_kernel]
pub fn vec_add_f32<const BLOCK: usize>(
    a: Ptr<f32>,
    b: Ptr<f32>,
    out: Ptr<f32>,
    n: i32,
) {
    let pid = program_id(0);
    let off = pid * (BLOCK as i32) + make_range(0, BLOCK as i32);
    let mask = off < n;
    let av = load(a + off, mask);
    let bv = load(b + off, mask);
    store(out + off, av + bv, mask);
}

/// Same shape, ferrum-compatible name. Useful when porting from
/// `ferrum-kernels::residual_add::residual_add_f32`.
#[triton_kernel]
pub fn residual_add_f32<const BLOCK: usize>(
    a: Ptr<f32>,
    b: Ptr<f32>,
    out: Ptr<f32>,
    n: i32,
) {
    let pid = program_id(0);
    let off = pid * (BLOCK as i32) + make_range(0, BLOCK as i32);
    let mask = off < n;
    let av = load(a + off, mask);
    let bv = load(b + off, mask);
    store(out + off, av + bv, mask);
}

/// In-place: `a[i] += b[i]`. Output goes back to `a`. Mirrors
/// ferrum's `residual_add_inplace_f32`.
#[triton_kernel]
pub fn residual_add_inplace_f32<const BLOCK: usize>(
    a: Ptr<f32>,
    b: Ptr<f32>,
    n: i32,
) {
    let pid = program_id(0);
    let off = pid * (BLOCK as i32) + make_range(0, BLOCK as i32);
    let mask = off < n;
    let a_ptrs = a + off;
    let av = load(a_ptrs, mask);
    let bv = load(b + off, mask);
    store(a_ptrs, av + bv, mask);
}

// ── activations ──

/// GELU (PyTorch default, erf-based):
/// `gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))`.
/// Used in BERT/CLIP/Whisper MLPs.
#[triton_kernel]
pub fn gelu_f32<const BLOCK: usize>(x: Ptr<f32>, out: Ptr<f32>, n: i32) {
    let pid = program_id(0);
    let off = pid * (BLOCK as i32) + make_range(0, BLOCK as i32);
    let mask = off < n;
    let xv = load(x + off, mask);

    // 1/sqrt(2) ≈ 0.70710677
    let scaled = xv * 0.707_106_77_f32;
    let erfed = erf(scaled);
    let result = (xv * 0.5_f32) * (erfed + 1.0_f32);
    store(out + off, result, mask);
}

/// Fused SiLU + multiply (LLaMA-style MLP gate projection):
/// `out[i] = silu(gate[i]) * up[i]`  where  `silu(x) = x / (1 + exp(-x))`.
/// Replaces 2 launches (silu + mul) with 1.
#[triton_kernel]
pub fn fused_silu_mul_f32<const BLOCK: usize>(
    gate: Ptr<f32>,
    up: Ptr<f32>,
    out: Ptr<f32>,
    n: i32,
) {
    let pid = program_id(0);
    let off = pid * (BLOCK as i32) + make_range(0, BLOCK as i32);
    let mask = off < n;
    let g = load(gate + off, mask);
    let u = load(up + off, mask);
    // silu(g) = g / (1 + exp(-g))
    let neg_g = g * -1.0_f32;
    let denom = exp(neg_g) + 1.0_f32;
    let silu_g = g / denom;
    store(out + off, silu_g * u, mask);
}

// ── biased linear post-processing ──

/// Broadcast bias add: `data[r, c] += bias[c]`. One block per row,
/// requires `cols <= BLOCK`. Used by Bert / CLIP / Whisper linear
/// projections (LLM path uses bias-free linear layers).
#[triton_kernel]
pub fn add_bias_f32<const BLOCK: usize>(
    data: Ptr<f32>,
    bias: Ptr<f32>,
    rows: i32,
    cols: i32,
) {
    let _ = rows;
    let row = program_id(0);
    let col_idx = make_range(0, BLOCK as i32);
    let mask = col_idx < cols;

    let abs_off = row * cols + col_idx;
    let data_ptrs = data + abs_off;
    let bias_ptrs = bias + col_idx;

    let dv = load(data_ptrs, mask);
    let bv = load(bias_ptrs, mask);
    store(data_ptrs, dv + bv, mask);
}
