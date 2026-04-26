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

/// f16 variant of [`vec_add_f32`]. Pure-f16 add (no upcast); matches
/// ferrum's residual-add path which is bandwidth-bound either way.
#[triton_kernel]
pub fn vec_add_f16<const BLOCK: usize>(
    a: Ptr<f16>,
    b: Ptr<f16>,
    out: Ptr<f16>,
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

/// f16 variant of [`residual_add_f32`].
#[triton_kernel]
pub fn residual_add_f16<const BLOCK: usize>(
    a: Ptr<f16>,
    b: Ptr<f16>,
    out: Ptr<f16>,
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

/// f16 in-place residual add — ferrum's `residual_add_inplace_f16`.
#[triton_kernel]
pub fn residual_add_inplace_f16<const BLOCK: usize>(
    a: Ptr<f16>,
    b: Ptr<f16>,
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

/// f16 GELU. Loads f16, upcasts to f32 for the erf-based formula
/// (preserves accuracy on small / large magnitudes), downcasts the
/// result back to f16 for the store.
#[triton_kernel]
pub fn gelu_f16<const BLOCK: usize>(x: Ptr<f16>, out: Ptr<f16>, n: i32) {
    let pid = program_id(0);
    let off = pid * (BLOCK as i32) + make_range(0, BLOCK as i32);
    let mask = off < n;
    let xv = to_f32(load(x + off, mask));

    let scaled = xv * 0.707_106_77_f32;
    let erfed = erf(scaled);
    let result = (xv * 0.5_f32) * (erfed + 1.0_f32);
    store(out + off, to_f16(result), mask);
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

/// f16 fused silu_mul. exp/div in f32 to keep silu accurate on small
/// activations; final multiply happens in f16.
#[triton_kernel]
pub fn fused_silu_mul_f16<const BLOCK: usize>(
    gate: Ptr<f16>,
    up: Ptr<f16>,
    out: Ptr<f16>,
    n: i32,
) {
    let pid = program_id(0);
    let off = pid * (BLOCK as i32) + make_range(0, BLOCK as i32);
    let mask = off < n;
    let g = to_f32(load(gate + off, mask));
    let u = to_f32(load(up + off, mask));
    let neg_g = g * -1.0_f32;
    let denom = exp(neg_g) + 1.0_f32;
    let silu_g = g / denom;
    store(out + off, to_f16(silu_g * u), mask);
}

/// f16 interleaved silu_mul (gate/up packed: `gate_up` is `[batch * 2 * inter]`).
#[triton_kernel]
pub fn fused_silu_mul_interleaved_f16<const BLOCK: usize>(
    gate_up: Ptr<f16>,
    out: Ptr<f16>,
    inter: i32,
    total: i32,
) {
    let pid = program_id(0);
    let cols = make_range(0, BLOCK as i32);
    let idx = pid * (BLOCK as i32) + cols;
    let mask = idx < total;

    let b = idx / inter;
    let i = idx % inter;
    let row_base = b * (2 * inter);

    let gate_off = row_base + i;
    let up_off = row_base + inter + i;
    let g = to_f32(load(gate_up + gate_off, mask));
    let u = to_f32(load(gate_up + up_off, mask));

    let neg_g = g * -1.0_f32;
    let denom = exp(neg_g) + 1.0_f32;
    let silu_g = g / denom;
    store(out + idx, to_f16(silu_g * u), mask);
}

/// Same as [`fused_silu_mul_f32`] but for the **interleaved** memory layout
/// where `gate` and `up` are concatenated into one buffer:
/// `gate_up` is `[batch * 2 * inter]`. For each output element
/// `out[b, i]` (flat index `b * inter + i`):
///   `g = gate_up[b * 2 * inter + i]`
///   `u = gate_up[b * 2 * inter + inter + i]`
///   `out[b, i] = silu(g) * u`
///
/// Used by ferrum's expert-parallel MLP where the GEMM emits gate/up
/// pre-interleaved per row, saving a split kernel before the activation.
#[triton_kernel]
pub fn fused_silu_mul_interleaved_f32<const BLOCK: usize>(
    gate_up: Ptr<f32>,
    out: Ptr<f32>,
    inter: i32,
    total: i32,
) {
    let pid = program_id(0);
    let cols = make_range(0, BLOCK as i32);
    let idx = pid * (BLOCK as i32) + cols;
    let mask = idx < total;

    // (b, i) = (idx / inter, idx % inter); recover the per-row stride.
    let b = idx / inter;
    let i = idx % inter;
    let row_base = b * (2 * inter);

    let gate_off = row_base + i;
    let up_off = row_base + inter + i;
    let g = load(gate_up + gate_off, mask);
    let u = load(gate_up + up_off, mask);

    let neg_g = g * -1.0_f32;
    let denom = exp(neg_g) + 1.0_f32;
    let silu_g = g / denom;
    store(out + idx, silu_g * u, mask);
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

/// f16 broadcast bias add (in-place).
#[triton_kernel]
pub fn add_bias_f16<const BLOCK: usize>(
    data: Ptr<f16>,
    bias: Ptr<f16>,
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
