//! Normalisation kernels — RMS norm, LayerNorm, and the fused
//! residual+RMS variant common in transformer blocks.

use triton_dsl::triton_kernel;

#[allow(dead_code)]
struct Ptr<T>(::std::marker::PhantomData<T>);

/// RMS normalisation:
/// `output[r, i] = input[r, i] / sqrt(mean(input[r]^2) + eps) * weight[i]`.
///
/// Caller supplies precomputed `inv_n = 1.0 / row_size` to avoid an
/// int->float cast inside the kernel. Launch: grid = (num_rows, 1, 1).
/// Requires `row_size <= BLOCK`.
#[triton_kernel]
pub fn rms_norm_f32<const BLOCK: usize>(
    input: Ptr<f32>,
    weight: Ptr<f32>,
    output: Ptr<f32>,
    row_size: i32,
    inv_n: f32,
    eps: f32,
) {
    let row = program_id(0);
    let row_off = row * row_size;

    let cols = make_range(0, BLOCK as i32);
    let mask = cols < row_size;
    let abs_off = row_off + cols;

    let xv = load(input + abs_off, mask);
    let sq = xv * xv;
    let sum_sq = reduce(sq, 0, |a, b| a + b);

    let mean = sum_sq * inv_n;
    let inv_rms = rsqrt(mean + eps);

    let wv = load(weight + cols, mask);
    let result = xv * inv_rms * wv;
    store(output + abs_off, result, mask);
}

/// Bert / CLIP / Whisper LayerNorm with affine `gamma` / `beta`:
/// `out[r, c] = (x[r, c] - mean(x[r])) / sqrt(var(x[r]) + eps) * gamma[c] + beta[c]`.
#[triton_kernel]
pub fn layer_norm_f32<const BLOCK: usize>(
    x: Ptr<f32>,
    gamma: Ptr<f32>,
    beta: Ptr<f32>,
    out: Ptr<f32>,
    dim: i32,
    inv_n: f32,
    eps: f32,
) {
    let row = program_id(0);
    let row_off = row * dim;

    let cols = make_range(0, BLOCK as i32);
    let mask = cols < dim;
    let abs_off = row_off + cols;

    let xv = load(x + abs_off, mask);

    let sum_x = reduce(xv, 0, |a, b| a + b);
    let mean = sum_x * inv_n;

    let centered = xv - mean;
    let sq = centered * centered;
    let sum_sq = reduce(sq, 0, |a, b| a + b);
    let var = sum_sq * inv_n;
    let inv_std = rsqrt(var + eps);

    let gv = load(gamma + cols, mask);
    let bv = load(beta + cols, mask);
    let result = centered * inv_std * gv + bv;
    store(out + abs_off, result, mask);
}

/// Fused residual-add + RMS normalisation: `residual_out = input +
/// residual; output = rms_norm(residual_out, weight, eps)`.
///
/// Critical LLM kernel — collapses 3 launches (add, variance,
/// normalize) into 1 in transformer blocks.
#[triton_kernel]
pub fn fused_add_rms_norm_f32<const BLOCK: usize>(
    input: Ptr<f32>,
    residual: Ptr<f32>,
    weight: Ptr<f32>,
    output: Ptr<f32>,
    residual_out: Ptr<f32>,
    hidden_size: i32,
    inv_n: f32,
    eps: f32,
) {
    let row = program_id(0);
    let row_off = row * hidden_size;

    let cols = make_range(0, BLOCK as i32);
    let mask = cols < hidden_size;
    let abs_off = row_off + cols;

    let xv = load(input + abs_off, mask);
    let rv = load(residual + abs_off, mask);
    let sum_v = xv + rv;
    store(residual_out + abs_off, sum_v, mask);

    let sq = sum_v * sum_v;
    let var_sum = reduce(sq, 0, |a, b| a + b);
    let var = var_sum * inv_n;
    let inv_rms = rsqrt(var + eps);

    let wv = load(weight + cols, mask);
    let result = sum_v * inv_rms * wv;
    store(output + abs_off, result, mask);
}
