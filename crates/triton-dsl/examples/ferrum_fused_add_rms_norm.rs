//! ferrum kernel port: `fused_add_rms_norm` — combines residual add and
//! RMS normalisation in one launch (3 launches → 1 in transformer blocks).
//!
//! Mirrors `ferrum-kernels/kernels/fused_add_rms_norm.cu`:
//!   residual_out = input + residual
//!   output       = rms_norm(residual_out, weight, eps)
//!
//! Same precomputed `inv_n` arg trick as `rms_norm` so we don't need a
//! sitofp op in the dialect yet.

use triton_dsl::triton_kernel;

#[allow(dead_code)]
struct Ptr<T>(::std::marker::PhantomData<T>);

#[triton_kernel]
fn fused_add_rms_norm_f32<const BLOCK: usize>(
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

    // Phase 1: add residual + compute variance in one pass.
    let xv = load(input + abs_off, mask);
    let rv = load(residual + abs_off, mask);
    let sum_v = xv + rv;

    // Write the residual_out (the input+residual sum) back.
    store(residual_out + abs_off, sum_v, mask);

    // Sum of squares for RMS.
    let sq = sum_v * sum_v;
    let var_sum = reduce(sq, 0, |a, b| a + b);

    // inv_rms = 1 / sqrt(var * inv_n + eps)
    let var = var_sum * inv_n;
    let inv_rms = rsqrt(var + eps);

    // Phase 2: normalize + apply weight.
    let wv = load(weight + cols, mask);
    let result = sum_v * inv_rms * wv;

    store(output + abs_off, result, mask);
}

fn main() {
    print!("{}", fused_add_rms_norm_f32::<1024>::mlir());
}
