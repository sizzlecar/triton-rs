//! ferrum kernel port: `rms_norm` — RMS normalisation, no residual.
//!
//! Mirrors `ferrum-kernels/kernels/rms_norm.cu`:
//!   output[r, i] = input[r, i] / sqrt(mean(input[r]^2) + eps) * weight[i]
//!
//! Launch: grid = (num_rows, 1, 1), one row per block. Requires
//! `row_size <= BLOCK`. The caller passes the precomputed
//! `inv_n = 1.0 / row_size` so we don't need an int→float cast op.

use triton_dsl::triton_kernel;

#[allow(dead_code)]
struct Ptr<T>(::std::marker::PhantomData<T>);

#[triton_kernel]
fn rms_norm_f32<const BLOCK: usize>(
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
    let mask = cols < splat_1d(row_size, BLOCK as i64);
    let abs_off = splat_1d(row_off, BLOCK as i64) + cols;

    // Load + sum-of-squares reduction.
    let in_ptrs = splat_1d(input, BLOCK as i64) + abs_off;
    let xv = load(in_ptrs, mask);
    let sq = xv * xv;
    let sum_sq = reduce(sq, 0, |a, b| a + b);

    // Compute scale factor (scalar).
    let mean = sum_sq * inv_n;
    let inv_rms = rsqrt(mean + eps);

    // Broadcast scale, multiply by weight, store.
    let inv_rms_v = splat_1d(inv_rms, BLOCK as i64);
    let w_ptrs = splat_1d(weight, BLOCK as i64) + cols;
    let wv = load(w_ptrs, mask);

    let result = xv * inv_rms_v * wv;
    let out_ptrs = splat_1d(output, BLOCK as i64) + abs_off;
    store(out_ptrs, result, mask);
}

fn main() {
    print!("{}", rms_norm_f32::<1024>::mlir());
}
