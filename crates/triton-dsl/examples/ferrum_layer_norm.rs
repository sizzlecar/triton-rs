//! ferrum kernel port: `layer_norm` (Bert/Whisper-style affine LayerNorm).
//!
//! Mirrors `ferrum-kernels/kernels/layer_norm.cu`:
//!   out[r, c] = (x[r, c] - mean(x[r])) / sqrt(var(x[r]) + eps)
//!               * gamma[c] + beta[c]
//!
//! Launch: grid = (rows, 1, 1). Same `inv_n = 1.0/dim` precomputed-arg
//! trick as `rms_norm` to avoid an int->float cast op.

use triton_dsl::triton_kernel;

#[allow(dead_code)]
struct Ptr<T>(::std::marker::PhantomData<T>);

#[triton_kernel]
fn layer_norm_f32<const BLOCK: usize>(
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
    let mask = cols < splat_1d(dim, BLOCK as i64);
    let abs_off = splat_1d(row_off, BLOCK as i64) + cols;

    let xv = load(splat_1d(x, BLOCK as i64) + abs_off, mask);

    // Mean.
    let sum_x = reduce(xv, 0, |a, b| a + b);
    let mean = sum_x * inv_n;
    let mean_v = splat_1d(mean, BLOCK as i64);

    // Variance.
    let centered = xv - mean_v;
    let sq = centered * centered;
    let sum_sq = reduce(sq, 0, |a, b| a + b);
    let var = sum_sq * inv_n;
    let inv_std = rsqrt(var + eps);

    // Apply gamma/beta.
    let gv = load(splat_1d(gamma, BLOCK as i64) + cols, mask);
    let bv = load(splat_1d(beta, BLOCK as i64) + cols, mask);
    let normalized = centered * splat_1d(inv_std, BLOCK as i64);
    let result = normalized * gv + bv;

    store(splat_1d(out, BLOCK as i64) + abs_off, result, mask);
}

fn main() {
    print!("{}", layer_norm_f32::<1024>::mlir());
}
