//! ferrum kernel port: `layer_norm` (Bert/Whisper-style affine LayerNorm).
//!
//! Mirrors `ferrum-kernels/kernels/layer_norm.cu`:
//!   out[r, c] = (x[r, c] - mean(x[r])) / sqrt(var(x[r]) + eps)
//!               * gamma[c] + beta[c]
//!
//! Launch: grid = (rows, 1, 1). Same `inv_n = 1.0/dim` precomputed-arg
//! trick as `rms_norm` to avoid an int->float cast op.
//!
//! ## Dtype-generic
//! Pointers `x`, `gamma`, `beta`, `out` all carry the same element type
//! `T`. Loads upcast to f32; the mean reduction, variance reduction,
//! `rsqrt`, and the centered/normalized multiplies all run in f32 (a
//! 1024-wide f16 reduction overflows easily). The final affine-applied
//! result is downcast back to T at the store boundary via `as_t::<T>`.

use triton_dsl::triton_kernel;
use triton_ir::ty::{bf16, f16, TritonElem};

#[allow(dead_code)]
struct Ptr<T>(::std::marker::PhantomData<T>);

#[triton_kernel]
fn layer_norm_typed<T: TritonElem, const BLOCK: usize>(
    x: Ptr<T>,
    gamma: Ptr<T>,
    beta: Ptr<T>,
    out: Ptr<T>,
    dim: i32,
    inv_n: f32,
    eps: f32,
) {
    let row = program_id(0);
    let row_off = row * dim;

    let cols = make_range(0, BLOCK as i32);
    let mask = cols < splat_1d(dim, BLOCK as i64);
    let abs_off = splat_1d(row_off, BLOCK as i64) + cols;

    let xv = to_f32(load(splat_1d(x, BLOCK as i64) + abs_off, mask));

    // Mean (f32).
    let sum_x = reduce(xv, 0, |a, b| a + b);
    let mean = sum_x * inv_n;
    let mean_v = splat_1d(mean, BLOCK as i64);

    // Variance (f32).
    let centered = xv - mean_v;
    let sq = centered * centered;
    let sum_sq = reduce(sq, 0, |a, b| a + b);
    let var = sum_sq * inv_n;
    let inv_std = rsqrt(var + eps);

    // Apply gamma/beta (also load+upcast). Result downcast to T.
    let gv = to_f32(load(splat_1d(gamma, BLOCK as i64) + cols, mask));
    let bv = to_f32(load(splat_1d(beta, BLOCK as i64) + cols, mask));
    let normalized = centered * splat_1d(inv_std, BLOCK as i64);
    let result = normalized * gv + bv;

    store(splat_1d(out, BLOCK as i64) + abs_off, as_t::<T>(result), mask);
}

fn main() {
    let dtype = std::env::args().nth(1).unwrap_or_else(|| "f32".into());
    match dtype.as_str() {
        "f32"  => print!("{}", layer_norm_typed::<f32, 1024>::mlir()),
        "f16"  => print!("{}", layer_norm_typed::<f16, 1024>::mlir()),
        "bf16" => print!("{}", layer_norm_typed::<bf16, 1024>::mlir()),
        other => {
            eprintln!("unknown dtype `{other}` — supported: f32, f16, bf16");
            std::process::exit(2);
        }
    }
}
