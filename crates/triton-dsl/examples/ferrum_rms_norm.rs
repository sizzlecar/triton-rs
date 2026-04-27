//! ferrum kernel port: `rms_norm` — RMS normalisation, no residual.
//!
//! Mirrors `ferrum-kernels/kernels/rms_norm.cu`:
//!   output[r, i] = input[r, i] / sqrt(mean(input[r]^2) + eps) * weight[i]
//!
//! Launch: grid = (num_rows, 1, 1), one row per block. Requires
//! `row_size <= BLOCK`. The caller passes the precomputed
//! `inv_n = 1.0 / row_size` so we don't need an int→float cast op.
//!
//! ## Dtype-generic
//! All three pointers (`input`, `weight`, `output`) carry the same
//! element type `T`. Loads upcast to f32 via `to_f32`; the
//! sum-of-squares reduction, mean, rsqrt, and the scaled multiply all
//! run in f32 (a 1024-wide f16 sum-of-squares would overflow on
//! activations of magnitude > 2^5). The final result is downcast back
//! to T via `as_t::<T>(...)` at the store boundary. For `T == f32` the
//! cast pairs collapse and the IR matches the original f32-only kernel.

use triton_dsl::triton_kernel;
use triton_ir::ty::{bf16, f16, TritonElem};

#[allow(dead_code)]
struct Ptr<T>(::std::marker::PhantomData<T>);

#[triton_kernel]
fn rms_norm_typed<T: TritonElem, const BLOCK: usize>(
    input: Ptr<T>,
    weight: Ptr<T>,
    output: Ptr<T>,
    row_size: i32,
    inv_n: f32,
    eps: f32,
) {
    let row = program_id(0);
    let row_off = row * row_size;

    let cols = make_range(0, BLOCK as i32);
    let mask = cols < splat_1d(row_size, BLOCK as i64);
    let abs_off = splat_1d(row_off, BLOCK as i64) + cols;

    // Load + sum-of-squares reduction, all in f32.
    let in_ptrs = splat_1d(input, BLOCK as i64) + abs_off;
    let xv = to_f32(load(in_ptrs, mask));
    let sq = xv * xv;
    let sum_sq = reduce(sq, 0, |a, b| a + b);

    // Compute scale factor (scalar, f32).
    let mean = sum_sq * inv_n;
    let inv_rms = rsqrt(mean + eps);

    // Broadcast scale, multiply by weight (also upcast to f32), store
    // downcast back to T.
    let inv_rms_v = splat_1d(inv_rms, BLOCK as i64);
    let w_ptrs = splat_1d(weight, BLOCK as i64) + cols;
    let wv = to_f32(load(w_ptrs, mask));

    let result = xv * inv_rms_v * wv;
    let out_ptrs = splat_1d(output, BLOCK as i64) + abs_off;
    store(out_ptrs, as_t::<T>(result), mask);
}

fn main() {
    let dtype = std::env::args().nth(1).unwrap_or_else(|| "f32".into());
    match dtype.as_str() {
        "f32"  => print!("{}", rms_norm_typed::<f32, 1024>::mlir()),
        "f16"  => print!("{}", rms_norm_typed::<f16, 1024>::mlir()),
        "bf16" => print!("{}", rms_norm_typed::<bf16, 1024>::mlir()),
        other => {
            eprintln!("unknown dtype `{other}` — supported: f32, f16, bf16");
            std::process::exit(2);
        }
    }
}
