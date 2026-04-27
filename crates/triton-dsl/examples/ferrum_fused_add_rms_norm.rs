//! ferrum kernel port: `fused_add_rms_norm` — combines residual add and
//! RMS normalisation in one launch (3 launches → 1 in transformer blocks).
//!
//! Mirrors `ferrum-kernels/kernels/fused_add_rms_norm.cu`:
//!   residual_out = input + residual
//!   output       = rms_norm(residual_out, weight, eps)
//!
//! Same precomputed `inv_n` arg trick as `rms_norm` so we don't need a
//! sitofp op in the dialect yet.
//!
//! ## Dtype-generic
//! All five pointers (`input`, `residual`, `weight`, `output`,
//! `residual_out`) carry the same element type `T`. Loads upcast to
//! f32; the residual add, sum-of-squares reduction, mean, rsqrt, and
//! the scaled multiply all run in f32. Both stores (`residual_out` for
//! the post-add value, `output` for the normalized value) downcast
//! back to T via `as_t::<T>(...)`. For `T == f32` the casts collapse.

use triton_dsl::triton_kernel;
use triton_ir::ty::{bf16, f16, TritonElem};

#[allow(dead_code)]
struct Ptr<T>(::std::marker::PhantomData<T>);

#[triton_kernel]
fn fused_add_rms_norm_typed<T: TritonElem, const BLOCK: usize>(
    input: Ptr<T>,
    residual: Ptr<T>,
    weight: Ptr<T>,
    output: Ptr<T>,
    residual_out: Ptr<T>,
    hidden_size: i32,
    inv_n: f32,
    eps: f32,
) {
    let row = program_id(0);
    let row_off = row * hidden_size;

    let cols = make_range(0, BLOCK as i32);
    let mask = cols < hidden_size;
    let abs_off = row_off + cols;

    // Phase 1: load+upcast both inputs, add in f32.
    let xv = to_f32(load(input + abs_off, mask));
    let rv = to_f32(load(residual + abs_off, mask));
    let sum_v = xv + rv;

    // Write the residual_out (input+residual sum) back, downcast to T.
    store(residual_out + abs_off, as_t::<T>(sum_v), mask);

    // Sum of squares for RMS, all in f32.
    let sq = sum_v * sum_v;
    let var_sum = reduce(sq, 0, |a, b| a + b);

    // inv_rms = 1 / sqrt(var * inv_n + eps)
    let var = var_sum * inv_n;
    let inv_rms = rsqrt(var + eps);

    // Phase 2: normalize + apply weight (load+upcast). Result downcast
    // to T at the store boundary.
    let wv = to_f32(load(weight + cols, mask));
    let result = sum_v * inv_rms * wv;

    store(output + abs_off, as_t::<T>(result), mask);
}

fn main() {
    let dtype = std::env::args().nth(1).unwrap_or_else(|| "f32".into());
    match dtype.as_str() {
        "f32"  => print!("{}", fused_add_rms_norm_typed::<f32, 1024>::mlir()),
        "f16"  => print!("{}", fused_add_rms_norm_typed::<f16, 1024>::mlir()),
        "bf16" => print!("{}", fused_add_rms_norm_typed::<bf16, 1024>::mlir()),
        other => {
            eprintln!("unknown dtype `{other}` — supported: f32, f16, bf16");
            std::process::exit(2);
        }
    }
}
