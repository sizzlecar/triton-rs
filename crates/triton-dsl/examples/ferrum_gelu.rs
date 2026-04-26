//! ferrum kernel port: `gelu` element-wise activation.
//!
//! Mirrors `ferrum-kernels/kernels/gelu.cu` (PyTorch default, erf-based):
//!   gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
//!
//! NOTE: signature differs from vec_add — this is 2-pointer + i32 (no
//! second input). Needs its own runner; see `run_gelu` example.

use triton_dsl::triton_kernel;

#[allow(dead_code)]
struct Ptr<T>(::std::marker::PhantomData<T>);

#[triton_kernel]
fn gelu_f32<const BLOCK: usize>(
    x: Ptr<f32>,
    out: Ptr<f32>,
    n: i32,
) {
    let pid = program_id(0);
    let off = splat_1d(pid * const_i32(BLOCK as i32), BLOCK as i64)
            + make_range(0, BLOCK as i32);
    let mask = off < splat_1d(n, BLOCK as i64);

    let xv = load(splat_1d(x, BLOCK as i64) + off, mask);

    // gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
    // 1/sqrt(2) ≈ 0.7071067811865475
    let inv_sqrt2 = splat_1d(const_f32(0.707_106_77), BLOCK as i64);
    let half      = splat_1d(const_f32(0.5),          BLOCK as i64);
    let one       = splat_1d(const_f32(1.0),          BLOCK as i64);

    let scaled = xv * inv_sqrt2;
    let erfed  = erf(scaled);
    let result = half * xv * (one + erfed);

    store(splat_1d(out, BLOCK as i64) + off, result, mask);
}

fn main() {
    print!("{}", gelu_f32::<1024>::mlir());
}
