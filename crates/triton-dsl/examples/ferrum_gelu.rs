//! ferrum kernel port: `gelu` element-wise activation.
//!
//! Mirrors `ferrum-kernels/kernels/gelu.cu` (PyTorch default, erf-based):
//!   gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
//!
//! NOTE: signature differs from vec_add — this is 2-pointer + i32 (no
//! second input). Needs its own runner; see `run_gelu` example.
//!
//! ## Dtype-generic
//! Loads upcast to f32; the `erf` math intrinsic and the surrounding
//! arithmetic all run in f32 (no native f16 `math.erf` on NVPTX).
//! Stores downcast back to T via `as_t::<T>`. For `T == f32` casts
//! collapse and the IR matches `gelu_f32`.

use triton_dsl::triton_kernel;
use triton_ir::ty::{bf16, f16, TritonElem};

#[allow(dead_code)]
struct Ptr<T>(::std::marker::PhantomData<T>);

#[triton_kernel]
fn gelu_typed<T: TritonElem, const BLOCK: usize>(
    x: Ptr<T>,
    out: Ptr<T>,
    n: i32,
) {
    let pid = program_id(0);
    let off = splat_1d(pid * const_i32(BLOCK as i32), BLOCK as i64)
            + make_range(0, BLOCK as i32);
    let mask = off < splat_1d(n, BLOCK as i64);

    let xv = to_f32(load(splat_1d(x, BLOCK as i64) + off, mask));

    // gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
    // 1/sqrt(2) ≈ 0.7071067811865475
    let inv_sqrt2 = splat_1d(const_f32(0.707_106_77), BLOCK as i64);
    let half      = splat_1d(const_f32(0.5),          BLOCK as i64);
    let one       = splat_1d(const_f32(1.0),          BLOCK as i64);

    let scaled = xv * inv_sqrt2;
    let erfed  = erf(scaled);
    let result = half * xv * (one + erfed);

    store(splat_1d(out, BLOCK as i64) + off, as_t::<T>(result), mask);
}

fn main() {
    let dtype = std::env::args().nth(1).unwrap_or_else(|| "f32".into());
    match dtype.as_str() {
        "f32"  => print!("{}", gelu_typed::<f32, 1024>::mlir()),
        "f16"  => print!("{}", gelu_typed::<f16, 1024>::mlir()),
        "bf16" => print!("{}", gelu_typed::<bf16, 1024>::mlir()),
        other => {
            eprintln!("unknown dtype `{other}` — supported: f32, f16, bf16");
            std::process::exit(2);
        }
    }
}
