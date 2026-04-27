//! ferrum kernel port: `residual_add` element-wise vector add.
//!
//! Mirrors `ferrum-kernels/kernels/residual_add.cu` semantically:
//!   `output[i] = a[i] + b[i]`  for `i in [0, n)`.
//!
//! Phase-1-lite end-to-end runner: `run_kernel residual_add_f32`.
//!
//! ## Dtype-generic
//! The kernel body is parameterized by `T: TritonElem`. Pure elementwise
//! add is bandwidth-bound, so we don't need an f32 accumulator — the
//! body just does `T + T → T`. For `T == f32` the IR matches the
//! original `residual_add_f32` byte-for-byte. For `T == f16` / `bf16`
//! the loads/stores carry the requested dtype directly.

use triton_dsl::triton_kernel;
use triton_ir::ty::{bf16, f16, TritonElem};

#[allow(dead_code)]
struct Ptr<T>(::std::marker::PhantomData<T>);

#[triton_kernel]
fn residual_add_typed<T: TritonElem, const BLOCK: usize>(
    a: Ptr<T>,
    b: Ptr<T>,
    out: Ptr<T>,
    n: i32,
) {
    let pid = program_id(0);
    let off = splat_1d(pid * const_i32(BLOCK as i32), BLOCK as i64)
            + make_range(0, BLOCK as i32);
    let mask = off < splat_1d(n, BLOCK as i64);
    let av = load(splat_1d(a, BLOCK as i64) + off, mask);
    let bv = load(splat_1d(b, BLOCK as i64) + off, mask);
    store(splat_1d(out, BLOCK as i64) + off, av + bv, mask);
}

fn main() {
    let dtype = std::env::args().nth(1).unwrap_or_else(|| "f32".into());
    match dtype.as_str() {
        "f32"  => print!("{}", residual_add_typed::<f32, 1024>::mlir()),
        "f16"  => print!("{}", residual_add_typed::<f16, 1024>::mlir()),
        "bf16" => print!("{}", residual_add_typed::<bf16, 1024>::mlir()),
        other => {
            eprintln!("unknown dtype `{other}` — supported: f32, f16, bf16");
            std::process::exit(2);
        }
    }
}
