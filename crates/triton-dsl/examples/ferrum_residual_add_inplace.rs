//! ferrum kernel port: `residual_add_inplace` — `a[i] += b[i]`.
//!
//! Mirrors `ferrum-kernels/kernels/residual_add.cu` (inplace variant).
//! Exists primarily to dodge Rust borrow conflicts when aliasing the
//! same buffer as input + output through an out-of-place call.
//!
//! Dtype-generic via `T: TritonElem`. Pure elementwise add is
//! bandwidth-bound — no f32 accumulator needed; the body just does
//! `T + T → T`.

use triton_dsl::triton_kernel;
use triton_ir::ty::{bf16, f16, TritonElem};

#[allow(dead_code)]
struct Ptr<T>(::std::marker::PhantomData<T>);

#[triton_kernel]
fn residual_add_inplace_typed<T: TritonElem, const BLOCK: usize>(
    a: Ptr<T>,
    b: Ptr<T>,
    n: i32,
) {
    let pid = program_id(0);
    let off = splat_1d(pid * const_i32(BLOCK as i32), BLOCK as i64)
            + make_range(0, BLOCK as i32);
    let mask = off < splat_1d(n, BLOCK as i64);

    let a_ptrs = splat_1d(a, BLOCK as i64) + off;
    let av = load(a_ptrs, mask);
    let bv = load(splat_1d(b, BLOCK as i64) + off, mask);

    store(a_ptrs, av + bv, mask);
}

fn main() {
    let dtype = std::env::args().nth(1).unwrap_or_else(|| "f32".into());
    match dtype.as_str() {
        "f32"  => print!("{}", residual_add_inplace_typed::<f32, 1024>::mlir()),
        "f16"  => print!("{}", residual_add_inplace_typed::<f16, 1024>::mlir()),
        "bf16" => print!("{}", residual_add_inplace_typed::<bf16, 1024>::mlir()),
        other => {
            eprintln!("unknown dtype `{other}` — supported: f32, f16, bf16");
            std::process::exit(2);
        }
    }
}
