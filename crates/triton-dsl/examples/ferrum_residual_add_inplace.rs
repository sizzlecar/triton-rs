//! ferrum kernel port: `residual_add_inplace` — `a[i] += b[i]`.
//!
//! Mirrors `ferrum-kernels/kernels/residual_add.cu` (inplace variant).
//! Exists primarily to dodge Rust borrow conflicts when aliasing the
//! same buffer as input + output through an out-of-place call.

use triton_dsl::triton_kernel;

#[allow(dead_code)]
struct Ptr<T>(::std::marker::PhantomData<T>);

#[triton_kernel]
fn residual_add_inplace_f32<const BLOCK: usize>(
    a: Ptr<f32>,
    b: Ptr<f32>,
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
    print!("{}", residual_add_inplace_f32::<1024>::mlir());
}
