//! Dump the natural-operator-syntax vec_add to stdout. Should produce the
//! same MLIR as `dump_vec_add` (which uses explicit add_i32 / mul_i32 /
//! addptr / lt_i32 helpers) — proving the operator-overloading layer is a
//! transparent ergonomic improvement, not a semantic shift.

use triton_dsl::triton_kernel;

#[allow(dead_code)]
struct Ptr<T>(::std::marker::PhantomData<T>);

#[triton_kernel]
fn vec_add(x: Ptr<f32>, y: Ptr<f32>, out: Ptr<f32>, n: i32) {
    let pid = program_id(0);
    let base = pid * const_i32(1024);
    let off = splat_1d(base, 1024) + make_range(0, 1024);
    let mask = off < splat_1d(n, 1024);
    let xv = load(splat_1d(x, 1024) + off, mask);
    let yv = load(splat_1d(y, 1024) + off, mask);
    let sum = xv + yv;
    store(splat_1d(out, 1024) + off, sum, mask);
}

fn main() {
    print!("{}", vec_add::mlir());
}
