//! Dump a const-generic vec_add kernel at two different BLOCK sizes,
//! proving each instantiation produces its own MLIR module.

use triton_dsl::triton_kernel;

#[allow(dead_code)]
struct Ptr<T>(::std::marker::PhantomData<T>);

#[triton_kernel]
fn vec_add<const BLOCK: usize>(x: Ptr<f32>, y: Ptr<f32>, out: Ptr<f32>, n: i32) {
    let pid = program_id(0);
    let off =
        splat_1d(pid * const_i32(BLOCK as i32), BLOCK as i64) + make_range(0, BLOCK as i32);
    let mask = off < splat_1d(n, BLOCK as i64);
    let xv = load(splat_1d(x, BLOCK as i64) + off, mask);
    let yv = load(splat_1d(y, BLOCK as i64) + off, mask);
    store(splat_1d(out, BLOCK as i64) + off, xv + yv, mask);
}

fn main() {
    print!("{}", vec_add::<512>::mlir());
}
