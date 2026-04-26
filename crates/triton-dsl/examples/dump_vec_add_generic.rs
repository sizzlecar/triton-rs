//! Dump a const-generic vec_add kernel at two different BLOCK sizes,
//! proving each instantiation produces its own MLIR module.

use triton_dsl::triton_kernel;

#[allow(dead_code)]
struct Ptr<T>(::std::marker::PhantomData<T>);

// With auto-broadcast (added to FuncBuilder.add/sub/mul/lt/...), the
// IR builder splats scalars / pointers to match a tensor operand
// automatically — `pid + range`, `off < n`, `x + off`, `out + off` all
// just work without manual splat_1d.
#[triton_kernel]
fn vec_add<const BLOCK: usize>(x: Ptr<f32>, y: Ptr<f32>, out: Ptr<f32>, n: i32) {
    let pid = program_id(0);
    let off  = pid * const_i32(BLOCK as i32) + make_range(0, BLOCK as i32);
    let mask = off < n;
    let xv   = load(x + off, mask);
    let yv   = load(y + off, mask);
    store(out + off, xv + yv, mask);
}

fn main() {
    // BLOCK=1024 matches the runtime example (`run_vec_add`) so the
    // compiled kernel and the launch parameters agree on tile width.
    print!("{}", vec_add::<1024>::mlir());
}
