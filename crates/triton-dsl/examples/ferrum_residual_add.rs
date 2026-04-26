//! ferrum kernel port: `residual_add` element-wise vector add.
//!
//! Mirrors `ferrum-kernels/kernels/residual_add.cu` semantically:
//!   `output[i] = a[i] + b[i]`  for `i in [0, n)`.
//!
//! Phase-1-lite end-to-end runner: `run_kernel residual_add_f32`.

use triton_dsl::triton_kernel;

#[allow(dead_code)]
struct Ptr<T>(::std::marker::PhantomData<T>);

#[triton_kernel]
fn residual_add_f32<const BLOCK: usize>(
    a: Ptr<f32>,
    b: Ptr<f32>,
    out: Ptr<f32>,
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

#[triton_kernel]
fn residual_add_f16<const BLOCK: usize>(
    a: Ptr<f16>,
    b: Ptr<f16>,
    out: Ptr<f16>,
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
        "f32" => print!("{}", residual_add_f32::<1024>::mlir()),
        "f16" => print!("{}", residual_add_f16::<1024>::mlir()),
        other => {
            eprintln!("unknown dtype `{other}` — supported: f32, f16");
            std::process::exit(2);
        }
    }
}
