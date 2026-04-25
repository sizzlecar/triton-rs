//! Dump the DSL-authored vector_add kernel to stdout. Round-trip-validate
//! against Triton 3.2.0 with `tools/validate_mlir.py`. The output should
//! be functionally identical to `crates/triton-ir/examples/dump_vector_add`
//! (which builds the same kernel by hand-rolled IR builder calls).

use triton_dsl::triton_kernel;

#[allow(dead_code)]
struct Ptr<T>(::std::marker::PhantomData<T>);

#[triton_kernel]
fn vec_add(x: Ptr<f32>, y: Ptr<f32>, out: Ptr<f32>, n: i32) {
    let pid = program_id(0);
    let block = const_i32(1024);
    let base = mul_i32(pid, block);
    let range = make_range(0, 1024);
    let base_v = splat_1d(base, 1024);
    let off = add_i32(base_v, range);

    let n_v = splat_1d(n, 1024);
    let mask = lt_i32(off, n_v);

    let xp = splat_1d(x, 1024);
    let xp_off = addptr(xp, off);
    let xv = load(xp_off, mask);

    let yp = splat_1d(y, 1024);
    let yp_off = addptr(yp, off);
    let yv = load(yp_off, mask);

    let sum = add_f32(xv, yv);

    let outp = splat_1d(out, 1024);
    let outp_off = addptr(outp, off);
    store(outp_off, sum, mask);
}

fn main() {
    print!("{}", vec_add::mlir());
}
