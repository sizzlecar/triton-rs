//! Dump a DSL kernel using `scf_for` with a multi-statement loop body
//! to stdout. Round-trip-validate against Triton 3.2.0.

use triton_dsl::triton_kernel;

#[allow(dead_code)]
struct Ptr<T>(::std::marker::PhantomData<T>);

#[triton_kernel]
fn sum_squares(out: Ptr<i32>) {
    let lb = const_i32(0);
    let ub = const_i32(10);
    let step = const_i32(1);
    let init = const_i32(0);

    let result = scf_for(lb, ub, step, init, |i, acc| {
        let i_sq = i * i;
        acc + i_sq
    });
    store(out, result);
}

fn main() {
    print!("{}", sum_squares::mlir());
}
