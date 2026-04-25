//! Dump a DSL-authored reduce kernel for Triton round-trip validation.

use triton_dsl::triton_kernel;

#[allow(dead_code)]
struct Ptr<T>(::std::marker::PhantomData<T>);

#[triton_kernel]
fn sum_block(out: Ptr<f32>) {
    let zero = const_f32(0.0);
    let tile = splat_1d(zero, 128);
    let total = reduce(tile, 0, |a, b| a + b);
    store(out, total);
}

fn main() {
    print!("{}", sum_block::mlir());
}
