//! Phase 3.4 step 2: `reduce(input, axis, |lhs, rhs| body)` via the
//! tt.reduce-with-region builder. Closure body uses standard DSL syntax
//! and yields its trailing expression as the combined value.

use triton_dsl::triton_kernel;

#[allow(dead_code)]
struct Ptr<T>(::std::marker::PhantomData<T>);

#[triton_kernel]
fn sum_block(x: Ptr<f32>, out: Ptr<f32>) {
    // Pretend we already loaded a tile of 128 f32s into `tile`.
    // We synthesise it via splat for testing — real kernels load() it.
    let zero = const_f32(0.0);
    let tile = splat_1d(zero, 128);
    let total = reduce(tile, 0, |a, b| a + b);
    store(out, total);
    // Suppress unused-arg warnings for x.
    let _ = x;
}

#[test]
fn reduce_emits_tt_reduce_and_return() {
    let text = sum_block::mlir();
    eprintln!("===== sum_block MLIR =====\n{text}\n==========================");

    assert!(text.contains("\"tt.reduce\""), "missing tt.reduce:\n{text}");
    assert!(text.contains("\"tt.reduce.return\""), "missing reduce.return:\n{text}");
    // The reduction body should call addf on the f32 inputs.
    assert!(text.contains("\"arith.addf\""), "missing addf inside reduce:\n{text}");
    // Result type: 1D tile reduced over axis 0 -> scalar f32.
    assert!(
        text.contains("(tensor<128xf32>) -> f32"),
        "reduce signature should drop axis 0:\n{text}"
    );
}
