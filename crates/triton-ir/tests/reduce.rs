//! Coverage for `tt.reduce` via the closure-based builder API.

use triton_ir::prelude::*;

fn build_sum_reduce(input_shape: Vec<i64>, axis: i32) -> Module {
    let mut m = Module::new();
    let mut f = m.func("sum_reduce");
    let input = f.arg("input", Type::tensor(input_shape, Type::f32()));

    let _result = f.reduce_with(input, axis, |fb, lhs, rhs| fb.add(lhs, rhs));
    f.op_void(tt::return_());
    f.finish();
    m
}

#[test]
fn reduce_1d_to_scalar() {
    let m = build_sum_reduce(vec![128], 0);
    let text = m.to_string();
    eprintln!("===== reduce 1d -> scalar =====\n{text}");

    assert!(text.contains("\"tt.reduce\""));
    assert!(text.contains("\"tt.reduce.return\""));
    assert!(text.contains("\"arith.addf\""));
    // Reducing a 1D tensor along axis 0 produces a scalar f32.
    assert!(text.contains("(tensor<128xf32>) -> f32"), "wrong reduce signature:\n{text}");
}

#[test]
fn reduce_2d_drops_one_axis() {
    let m = build_sum_reduce(vec![32, 64], 1);
    let text = m.to_string();
    eprintln!("===== reduce 2d axis=1 =====\n{text}");

    assert!(text.contains("\"tt.reduce\""));
    // Reducing axis=1 of <32x64> should yield <32xf32>.
    assert!(
        text.contains("(tensor<32x64xf32>) -> tensor<32xf32>"),
        "wrong reduce signature:\n{text}"
    );
}
