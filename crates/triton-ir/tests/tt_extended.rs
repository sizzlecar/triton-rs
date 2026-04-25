//! Coverage for the extended tt dialect helpers (dot, broadcast,
//! expand_dims, reshape). Exercises each individually plus a synthetic
//! one-block matmul that wires them together.

use triton_ir::prelude::*;

#[test]
fn tt_dot_emits_correct_shape() {
    let mut m = Module::new();
    let mut f = m.func("dot_only");
    // Tensor args are unusual in real Triton kernels (which take scalars/
    // pointers/constexpr at the launch boundary) but are legal in MLIR
    // and let us test tt.dot in isolation without building an entire
    // tile-pointer pipeline.
    let a = f.arg("a", Type::tensor(vec![16, 32], Type::f16()));
    let b = f.arg("b", Type::tensor(vec![32, 16], Type::f16()));
    let c = f.arg("c", Type::tensor(vec![16, 16], Type::f32()));
    let _r = f.op_one(tt::dot(a, b, c));
    f.op_void(tt::return_());
    f.finish();

    let text = m.to_string();
    assert!(text.contains("\"tt.dot\""), "missing tt.dot:\n{text}");
    assert!(
        text.contains("(tensor<16x32xf16>, tensor<32x16xf16>, tensor<16x16xf32>) -> tensor<16x16xf32>"),
        "wrong dot signature:\n{text}"
    );
}

#[test]
fn tt_broadcast_preserves_elem_type() {
    let mut m = Module::new();
    let mut f = m.func("bcast");
    let x = f.arg("x", Type::tensor(vec![1, 128], Type::f32()));
    let _y = f.op_one(tt::broadcast(x, vec![64, 128]));
    f.op_void(tt::return_());
    f.finish();

    let text = m.to_string();
    assert!(text.contains("\"tt.broadcast\""));
    assert!(text.contains("(tensor<1x128xf32>) -> tensor<64x128xf32>"));
}

#[test]
fn tt_expand_dims_inserts_at_axis() {
    let mut m = Module::new();
    let mut f = m.func("exp");
    let x = f.arg("x", Type::tensor(vec![4], Type::i32()));
    // Insert a leading singleton dim → tensor<1x4xi32>.
    let _y = f.op_one(tt::expand_dims(x.clone(), 0));
    // Insert a trailing singleton dim → tensor<4x1xi32>.
    let _z = f.op_one(tt::expand_dims(x, 1));
    f.op_void(tt::return_());
    f.finish();

    let text = m.to_string();
    eprintln!("{text}");
    assert!(text.contains("\"tt.expand_dims\""));
    assert!(text.contains("{axis = 0 : i32}"));
    assert!(text.contains("{axis = 1 : i32}"));
    assert!(text.contains("-> tensor<1x4xi32>"));
    assert!(text.contains("-> tensor<4x1xi32>"));
}

#[test]
fn tt_reshape_changes_layout_keeps_elem() {
    let mut m = Module::new();
    let mut f = m.func("rshp");
    let x = f.arg("x", Type::tensor(vec![512], Type::i32()));
    let _y = f.op_one(tt::reshape(x, vec![16, 32]));
    f.op_void(tt::return_());
    f.finish();

    let text = m.to_string();
    assert!(text.contains("\"tt.reshape\""));
    assert!(text.contains("(tensor<512xi32>) -> tensor<16x32xi32>"));
}

#[test]
fn one_block_matmul_chains_dot_with_zero_accumulator() {
    // Synthetic 16×32 * 32×16 matmul into a zero accumulator. Exercises
    // dot with a real broadcast pipeline for the accumulator init.
    let mut m = Module::new();
    let mut f = m.func("matmul_one_block");
    let a = f.arg("a", Type::tensor(vec![16, 32], Type::f16()));
    let b = f.arg("b", Type::tensor(vec![32, 16], Type::f16()));

    // c_init = broadcast(splat_1d(0.0, 1), [16,16]) — but our splat is 1D.
    // Easier: splat to 1×1, expand to 16×16 via two broadcasts.
    let zero = f.op_one(arith::constant_f32(0.0));
    let zero_1 = f.op_one(tt::splat(zero, vec![1])); // tensor<1xf32>
    let zero_2d = f.op_one(tt::expand_dims(zero_1, 0)); // tensor<1x1xf32>
    let c_init = f.op_one(tt::broadcast(zero_2d, vec![16, 16])); // tensor<16x16xf32>

    let _result = f.op_one(tt::dot(a, b, c_init));
    f.op_void(tt::return_());
    f.finish();

    let text = m.to_string();
    eprintln!("===== matmul_one_block =====\n{text}");
    assert!(text.contains("\"tt.dot\""));
    assert!(text.contains("\"tt.broadcast\""));
    assert!(text.contains("\"tt.expand_dims\""));
    assert!(text.contains("tensor<16x16xf32>"));
}
