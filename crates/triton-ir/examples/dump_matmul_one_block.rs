//! Dump a synthetic 16×32 * 32×16 matmul that exercises tt.dot,
//! tt.broadcast, tt.expand_dims, and tt.splat. Validate via
//! `tools/validate_mlir.py`.

use triton_ir::prelude::*;

fn main() {
    let mut m = Module::new();
    let mut f = m.func("matmul_one_block");
    let a = f.arg("a", Type::tensor(vec![16, 32], Type::f16()));
    let b = f.arg("b", Type::tensor(vec![32, 16], Type::f16()));

    let zero = f.op_one(arith::constant_f32(0.0));
    let zero_1 = f.op_one(tt::splat(zero, vec![1]));
    let zero_2d = f.op_one(tt::expand_dims(zero_1, 0));
    let c_init = f.op_one(tt::broadcast(zero_2d, vec![16, 16]));

    let _result = f.op_one(tt::dot(a, b, c_init));
    f.op_void(tt::return_());
    f.finish();

    print!("{}", m);
}
