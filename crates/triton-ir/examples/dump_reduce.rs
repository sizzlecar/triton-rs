//! Dump a reduce kernel for Triton round-trip validation.

use triton_ir::prelude::*;

fn main() {
    let mut m = Module::new();
    let mut f = m.func("sum_reduce_2d");
    let input = f.arg("input", Type::tensor(vec![32, 64], Type::f32()));
    let _result = f.reduce_with(input, 1, |fb, lhs, rhs| fb.add(lhs, rhs));
    f.op_void(tt::return_());
    f.finish();
    print!("{}", m);
}
