//! Dump the `sum_0_to_10` scf.for kernel to stdout for round-trip
//! validation against a real Triton install (see `tools/validate_mlir.py`).

use triton_ir::prelude::*;

fn main() {
    let mut m = Module::new();
    let mut f = m.func("sum_0_to_10");
    let out_ptr = f.arg("out", Type::ptr(Type::i32()));

    let c0 = f.op_one(arith::constant_i32(0));
    let c1 = f.op_one(arith::constant_i32(1));
    let c10 = f.op_one(arith::constant_i32(10));

    let (mut body, body_args) = f.new_region(vec![Type::i32(), Type::i32()]);
    let i = body_args[0].clone();
    let acc = body_args[1].clone();

    let new_acc = f.append_to_region_one(&mut body, arith::addi(acc, i));
    f.append_to_region_void(&mut body, scf::yield_(vec![new_acc]));

    let result = f.op_one(scf::for_loop(c0.clone(), c10, c1, vec![c0], body));

    f.op_void(tt::store(out_ptr, result, None));
    f.op_void(tt::return_());
    f.finish();

    print!("{}", m);
}
