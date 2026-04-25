//! Print the canonical vector_add MLIR module to stdout. Used by
//! `tools/validate_mlir.py` to feed the generated IR into a real Triton
//! parser for round-trip validation.

use triton_ir::dialect::arith::CmpiPred;
use triton_ir::prelude::*;

const BLOCK: i64 = 1024;

fn main() {
    let mut m = Module::new();
    let mut f = m.func("vector_add");
    let x_ptr = f.arg("x", Type::ptr(Type::f32()));
    let y_ptr = f.arg("y", Type::ptr(Type::f32()));
    let out_ptr = f.arg("out", Type::ptr(Type::f32()));
    let n = f.arg("n", Type::i32());

    let pid = f.op_one(tt::get_program_id(0));
    let block_const = f.op_one(arith::constant_i32(BLOCK as i32));
    let base = f.op_one(arith::muli(pid, block_const));
    let range = f.op_one(tt::make_range(0, BLOCK as i32));
    let base_v = f.op_one(tt::splat(base, vec![BLOCK]));
    let off = f.op_one(arith::addi(base_v, range));

    let n_v = f.op_one(tt::splat(n, vec![BLOCK]));
    let mask = f.op_one(arith::cmpi(CmpiPred::Slt, off.clone(), n_v));

    let xp = f.op_one(tt::splat(x_ptr, vec![BLOCK]));
    let xp_off = f.op_one(tt::addptr(xp, off.clone()));
    let xv = f.op_one(tt::load(xp_off, Some(mask.clone())));

    let yp = f.op_one(tt::splat(y_ptr, vec![BLOCK]));
    let yp_off = f.op_one(tt::addptr(yp, off.clone()));
    let yv = f.op_one(tt::load(yp_off, Some(mask.clone())));

    let sum = f.op_one(arith::addf(xv, yv));

    let outp = f.op_one(tt::splat(out_ptr, vec![BLOCK]));
    let outp_off = f.op_one(tt::addptr(outp, off));
    f.op_void(tt::store(outp_off, sum, Some(mask)));

    f.op_void(tt::return_());
    f.finish();

    print!("{}", m);
}
