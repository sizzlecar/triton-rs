//! Dialect-specific op constructors. Each submodule mirrors a Triton-
//! consumed MLIR dialect and exports free functions returning `OpSpec`.
//!
//! Usage:
//! ```ignore
//! let pid = f.op_one(tt::get_program_id(0));
//! let c = f.op_one(arith::constant_i32(1024));
//! let mul = f.op_one(arith::muli(pid.clone(), c.clone()));
//! ```

pub mod arith;
pub mod scf;
pub mod tt;
