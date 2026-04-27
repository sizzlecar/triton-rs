//! Dump decode_attention_typed f16 with HEAD_DIM=64 (TinyLlama / smaller
//! Llamas). dump_all defaults to HEAD_DIM=128 (Llama-3 / Qwen sizes), so
//! this is the explicit smaller-head variant.

use triton_kernels::attention::decode_attention_typed;
use triton_ir::ty::f16;

fn main() {
    print!("{}", decode_attention_typed::<f16, 64, 32>::mlir());
}
