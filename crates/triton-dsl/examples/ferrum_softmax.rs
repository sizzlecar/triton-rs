//! ferrum kernel port: `softmax` over the last dim.
//!
//! Mirrors `ferrum-kernels/kernels/softmax.cu`:
//!   out[r, c] = exp(x[r, c] - max(x[r])) / sum(exp(x[r, :] - max(x[r])))
//!
//! Launch: grid = (rows, 1, 1). Assumes `cols == BLOCK` for now (the mask
//! path needs the optional `other` operand on tt.load to set masked
//! lanes to -inf for the max-reduce pass; deferred).
//!
//! ## Dtype-generic
//! Loads upcast to f32; `max`, `exp`, `sum`, and the divide all run in
//! f32 (NVPTX has no native f16 `math.exp`; an f16 sum-of-exps would
//! overflow on logits with magnitude > 5). Stores downcast back to T
//! via `as_t::<T>`. For `T == f32` casts collapse.

use triton_dsl::triton_kernel;
use triton_ir::ty::{bf16, f16, TritonElem};

#[allow(dead_code)]
struct Ptr<T>(::std::marker::PhantomData<T>);

#[triton_kernel]
fn softmax_typed<T: TritonElem, const BLOCK: usize>(
    input: Ptr<T>,
    output: Ptr<T>,
    rows: i32,
    cols: i32,
) {
    let _ = rows;
    let row = program_id(0);
    let row_off = row * cols;
    let col_idx = make_range(0, BLOCK as i32);
    let mask = col_idx < splat_1d(cols, BLOCK as i64);
    let abs_off = splat_1d(row_off, BLOCK as i64) + col_idx;

    let xv = to_f32(load(splat_1d(input, BLOCK as i64) + abs_off, mask));

    // Phase 1: row max (f32).
    let row_max = reduce(xv, 0, |a, b| max(a, b));

    // Phase 2: exp(x - max), sum (f32).
    let max_v = splat_1d(row_max, BLOCK as i64);
    let shifted = xv - max_v;
    let exp_v = exp(shifted);
    let sum_e = reduce(exp_v, 0, |a, b| a + b);

    // Phase 3: divide (f32) → downcast to T at the store boundary.
    let inv_sum = const_f32(1.0) / sum_e;
    let result = exp_v * splat_1d(inv_sum, BLOCK as i64);

    store(splat_1d(output, BLOCK as i64) + abs_off, as_t::<T>(result), mask);
}

fn main() {
    let dtype = std::env::args().nth(1).unwrap_or_else(|| "f32".into());
    match dtype.as_str() {
        "f32"  => print!("{}", softmax_typed::<f32, 1024>::mlir()),
        "f16"  => print!("{}", softmax_typed::<f16, 1024>::mlir()),
        "bf16" => print!("{}", softmax_typed::<bf16, 1024>::mlir()),
        other => {
            eprintln!("unknown dtype `{other}` — supported: f32, f16, bf16");
            std::process::exit(2);
        }
    }
}
