//! ferrum kernel port: `softmax` over the last dim.
//!
//! Mirrors `ferrum-kernels/kernels/softmax.cu`:
//!   out[r, c] = exp(x[r, c] - max(x[r])) / sum(exp(x[r, :] - max(x[r])))
//!
//! Launch: grid = (rows, 1, 1). Assumes `cols == BLOCK` for now (the mask
//! path needs the optional `other` operand on tt.load to set masked
//! lanes to -inf for the max-reduce pass; deferred).

use triton_dsl::triton_kernel;

#[allow(dead_code)]
struct Ptr<T>(::std::marker::PhantomData<T>);

#[triton_kernel]
fn softmax_f32<const BLOCK: usize>(
    input: Ptr<f32>,
    output: Ptr<f32>,
    rows: i32,
    cols: i32,
) {
    let _ = rows;
    let row = program_id(0);
    let row_off = row * cols;
    let col_idx = make_range(0, BLOCK as i32);
    let mask = col_idx < splat_1d(cols, BLOCK as i64);
    let abs_off = splat_1d(row_off, BLOCK as i64) + col_idx;

    let xv = load(splat_1d(input, BLOCK as i64) + abs_off, mask);

    // Phase 1: row max.
    let row_max = reduce(xv, 0, |a, b| max(a, b));

    // Phase 2: exp(x - max), sum.
    let max_v = splat_1d(row_max, BLOCK as i64);
    let shifted = xv - max_v;
    let exp_v = exp(shifted);
    let sum_e = reduce(exp_v, 0, |a, b| a + b);

    // Phase 3: divide.
    let inv_sum = const_f32(1.0) / sum_e;
    let result = exp_v * splat_1d(inv_sum, BLOCK as i64);

    store(splat_1d(output, BLOCK as i64) + abs_off, result, mask);
}

fn main() {
    print!("{}", softmax_f32::<1024>::mlir());
}
