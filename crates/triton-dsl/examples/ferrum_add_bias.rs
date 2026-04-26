//! ferrum kernel port: `add_bias` — broadcast bias add.
//!
//! Mirrors `ferrum-kernels/kernels/add_bias.cu`:
//!   data[r, c] += bias[c]   for r in [0, rows), c in [0, cols).
//!
//! Launch: grid = (rows, 1, 1), one tile = one row. Requires `cols <= BLOCK`.

use triton_dsl::triton_kernel;

#[allow(dead_code)]
struct Ptr<T>(::std::marker::PhantomData<T>);

#[triton_kernel]
fn add_bias_f32<const BLOCK: usize>(
    data: Ptr<f32>,
    bias: Ptr<f32>,
    rows: i32,
    cols: i32,
) {
    let _ = rows; // bound implicitly by grid_dim.x

    let row = program_id(0);
    let row_off = row * cols;
    let col_idx = make_range(0, BLOCK as i32);
    let mask = col_idx < splat_1d(cols, BLOCK as i64);

    let abs_off = splat_1d(row_off, BLOCK as i64) + col_idx;
    let data_ptrs = splat_1d(data, BLOCK as i64) + abs_off;
    let bias_ptrs = splat_1d(bias, BLOCK as i64) + col_idx;

    let dv = load(data_ptrs, mask);
    let bv = load(bias_ptrs, mask);

    store(data_ptrs, dv + bv, mask);
}

fn main() {
    print!("{}", add_bias_f32::<1024>::mlir());
}
