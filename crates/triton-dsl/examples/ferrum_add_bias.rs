//! ferrum kernel port: `add_bias` — broadcast bias add.
//!
//! Mirrors `ferrum-kernels/kernels/add_bias.cu`:
//!   data[r, c] += bias[c]   for r in [0, rows), c in [0, cols).
//!
//! Launch: grid = (rows, 1, 1), one tile = one row. Requires `cols <= BLOCK`.
//!
//! Dtype-generic via `T: TritonElem`. Pure elementwise add — no f32
//! accumulator needed. For `T == f32` the IR matches `add_bias_f32`.

use triton_dsl::triton_kernel;
use triton_ir::ty::{bf16, f16, TritonElem};

#[allow(dead_code)]
struct Ptr<T>(::std::marker::PhantomData<T>);

#[triton_kernel]
fn add_bias_typed<T: TritonElem, const BLOCK: usize>(
    data: Ptr<T>,
    bias: Ptr<T>,
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
    let dtype = std::env::args().nth(1).unwrap_or_else(|| "f32".into());
    match dtype.as_str() {
        "f32"  => print!("{}", add_bias_typed::<f32, 1024>::mlir()),
        "f16"  => print!("{}", add_bias_typed::<f16, 1024>::mlir()),
        "bf16" => print!("{}", add_bias_typed::<bf16, 1024>::mlir()),
        other => {
            eprintln!("unknown dtype `{other}` — supported: f32, f16, bf16");
            std::process::exit(2);
        }
    }
}
