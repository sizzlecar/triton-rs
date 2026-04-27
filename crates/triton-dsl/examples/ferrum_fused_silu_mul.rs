//! ferrum kernel port: `fused_silu_mul` — `output[i] = silu(gate[i]) * up[i]`
//! where `silu(x) = x / (1 + exp(-x))`. Used in MLP gate projection
//! (`down_proj(silu(gate) * up)`).
//!
//! Mirrors `ferrum-kernels/kernels/fused_silu_mul.cu`.
//!
//! ## Dtype-generic
//! Body is parameterized by `T: TritonElem`. Loads upcast to f32 via
//! `to_f32`; the `exp(-g)`, division, and multiply all happen in f32
//! (NVPTX has no native f16 division and `math.exp` operates on f32);
//! the final result is downcast back to T via `as_t::<T>(...)` at the
//! store boundary. For `T == f32` the cast pairs collapse and the IR
//! matches the original f32-only kernel.

use triton_dsl::triton_kernel;
use triton_ir::ty::{bf16, f16, TritonElem};

#[allow(dead_code)]
struct Ptr<T>(::std::marker::PhantomData<T>);

#[triton_kernel]
fn fused_silu_mul_typed<T: TritonElem, const BLOCK: usize>(
    gate: Ptr<T>,
    up: Ptr<T>,
    out: Ptr<T>,
    n: i32,
) {
    let pid = program_id(0);
    let off = splat_1d(pid * const_i32(BLOCK as i32), BLOCK as i64)
            + make_range(0, BLOCK as i32);
    let mask = off < splat_1d(n, BLOCK as i64);

    let gv = to_f32(load(splat_1d(gate, BLOCK as i64) + off, mask));
    let uv = to_f32(load(splat_1d(up,   BLOCK as i64) + off, mask));

    // silu(g) = g / (1 + exp(-g)). Body runs entirely in f32 — same
    // reasoning as the matmul_typed accumulator: f16 division +
    // math.exp don't have native NVPTX implementations, so the
    // optimizer would have to emit upcasts anyway.
    let one = splat_1d(const_f32(1.0), BLOCK as i64);
    let zero = splat_1d(const_f32(0.0), BLOCK as i64);
    let neg_g = zero - gv;
    let exp_neg_g = exp(neg_g);
    let denom = one + exp_neg_g;
    let silu_g = gv / denom;

    let result = silu_g * uv;
    store(splat_1d(out, BLOCK as i64) + off, as_t::<T>(result), mask);
}

fn main() {
    let dtype = std::env::args().nth(1).unwrap_or_else(|| "f32".into());
    match dtype.as_str() {
        "f32"  => print!("{}", fused_silu_mul_typed::<f32, 1024>::mlir()),
        "f16"  => print!("{}", fused_silu_mul_typed::<f16, 1024>::mlir()),
        "bf16" => print!("{}", fused_silu_mul_typed::<bf16, 1024>::mlir()),
        other => {
            eprintln!("unknown dtype `{other}` — supported: f32, f16, bf16");
            std::process::exit(2);
        }
    }
}
