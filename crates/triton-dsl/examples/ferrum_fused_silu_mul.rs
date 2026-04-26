//! ferrum kernel port: `fused_silu_mul` — `output[i] = silu(gate[i]) * up[i]`
//! where `silu(x) = x / (1 + exp(-x))`. Used in MLP gate projection
//! (`down_proj(silu(gate) * up)`).
//!
//! Mirrors `ferrum-kernels/kernels/fused_silu_mul.cu`. f32 variant only —
//! f16 needs the runner host buffer to use `half::f16`, which is a
//! follow-up.

use triton_dsl::triton_kernel;

#[allow(dead_code)]
struct Ptr<T>(::std::marker::PhantomData<T>);

#[triton_kernel]
fn fused_silu_mul_f32<const BLOCK: usize>(
    gate: Ptr<f32>,
    up: Ptr<f32>,
    out: Ptr<f32>,
    n: i32,
) {
    let pid = program_id(0);
    let off = splat_1d(pid * const_i32(BLOCK as i32), BLOCK as i64)
            + make_range(0, BLOCK as i32);
    let mask = off < splat_1d(n, BLOCK as i64);

    let gv = load(splat_1d(gate, BLOCK as i64) + off, mask);
    let uv = load(splat_1d(up,   BLOCK as i64) + off, mask);

    // silu(g) = g / (1 + exp(-g)). The operator overloads dispatch into
    // the float variant of each arith op based on the tensor element type.
    let one = splat_1d(const_f32(1.0), BLOCK as i64);
    let zero = splat_1d(const_f32(0.0), BLOCK as i64);
    let neg_g = zero - gv;
    let exp_neg_g = exp(neg_g);
    let denom = one + exp_neg_g;
    let silu_g = gv / denom;

    let result = silu_g * uv;
    store(splat_1d(out, BLOCK as i64) + off, result, mask);
}

fn main() {
    print!("{}", fused_silu_mul_f32::<1024>::mlir());
}
