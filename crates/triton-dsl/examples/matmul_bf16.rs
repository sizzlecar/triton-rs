//! GEMM (general matrix multiplication) kernel: `C = A @ B` for bf16.
//!
//! Same algorithm as `matmul_f32` (canonical Triton tutorial 03), but with
//! bf16-typed pointers and a `to_f32 → dot → as_t::<bf16>` cast chain at
//! the load/store boundaries.
//!
//! ## Why bf16 with f32-internal compute?
//! bf16 has the same 8-bit exponent as f32, so partial-sum overflow is
//! much rarer than f16 — but mantissa precision is only 7 bits, so a
//! BK-long inner-product sum still loses precision fast in bf16. Same
//! mitigation as the f16 path: load bf16, upcast to f32 immediately,
//! accumulate in f32, downcast at store.
//!
//! The kernel body is byte-identical to `matmul_f16`'s `matmul_typed` —
//! the proc-macro generates a fresh struct per file, so duplicating the
//! body is the simplest way to keep both examples self-contained without
//! needing a shared library crate.

use triton_dsl::triton_kernel;
use triton_ir::ty::{bf16, TritonElem};

#[allow(dead_code)]
struct Ptr<T>(::std::marker::PhantomData<T>);

/// Block matmul `C[m, n] = sum_k A[m, k] * B[k, n]` — dtype-generic.
/// See `matmul_f16.rs` for the full discussion.
#[triton_kernel]
pub fn matmul_typed<T: TritonElem, const BM: usize, const BN: usize, const BK: usize>(
    a_ptr: Ptr<T>, // [M, K]
    b_ptr: Ptr<T>, // [K, N]
    c_ptr: Ptr<T>, // [M, N]
    m_size: i32,
    n_size: i32,
    k_size: i32,
    stride_am: i32,
    stride_ak: i32,
    stride_bk: i32,
    stride_bn: i32,
    stride_cm: i32,
    stride_cn: i32,
) {
    let pid_m = program_id(0);
    let pid_n = program_id(1);

    let offs_m = pid_m * (BM as i32) + make_range(0, BM as i32);
    let offs_n = pid_n * (BN as i32) + make_range(0, BN as i32);
    let offs_k0 = make_range(0, BK as i32);

    let mask_m = offs_m < m_size;
    let mask_n = offs_n < n_size;

    let offs_m_2d = expand_dims(offs_m, 1);
    let offs_n_2d = expand_dims(offs_n, 0);

    let zero = const_f32(0.0_f32);
    let zero_1 = splat_1d(zero, 1);
    let zero_2d = expand_dims(zero_1, 0);
    let acc_init = broadcast_2d(zero_2d, BM as i64, BN as i64);

    let k_blocks = (k_size + (BK as i32) - 1) / (BK as i32);

    let acc = scf_for(const_i32(0), k_blocks, const_i32(1), acc_init, |kb, acc| {
        let offs_k = kb * (BK as i32) + offs_k0;
        let mask_k = offs_k < k_size;
        let offs_k_row = expand_dims(offs_k, 0);
        let offs_k_col = expand_dims(offs_k, 1);

        let a_off = offs_m_2d * stride_am + offs_k_row * stride_ak;
        let mask_a = expand_dims(mask_m, 1) & expand_dims(mask_k, 0);
        let a_block = load(a_ptr + a_off, mask_a);

        let b_off = offs_k_col * stride_bk + offs_n_2d * stride_bn;
        let mask_b = expand_dims(mask_k, 1) & expand_dims(mask_n, 0);
        let b_block = load(b_ptr + b_off, mask_b);

        // dot has bf16 inputs + f32 accumulator → native bf16-bf16-f32
        // Tensor-Core mma. No extf chain, no f32 mma fallback. (For
        // T == f32 instantiations this collapses to f32-f32-f32 dot.)
        dot(a_block, b_block, acc)
    });

    let acc_t = as_t::<T>(acc);
    let c_off = offs_m_2d * stride_cm + offs_n_2d * stride_cn;
    let mask_c = expand_dims(mask_m, 1) & expand_dims(mask_n, 0);
    store(c_ptr + c_off, acc_t, mask_c);
}

fn main() {
    // 64×64×32 — same default as matmul_f32 / matmul_f16. bf16 input
    // tiles are 4 KB each; f32 accumulator stays at 16 KB.
    print!("{}", matmul_typed::<bf16, 64, 64, 32>::mlir());
}
