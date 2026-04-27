//! GEMM (general matrix multiplication) kernel: `C = A @ B` for f16.
//!
//! Same algorithm as `matmul_f32` (canonical Triton tutorial 03), but with
//! f16-typed pointers and a `to_f32 → dot → as_t::<f16>` cast chain at the
//! load/store boundaries.
//!
//! ## Why an f16 / f32-internal split?
//! Real LLM weights are stored as f16 (or bf16), but accumulating a sum of
//! BK f16 products in f16 loses precision fast — once partial sums hit
//! magnitudes above ±2^10 the next term gets rounded to the wrong half-ulp.
//! So we load f16, immediately upcast to f32, accumulate in f32, and
//! truncate back to f16 only at the store boundary. This matches what
//! Python `@triton.jit` does when input pointers are f16: Triton's
//! `tt.dot` emits `inputPrecision = ieee` and the optimizer fuses the
//! upcasts.
//!
//! In *this* DSL, `tt.dot`'s `inputPrecision` attribute isn't yet exposed,
//! so we keep the dot itself in f32 (a/b are both upcast first). On
//! NVIDIA hardware the optimizer will still pattern-match the
//! `extf → dot → truncf` chain into native f16 mma when it can; if it
//! can't, perf falls back to f32 mma. The follow-up — exposing
//! `inputPrecision` so `tt.dot` runs natively on f16 tensors with f32
//! accumulator — is captured in the kernel doc on `matmul_typed`.
//!
//! ## v0 limitations (vs matmul_f32)
//! - Same masking + tile shape semantics as matmul_f32.
//! - Tile size `<64, 64, 32>` chosen so the f32 accumulator stays at
//!   16 KB; the f16 input tiles are smaller (8 KB and 4 KB) and easily
//!   fit alongside.
//! - Native f16 mma via `tt.dot` `inputPrecision` attribute is a follow-up.

use triton_dsl::triton_kernel;
use triton_ir::ty::{f16, TritonElem};

#[allow(dead_code)]
struct Ptr<T>(::std::marker::PhantomData<T>);

/// Block matmul `C[m, n] = sum_k A[m, k] * B[k, n]` — dtype-generic.
///
/// `T` is the element type of all three pointers (A, B, C). Internal
/// compute is f32: the kernel loads at T, upcasts to f32 via `to_f32`,
/// accumulates in an f32 register tile, and downcasts back to T via
/// `as_t::<T>(...)` at store time. For T == f32 every cast collapses
/// and the IR matches `matmul_f32` exactly.
///
/// `BM`, `BN`, `BK` are the per-program tile dimensions; pick powers of
/// two for the canonical Triton lowering to vectorize cleanly.
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
    // ── per-program tile coordinates ──
    let pid_m = program_id(0);
    let pid_n = program_id(1);

    let offs_m = pid_m * (BM as i32) + make_range(0, BM as i32); // [BM]   i32
    let offs_n = pid_n * (BN as i32) + make_range(0, BN as i32); // [BN]   i32
    let offs_k0 = make_range(0, BK as i32); // [BK]   i32

    let mask_m = offs_m < m_size; // [BM]   i1
    let mask_n = offs_n < n_size; // [BN]   i1

    let offs_m_2d = expand_dims(offs_m, 1); // [BM, 1] i32
    let offs_n_2d = expand_dims(offs_n, 0); // [1, BN] i32

    // ── accumulator init: tensor<BM x BN x f32> of zeros ──
    // Build the accumulator in *f32* regardless of T — same precision
    // reasoning as flash_attn_full's online-softmax state (a stream of
    // BK*K_BLOCKS f16 multiplications loses precision in f16 around
    // magnitudes > 2^10).
    let zero = const_f32(0.0_f32);
    let zero_1 = splat_1d(zero, 1); // [1]    f32
    let zero_2d = expand_dims(zero_1, 0); // [1, 1] f32
    let acc_init = broadcast_2d(zero_2d, BM as i64, BN as i64); // [BM, BN] f32

    // ── streaming K loop ──
    let k_blocks = (k_size + (BK as i32) - 1) / (BK as i32);

    let acc = scf_for(const_i32(0), k_blocks, const_i32(1), acc_init, |kb, acc| {
        let offs_k = kb * (BK as i32) + offs_k0; // [BK] i32
        let mask_k = offs_k < k_size; // [BK] i1
        let offs_k_row = expand_dims(offs_k, 0); // [1, BK]  i32
        let offs_k_col = expand_dims(offs_k, 1); // [BK, 1]  i32

        // ── A tile [BM, BK] address: m * stride_am + k * stride_ak ──
        let a_off = offs_m_2d * stride_am + offs_k_row * stride_ak; // [BM, BK] i32
        let mask_k_row = expand_dims(mask_k, 0); // [1, BK]
        let mask_m_col = expand_dims(mask_m, 1); // [BM, 1]
        let mask_a = mask_m_col & mask_k_row; // [BM, BK] i1
        let a_block_t = load(a_ptr + a_off, mask_a); // [BM, BK] T
        let a_block = to_f32(a_block_t); // [BM, BK] f32

        // ── B tile [BK, BN] address: k * stride_bk + n * stride_bn ──
        let b_off = offs_k_col * stride_bk + offs_n_2d * stride_bn; // [BK, BN] i32
        let mask_k_col = expand_dims(mask_k, 1); // [BK, 1]
        let mask_n_row = expand_dims(mask_n, 0); // [1, BN]
        let mask_b = mask_k_col & mask_n_row; // [BK, BN] i1
        let b_block_t = load(b_ptr + b_off, mask_b); // [BK, BN] T
        let b_block = to_f32(b_block_t); // [BK, BN] f32

        // ── block matmul: acc += A_block @ B_block ──
        // a/b are now f32, c is f32 — dot result is f32. For T == f32
        // this matches matmul_f32 exactly; for T == f16 / bf16 the load
        // boundary upcast and the store boundary downcast bracket the
        // f32-internal compute.
        dot(a_block, b_block, acc)
    });

    // ── store C tile [BM, BN] ──
    // Downcast the f32 accumulator to T so the store matches c_ptr's
    // pointee type. For T == f32 this is the identity (the cast collapses
    // in the IR).
    let acc_t = as_t::<T>(acc);
    let c_off = offs_m_2d * stride_cm + offs_n_2d * stride_cn; // [BM, BN] i32
    let mask_m_col = expand_dims(mask_m, 1); // [BM, 1]
    let mask_n_row = expand_dims(mask_n, 0); // [1, BN]
    let mask_c = mask_m_col & mask_n_row; // [BM, BN] i1
    store(c_ptr + c_off, acc_t, mask_c);
}

fn main() {
    // 64×64×32 — same default as matmul_f32. With T = f16 the input
    // tiles are 64*32*2 = 4 KB (A) + 32*64*2 = 4 KB (B), well below
    // shared memory budget; the f32 accumulator stays at 16 KB.
    print!("{}", matmul_typed::<f16, 64, 64, 32>::mlir());
}
