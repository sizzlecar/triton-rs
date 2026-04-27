//! GEMM (general matrix multiplication) kernel: `C = A @ B` for f32.
//!
//! Mirrors the canonical Triton tutorial pattern
//! (`python/tutorials/03-matrix-multiplication.py`) translated to triton-rs DSL.
//!
//! ## Shapes (row-major)
//! - `A` : `[M, K]` element pointer `Ptr<f32>`
//! - `B` : `[K, N]`
//! - `C` : `[M, N]`
//!
//! ## Strides
//! Strides are passed in *elements*, not bytes — same convention as Triton.
//! Row-major matrices satisfy `stride_*m == cols` and `stride_*n == 1`, but
//! the kernel does not assume row-major: any combination of strides works
//! (a `[K, M]` column-major A is just `stride_am=1, stride_ak=M`).
//!
//! ## Launch
//! Grid: `(ceil(M / BM), ceil(N / BN), 1)`. One program produces one
//! `BM × BN` output tile by streaming `BK`-wide tiles of A and B and
//! accumulating into a register-resident `BM × BN` f32 tile.
//!
//! ## Tile sizes
//! `<64, 64, 32>` is a safe starting point for f32 — keeps the accumulator
//! at 64*64*4 = 16 KB which easily fits per-CTA in registers/shared memory.
//! Tune up (128/128/32, 128/64/64) once the kernel runs end-to-end.
//!
//! ## v0 limitations
//! - f32 only. f16 / bf16 are a small extension once we settle on whether
//!   to expose the `inputPrecision` attribute on `tt.dot` from the DSL.
//! - Boundary handling uses 2D masks built from outer-AND of 1D bounds
//!   tensors, so `M`, `N`, `K` may be any positive integer (no padding
//!   requirement).
//! - No `tt.make_block_ptr` — the kernel uses manual pointer arithmetic
//!   (splat scalar pointer + add 2D offset tensor). This matches what the
//!   Triton tutorial generates after canonicalization, just without going
//!   through the syntactic block-pointer sugar.

use triton_dsl::triton_kernel;

#[allow(dead_code)]
struct Ptr<T>(::std::marker::PhantomData<T>);

/// Block matmul `C[m, n] = sum_k A[m, k] * B[k, n]`. f32 throughout.
///
/// `BM`, `BN`, `BK` are the per-program tile dimensions. They must be powers
/// of two for the canonical Triton lowering to vectorize cleanly; the kernel
/// itself doesn't enforce this.
#[triton_kernel]
pub fn matmul_f32<const BM: usize, const BN: usize, const BK: usize>(
    a_ptr: Ptr<f32>, // [M, K]
    b_ptr: Ptr<f32>, // [K, N]
    c_ptr: Ptr<f32>, // [M, N]
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

    // 1D index ranges along each tile dim.
    let offs_m = pid_m * (BM as i32) + make_range(0, BM as i32); // [BM]   i32
    let offs_n = pid_n * (BN as i32) + make_range(0, BN as i32); // [BN]   i32
    let offs_k0 = make_range(0, BK as i32); // [BK]   i32

    // Boundary masks (1D). We expand-dims them to 2D inside each load/store
    // so we don't pay for the broadcast unless we use it.
    let mask_m = offs_m < m_size; // [BM]   i1
    let mask_n = offs_n < n_size; // [BN]   i1

    // Lift M and N coords to 2D — these are reused both for the A/B tile
    // pointer construction and the output tile mask.
    let offs_m_2d = expand_dims(offs_m, 1); // [BM, 1] i32
    let offs_n_2d = expand_dims(offs_n, 0); // [1, BN] i32

    // ── accumulator init: tensor<BM x BN x f32> of zeros ──
    // Build a 1×1 zero tile then broadcast — same pattern as
    // `dump_matmul_one_block.rs`, just larger.
    let zero = const_f32(0.0_f32);
    let zero_1 = splat_1d(zero, 1); // [1]    f32
    let zero_2d = expand_dims(zero_1, 0); // [1, 1] f32
    let acc_init = broadcast_2d(zero_2d, BM as i64, BN as i64); // [BM, BN] f32

    // ── streaming K loop ──
    let k_blocks = (k_size + (BK as i32) - 1) / (BK as i32);

    let acc = scf_for(const_i32(0), k_blocks, const_i32(1), acc_init, |kb, acc| {
        // Per-tile K offsets.
        let offs_k = kb * (BK as i32) + offs_k0; // [BK] i32
        let mask_k = offs_k < k_size; // [BK] i1
        let offs_k_row = expand_dims(offs_k, 0); // [1, BK]  i32  — for A's col axis
        let offs_k_col = expand_dims(offs_k, 1); // [BK, 1]  i32  — for B's row axis

        // ── A tile [BM, BK] address: m * stride_am + k * stride_ak ──
        let a_off = offs_m_2d * stride_am + offs_k_row * stride_ak; // [BM, BK] i32
        let mask_k_row = expand_dims(mask_k, 0); // [1, BK]
        let mask_m_col = expand_dims(mask_m, 1); // [BM, 1]
        let mask_a = mask_m_col & mask_k_row; // [BM, BK] i1
        let a_block = load(a_ptr + a_off, mask_a); // [BM, BK] f32

        // ── B tile [BK, BN] address: k * stride_bk + n * stride_bn ──
        let b_off = offs_k_col * stride_bk + offs_n_2d * stride_bn; // [BK, BN] i32
        let mask_k_col = expand_dims(mask_k, 1); // [BK, 1]
        let mask_n_row = expand_dims(mask_n, 0); // [1, BN]
        let mask_b = mask_k_col & mask_n_row; // [BK, BN] i1
        let b_block = load(b_ptr + b_off, mask_b); // [BK, BN] f32

        // ── block matmul: acc += A_block @ B_block ──
        // tt.dot signature: `(a, b, c) -> result_ty == c.ty()`.
        // Result shape: [BM, BN] f32.
        dot(a_block, b_block, acc)
    });

    // ── store C tile [BM, BN] ──
    let c_off = offs_m_2d * stride_cm + offs_n_2d * stride_cn; // [BM, BN] i32
    let mask_m_col = expand_dims(mask_m, 1); // [BM, 1]
    let mask_n_row = expand_dims(mask_n, 0); // [1, BN]
    let mask_c = mask_m_col & mask_n_row; // [BM, BN] i1
    store(c_ptr + c_off, acc, mask_c);
}

fn main() {
    // 64×64×32 — safe default for f32 GEMM. Accumulator tile is
    // 64*64*4 = 16 KB, well within per-CTA register / shared memory budget
    // on every CUDA gen ≥ Turing.
    print!("{}", matmul_f32::<64, 64, 32>::mlir());
}
