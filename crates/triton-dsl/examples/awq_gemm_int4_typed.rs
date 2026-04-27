//! AWQ INT4 weight × FP16 activation GEMM kernel — DSL port of vLLM's
//! `awq_gemm_kernel` (see `vllm/model_executor/layers/quantization/awq_triton.py`).
//!
//! This is the foundation for replacing Marlin (hand-tuned CUTLASS C++) with
//! a triton-rs path on Blackwell. AWQ is the second 4-bit format in the
//! ferrum stack — sibling to `ferrum_dequant_int4.rs` (GPTQ format), but
//! packed along **N** (not K) and using the AWQ "reverse order" lane
//! layout.
//!
//! ## Layout (different from GPTQ — read carefully)
//!
//! | tensor   | shape           | dtype | packing axis      |
//! |----------|-----------------|-------|-------------------|
//! | input    | `[M, K]`        | f16   | (none, dense)     |
//! | qweight  | `[K, N/8]`      | i32   | 8 nibbles per N   |
//! | scales   | `[K/G, N]`      | f16   | (none, dense)     |
//! | qzeros   | `[K/G, N/8]`    | i32   | 8 nibbles per N   |
//! | output   | `[M, N]`        | f16   | (none, dense)     |
//!
//! `G = group_size`, typically 128. For each output column `n`:
//!   - source int32 lane index in the packed dim = `n / 8`
//!   - bit position within int32 = `awq_order[n % 8] * 4`
//!   - dequant value = `((qw_packed >> shift) & 0xF) - ((qz_packed >> shift) & 0xF)`
//!     scaled by `scales[k/G, n]`
//!
//! AWQ's `reverse_awq_order_tensor = [0, 4, 1, 5, 2, 6, 3, 7]`. For column
//! `n`, the lane within the int32 follows `(n % 8 / 2) + (n % 8 % 2) * 4`,
//! which matches the table at indices 0..8. This deliberately differs from
//! GPTQ's straight `(col % 8)` ordering — AWQ ships its weights this way
//! because the reorder reduces shared-memory bank conflicts on Volta/Turing
//! INT8/INT4 mma layouts, and we have to honor it bit-for-bit.
//!
//! ## Algorithm (per program)
//!
//! 1. Tile `[BM, BN]` of the output. Stream BK-wide K-tiles.
//! 2. Each K-tile:
//!    - Load A `[BM, BK]` fp16 (dense).
//!    - Load qweight `[BK, BN]` int32 — lanes `(k, n)` read packed int32 at
//!      `qweight[k, n/8]`, so lanes 0..8 along N share one int32. We do
//!      this by computing a 2D offset tensor with `n_idx / 8` along the N
//!      axis (multiple lanes alias the same address — load is idempotent).
//!    - Compute per-N shift = `awq_order[n % 8] * 4` once outside the loop
//!      (depends only on N coords, loop-invariant).
//!    - Load qzeros `[BN]` int32 + unpack to `[BN]` int32 — single group per
//!      K-tile (assumes `BK <= group_size`, the common case for G=128).
//!    - Load scales `[BN]` f16 (broadcast along K to `[BK, BN]` at use).
//!    - Dequant inline: `((qw >> shift) & 0xF) - zero` → cast to f16 → `* scale`.
//!    - `acc += dot(a_f16, dequant_f16, acc_f32)` — native f16-f16-f32 mma.
//! 3. Store: `truncf(acc_f32) → f16`.
//!
//! ## Skipped from vLLM
//!
//! - **SPLIT_K**: vLLM uses a 2D grid `(M*N, split_k_iters)` and writes a
//!   `[split_k, M, N]` partial output that's reduced on the host. v0 here
//!   uses single-K (split_k_iters = 1) for clarity. Split-K is mechanical
//!   to bolt on once the base kernel is correct: add a `pid_z` dim, scale
//!   the K-loop bound, and offset the C tile by `pid_z * M * N`.
//! - **`tl.interleave`**: vLLM 3x interleaves the loaded packed tile to
//!   broadcast each int32 across its 8 N-lanes. We achieve the same via 2D
//!   pointer arithmetic (`n_idx / 8` along N) — the lanes alias the same
//!   address on load.
//! - **Per-token scale / tensor-rank scale variants**: AWQ's standard fp16
//!   path doesn't need them; reserved for follow-up if AWQ-MoE needs it.
//!
//! ## v0 limitations
//!
//! - `BK <= group_size` assumed (single group per K-tile load). For G=128
//!   and BK=32 this is comfortable. If a model ships G=32, BK must be ≤ 32.
//! - Single-precision accumulator only (matches vLLM); INT32 accumulation
//!   variant is a follow-up.
//!
//! ## Note on shape coercion
//!
//! Rust binary operators (`+ - * / % & | ^`) auto-broadcast singleton dims
//! through `coerce_elemwise`. Named DSL calls like `bit_and(...)` and
//! `shr_u_i32(...)` go through the raw `arith::andi` / `arith::shrui`
//! constructors and *don't* auto-broadcast — operand shapes must match
//! exactly. For 2D nibble unpack we explicitly `broadcast_2d` shifts and
//! mask tensors to `[BK, BN]` before the bitwise ops.

use triton_dsl::triton_kernel;
use triton_ir::ty::{f16, TritonElem};

#[allow(dead_code)]
struct Ptr<T>(::std::marker::PhantomData<T>);

/// AWQ INT4-weight × FP16-act GEMM. `acc[m, n] = sum_k a[m, k] * dequant_b[k, n]`.
///
/// Type-generic over `T` for activation/scale/output dtype (typical: f16).
/// qweight and qzeros are always int32 (8 nibbles per int32).
///
/// `BM`, `BN`, `BK` are per-program tile dims. `BN` must be a multiple of 8.
/// `BK` must be ≤ `group_size` (vLLM constraint, usually trivially true).
#[triton_kernel]
pub fn awq_gemm_int4_typed<T: TritonElem, const BM: usize, const BN: usize, const BK: usize>(
    a_ptr: Ptr<T>,         // [M, K] fp16 input (activations)
    qweight_ptr: Ptr<i32>, // [K, N/8] int32 packed (8 nibbles per int32 along N)
    scales_ptr: Ptr<T>,    // [K/group_size, N] fp16
    qzeros_ptr: Ptr<i32>,  // [K/group_size, N/8] int32 packed
    c_ptr: Ptr<T>,         // [M, N] fp16 output
    m_size: i32,
    n_size: i32,
    k_size: i32,
    group_size: i32,
    stride_am: i32,
    stride_ak: i32,
    stride_qwk: i32, // typically N/8
    stride_qwn: i32, // typically 1
    stride_sk: i32,  // typically N
    stride_sn: i32,  // typically 1
    stride_qzk: i32, // typically N/8
    stride_qzn: i32, // typically 1
    stride_cm: i32,
    stride_cn: i32,
) {
    // ── per-program tile coordinates ──
    let pid_m = program_id(0);
    let pid_n = program_id(1);

    // M-axis offsets (rows of A and C). [BM] i32.
    let offs_m = pid_m * (BM as i32) + make_range(0, BM as i32);
    let mask_m = offs_m < m_size; // [BM] i1

    // N-axis offsets (cols of dequant_b and C). [BN] i32.
    let offs_n = pid_n * (BN as i32) + make_range(0, BN as i32);
    let mask_n = offs_n < n_size; // [BN] i1

    // K-axis offsets within one tile, base 0 — re-base per K iteration. [BK] i32.
    let offs_k0 = make_range(0, BK as i32);

    // 2D-broadcast forms of the M/N offsets (reused every K iteration).
    let offs_m_2d = expand_dims(offs_m, 1); // [BM, 1] i32
    let offs_n_2d = expand_dims(offs_n, 0); // [1, BN] i32

    // ── AWQ shift table per N column (loop-invariant) ──
    //
    // For N column `n`: int32 lane = `n / 8`, shift = `awq_order[n % 8] * 4`.
    //
    //   awq_order = [0, 4, 1, 5, 2, 6, 3, 7]
    //   awq_order[i] = (i / 2) + (i % 2) * 4
    //
    // So `shift[n] = ((n % 8) / 2) + ((n % 8) % 2) * 4) * 4`.
    //                                     └ value 0..7   ┘  └ * 4 → bit position 0..28
    let bn_eight = splat_1d(const_i32(8), BN as i64);
    let bn_two = splat_1d(const_i32(2), BN as i64);
    let bn_four = splat_1d(const_i32(4), BN as i64);
    let lane_in_int32 = offs_n % bn_eight; // [BN] = n % 8 (range 0..7)
    let lane_high = lane_in_int32 / bn_two; // [BN] = (n%8) / 2 (range 0..3)
    let lane_low = lane_in_int32 % bn_two; // [BN] = (n%8) % 2 (range 0..1)
    let awq_lane = lane_high + lane_low * bn_four; // [BN] = awq_order[n%8] (range 0..7)
    let shifts_1d = awq_lane * bn_four; // [BN] = awq_order[n%8] * 4 (bit pos 0..28)
    let shifts_row = expand_dims(shifts_1d, 0); // [1, BN] i32
    let shifts_2d = broadcast_2d(shifts_row, BK as i64, BN as i64); // [BK, BN] i32

    // 0xF mask broadcast to [BK, BN] for the 2D nibble extract; qzero
    // needs only the [BN]-shape variant, so we keep both.
    let f_mask_1d = splat_1d(const_i32(15), BN as i64); // [BN]
    let f_mask_row = expand_dims(f_mask_1d, 0); // [1, BN]
    let f_mask_2d = broadcast_2d(f_mask_row, BK as i64, BN as i64); // [BK, BN]

    // Per-N column-index in the packed tensors: `n_idx / 8`. [BN] i32.
    let packed_col = offs_n / bn_eight; // [BN]
    let packed_col_2d = expand_dims(packed_col, 0); // [1, BN] i32

    // ── accumulator init: [BM, BN] f32 of zeros ──
    let zero = const_f32(0.0_f32);
    let zero_1 = splat_1d(zero, 1);
    let zero_2d_init = expand_dims(zero_1, 0);
    let acc_init = broadcast_2d(zero_2d_init, BM as i64, BN as i64); // [BM, BN] f32

    // Number of K-tiles. Round up so the masked tail handles K not divisible by BK.
    let k_blocks = (k_size + (BK as i32) - 1) / (BK as i32);

    // ── streaming K loop ──
    let acc = scf_for(const_i32(0), k_blocks, const_i32(1), acc_init, |kb, acc| {
        // Global K offset for this tile's first row.
        let k_base = kb * (BK as i32); // i32 scalar
        let offs_k = splat_1d(k_base, BK as i64) + offs_k0; // [BK] i32
        let mask_k = offs_k < k_size; // [BK] i1

        let offs_k_row = expand_dims(offs_k, 0); // [1, BK] i32
        let offs_k_col = expand_dims(offs_k, 1); // [BK, 1] i32

        // ── Load A tile [BM, BK] at native dtype T ──
        let a_off = offs_m_2d * stride_am + offs_k_row * stride_ak; // [BM, BK] i32
        let mask_a = expand_dims(mask_m, 1) & expand_dims(mask_k, 0); // [BM, BK] i1
        let a_block = load(a_ptr + a_off, mask_a); // [BM, BK] T

        // ── Load qweight packed tile [BK, BN] int32 ──
        // For each lane (k, n), read qweight[k, n/8]. Lanes that share an
        // n/8 alias the same int32 — load is idempotent and Triton's load
        // coalescing handles the redundant addresses.
        let qw_off = offs_k_col * stride_qwk + packed_col_2d * stride_qwn; // [BK, BN] i32
        let mask_qw = expand_dims(mask_k, 1) & expand_dims(mask_n, 0); // [BK, BN] i1
        let qw_block = load(qweight_ptr + qw_off, mask_qw); // [BK, BN] i32

        // ── Compute group index and load scales / qzeros (single group per tile) ──
        // Assume BK <= group_size, so the entire BK-tile shares one group.
        // (vLLM makes the same assumption — see AWQ_TRITON_SUPPORTED_GROUP_SIZES.)
        let group = k_base / group_size; // i32 scalar

        // qzeros[group, n/8] → [BN] i32. Then unpack with the same shift as qweight.
        let qz_off_1d = splat_1d(group * stride_qzk, BN as i64) + packed_col * stride_qzn; // [BN] i32
        let qz_packed = load(qzeros_ptr + qz_off_1d, mask_n); // [BN] i32
        let zero_int4 = bit_and(shr_u_i32(qz_packed, shifts_1d), f_mask_1d); // [BN] i32

        // scales[group, n] → [BN] T.
        let s_off_1d = splat_1d(group * stride_sk, BN as i64) + offs_n * stride_sn; // [BN] i32
        let scale_t = load(scales_ptr + s_off_1d, mask_n); // [BN] T

        // ── Dequant qweight tile: ((qw >> shift) & 0xF) - zero, then * scale ──
        // shifts_2d / f_mask_2d are pre-broadcast to [BK, BN] above.
        let qw_shifted = shr_u_i32(qw_block, shifts_2d); // [BK, BN] i32
        let qw_nibble = bit_and(qw_shifted, f_mask_2d); // [BK, BN] i32, range 0..15

        // Subtract zero-point per N column (auto-broadcasts [1, BN] → [BK, BN]).
        let zero_int4_row = expand_dims(zero_int4, 0); // [1, BN] i32
        let centered = qw_nibble - zero_int4_row; // [BK, BN] i32 (auto-broadcast)

        // Cast to f32, scale, then back to T (= f16). Multiplying in f32
        // before downcast keeps the (nibble - zero) * scale precision
        // identical to vLLM's path. The store-time truncf collapses
        // automatically when T == f32.
        let centered_f32 = to_f32(centered); // [BK, BN] f32
        let scale_f32 = to_f32(scale_t); // [BN] f32
        let scale_f32_row = expand_dims(scale_f32, 0); // [1, BN] f32 — auto-broadcasts K
        let dequant_f32 = centered_f32 * scale_f32_row; // [BK, BN] f32 (broadcast K)
        let dequant_t = as_t::<T>(dequant_f32); // [BK, BN] T (f16: truncf, f32: id)

        // ── acc += A @ dequant_b ──
        // Native f16-f16-f32 mma when T == f16: a_block stays at T, dequant_t is T,
        // acc is f32. tt.dot lowers to mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32.
        dot(a_block, dequant_t, acc)
    });

    // ── store C tile [BM, BN] ──
    // Downcast f32 acc to T at the boundary (truncf for f16; identity for f32).
    let acc_t = as_t::<T>(acc);
    let c_off = offs_m_2d * stride_cm + offs_n_2d * stride_cn; // [BM, BN] i32
    let mask_c = expand_dims(mask_m, 1) & expand_dims(mask_n, 0); // [BM, BN] i1
    store(c_ptr + c_off, acc_t, mask_c);
}

fn main() {
    // Default tile: f16 act/output, BM=64, BN=64, BK=32. Same shape as
    // matmul_typed; exercises the same Tensor-Core path and keeps the
    // inner loop's register pressure modest. BN=64 is comfortably ≥ 8 so
    // the n/8 packing math is meaningful.
    print!("{}", awq_gemm_int4_typed::<f16, 64, 64, 32>::mlir());
}
