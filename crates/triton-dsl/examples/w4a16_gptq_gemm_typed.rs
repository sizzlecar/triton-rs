//! GPTQ INT4-weight × FP16-act GEMM kernel — DSL port of vLLM's
//! `triton_w4a16_gemm_kernel` adapted to the **GPTQ on-disk layout**
//! (qweight packed along K, NOT along N like vLLM/AWQ).
//!
//! This kernel is the GPTQ sibling to `awq_gemm_int4_typed.rs`. They share
//! the same outer skeleton (load act, load packed weight, dequant inline,
//! dot, accumulate), but the unpack axis is swapped:
//!   - AWQ:  qweight `[K, N/8]` packed along N → shifts are per-N column
//!   - GPTQ: qweight `[K/8, N]` packed along K → shifts are per-K row
//!
//! Matching ferrum's existing GPTQ tensor layout means we can wire this in
//! without a host-side repack on the hot path — vLLM's reference kernel
//! requires that repack because *it* is packed-along-N. Ours skips it.
//!
//! ## Layout
//!
//! | tensor   | shape           | dtype | packing axis           |
//! |----------|-----------------|-------|------------------------|
//! | input    | `[M, K]`        | f16   | (none, dense)          |
//! | qweight  | `[K/8, N]`      | i32   | 8 nibbles per K        |
//! | scales   | `[K/G, N]`      | f16   | (none, dense)          |
//! | qzeros   | `[K/G, N/8]`    | i32   | 8 nibbles per N        |
//! | output   | `[M, N]`        | f16   | (none, dense)          |
//!
//! `G = group_size`, typically 128. For each (k, n) inside a tile:
//!   - source int32 lane index in qweight = `k / 8` (row),  `n` (col)
//!   - bit position within int32 = `(k % 8) * 4`            (straight, no AWQ permutation)
//!   - dequant value = `((qw_packed >> shift_k) & 0xF) - zero[k/G, n]` * `scales[k/G, n]`
//!
//! qzeros packed along N still uses the straight `(n % 8) * 4` ordering —
//! GPTQ does NOT use AWQ's reverse permutation `[0, 4, 1, 5, 2, 6, 3, 7]`.
//! The straight order matches `ferrum_dequant_int4.rs` exactly.
//!
//! ## Algorithm (per program)
//!
//! 1. Tile `[BM, BN]` of the output. Stream BK-wide K-tiles.
//! 2. Each K-tile:
//!    - Load A `[BM, BK]` fp16 (dense).
//!    - Load qweight tile `[BK, BN]` int32 — lanes `(k, n)` read packed
//!      int32 at `qweight[k/8, n]`. Multiple K-lanes that share `k/8` alias
//!      the same int32 (same as AWQ does for N): we compute a 2D offset
//!      with `k_idx / 8` along the K axis. The redundant addresses are
//!      coalesced by Triton's load.
//!    - Compute per-K shift = `(k % 8) * 4` once per outer K-tile (loop-
//!      invariant for the BK rows of the tile).
//!    - Load qzeros `[BN]` int32 (one packed row per group) + unpack with
//!      the standard GPTQ shift `((n % 8) * 4)` — straight, no permutation.
//!    - Load scales `[BN]` f16 (broadcast along K to `[BK, BN]` at use).
//!    - Dequant inline: `((qw >> shift_k) & 0xF) - zero` → cast to f16 → `* scale`.
//!    - `acc += dot(a_f16, dequant_f16, acc_f32)` — native f16-f16-f32 mma.
//! 3. Store: `truncf(acc_f32) → f16`.
//!
//! ## Skipped from vLLM
//!
//! - **HAS_ZP=False symmetric path with `ZP_BIAS`**: GPTQ checkpoints in
//!   the wild (Qwen2.5-3B-Instruct-GPTQ-Int4 et al.) all carry an explicit
//!   `qzeros` tensor — even uint4b8-style symmetric quant ships zeros set
//!   to the bias value. We always load qzeros, mirroring ferrum's existing
//!   `gptq.rs` which does the same. The constexpr branch isn't observable
//!   by ferrum's loader.
//! - **`tl.interleave`**: vLLM's reference code does `interleave x 3` to
//!   broadcast each int32 across its 8 N-lanes. We do the same trick AWQ
//!   does — a 2D pointer arithmetic with `k_idx / 8` along the K axis. The
//!   redundant addresses in adjacent K-lanes coalesce on load.
//! - **bf16**: kernel is dtype-generic over `T: TritonElem` so `bf16` works
//!   by changing the const generic instantiation. v0 ships f16 only because
//!   ferrum's GPTQ models all run f16 activations.
//!
//! ## v0 limitations
//!
//! - `BK <= group_size` assumed (single group per K-tile load). For the
//!   common GPTQ G=128 + BK=32 this is comfortable. Smaller groups (G=32)
//!   require BK ≤ 32 to stay within one group.
//! - `BK % 8 == 0` and `BN % 8 == 0` required for the packing math to
//!   line up cleanly (same as AWQ). A BN that isn't a multiple of 8 would
//!   spill into a partial qzero int32 and require special handling.
//!
//! ## Why we keep auto-broadcast vs explicit broadcast_2d
//!
//! Same rule as `awq_gemm_int4_typed`: Rust binary operators (`+ - * /
//! % & | ^`) auto-broadcast singleton dims through `coerce_elemwise`,
//! while named DSL calls like `bit_and(...)` and `shr_u_i32(...)` route
//! through the raw `arith::andi` / `arith::shrui` constructors and
//! *don't* auto-broadcast. So for the 2D nibble unpack we explicitly
//! `broadcast_2d` shift / mask tensors to `[BK, BN]` before the bitwise
//! ops; the subtract / multiply / cast chain after that uses native
//! arithmetic operators and broadcasts itself.

use triton_dsl::triton_kernel;
use triton_ir::ty::{f16, TritonElem};

#[allow(dead_code)]
struct Ptr<T>(::std::marker::PhantomData<T>);

/// GPTQ INT4-weight × FP16-act GEMM. `acc[m, n] = sum_k a[m, k] * dequant_b[k, n]`.
///
/// Type-generic over `T` for activation/scale/output dtype (typical: f16).
/// qweight and qzeros are always int32 (8 nibbles per int32).
///
/// `BM`, `BN`, `BK` are per-program tile dims. `BK` and `BN` must be
/// multiples of 8. `BK` must be ≤ `group_size` (so a single group per
/// K-tile load holds — vLLM constraint, usually trivially true for G=128).
#[triton_kernel]
pub fn w4a16_gptq_gemm_typed<
    T: TritonElem,
    const BM: usize,
    const BN: usize,
    const BK: usize,
>(
    a_ptr: Ptr<T>,         // [M, K] fp16 input (activations)
    qweight_ptr: Ptr<i32>, // [K/8, N] int32 packed (8 nibbles per int32 along K)
    scales_ptr: Ptr<T>,    // [K/group_size, N] fp16
    qzeros_ptr: Ptr<i32>,  // [K/group_size, N/8] int32 packed (along N)
    c_ptr: Ptr<T>,         // [M, N] fp16 output
    m_size: i32,
    n_size: i32,
    k_size: i32,
    group_size: i32,
    stride_am: i32,
    stride_ak: i32,
    stride_qwk: i32, // typically N (qweight is [K/8, N], stride along packed-K rows)
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

    // 2D-broadcast forms of M/N offsets (reused every K iteration).
    let offs_m_2d = expand_dims(offs_m, 1); // [BM, 1] i32
    let offs_n_2d = expand_dims(offs_n, 0); // [1, BN] i32

    // ── 0xF mask for nibble extract ──
    // Used both for the 2D weight unpack ([BK, BN]) and the 1D qzero
    // unpack ([BN]). Built once outside the K loop.
    let f_mask_1d_bn = splat_1d(const_i32(15), BN as i64); // [BN]
    let f_mask_row_bn = expand_dims(f_mask_1d_bn, 0); // [1, BN]
    let f_mask_2d = broadcast_2d(f_mask_row_bn, BK as i64, BN as i64); // [BK, BN]

    // ── qzero shift table per N column (loop-invariant) ──
    //
    // For N column `n` in qzeros: int32 lane = `n / 8`, shift =
    // `(n % 8) * 4`. Straight order — GPTQ does NOT use AWQ's reverse
    // permutation. Matches `ferrum_dequant_int4.rs`.
    let bn_eight = splat_1d(const_i32(8), BN as i64);
    let bn_four = splat_1d(const_i32(4), BN as i64);
    let qz_lane = offs_n % bn_eight; // [BN] = n % 8 (range 0..7)
    let qz_shifts_1d = qz_lane * bn_four; // [BN] = (n % 8) * 4 (bit pos 0..28)

    // Per-N column-index in qzeros: `n_idx / 8`. [BN] i32.
    let qz_packed_col = offs_n / bn_eight; // [BN]

    // ── qweight per-K-row shift table (also loop-invariant) ──
    //
    // For K row `k` (0..BK-1) within a K-tile: shift = `(k % 8) * 4`.
    // Built from offs_k0, which goes 0..BK-1 — that pattern repeats
    // identically every K iteration since `(kb*BK + i) % 8 == i % 8` when
    // `BK % 8 == 0` (and we require this for clean packing math).
    let bk_eight = splat_1d(const_i32(8), BK as i64);
    let bk_four = splat_1d(const_i32(4), BK as i64);
    let k_lane = offs_k0 % bk_eight; // [BK] = k % 8 (range 0..7)
    let qw_shifts_1d = k_lane * bk_four; // [BK] = (k % 8) * 4 (bit pos 0..28)
    let qw_shifts_col = expand_dims(qw_shifts_1d, 1); // [BK, 1] i32
    let qw_shifts_2d = broadcast_2d(qw_shifts_col, BK as i64, BN as i64); // [BK, BN] i32

    // ── accumulator init: [BM, BN] f32 of zeros ──
    let zero = const_f32(0.0_f32);
    let zero_1 = splat_1d(zero, 1);
    let zero_2d_init = expand_dims(zero_1, 0);
    let acc_init = broadcast_2d(zero_2d_init, BM as i64, BN as i64); // [BM, BN] f32

    // Number of K-tiles. Round up so the masked tail handles K not
    // divisible by BK. (For the common case where K is a multiple of
    // BK == 32, the tail mask is fully active and adds zero overhead.)
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
        // For each lane (k, n), read qweight[k/8, n]. Adjacent K-lanes
        // that share `k/8` alias the same int32 — load is idempotent and
        // Triton's load coalescing handles the redundant addresses.
        // (Same trick AWQ uses for `n/8` along N, just rotated.)
        let packed_k_row = offs_k / bk_eight; // [BK] = k / 8
        let packed_k_col = expand_dims(packed_k_row, 1); // [BK, 1]
        let qw_off = packed_k_col * stride_qwk + offs_n_2d * stride_qwn; // [BK, BN] i32
        let mask_qw = expand_dims(mask_k, 1) & expand_dims(mask_n, 0); // [BK, BN] i1
        let qw_block = load(qweight_ptr + qw_off, mask_qw); // [BK, BN] i32

        // ── Compute group index and load scales / qzeros (single group per tile) ──
        // Assume BK <= group_size, so the entire BK-tile shares one group.
        let group = k_base / group_size; // i32 scalar

        // qzeros[group, n/8] → [BN] i32. Then unpack with `(n % 8) * 4` shift.
        let qz_off_1d =
            splat_1d(group * stride_qzk, BN as i64) + qz_packed_col * stride_qzn; // [BN] i32
        let qz_packed = load(qzeros_ptr + qz_off_1d, mask_n); // [BN] i32
        let zero_int4 = bit_and(shr_u_i32(qz_packed, qz_shifts_1d), f_mask_1d_bn); // [BN] i32

        // scales[group, n] → [BN] T.
        let s_off_1d = splat_1d(group * stride_sk, BN as i64) + offs_n * stride_sn; // [BN] i32
        let scale_t = load(scales_ptr + s_off_1d, mask_n); // [BN] T

        // ── Dequant qweight tile: ((qw >> shift_k) & 0xF) - zero, then * scale ──
        // shifts_2d / f_mask_2d are pre-broadcast to [BK, BN] above. The
        // shift varies along K (rows of qw), the zero/scale broadcast
        // along K from a single [BN] vector.
        let qw_shifted = shr_u_i32(qw_block, qw_shifts_2d); // [BK, BN] i32
        let qw_nibble = bit_and(qw_shifted, f_mask_2d); // [BK, BN] i32, range 0..15

        // Subtract zero-point per N column. [1, BN] auto-broadcasts to [BK, BN].
        let zero_int4_row = expand_dims(zero_int4, 0); // [1, BN] i32
        let centered = qw_nibble - zero_int4_row; // [BK, BN] i32

        // Cast to f32, scale, then back to T (f16 path: truncf at the end;
        // f32 path: identity). Multiplying in f32 before downcast keeps
        // the (nibble - zero) * scale precision identical to vLLM's path.
        let centered_f32 = to_f32(centered); // [BK, BN] f32
        let scale_f32 = to_f32(scale_t); // [BN] f32
        let scale_f32_row = expand_dims(scale_f32, 0); // [1, BN] f32
        let dequant_f32 = centered_f32 * scale_f32_row; // [BK, BN] f32 (broadcast K)
        let dequant_t = as_t::<T>(dequant_f32); // [BK, BN] T

        // ── acc += A @ dequant_b ──
        // Native f16-f16-f32 mma when T == f16: a_block stays at T,
        // dequant_t is T, acc is f32. tt.dot lowers to mma.sync.aligned.
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
    // matmul_typed and awq_gemm_int4_typed; exercises the same Tensor-Core
    // path. BK=32 keeps the BK ≤ group_size constraint trivially true for
    // G=128, and BN=64 is comfortably ≥ 8 so the n/8 packing math is
    // meaningful for the qzero unpack.
    print!("{}", w4a16_gptq_gemm_typed::<f16, 64, 64, 32>::mlir());
}
