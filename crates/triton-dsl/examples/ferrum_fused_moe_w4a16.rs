//! Fused MoE GPTQ INT4×FP16 GEMM — DSL port of vLLM's
//! `vllm/model_executor/layers/fused_moe/fused_moe.py` (`fused_moe_kernel`).
//!
//! Single launch processes ALL `(token, expert)` pairs of one MoE phase
//! by reading per-tile expert id from `expert_ids[pid_m]` and gathering
//! the M-axis input rows via `sorted_token_ids[pid_m * BM .. ]`.
//!
//! Adapted to ferrum's GPTQ on-disk layout (qweight packed along K), so
//! the weight loader can stack `num_experts` copies of the raw GPTQ
//! tensors with a fixed per-expert stride and skip Marlin-style repack
//! entirely. This bypasses the byte-layout incompatibility that blocked
//! the vLLM marlin_moe_wna16 C++ port.
//!
//! ## Layout
//!
//! | tensor              | shape                          | dtype | role                               |
//! |---------------------|--------------------------------|-------|------------------------------------|
//! | input               | `[size_m, K]`                  | f16   | dense, untouched per pair          |
//! | qweight (stacked)   | `[E, K/8, N]`                  | i32   | per-expert raw GPTQ                |
//! | scales (stacked)    | `[E, K/G, N]`                  | f16   | per-expert                         |
//! | qzeros (stacked)    | `[E, K/G, N/8]`                | i32   | per-expert                         |
//! | sorted_token_ids    | `[N_padded]`                   | i32   | per-tile-row → input row idx       |
//! | expert_ids          | `[N_padded / BM]`              | i32   | per-tile expert assignment         |
//! | output              | `[size_m * top_k, N]`          | f16   | dense; gather/scatter via sorted   |
//!
//! Sentinel rows (`sorted_token_ids[i] == size_m`) are masked out at
//! load time so they contribute zeros to acc and skip the C write.
//!
//! ## Per-program work
//!
//! 1. `expert_id = expert_ids[pid_m]`. Compute B/s/zp expert offsets:
//!    `qw_expert_off = expert_id * (K/8 * N)`, etc.
//! 2. Read `m_idx[BM] = sorted_token_ids[pid_m * BM .. pid_m * BM + BM]`.
//!    For lanes where `m_idx == size_m * top_k` (sentinel), mask off.
//! 3. Standard streaming K-tile loop (same as w4a16_gptq_gemm_typed):
//!    A loaded via `m_idx`-gather, B/scales/qzeros via the fixed
//!    expert offsets.
//! 4. Store `c[m_idx, n_offs]` for non-sentinel lanes.
//!
//! `BM` MUST equal vLLM's `moe_block_size` (16/32/48/64). The N grid
//! dimension is independent: `pid_n` indexes BN-wide column tiles.
//!
//! Constraints (inherited from w4a16_gptq_gemm_typed):
//!   - `K % BK == 0` and `BK <= group_size`
//!   - `BN % 8 == 0` (qzero unpack along N)
//!   - `BK % 8 == 0` (qweight unpack along K)

use triton_dsl::triton_kernel;
use triton_ir::ty::{f16, TritonElem};

#[triton_kernel]
pub fn fused_moe_w4a16_typed<
    T: TritonElem,
    const BM: usize,
    const BN: usize,
    const BK: usize,
>(
    a_ptr: Ptr<T>,                  // [size_m, K] fp16, dense
    qweight_ptr: Ptr<i32>,           // [E, K/8, N] int32 stacked per-expert
    scales_ptr: Ptr<T>,              // [E, K/G, N] fp16 stacked
    qzeros_ptr: Ptr<i32>,            // [E, K/G, N/8] int32 stacked
    sorted_token_ids_ptr: Ptr<i32>,  // [N_padded] — gather index for A rows
    expert_ids_ptr: Ptr<i32>,        // [N_padded / BM] — expert per tile
    c_ptr: Ptr<T>,                   // [size_m * top_k, N] fp16 output
    num_valid_tokens: i32,           // size_m * top_k (sentinel boundary)
    n_size: i32,
    k_size: i32,
    group_size: i32,
    qw_expert_stride: i32,           // (K/8) * N: int32-elements per expert in qweight
    s_expert_stride: i32,            // (K/G) * N: f16-elements per expert in scales
    qz_expert_stride: i32,           // (K/G) * (N/8): int32-elements per expert in qzeros
    stride_am: i32,                  // typically K
    stride_ak: i32,                  // typically 1
    stride_qwk: i32,                 // typically N
    stride_qwn: i32,                 // typically 1
    stride_sk: i32,                  // typically N
    stride_sn: i32,                  // typically 1
    stride_qzk: i32,                 // typically N/8
    stride_qzn: i32,                 // typically 1
    stride_cm: i32,                  // typically N
    stride_cn: i32,                  // typically 1
) {
    // ── per-program tile coordinates ──
    let pid_m = program_id(0);
    let pid_n = program_id(1);

    // Read per-tile expert id (scalar i32).
    // load(ptr) with a single (non-tensor) pointer returns a scalar value.
    let expert_id_scalar = load(expert_ids_ptr + pid_m);

    // ── Apply per-expert offsets to weight pointers ──
    let qw_base_off = expert_id_scalar * qw_expert_stride;
    let s_base_off = expert_id_scalar * s_expert_stride;
    let qz_base_off = expert_id_scalar * qz_expert_stride;

    // ── Gather M-axis row indices from sorted_token_ids ──
    // Unmasked tensor load — sentinel rows still read from sorted_token_ids
    // (those slots hold the sentinel value `num_valid_tokens` itself,
    // which is valid memory). The sentinel comparison filters both A
    // gather and C scatter below.
    let offs_m_local = pid_m * (BM as i32) + make_range(0, BM as i32); // [BM]
    let m_idx = load(sorted_token_ids_ptr + offs_m_local); // [BM] i32
    let num_valid_v = splat_1d(num_valid_tokens, BM as i64); // [BM] i32
    let mask_m = m_idx < num_valid_v; // [BM] i1

    // Effective row index used for A gather and C scatter.
    let offs_m = m_idx; // [BM] i32

    // N-axis offsets (cols of dequant_b and C). [BN] i32.
    let offs_n = pid_n * (BN as i32) + make_range(0, BN as i32);
    let mask_n = offs_n < n_size; // [BN] i1

    // K-axis offsets within one tile, base 0.
    let offs_k0 = make_range(0, BK as i32);

    // 2D-broadcast forms.
    let offs_m_2d = expand_dims(offs_m, 1); // [BM, 1] i32
    let offs_n_2d = expand_dims(offs_n, 0); // [1, BN] i32

    // ── Loop-invariant tables (same as w4a16_gptq_gemm_typed) ──
    let f_mask_1d_bn = splat_1d(const_i32(15), BN as i64);
    let f_mask_row_bn = expand_dims(f_mask_1d_bn, 0);
    let f_mask_2d = broadcast_2d(f_mask_row_bn, BK as i64, BN as i64);

    let bn_eight = splat_1d(const_i32(8), BN as i64);
    let bn_four = splat_1d(const_i32(4), BN as i64);
    let qz_lane = offs_n % bn_eight;
    let qz_shifts_1d = qz_lane * bn_four;
    let qz_packed_col = offs_n / bn_eight;

    let bk_eight = splat_1d(const_i32(8), BK as i64);
    let bk_four = splat_1d(const_i32(4), BK as i64);
    let k_lane = offs_k0 % bk_eight;
    let qw_shifts_1d = k_lane * bk_four;
    let qw_shifts_col = expand_dims(qw_shifts_1d, 1);
    let qw_shifts_2d = broadcast_2d(qw_shifts_col, BK as i64, BN as i64);

    // ── accumulator init ──
    let zero = const_f32(0.0_f32);
    let zero_1 = splat_1d(zero, 1);
    let zero_2d_init = expand_dims(zero_1, 0);
    let acc_init = broadcast_2d(zero_2d_init, BM as i64, BN as i64);

    let k_blocks = (k_size + (BK as i32) - 1) / (BK as i32);

    // ── streaming K loop (with per-expert pointer offsets applied) ──
    let acc = scf_for(const_i32(0), k_blocks, const_i32(1), acc_init, |kb, acc| {
        let k_base = kb * (BK as i32);
        let offs_k = splat_1d(k_base, BK as i64) + offs_k0;
        let mask_k = offs_k < k_size;

        let offs_k_row = expand_dims(offs_k, 0);
        let _offs_k_col = expand_dims(offs_k, 1);

        // ── Load A tile [BM, BK] via gathered m_idx ──
        // Mask combines sentinel mask_m with mask_k.
        let a_off = offs_m_2d * stride_am + offs_k_row * stride_ak;
        let mask_a = expand_dims(mask_m, 1) & expand_dims(mask_k, 0);
        let a_block = load(a_ptr + a_off, mask_a);

        // ── Load qweight packed tile [BK, BN] int32, with expert offset ──
        let packed_k_row = offs_k / bk_eight;
        let packed_k_col = expand_dims(packed_k_row, 1);
        let qw_off_local = packed_k_col * stride_qwk + offs_n_2d * stride_qwn;
        // Build [BK, BN] broadcast of qw_base_off (scalar).
        let qw_off_1d = splat_1d(qw_base_off, BN as i64); // [BN]
        let qw_off_row = expand_dims(qw_off_1d, 0); // [1, BN]
        let qw_off_2d = broadcast_2d(qw_off_row, BK as i64, BN as i64); // [BK, BN]
        let qw_off = qw_off_local + qw_off_2d;
        let mask_qw = expand_dims(mask_k, 1) & expand_dims(mask_n, 0);
        let qw_block = load(qweight_ptr + qw_off, mask_qw);

        let group = k_base / group_size;

        // qzeros[expert, group, n/8] → [BN].
        let qz_off_1d_local =
            splat_1d(group * stride_qzk, BN as i64) + qz_packed_col * stride_qzn;
        let qz_off_1d = qz_off_1d_local + splat_1d(qz_base_off, BN as i64);
        let qz_packed = load(qzeros_ptr + qz_off_1d, mask_n);
        let zero_raw = bit_and(shr_u_i32(qz_packed, qz_shifts_1d), f_mask_1d_bn);
        let one_bn = splat_1d(const_i32(1), BN as i64);
        let zero_int4 = zero_raw + one_bn;

        // scales[expert, group, n] → [BN] T.
        let s_off_1d_local =
            splat_1d(group * stride_sk, BN as i64) + offs_n * stride_sn;
        let s_off_1d = s_off_1d_local + splat_1d(s_base_off, BN as i64);
        let scale_t = load(scales_ptr + s_off_1d, mask_n);

        // ── Dequant ──
        let qw_shifted = shr_u_i32(qw_block, qw_shifts_2d);
        let qw_nibble = bit_and(qw_shifted, f_mask_2d);
        let zero_int4_row = expand_dims(zero_int4, 0);
        let centered = qw_nibble - zero_int4_row;
        let centered_f32 = to_f32(centered);
        let scale_f32 = to_f32(scale_t);
        let scale_f32_row = expand_dims(scale_f32, 0);
        let dequant_f32 = centered_f32 * scale_f32_row;
        let dequant_t = as_t::<T>(dequant_f32);

        dot(a_block, dequant_t, acc)
    });

    // ── store C tile [BM, BN] (gathered scatter via offs_m == m_idx) ──
    let acc_t = as_t::<T>(acc);
    let c_off = offs_m_2d * stride_cm + offs_n_2d * stride_cn;
    let mask_c = expand_dims(mask_m, 1) & expand_dims(mask_n, 0);
    store(c_ptr + c_off, acc_t, mask_c);
}

fn main() {
    // Default: BM=16, BN=64, BK=32. To emit a different tile, pass
    // FERRUM_FUSED_MOE_TILE=BMxBNxBK as an env var, e.g.
    //   FERRUM_FUSED_MOE_TILE=16x128x64 cargo run --example ...
    let tile = std::env::var("FERRUM_FUSED_MOE_TILE")
        .unwrap_or_else(|_| "16x64x32".to_string());
    let mlir = match tile.as_str() {
        "16x64x32" => fused_moe_w4a16_typed::<f16, 16, 64, 32>::mlir(),
        "16x128x32" => fused_moe_w4a16_typed::<f16, 16, 128, 32>::mlir(),
        "16x128x64" => fused_moe_w4a16_typed::<f16, 16, 128, 64>::mlir(),
        "16x256x64" => fused_moe_w4a16_typed::<f16, 16, 256, 64>::mlir(),
        "16x64x64" => fused_moe_w4a16_typed::<f16, 16, 64, 64>::mlir(),
        "32x64x32" => fused_moe_w4a16_typed::<f16, 32, 64, 32>::mlir(),
        "32x128x32" => fused_moe_w4a16_typed::<f16, 32, 128, 32>::mlir(),
        "32x128x64" => fused_moe_w4a16_typed::<f16, 32, 128, 64>::mlir(),
        "32x256x64" => fused_moe_w4a16_typed::<f16, 32, 256, 64>::mlir(),
        "64x128x32" => fused_moe_w4a16_typed::<f16, 64, 128, 32>::mlir(),
        "64x128x64" => fused_moe_w4a16_typed::<f16, 64, 128, 64>::mlir(),
        "64x256x64" => fused_moe_w4a16_typed::<f16, 64, 256, 64>::mlir(),
        _ => panic!("unknown tile {tile} — add to match arms"),
    };
    print!("{mlir}");
}
