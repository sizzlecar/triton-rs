//! Smoke test for the AWQ INT4 × FP16 GEMM kernel — sibling to
//! `proc_macro_matmul.rs` (which covers the dense matmul) but with a
//! 4-bit packed weight tile that needs inline dequantization inside the
//! K loop.
//!
//! Asserts the load-time-invariant:
//!   - the K loop streams a packed `[BK, BN]` int32 tile of qweight
//!   - the dequant pipeline emits `arith.shrui` (nibble unpack),
//!     `arith.andi` (0xF mask), `arith.subi` (centering by zero-point),
//!     `arith.sitofp` + `arith.mulf` (scale apply), and `arith.truncf`
//!     to bring the dequant tile back to f16 before the dot
//!   - the dot itself is native f16 mma:
//!     `tt.dot %a_f16, %dequant_f16, %acc_f32 -> %acc_f32`
//!   - the final store rounds the f32 accumulator to f16 via truncf
//!
//! These together prove the kernel is structurally what we expect — Tensor
//! Core mma path active, dequant inlined into the inner loop, no f32
//! "secondary" dot fallback.
//!
//! See `examples/awq_gemm_int4_typed.rs` for the kernel itself + design
//! rationale.

use triton_dsl::triton_kernel;
use triton_ir::ty::{f16, TritonElem};

#[allow(dead_code)]
struct Ptr<T>(::std::marker::PhantomData<T>);

#[triton_kernel]
fn awq_gemm_int4_typed<T: TritonElem, const BM: usize, const BN: usize, const BK: usize>(
    a_ptr: Ptr<T>,
    qweight_ptr: Ptr<i32>,
    scales_ptr: Ptr<T>,
    qzeros_ptr: Ptr<i32>,
    c_ptr: Ptr<T>,
    m_size: i32,
    n_size: i32,
    k_size: i32,
    group_size: i32,
    stride_am: i32,
    stride_ak: i32,
    stride_qwk: i32,
    stride_qwn: i32,
    stride_sk: i32,
    stride_sn: i32,
    stride_qzk: i32,
    stride_qzn: i32,
    stride_cm: i32,
    stride_cn: i32,
) {
    let pid_m = program_id(0);
    let pid_n = program_id(1);

    let offs_m = pid_m * (BM as i32) + make_range(0, BM as i32);
    let mask_m = offs_m < m_size;

    let offs_n = pid_n * (BN as i32) + make_range(0, BN as i32);
    let mask_n = offs_n < n_size;

    let offs_k0 = make_range(0, BK as i32);

    let offs_m_2d = expand_dims(offs_m, 1);
    let offs_n_2d = expand_dims(offs_n, 0);

    // AWQ shift table per N column.
    let bn_eight = splat_1d(const_i32(8), BN as i64);
    let bn_two = splat_1d(const_i32(2), BN as i64);
    let bn_four = splat_1d(const_i32(4), BN as i64);
    let lane_in_int32 = offs_n % bn_eight;
    let lane_high = lane_in_int32 / bn_two;
    let lane_low = lane_in_int32 % bn_two;
    let awq_lane = lane_high + lane_low * bn_four;
    let shifts_1d = awq_lane * bn_four;
    let shifts_row = expand_dims(shifts_1d, 0);
    let shifts_2d = broadcast_2d(shifts_row, BK as i64, BN as i64);

    let f_mask_1d = splat_1d(const_i32(15), BN as i64);
    let f_mask_row = expand_dims(f_mask_1d, 0);
    let f_mask_2d = broadcast_2d(f_mask_row, BK as i64, BN as i64);

    let packed_col = offs_n / bn_eight;
    let packed_col_2d = expand_dims(packed_col, 0);

    let zero = const_f32(0.0_f32);
    let zero_1 = splat_1d(zero, 1);
    let zero_2d_init = expand_dims(zero_1, 0);
    let acc_init = broadcast_2d(zero_2d_init, BM as i64, BN as i64);

    let k_blocks = (k_size + (BK as i32) - 1) / (BK as i32);

    let acc = scf_for(const_i32(0), k_blocks, const_i32(1), acc_init, |kb, acc| {
        let k_base = kb * (BK as i32);
        let offs_k = splat_1d(k_base, BK as i64) + offs_k0;
        let mask_k = offs_k < k_size;

        let offs_k_row = expand_dims(offs_k, 0);
        let offs_k_col = expand_dims(offs_k, 1);

        let a_off = offs_m_2d * stride_am + offs_k_row * stride_ak;
        let mask_a = expand_dims(mask_m, 1) & expand_dims(mask_k, 0);
        let a_block = load(a_ptr + a_off, mask_a);

        let qw_off = offs_k_col * stride_qwk + packed_col_2d * stride_qwn;
        let mask_qw = expand_dims(mask_k, 1) & expand_dims(mask_n, 0);
        let qw_block = load(qweight_ptr + qw_off, mask_qw);

        let group = k_base / group_size;

        let qz_off_1d = splat_1d(group * stride_qzk, BN as i64) + packed_col * stride_qzn;
        let qz_packed = load(qzeros_ptr + qz_off_1d, mask_n);
        let zero_int4 = bit_and(shr_u_i32(qz_packed, shifts_1d), f_mask_1d);

        let s_off_1d = splat_1d(group * stride_sk, BN as i64) + offs_n * stride_sn;
        let scale_t = load(scales_ptr + s_off_1d, mask_n);

        let qw_shifted = shr_u_i32(qw_block, shifts_2d);
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

    let acc_t = as_t::<T>(acc);
    let c_off = offs_m_2d * stride_cm + offs_n_2d * stride_cn;
    let mask_c = expand_dims(mask_m, 1) & expand_dims(mask_n, 0);
    store(c_ptr + c_off, acc_t, mask_c);
}

#[test]
fn awq_gemm_f16_emits_dequant_and_native_f16_dot() {
    let text = awq_gemm_int4_typed::<f16, 64, 64, 32>::mlir();
    eprintln!(
        "===== awq_gemm_int4_typed<f16,64,64,32> MLIR =====\n{text}\n=================================="
    );

    // ── pointer types ──
    assert!(
        text.contains("!tt.ptr<f16>"),
        "expected f16 pointer for A/scales/C:\n{text}"
    );
    assert!(
        text.contains("!tt.ptr<i32>"),
        "expected i32 pointer for qweight/qzeros:\n{text}"
    );

    // ── K-loop carrier: f32 accumulator survives the loop ──
    assert!(
        text.contains("\"scf.for\""),
        "missing scf.for K loop:\n{text}"
    );
    assert!(
        text.contains("tensor<64x64xf32>"),
        "missing f32 accumulator tile:\n{text}"
    );
    assert!(text.contains("\"scf.yield\""), "missing scf.yield:\n{text}");

    // ── packed qweight tile loads as int32 ──
    assert!(
        text.contains("tensor<32x64xi32>"),
        "missing [BK=32, BN=64] qweight tile:\n{text}"
    );

    // ── nibble unpack: shrui (logical right shift) + andi 0xF ──
    // 2D shrui on the qweight tile. There are also 1D shrui calls for
    // the qzero tile; the 2D form proves the dequant operates on the
    // full [BK, BN] qweight tile, not just its zero-point slice.
    let has_2d_shrui = text.lines().any(|l| {
        l.contains("arith.shrui")
            && l.contains("tensor<32x64xi32>")
            && l.matches("tensor<32x64xi32>").count() >= 2
    });
    assert!(
        has_2d_shrui,
        "missing 2D arith.shrui (qweight nibble unpack):\n{text}"
    );

    // 0xF mask via 2D andi.
    let has_2d_andi = text.lines().any(|l| {
        l.contains("arith.andi")
            && l.contains("tensor<32x64xi32>")
            && l.matches("tensor<32x64xi32>").count() >= 2
    });
    assert!(
        has_2d_andi,
        "missing 2D arith.andi (qweight nibble mask):\n{text}"
    );

    // ── subi to center by qzero ──
    let has_2d_subi = text
        .lines()
        .any(|l| l.contains("arith.subi") && l.contains("tensor<32x64xi32>"));
    assert!(
        has_2d_subi,
        "missing 2D arith.subi (qweight - zero_int4):\n{text}"
    );

    // ── int → float for the dequant value ──
    assert!(
        text.contains("\"arith.sitofp\""),
        "missing sitofp on (qw - zero) for scale apply:\n{text}"
    );

    // ── apply per-(group, col) scale ──
    let has_2d_mulf = text
        .lines()
        .any(|l| l.contains("arith.mulf") && l.contains("tensor<32x64xf32>"));
    assert!(
        has_2d_mulf,
        "missing 2D arith.mulf for dequant * scale:\n{text}"
    );

    // ── truncf back to f16 before dot ──
    // Two truncfs total: one in-loop (f32 dequant -> f16) and one at store
    // (f32 acc -> f16). We require both.
    let truncf_count = text.matches("\"arith.truncf\"").count();
    assert!(
        truncf_count >= 2,
        "expected ≥2 truncf (in-loop dequant + store boundary), got {truncf_count}:\n{text}"
    );

    // ── tt.dot: native f16-f16-f32 mma path ──
    // a:f16 @ dequant_b:f16 + acc:f32 -> f32. Same shape as the dense
    // matmul_typed test — proves the inline dequant didn't accidentally
    // upcast inputs to f32 and trigger the f32 dot fallback.
    assert!(
        text.contains(
            "(tensor<64x32xf16>, tensor<32x64xf16>, tensor<64x64xf32>) -> tensor<64x64xf32>",
        ),
        "tt.dot must be native f16 mma (a:f16 @ b:f16 + c:f32 → f32):\n{text}"
    );

    // ── final store on f16 ptr ──
    assert!(
        text.contains("tensor<64x64x!tt.ptr<f16>>"),
        "missing f16 C tile ptr:\n{text}"
    );
    assert!(text.contains("\"tt.store\""), "missing tt.store:\n{text}");
    assert!(text.contains("\"tt.return\""), "missing tt.return:\n{text}");
}

#[test]
fn awq_gemm_block_sizes_propagate_via_const_generics() {
    // Sanity: changing tile sizes changes the dot signature.
    let text = awq_gemm_int4_typed::<f16, 32, 32, 32>::mlir();
    assert!(
        text.contains(
            "(tensor<32x32xf16>, tensor<32x32xf16>, tensor<32x32xf32>) -> tensor<32x32xf32>",
        ),
        "tt.dot signature should match BM=32, BK=32, BN=32:\n{text}"
    );
}

#[test]
fn awq_gemm_uses_n_div_8_for_packed_lane_indexing() {
    // Cardinal AWQ-vs-GPTQ check: the packed dim is N (not K).
    // We expect the kernel to compute `n / 8` on the offs_n tensor.
    // Concretely: an arith.divsi with rhs = splat(8) appears in the
    // pre-loop hoists.
    let text = awq_gemm_int4_typed::<f16, 64, 64, 32>::mlir();
    assert!(
        text.contains("\"arith.divsi\""),
        "missing arith.divsi for n/8 packed lane index:\n{text}"
    );
    // The 8 constant for n%8 lane mask.
    assert!(
        text.contains("\"arith.constant\"() {value = 8 : i32}"),
        "missing constant 8 for n%8 / n/8 lane indexing:\n{text}"
    );
    // The 0xF mask — it is the load-bearing constant for "this is a
    // 4-bit unpack, not 2-bit or 8-bit".
    assert!(
        text.contains("\"arith.constant\"() {value = 15 : i32}"),
        "missing 0xF (15) constant for nibble mask:\n{text}"
    );
}

#[test]
fn awq_gemm_preserves_awq_reverse_order_via_lane_remap() {
    // The AWQ format ships its 8 nibbles in [0,4,1,5,2,6,3,7] order
    // (not straight [0..8]). Our kernel computes
    //   awq_lane = (n%8 / 2) + (n%8 % 2) * 4
    // which produces that table at indices 0..7. The presence of
    //   - remsi by 2 (n%8 % 2)
    //   - divsi by 2 (n%8 / 2)
    //   - muli by 4 (lane_low * 4)
    // on the 1D BN-shape tensor proves the remap landed.
    let text = awq_gemm_int4_typed::<f16, 64, 64, 32>::mlir();

    let constant_2 = text.contains("\"arith.constant\"() {value = 2 : i32}");
    let constant_4 = text.contains("\"arith.constant\"() {value = 4 : i32}");
    assert!(
        constant_2,
        "missing constant 2 for AWQ lane remap (n%8 / 2 and n%8 % 2):\n{text}"
    );
    assert!(
        constant_4,
        "missing constant 4 for AWQ lane remap (lane_low * 4):\n{text}"
    );

    // The lane remap arith on the offs_n (BN-shape) tensor: at least one
    // remsi *and* one divsi where both operands are tensor<64xi32>.
    let has_1d_remsi = text
        .lines()
        .any(|l| l.contains("arith.remsi") && l.contains("tensor<64xi32>"));
    let has_1d_divsi = text
        .lines()
        .any(|l| l.contains("arith.divsi") && l.contains("tensor<64xi32>"));
    assert!(
        has_1d_remsi,
        "missing 1D remsi on offs_n tensor (AWQ lane remap):\n{text}"
    );
    assert!(
        has_1d_divsi,
        "missing 1D divsi on offs_n tensor (AWQ lane remap / packed_col):\n{text}"
    );
}
