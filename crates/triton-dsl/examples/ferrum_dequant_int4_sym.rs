//! ferrum kernel port: GPTQ INT4 → FP16 dequantization (symmetric variant).
//!
//! Mirrors `ferrum-kernels/kernels/dequant_int4.cu :: dequant_int4_sym_to_fp16`:
//!   - qweight: [K/8, N] packed int32 (8x INT4 per int32, packed along K)
//!   - scales:  [K/group_size, N] fp16
//!   - output:  [K, N] fp16 — accessed as output[col*K + (base_k+i)]
//!
//! Per-program: one packed_row × one BLOCK-wide tile of N cols.
//! Each lane unpacks 8 K-values from one packed int32 and writes them
//! at output[col, base_k..base_k+8].
//!
//! Symmetric: zero_point is fixed at 8 (no qzeros tensor needed).
//! For the asymmetric variant see `ferrum_dequant_int4.rs` (per-group
//! zero-point unpacked from a packed qzeros tensor).
//!
//! This is the first triton-rs DSL port that exercises the new bitwise
//! ops added in `feat/dsl-bitwise-ops` (`shr_u_i32`, `bit_and`, `shl_i32`).

use triton_dsl::triton_kernel;

#[allow(dead_code)]
struct Ptr<T>(::std::marker::PhantomData<T>);

#[triton_kernel]
fn dequant_int4_sym_to_fp16<const BLOCK: usize>(
    qweight: Ptr<i32>,
    scales: Ptr<f16>,
    output: Ptr<f16>,
    k_dim: i32,
    n_dim: i32,
    group_size: i32,
) {
    // Grid: (ceil(N/BLOCK), K/8). One program = one (col-tile, packed_row).
    let pid_n = program_id(0);
    let packed_row = program_id(1);

    // Scalar host-side computation per program. base_k is the first K row
    // this program writes; group is the scale-row index for it. Rust `/`
    // dispatches to `arith.divsi` (signed integer divide).
    let base_k = packed_row * const_i32(8);
    let group = base_k / group_size;

    // Tile of cols along N.
    let col_start = pid_n * (BLOCK as i32);
    let cols = splat_1d(col_start, BLOCK as i64) + make_range(0, BLOCK as i32);
    let n_v = splat_1d(n_dim, BLOCK as i64);
    let n_mask = cols < n_v;

    // Load packed int32 per col: qweight[packed_row, col].
    let qw_offset = splat_1d(packed_row * n_dim, BLOCK as i64) + cols;
    let packed = load(splat_1d(qweight, BLOCK as i64) + qw_offset, n_mask);

    // Load scale (fp16) per col: scales[group, col].
    let scale_offset = splat_1d(group * n_dim, BLOCK as i64) + cols;
    let s_f16 = load(splat_1d(scales, BLOCK as i64) + scale_offset, n_mask);
    let s_f32 = to_f32(s_f16);

    // Constants reused across the 8 unrolled iterations.
    let f_mask = splat_1d(const_i32(15), BLOCK as i64); // 0xF
    let zero = splat_1d(const_i32(8), BLOCK as i64); // symmetric zero-point
    let k_v = splat_1d(k_dim, BLOCK as i64);
    let cols_times_k = cols * k_v;
    let out_base = splat_1d(output, BLOCK as i64);
    let base_k_v = splat_1d(base_k, BLOCK as i64);

    // Manually unrolled 8 iterations. Triton's optimizer would fold an
    // `scf.for` over a constant trip count, but explicit unroll keeps the
    // generated MLIR readable and matches the .cu's `#pragma unroll 8`.
    //
    // For each i in 0..8:
    //   val_i      = (packed >> (i*4)) & 0xF       (logical right shift)
    //   centered_i = val_i - 8
    //   deq_f16_i  = (float(centered_i) * scale).to_f16()
    //   output[col*K + (base_k + i)] = deq_f16_i

    // i = 0
    let s0 = splat_1d(const_i32(0), BLOCK as i64);
    let v0 = bit_and(shr_u_i32(packed, s0), f_mask);
    let d0 = to_f16(to_f32(v0 - zero) * s_f32);
    let o0 = cols_times_k + (base_k_v + splat_1d(const_i32(0), BLOCK as i64));
    store(out_base + o0, d0, n_mask);

    // i = 1
    let s1 = splat_1d(const_i32(4), BLOCK as i64);
    let v1 = bit_and(shr_u_i32(packed, s1), f_mask);
    let d1 = to_f16(to_f32(v1 - zero) * s_f32);
    let o1 = cols_times_k + (base_k_v + splat_1d(const_i32(1), BLOCK as i64));
    store(out_base + o1, d1, n_mask);

    // i = 2
    let s2 = splat_1d(const_i32(8), BLOCK as i64);
    let v2 = bit_and(shr_u_i32(packed, s2), f_mask);
    let d2 = to_f16(to_f32(v2 - zero) * s_f32);
    let o2 = cols_times_k + (base_k_v + splat_1d(const_i32(2), BLOCK as i64));
    store(out_base + o2, d2, n_mask);

    // i = 3
    let s3 = splat_1d(const_i32(12), BLOCK as i64);
    let v3 = bit_and(shr_u_i32(packed, s3), f_mask);
    let d3 = to_f16(to_f32(v3 - zero) * s_f32);
    let o3 = cols_times_k + (base_k_v + splat_1d(const_i32(3), BLOCK as i64));
    store(out_base + o3, d3, n_mask);

    // i = 4
    let s4 = splat_1d(const_i32(16), BLOCK as i64);
    let v4 = bit_and(shr_u_i32(packed, s4), f_mask);
    let d4 = to_f16(to_f32(v4 - zero) * s_f32);
    let o4 = cols_times_k + (base_k_v + splat_1d(const_i32(4), BLOCK as i64));
    store(out_base + o4, d4, n_mask);

    // i = 5
    let s5 = splat_1d(const_i32(20), BLOCK as i64);
    let v5 = bit_and(shr_u_i32(packed, s5), f_mask);
    let d5 = to_f16(to_f32(v5 - zero) * s_f32);
    let o5 = cols_times_k + (base_k_v + splat_1d(const_i32(5), BLOCK as i64));
    store(out_base + o5, d5, n_mask);

    // i = 6
    let s6 = splat_1d(const_i32(24), BLOCK as i64);
    let v6 = bit_and(shr_u_i32(packed, s6), f_mask);
    let d6 = to_f16(to_f32(v6 - zero) * s_f32);
    let o6 = cols_times_k + (base_k_v + splat_1d(const_i32(6), BLOCK as i64));
    store(out_base + o6, d6, n_mask);

    // i = 7
    let s7 = splat_1d(const_i32(28), BLOCK as i64);
    let v7 = bit_and(shr_u_i32(packed, s7), f_mask);
    let d7 = to_f16(to_f32(v7 - zero) * s_f32);
    let o7 = cols_times_k + (base_k_v + splat_1d(const_i32(7), BLOCK as i64));
    store(out_base + o7, d7, n_mask);
}

fn main() {
    print!("{}", dequant_int4_sym_to_fp16::<128>::mlir());
}
