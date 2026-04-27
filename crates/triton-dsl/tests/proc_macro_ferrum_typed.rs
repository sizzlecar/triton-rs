//! Dtype-generic versions of the ferrum-port DSL kernels, mirroring
//! `crates/triton-dsl/examples/ferrum_*.rs`. Covers the
//! `T: TritonElem`-parameterised pattern: load at T → upcast to f32 →
//! compute in f32 → downcast to T at store via `as_t::<T>`.
//!
//! These tests lock down the IR contract for f32 / f16 / bf16
//! instantiations of every ferrum example kernel. The bodies here are
//! intentionally byte-identical to the example files — duplicated
//! because Cargo example binaries can't be imported from a test crate.
//!
//! Two invariants per kernel:
//! 1. f32 instantiation matches the original f32-only IR (no
//!    `arith.extf` / `arith.truncf`).
//! 2. f16 / bf16 instantiation carries `!tt.ptr<f16>` / `!tt.ptr<bf16>`
//!    parameters, and the kernels with explicit f32-internal compute
//!    (`to_f32` / `as_t::<T>`) emit the boundary casts.

use triton_dsl::triton_kernel;
use triton_ir::ty::{bf16, f16, TritonElem};

#[allow(dead_code)]
struct Ptr<T>(::std::marker::PhantomData<T>);

// ── pure-elementwise kernels (no boundary casts) ────────────────────
//
// residual_add / residual_add_inplace / add_bias just do `T + T → T`.
// At T == f16 the load / add / store all stay in f16 (bandwidth-bound,
// no precision risk for a single add). Hence no extf / truncf in
// the f16 IR either.

#[triton_kernel]
fn residual_add_typed<T: TritonElem, const BLOCK: usize>(
    a: Ptr<T>,
    b: Ptr<T>,
    out: Ptr<T>,
    n: i32,
) {
    let pid = program_id(0);
    let off = splat_1d(pid * const_i32(BLOCK as i32), BLOCK as i64)
            + make_range(0, BLOCK as i32);
    let mask = off < splat_1d(n, BLOCK as i64);
    let av = load(splat_1d(a, BLOCK as i64) + off, mask);
    let bv = load(splat_1d(b, BLOCK as i64) + off, mask);
    store(splat_1d(out, BLOCK as i64) + off, av + bv, mask);
}

#[triton_kernel]
fn residual_add_inplace_typed<T: TritonElem, const BLOCK: usize>(
    a: Ptr<T>,
    b: Ptr<T>,
    n: i32,
) {
    let pid = program_id(0);
    let off = splat_1d(pid * const_i32(BLOCK as i32), BLOCK as i64)
            + make_range(0, BLOCK as i32);
    let mask = off < splat_1d(n, BLOCK as i64);
    let a_ptrs = splat_1d(a, BLOCK as i64) + off;
    let av = load(a_ptrs, mask);
    let bv = load(splat_1d(b, BLOCK as i64) + off, mask);
    store(a_ptrs, av + bv, mask);
}

#[triton_kernel]
fn add_bias_typed<T: TritonElem, const BLOCK: usize>(
    data: Ptr<T>,
    bias: Ptr<T>,
    rows: i32,
    cols: i32,
) {
    let _ = rows;
    let row = program_id(0);
    let row_off = row * cols;
    let col_idx = make_range(0, BLOCK as i32);
    let mask = col_idx < splat_1d(cols, BLOCK as i64);
    let abs_off = splat_1d(row_off, BLOCK as i64) + col_idx;
    let data_ptrs = splat_1d(data, BLOCK as i64) + abs_off;
    let bias_ptrs = splat_1d(bias, BLOCK as i64) + col_idx;
    let dv = load(data_ptrs, mask);
    let bv = load(bias_ptrs, mask);
    store(data_ptrs, dv + bv, mask);
}

// ── f32-internal kernels (load → to_f32 → compute → as_t::<T> → store) ──

#[triton_kernel]
fn rms_norm_typed<T: TritonElem, const BLOCK: usize>(
    input: Ptr<T>,
    weight: Ptr<T>,
    output: Ptr<T>,
    row_size: i32,
    inv_n: f32,
    eps: f32,
) {
    let row = program_id(0);
    let row_off = row * row_size;
    let cols = make_range(0, BLOCK as i32);
    let mask = cols < splat_1d(row_size, BLOCK as i64);
    let abs_off = splat_1d(row_off, BLOCK as i64) + cols;

    let in_ptrs = splat_1d(input, BLOCK as i64) + abs_off;
    let xv = to_f32(load(in_ptrs, mask));
    let sq = xv * xv;
    let sum_sq = reduce(sq, 0, |a, b| a + b);

    let mean = sum_sq * inv_n;
    let inv_rms = rsqrt(mean + eps);

    let inv_rms_v = splat_1d(inv_rms, BLOCK as i64);
    let w_ptrs = splat_1d(weight, BLOCK as i64) + cols;
    let wv = to_f32(load(w_ptrs, mask));

    let result = xv * inv_rms_v * wv;
    let out_ptrs = splat_1d(output, BLOCK as i64) + abs_off;
    store(out_ptrs, as_t::<T>(result), mask);
}

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

    let one = splat_1d(const_f32(1.0), BLOCK as i64);
    let zero = splat_1d(const_f32(0.0), BLOCK as i64);
    let neg_g = zero - gv;
    let exp_neg_g = exp(neg_g);
    let denom = one + exp_neg_g;
    let silu_g = gv / denom;

    let result = silu_g * uv;
    store(splat_1d(out, BLOCK as i64) + off, as_t::<T>(result), mask);
}

#[triton_kernel]
fn fused_add_rms_norm_typed<T: TritonElem, const BLOCK: usize>(
    input: Ptr<T>,
    residual: Ptr<T>,
    weight: Ptr<T>,
    output: Ptr<T>,
    residual_out: Ptr<T>,
    hidden_size: i32,
    inv_n: f32,
    eps: f32,
) {
    let row = program_id(0);
    let row_off = row * hidden_size;
    let cols = make_range(0, BLOCK as i32);
    let mask = cols < hidden_size;
    let abs_off = row_off + cols;

    let xv = to_f32(load(input + abs_off, mask));
    let rv = to_f32(load(residual + abs_off, mask));
    let sum_v = xv + rv;
    store(residual_out + abs_off, as_t::<T>(sum_v), mask);

    let sq = sum_v * sum_v;
    let var_sum = reduce(sq, 0, |a, b| a + b);
    let var = var_sum * inv_n;
    let inv_rms = rsqrt(var + eps);

    let wv = to_f32(load(weight + cols, mask));
    let result = sum_v * inv_rms * wv;
    store(output + abs_off, as_t::<T>(result), mask);
}

#[triton_kernel]
fn layer_norm_typed<T: TritonElem, const BLOCK: usize>(
    x: Ptr<T>,
    gamma: Ptr<T>,
    beta: Ptr<T>,
    out: Ptr<T>,
    dim: i32,
    inv_n: f32,
    eps: f32,
) {
    let row = program_id(0);
    let row_off = row * dim;
    let cols = make_range(0, BLOCK as i32);
    let mask = cols < splat_1d(dim, BLOCK as i64);
    let abs_off = splat_1d(row_off, BLOCK as i64) + cols;

    let xv = to_f32(load(splat_1d(x, BLOCK as i64) + abs_off, mask));

    let sum_x = reduce(xv, 0, |a, b| a + b);
    let mean = sum_x * inv_n;
    let mean_v = splat_1d(mean, BLOCK as i64);

    let centered = xv - mean_v;
    let sq = centered * centered;
    let sum_sq = reduce(sq, 0, |a, b| a + b);
    let var = sum_sq * inv_n;
    let inv_std = rsqrt(var + eps);

    let gv = to_f32(load(splat_1d(gamma, BLOCK as i64) + cols, mask));
    let bv = to_f32(load(splat_1d(beta, BLOCK as i64) + cols, mask));
    let normalized = centered * splat_1d(inv_std, BLOCK as i64);
    let result = normalized * gv + bv;

    store(splat_1d(out, BLOCK as i64) + abs_off, as_t::<T>(result), mask);
}

#[triton_kernel]
fn softmax_typed<T: TritonElem, const BLOCK: usize>(
    input: Ptr<T>,
    output: Ptr<T>,
    rows: i32,
    cols: i32,
) {
    let _ = rows;
    let row = program_id(0);
    let row_off = row * cols;
    let col_idx = make_range(0, BLOCK as i32);
    let mask = col_idx < splat_1d(cols, BLOCK as i64);
    let abs_off = splat_1d(row_off, BLOCK as i64) + col_idx;

    let xv = to_f32(load(splat_1d(input, BLOCK as i64) + abs_off, mask));

    let row_max = reduce(xv, 0, |a, b| max(a, b));
    let max_v = splat_1d(row_max, BLOCK as i64);
    let shifted = xv - max_v;
    let exp_v = exp(shifted);
    let sum_e = reduce(exp_v, 0, |a, b| a + b);

    let inv_sum = const_f32(1.0) / sum_e;
    let result = exp_v * splat_1d(inv_sum, BLOCK as i64);

    store(splat_1d(output, BLOCK as i64) + abs_off, as_t::<T>(result), mask);
}

#[triton_kernel]
fn gelu_typed<T: TritonElem, const BLOCK: usize>(
    x: Ptr<T>,
    out: Ptr<T>,
    n: i32,
) {
    let pid = program_id(0);
    let off = splat_1d(pid * const_i32(BLOCK as i32), BLOCK as i64)
            + make_range(0, BLOCK as i32);
    let mask = off < splat_1d(n, BLOCK as i64);

    let xv = to_f32(load(splat_1d(x, BLOCK as i64) + off, mask));

    let inv_sqrt2 = splat_1d(const_f32(0.707_106_77), BLOCK as i64);
    let half      = splat_1d(const_f32(0.5),          BLOCK as i64);
    let one       = splat_1d(const_f32(1.0),          BLOCK as i64);

    let scaled = xv * inv_sqrt2;
    let erfed  = erf(scaled);
    let result = half * xv * (one + erfed);

    store(splat_1d(out, BLOCK as i64) + off, as_t::<T>(result), mask);
}

// ── helpers ─────────────────────────────────────────────────────────

fn assert_no_casts(text: &str, who: &str) {
    assert!(
        !text.contains("\"arith.extf\""),
        "{who}: f32 path should not contain arith.extf:\n{text}"
    );
    assert!(
        !text.contains("\"arith.truncf\""),
        "{who}: f32 path should not contain arith.truncf:\n{text}"
    );
    assert!(
        !text.contains("xf16>"),
        "{who}: f32 path must not contain f16 tensors:\n{text}"
    );
    assert!(
        !text.contains("xbf16>"),
        "{who}: f32 path must not contain bf16 tensors:\n{text}"
    );
}

fn assert_has_ptr(text: &str, ptr_kind: &str, who: &str) {
    let needle = format!("!tt.ptr<{ptr_kind}>");
    assert!(
        text.contains(&needle),
        "{who}: expected {needle} in IR:\n{text}"
    );
}

// ── pure-elementwise tests ─────────────────────────────────────────

#[test]
fn residual_add_typed_f32_matches_pure_f32() {
    let text = residual_add_typed::<f32, 1024>::mlir();
    assert_no_casts(&text, "residual_add_typed<f32>");
    assert!(text.contains("tensor<1024xf32>"));
    assert!(text.contains("\"arith.addf\""));
}

#[test]
fn residual_add_typed_f16_keeps_f16_throughout() {
    // Pure elementwise: no boundary casts; load+add+store all f16.
    let text = residual_add_typed::<f16, 1024>::mlir();
    assert_has_ptr(&text, "f16", "residual_add_typed<f16>");
    assert!(text.contains("tensor<1024xf16>"));
    assert!(text.contains("\"arith.addf\""));
    assert!(
        !text.contains("\"arith.extf\""),
        "pure f16 elementwise add should not need extf:\n{text}"
    );
    assert!(
        !text.contains("\"arith.truncf\""),
        "pure f16 elementwise add should not need truncf:\n{text}"
    );
}

#[test]
fn residual_add_typed_bf16_keeps_bf16_throughout() {
    let text = residual_add_typed::<bf16, 1024>::mlir();
    assert_has_ptr(&text, "bf16", "residual_add_typed<bf16>");
    assert!(text.contains("tensor<1024xbf16>"));
    assert!(text.contains("\"arith.addf\""));
}

#[test]
fn residual_add_inplace_typed_dtypes_distinct() {
    let f32_t = residual_add_inplace_typed::<f32, 1024>::mlir();
    let f16_t = residual_add_inplace_typed::<f16, 1024>::mlir();
    let bf16_t = residual_add_inplace_typed::<bf16, 1024>::mlir();
    assert_no_casts(&f32_t, "residual_add_inplace_typed<f32>");
    assert_has_ptr(&f16_t, "f16", "residual_add_inplace_typed<f16>");
    assert_has_ptr(&bf16_t, "bf16", "residual_add_inplace_typed<bf16>");
    assert_ne!(f32_t, f16_t);
    assert_ne!(f32_t, bf16_t);
}

#[test]
fn add_bias_typed_dtypes_distinct() {
    let f32_t = add_bias_typed::<f32, 1024>::mlir();
    let f16_t = add_bias_typed::<f16, 1024>::mlir();
    let bf16_t = add_bias_typed::<bf16, 1024>::mlir();
    assert_no_casts(&f32_t, "add_bias_typed<f32>");
    assert_has_ptr(&f16_t, "f16", "add_bias_typed<f16>");
    assert_has_ptr(&bf16_t, "bf16", "add_bias_typed<bf16>");
    assert_ne!(f32_t, f16_t);
    assert_ne!(f32_t, bf16_t);
}

// ── f32-internal kernel tests ──────────────────────────────────────

#[test]
fn rms_norm_typed_f32_collapses_casts() {
    let text = rms_norm_typed::<f32, 1024>::mlir();
    assert_no_casts(&text, "rms_norm_typed<f32>");
    assert!(text.contains("\"tt.reduce\""));
}

#[test]
fn rms_norm_typed_f16_loads_at_f16_computes_in_f32_stores_at_f16() {
    let text = rms_norm_typed::<f16, 1024>::mlir();
    assert_has_ptr(&text, "f16", "rms_norm_typed<f16>");
    assert!(text.contains("tensor<1024xf16>"));
    assert!(text.contains("tensor<1024xf32>"));
    assert!(text.contains("\"arith.extf\""), "missing extf:\n{text}");
    assert!(text.contains("\"arith.truncf\""), "missing truncf:\n{text}");
}

#[test]
fn rms_norm_typed_bf16_loads_at_bf16_computes_in_f32_stores_at_bf16() {
    let text = rms_norm_typed::<bf16, 1024>::mlir();
    assert_has_ptr(&text, "bf16", "rms_norm_typed<bf16>");
    assert!(text.contains("tensor<1024xbf16>"));
    assert!(text.contains("tensor<1024xf32>"));
    assert!(text.contains("\"arith.extf\""), "missing extf:\n{text}");
    assert!(text.contains("\"arith.truncf\""), "missing truncf:\n{text}");
}

#[test]
fn fused_silu_mul_typed_f16_emits_extf_and_truncf() {
    let text = fused_silu_mul_typed::<f16, 1024>::mlir();
    assert_has_ptr(&text, "f16", "fused_silu_mul_typed<f16>");
    assert!(text.contains("\"arith.extf\""));
    assert!(text.contains("\"arith.truncf\""));
    // exp must operate on f32 (no native f16 math.exp on NVPTX).
    assert!(text.contains("\"math.exp\""));
}

#[test]
fn fused_silu_mul_typed_f32_no_casts() {
    let text = fused_silu_mul_typed::<f32, 1024>::mlir();
    assert_no_casts(&text, "fused_silu_mul_typed<f32>");
}

#[test]
fn fused_add_rms_norm_typed_f16_double_store_downcast() {
    let text = fused_add_rms_norm_typed::<f16, 1024>::mlir();
    assert_has_ptr(&text, "f16", "fused_add_rms_norm_typed<f16>");
    // Two stores (residual_out + output), both downcast → at least
    // two truncf occurrences in the IR.
    let truncf_count = text.matches("\"arith.truncf\"").count();
    assert!(
        truncf_count >= 2,
        "expected ≥2 truncf (one per store), got {truncf_count}:\n{text}"
    );
    assert!(text.contains("\"arith.extf\""), "missing extf:\n{text}");
}

#[test]
fn fused_add_rms_norm_typed_f32_no_casts() {
    let text = fused_add_rms_norm_typed::<f32, 1024>::mlir();
    assert_no_casts(&text, "fused_add_rms_norm_typed<f32>");
}

#[test]
fn layer_norm_typed_f16_full_cast_chain() {
    let text = layer_norm_typed::<f16, 1024>::mlir();
    assert_has_ptr(&text, "f16", "layer_norm_typed<f16>");
    assert!(text.contains("\"arith.extf\""));
    assert!(text.contains("\"arith.truncf\""));
}

#[test]
fn layer_norm_typed_f32_no_casts() {
    let text = layer_norm_typed::<f32, 1024>::mlir();
    assert_no_casts(&text, "layer_norm_typed<f32>");
}

#[test]
fn softmax_typed_f16_emits_extf_truncf_and_keeps_exp_in_f32() {
    let text = softmax_typed::<f16, 1024>::mlir();
    assert_has_ptr(&text, "f16", "softmax_typed<f16>");
    assert!(text.contains("\"arith.extf\""));
    assert!(text.contains("\"arith.truncf\""));
    assert!(text.contains("\"math.exp\""));
}

#[test]
fn softmax_typed_f32_no_casts() {
    let text = softmax_typed::<f32, 1024>::mlir();
    assert_no_casts(&text, "softmax_typed<f32>");
}

#[test]
fn gelu_typed_f16_emits_extf_truncf_and_keeps_erf_in_f32() {
    let text = gelu_typed::<f16, 1024>::mlir();
    assert_has_ptr(&text, "f16", "gelu_typed<f16>");
    assert!(text.contains("\"arith.extf\""));
    assert!(text.contains("\"arith.truncf\""));
    assert!(text.contains("\"math.erf\""));
}

#[test]
fn gelu_typed_f32_no_casts() {
    let text = gelu_typed::<f32, 1024>::mlir();
    assert_no_casts(&text, "gelu_typed<f32>");
}

// ── three-dtype sanity check ───────────────────────────────────────

#[test]
fn ferrum_kernels_all_three_dtypes_yield_distinct_funcs() {
    // Spot-check each kernel: f32 / f16 / bf16 must produce different IR.
    let cases: [(&str, [String; 3]); 9] = [
        ("residual_add", [
            residual_add_typed::<f32, 1024>::mlir(),
            residual_add_typed::<f16, 1024>::mlir(),
            residual_add_typed::<bf16, 1024>::mlir(),
        ]),
        ("residual_add_inplace", [
            residual_add_inplace_typed::<f32, 1024>::mlir(),
            residual_add_inplace_typed::<f16, 1024>::mlir(),
            residual_add_inplace_typed::<bf16, 1024>::mlir(),
        ]),
        ("add_bias", [
            add_bias_typed::<f32, 1024>::mlir(),
            add_bias_typed::<f16, 1024>::mlir(),
            add_bias_typed::<bf16, 1024>::mlir(),
        ]),
        ("rms_norm", [
            rms_norm_typed::<f32, 1024>::mlir(),
            rms_norm_typed::<f16, 1024>::mlir(),
            rms_norm_typed::<bf16, 1024>::mlir(),
        ]),
        ("fused_silu_mul", [
            fused_silu_mul_typed::<f32, 1024>::mlir(),
            fused_silu_mul_typed::<f16, 1024>::mlir(),
            fused_silu_mul_typed::<bf16, 1024>::mlir(),
        ]),
        ("fused_add_rms_norm", [
            fused_add_rms_norm_typed::<f32, 1024>::mlir(),
            fused_add_rms_norm_typed::<f16, 1024>::mlir(),
            fused_add_rms_norm_typed::<bf16, 1024>::mlir(),
        ]),
        ("layer_norm", [
            layer_norm_typed::<f32, 1024>::mlir(),
            layer_norm_typed::<f16, 1024>::mlir(),
            layer_norm_typed::<bf16, 1024>::mlir(),
        ]),
        ("softmax", [
            softmax_typed::<f32, 1024>::mlir(),
            softmax_typed::<f16, 1024>::mlir(),
            softmax_typed::<bf16, 1024>::mlir(),
        ]),
        ("gelu", [
            gelu_typed::<f32, 1024>::mlir(),
            gelu_typed::<f16, 1024>::mlir(),
            gelu_typed::<bf16, 1024>::mlir(),
        ]),
    ];

    for (name, [a, b, c]) in &cases {
        assert_ne!(a, b, "{name}: f32 vs f16 IR identical");
        assert_ne!(a, c, "{name}: f32 vs bf16 IR identical");
        assert_ne!(b, c, "{name}: f16 vs bf16 IR identical");
    }
}
