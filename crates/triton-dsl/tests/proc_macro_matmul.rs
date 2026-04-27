//! Smoke test for the GEMM kernel: covers 2D pointer arithmetic +
//! `tt.dot` + `scf.for` over K with a tile accumulator iter_arg, all
//! authored through the DSL surface (no hand-rolled IR builder calls).
//!
//! Also covers the dtype-generic `matmul_typed<T>` variant — same kernel
//! body parameterized over `T: TritonElem`, with `to_f32` upcasts at the
//! load boundary and `as_t::<T>` downcast at store. Mirrors the pattern
//! used by `flash_attn_full` in triton-kernels.

use triton_dsl::triton_kernel;
use triton_ir::ty::{bf16, f16, TritonElem};

#[allow(dead_code)]
struct Ptr<T>(::std::marker::PhantomData<T>);

#[triton_kernel]
fn matmul_f32<const BM: usize, const BN: usize, const BK: usize>(
    a_ptr: Ptr<f32>,
    b_ptr: Ptr<f32>,
    c_ptr: Ptr<f32>,
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

        dot(a_block, b_block, acc)
    });

    let c_off = offs_m_2d * stride_cm + offs_n_2d * stride_cn;
    let mask_c = expand_dims(mask_m, 1) & expand_dims(mask_n, 0);
    store(c_ptr + c_off, acc, mask_c);
}

#[test]
fn matmul_f32_emits_dot_with_correct_block_shapes() {
    let text = matmul_f32::<64, 64, 32>::mlir();
    eprintln!(
        "===== matmul_f32<64,64,32> MLIR =====\n{text}\n====================================="
    );

    // Body is a streaming K loop carrying the BM×BN accumulator.
    assert!(
        text.contains("\"scf.for\""),
        "missing scf.for K loop:\n{text}"
    );
    assert!(
        text.contains("\"scf.yield\""),
        "missing scf.yield in K loop:\n{text}"
    );

    // Core block matmul: A[BM,BK] @ B[BK,BN] + C[BM,BN] -> C[BM,BN].
    assert!(text.contains("\"tt.dot\""), "missing tt.dot:\n{text}");
    assert!(
        text.contains(
            "(tensor<64x32xf32>, tensor<32x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>",
        ),
        "tt.dot signature does not match BM=64, BK=32, BN=64:\n{text}",
    );

    // 2D pointer arithmetic landed in the body — addptr on tensor-of-ptr
    // for A (BM×BK), B (BK×BN), and C (BM×BN).
    assert!(
        text.contains("tensor<64x32x!tt.ptr<f32>>"),
        "missing A tile ptr type:\n{text}"
    );
    assert!(
        text.contains("tensor<32x64x!tt.ptr<f32>>"),
        "missing B tile ptr type:\n{text}"
    );
    assert!(
        text.contains("tensor<64x64x!tt.ptr<f32>>"),
        "missing C tile ptr type:\n{text}"
    );

    // 2D mask via outer-AND of 1D bounds tensors. arith.andi on i1 tensors.
    assert!(
        text.contains("\"arith.andi\""),
        "missing andi for 2D masking:\n{text}"
    );

    // Final store at the end of the kernel.
    assert!(text.contains("\"tt.store\""), "missing tt.store:\n{text}");
    assert!(text.contains("\"tt.return\""), "missing tt.return:\n{text}");
}

#[test]
fn matmul_f32_block_sizes_propagate_via_const_generics() {
    // Different instantiation should change the tile shape attributes
    // (proves the const-generic plumbing isn't accidentally hard-coded).
    let text = matmul_f32::<32, 32, 16>::mlir();
    assert!(
        text.contains(
            "(tensor<32x16xf32>, tensor<16x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>",
        ),
        "tt.dot signature does not match BM=32, BK=16, BN=32:\n{text}",
    );
}

// ── dtype-generic matmul_typed<T> ──────────────────────────────────────
//
// Same kernel body as matmul_f32, but the element type is a generic
// `T: TritonElem`. Inputs are loaded at T then upcast to f32 via
// `to_f32`; the f32 accumulator runs through `tt.dot` in f32; the result
// is downcast back to T via `as_t::<T>` at store time. This is the
// pattern flash_attn_full uses for its f16 instantiation: NVPTX has no
// native f16 division / `math.exp` so the loop body must run in f32.
//
// For T == f32 the upcast/downcast pairs collapse and the IR matches
// matmul_f32 byte-for-byte.

#[triton_kernel]
fn matmul_typed<T: TritonElem, const BM: usize, const BN: usize, const BK: usize>(
    a_ptr: Ptr<T>,
    b_ptr: Ptr<T>,
    c_ptr: Ptr<T>,
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

    // Accumulator stays in f32 regardless of T — same precision argument
    // as flash_attn_full's online-softmax state.
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
        let a_block_t = load(a_ptr + a_off, mask_a);
        let a_block = to_f32(a_block_t);

        let b_off = offs_k_col * stride_bk + offs_n_2d * stride_bn;
        let mask_b = expand_dims(mask_k, 1) & expand_dims(mask_n, 0);
        let b_block_t = load(b_ptr + b_off, mask_b);
        let b_block = to_f32(b_block_t);

        dot(a_block, b_block, acc)
    });

    let acc_t = as_t::<T>(acc);
    let c_off = offs_m_2d * stride_cm + offs_n_2d * stride_cn;
    let mask_c = expand_dims(mask_m, 1) & expand_dims(mask_n, 0);
    store(c_ptr + c_off, acc_t, mask_c);
}

#[test]
fn matmul_typed_f16_loads_at_f16_accumulates_in_f32_stores_at_f16() {
    let text = matmul_typed::<f16, 64, 64, 32>::mlir();
    eprintln!("===== matmul_typed<f16,64,64,32> MLIR =====\n{text}\n=====================================");

    // Pointer params are f16.
    assert!(
        text.contains("!tt.ptr<f16>"),
        "expected f16 pointer parameters:\n{text}"
    );

    // Loads return f16 tile tensors.
    assert!(
        text.contains("tensor<64x32xf16>"),
        "expected f16 A tile:\n{text}"
    );
    assert!(
        text.contains("tensor<32x64xf16>"),
        "expected f16 B tile:\n{text}"
    );

    // The to_f32 boundary upcast is `arith.extf` from f16 -> f32 tile.
    assert!(
        text.contains("\"arith.extf\""),
        "missing extf upcast on load boundary:\n{text}"
    );

    // tt.dot runs in f32 — accumulator and inputs are all f32 tensors.
    assert!(
        text.contains(
            "(tensor<64x32xf32>, tensor<32x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>",
        ),
        "tt.dot must operate on f32 tensors with f32 accumulator:\n{text}"
    );

    // Final cast back to f16 via arith.truncf, then 2D store on f16 ptr.
    assert!(
        text.contains("\"arith.truncf\""),
        "missing truncf at store boundary:\n{text}"
    );
    assert!(
        text.contains("tensor<64x64x!tt.ptr<f16>>"),
        "expected f16 C tile ptr type:\n{text}"
    );
    assert!(
        text.contains("tensor<64x64xf16>"),
        "expected final f16 C tile:\n{text}"
    );
}

#[test]
fn matmul_typed_bf16_loads_at_bf16_accumulates_in_f32_stores_at_bf16() {
    let text = matmul_typed::<bf16, 64, 64, 32>::mlir();
    eprintln!("===== matmul_typed<bf16,64,64,32> MLIR =====\n{text}\n=====================================");

    // Pointer params are bf16.
    assert!(
        text.contains("!tt.ptr<bf16>"),
        "expected bf16 pointer parameters:\n{text}"
    );

    // Load tiles are bf16; accumulator inside dot is f32.
    assert!(
        text.contains("tensor<64x32xbf16>"),
        "expected bf16 A tile:\n{text}"
    );
    assert!(
        text.contains("tensor<32x64xbf16>"),
        "expected bf16 B tile:\n{text}"
    );

    // bf16 -> f32 upcast (extf) then f32 dot.
    assert!(
        text.contains("\"arith.extf\""),
        "missing extf upcast on load boundary:\n{text}"
    );
    assert!(
        text.contains(
            "(tensor<64x32xf32>, tensor<32x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>",
        ),
        "tt.dot must operate on f32 tensors with f32 accumulator:\n{text}"
    );

    // Cast back to bf16 + bf16 store.
    assert!(
        text.contains("\"arith.truncf\""),
        "missing truncf at store boundary:\n{text}"
    );
    assert!(
        text.contains("tensor<64x64x!tt.ptr<bf16>>"),
        "expected bf16 C tile ptr type:\n{text}"
    );
    assert!(
        text.contains("tensor<64x64xbf16>"),
        "expected final bf16 C tile:\n{text}"
    );
}

#[test]
fn matmul_typed_f32_collapses_casts_and_matches_matmul_f32() {
    // Sanity check: with T = f32, all upcasts / downcasts become the
    // identity. The IR should contain neither extf nor truncf.
    let text = matmul_typed::<f32, 64, 64, 32>::mlir();
    eprintln!("===== matmul_typed<f32,64,64,32> MLIR =====\n{text}\n=====================================");

    assert!(
        text.contains("!tt.ptr<f32>"),
        "expected f32 pointer parameters:\n{text}"
    );
    assert!(
        text.contains(
            "(tensor<64x32xf32>, tensor<32x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>",
        ),
        "tt.dot signature must match matmul_f32:\n{text}"
    );
    assert!(
        !text.contains("\"arith.extf\""),
        "T == f32 should have no extf casts:\n{text}"
    );
    assert!(
        !text.contains("\"arith.truncf\""),
        "T == f32 should have no truncf casts:\n{text}"
    );
    assert!(
        !text.contains("xf16>"),
        "T == f32 must not contain f16 tensors:\n{text}"
    );
}

#[test]
fn matmul_typed_two_dtypes_yield_distinct_funcs() {
    // Same source, three element types, three different MLIR modules.
    let f32_text = matmul_typed::<f32, 64, 64, 32>::mlir();
    let f16_text = matmul_typed::<f16, 64, 64, 32>::mlir();
    let bf16_text = matmul_typed::<bf16, 64, 64, 32>::mlir();
    assert_ne!(f32_text, f16_text);
    assert_ne!(f32_text, bf16_text);
    assert_ne!(f16_text, bf16_text);
}
