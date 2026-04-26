//! Phase 3.6: dtype-generic kernels via `T: TritonElem`.
//!
//! Mirrors Python @triton.jit's "one source per algorithm, one cubin per
//! dtype" pattern: write the kernel body once, parameterize over the
//! element type. The proc-macro emits `<T as TritonElem>::ir_type()`
//! wherever the type appears in the signature; instantiating
//! `kernel::<f32, ...>::mlir()` resolves T to f32 and builds the right
//! IR; `kernel::<f16, ...>::mlir()` builds the f16 variant from the
//! same body.

use triton_dsl::triton_kernel;
use triton_ir::ty::{f16, TritonElem};

#[allow(dead_code)]
struct Ptr<T>(::std::marker::PhantomData<T>);

#[triton_kernel]
fn vec_add_generic<T: TritonElem, const BLOCK: usize>(
    a: Ptr<T>,
    b: Ptr<T>,
    out: Ptr<T>,
    n: i32,
) {
    let pid = program_id(0);
    let off = pid * (BLOCK as i32) + make_range(0, BLOCK as i32);
    let mask = off < n;
    let av = load(a + off, mask);
    let bv = load(b + off, mask);
    store(out + off, av + bv, mask);
}

#[test]
fn dtype_generic_f32_emits_f32_tensors_and_addf() {
    let text = vec_add_generic::<f32, 1024>::mlir();
    eprintln!("===== f32 instantiation =====\n{text}");
    assert!(text.contains("tensor<1024xf32>"));
    assert!(text.contains("\"arith.addf\""));
    assert!(!text.contains("xf16>"), "should not contain f16 tensors:\n{text}");
}

#[test]
fn dtype_generic_f16_emits_f16_tensors_and_addf() {
    let text = vec_add_generic::<f16, 1024>::mlir();
    eprintln!("===== f16 instantiation =====\n{text}");
    assert!(text.contains("tensor<1024xf16>"));
    assert!(text.contains("\"arith.addf\""));
    assert!(!text.contains("xf32>"), "should not contain f32 tensors:\n{text}");
}

#[test]
fn dtype_generic_two_instances_yield_distinct_funcs() {
    // Same source, different MLIR — confirms the type param actually
    // propagates through the IR builder.
    let f32_text = vec_add_generic::<f32, 1024>::mlir();
    let f16_text = vec_add_generic::<f16, 1024>::mlir();
    assert_ne!(f32_text, f16_text);
}
