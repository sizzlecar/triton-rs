//! Phase 3.5: element-type cast vocabulary — `to_f32`, `to_f16`, `to_i32`.
//!
//! These wrap arith.sitofp / arith.fptosi / arith.extf / arith.truncf /
//! arith.extsi / arith.trunci. The DSL surface lets a kernel write
//! `to_f32(int_value)` in the natural way; the proc-macro routes it to
//! FuncBuilder's cast_with_elem dispatcher which picks the correct op.

use triton_dsl::triton_kernel;

#[allow(dead_code)]
struct Ptr<T>(::std::marker::PhantomData<T>);

// int → float: a tile of i32 offsets gets converted to f32 before storing.
#[triton_kernel]
fn cast_i32_to_f32(out: Ptr<f32>) {
    let pid = program_id(0);
    let cols = make_range(0, 64);
    let off = pid * 64 + cols;
    let as_f = to_f32(off);
    store(out + off, as_f);
}

// float → int: a tile of f32 loaded values is truncated to i32 and stored.
#[triton_kernel]
fn cast_f32_to_i32(x: Ptr<f32>, out: Ptr<i32>) {
    let pid = program_id(0);
    let cols = make_range(0, 64);
    let off = pid * 64 + cols;
    let xv = load(x + off);
    let xi = to_i32(xv);
    store(out + off, xi);
}

// float-narrow: f32 → f16 via arith.truncf.
#[triton_kernel]
fn cast_f32_to_f16(x: Ptr<f32>, out: Ptr<f16>) {
    let pid = program_id(0);
    let cols = make_range(0, 64);
    let off = pid * 64 + cols;
    let xv = load(x + off);
    let xh = to_f16(xv);
    store(out + off, xh);
}

#[test]
fn int_to_float_emits_sitofp() {
    let text = cast_i32_to_f32::mlir();
    eprintln!("===== cast_i32_to_f32 =====\n{text}");
    assert!(
        text.contains("\"arith.sitofp\""),
        "missing arith.sitofp for i32→f32:\n{text}"
    );
    // Result tensor type must be f32-element of the same shape.
    assert!(
        text.contains("tensor<64xf32>"),
        "missing tensor<64xf32> result of cast:\n{text}"
    );
}

#[test]
fn float_to_int_emits_fptosi() {
    let text = cast_f32_to_i32::mlir();
    eprintln!("===== cast_f32_to_i32 =====\n{text}");
    assert!(
        text.contains("\"arith.fptosi\""),
        "missing arith.fptosi for f32→i32:\n{text}"
    );
}

#[test]
fn float_narrow_emits_truncf() {
    let text = cast_f32_to_f16::mlir();
    eprintln!("===== cast_f32_to_f16 =====\n{text}");
    assert!(
        text.contains("\"arith.truncf\""),
        "missing arith.truncf for f32→f16:\n{text}"
    );
    assert!(
        text.contains("tensor<64xf16>"),
        "missing tensor<64xf16> result of narrowing cast:\n{text}"
    );
}
