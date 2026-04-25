//! Integration test: write a kernel using `#[triton_kernel]` and check the
//! emitted MLIR matches what we'd produce by hand-rolling the IR builder.
//!
//! This validates Phase 3.1 of the DSL: signature parsing + skeleton MLIR
//! emission. Function body translation arrives in Phase 3.3.

use triton_dsl::triton_kernel;

// `Ptr<T>` is a marker the proc-macro recognises in argument types — it
// never appears in the expanded output, so we don't need a real definition.
// The original `fn` body is replaced by the macro, so the only remaining
// reference to `Ptr<T>` is in the (discarded) Rust signature. Provide a
// trivial alias to keep tooling that scans the source happy.
#[allow(dead_code)]
struct Ptr<T>(::std::marker::PhantomData<T>);

#[triton_kernel]
fn vec_add_signature(x: Ptr<f32>, y: Ptr<f32>, out: Ptr<f32>, n: i32) {}

#[triton_kernel]
fn dot_signature(a: Ptr<f16>, b: Ptr<f16>, c: Ptr<f32>, m: i32, n: i32, k: i32) {}

#[test]
fn vec_add_signature_emits_well_formed_func() {
    let text = vec_add_signature::mlir();
    eprintln!("===== vec_add_signature MLIR =====\n{text}\n==================================");

    assert!(
        text.contains("tt.func @vec_add_signature("),
        "missing kernel func header:\n{text}"
    );
    // Three pointer args + one i32 — three `!tt.ptr<f32>` substrings expected.
    assert_eq!(
        text.matches("!tt.ptr<f32>").count(),
        3,
        "expected exactly 3 !tt.ptr<f32> args:\n{text}"
    );
    assert!(text.contains(": i32"), "missing i32 arg:\n{text}");
    // Body is empty in Phase 3.1 — only the terminator should be present.
    assert!(text.contains("\"tt.return\""), "missing terminator:\n{text}");
    assert!(
        !text.contains("\"tt.load\"") && !text.contains("\"tt.store\""),
        "body should not yet be translated in Phase 3.1:\n{text}"
    );
}

#[test]
fn dot_signature_handles_mixed_pointer_types() {
    let text = dot_signature::mlir();
    eprintln!("===== dot_signature MLIR =====\n{text}\n==============================");

    assert!(text.contains("tt.func @dot_signature("));
    assert_eq!(text.matches("!tt.ptr<f16>").count(), 2);
    assert_eq!(text.matches("!tt.ptr<f32>").count(), 1);
    assert_eq!(text.matches(": i32").count(), 3);
}

#[test]
fn module_constructor_round_trips_through_to_string() {
    // `module()` returns an owned `Module`; converting to string should
    // give the same text as the `mlir()` shortcut.
    let m = vec_add_signature::module();
    assert_eq!(m.to_string(), vec_add_signature::mlir());
}
