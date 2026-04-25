//! Integration test: body translation. Validates that a `#[triton_kernel]`
//! function with explicit-typed builder helpers (Phase 3.3 step 1
//! vocabulary) produces the same MLIR a hand-rolled IR builder would.

use triton_dsl::triton_kernel;

#[allow(dead_code)]
struct Ptr<T>(::std::marker::PhantomData<T>);

#[triton_kernel]
fn hello_kernel(out: Ptr<i32>) {
    let pid = program_id(0);
    let c = const_i32(42);
    let v = add_i32(pid, c);
    store(out, v);
}

#[triton_kernel]
fn nested_calls_kernel(out: Ptr<i32>) {
    // Tests recursive translate_call: arguments are themselves recognized
    // calls. Each inner call should also become an IR builder invocation.
    let v = add_i32(program_id(0), const_i32(7));
    store(out, v);
}

#[test]
fn hello_body_emits_op_chain() {
    let text = hello_kernel::mlir();
    eprintln!("===== hello_kernel MLIR =====\n{text}\n=============================");

    assert!(text.contains("\"tt.get_program_id\""), "missing get_program_id:\n{text}");
    assert!(
        text.contains("\"arith.constant\"() {value = 42 : i32}"),
        "missing constant 42:\n{text}"
    );
    assert!(text.contains("\"arith.addi\""), "missing addi:\n{text}");
    assert!(text.contains("\"tt.store\""), "missing store:\n{text}");
    assert!(text.contains("\"tt.return\""), "missing terminator:\n{text}");
}

#[test]
fn nested_calls_translate_recursively() {
    let text = nested_calls_kernel::mlir();
    eprintln!("===== nested_calls_kernel MLIR =====\n{text}\n====================================");

    // Every inner call should have produced its own op.
    assert!(text.contains("\"tt.get_program_id\""));
    assert_eq!(
        text.matches("\"arith.constant\"").count(),
        1,
        "nested const_i32 should have produced exactly one constant op"
    );
    assert!(text.contains("\"arith.addi\""));
    assert!(text.contains("\"tt.store\""));
}

#[test]
fn body_terminator_inserted_automatically() {
    // The user didn't write `return_()`, but the codegen always appends one.
    let text = hello_kernel::mlir();
    let return_count = text.matches("\"tt.return\"").count();
    assert_eq!(return_count, 1, "expected exactly one tt.return:\n{text}");
}
