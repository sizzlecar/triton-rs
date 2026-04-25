//! Dump the proc-macro-translated `hello_kernel` MLIR to stdout, so it
//! can be fed to `tools/validate_mlir.py` for round-trip parsing through
//! a real Triton install. Proves the DSL → IR → text → Triton path works
//! end to end.

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

fn main() {
    print!("{}", hello_kernel::mlir());
}
