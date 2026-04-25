//! Phase 3.4 step 3: const generics in #[triton_kernel] signatures.
//! The kernel becomes parameterised on a tile size (or any other usize/i32
//! const-generic carrier). Each instantiation produces its own MLIR module.

use triton_dsl::triton_kernel;

#[allow(dead_code)]
struct Ptr<T>(::std::marker::PhantomData<T>);

#[triton_kernel]
fn vec_add_const_generic<const BLOCK: usize>(
    x: Ptr<f32>,
    y: Ptr<f32>,
    out: Ptr<f32>,
    n: i32,
) {
    let pid = program_id(0);
    let block_const = const_i32(BLOCK as i32);
    let off = splat_1d(pid * block_const, BLOCK as i64) + make_range(0, BLOCK as i32);
    let mask = off < splat_1d(n, BLOCK as i64);
    let xv = load(splat_1d(x, BLOCK as i64) + off, mask);
    let yv = load(splat_1d(y, BLOCK as i64) + off, mask);
    store(splat_1d(out, BLOCK as i64) + off, xv + yv, mask);
}

#[test]
fn vec_add_const_generic_block_1024() {
    let text = vec_add_const_generic::<1024>::mlir();
    eprintln!("===== vec_add<1024> =====\n{text}\n=========================");

    assert!(text.contains("\"arith.constant\"() {value = 1024 : i32}"));
    assert!(text.contains("tensor<1024xi32>"));
    assert!(text.contains("tensor<1024xi1>"));
    assert!(text.contains("tensor<1024xf32>"));
    assert!(text.contains("tensor<1024x!tt.ptr<f32>>"));
}

#[test]
fn vec_add_const_generic_block_256() {
    let text = vec_add_const_generic::<256>::mlir();
    eprintln!("===== vec_add<256> =====\n{text}\n========================");

    // The same source produces a different module per instantiation.
    assert!(text.contains("\"arith.constant\"() {value = 256 : i32}"));
    assert!(text.contains("tensor<256xi32>"));
    assert!(text.contains("tensor<256xf32>"));
    assert!(text.contains("tensor<256x!tt.ptr<f32>>"));
    // And does NOT mention 1024 anywhere.
    assert!(
        !text.contains("1024"),
        "vec_add<256> output should not mention block 1024:\n{text}"
    );
}

#[test]
fn different_instantiations_produce_distinct_modules() {
    let m_1024 = vec_add_const_generic::<1024>::mlir();
    let m_256 = vec_add_const_generic::<256>::mlir();
    assert_ne!(m_1024, m_256, "different BLOCK values must produce different MLIR");
}
