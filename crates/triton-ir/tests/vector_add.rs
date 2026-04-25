//! End-to-end IR builder test: hand-write the canonical Triton vector_add
//! kernel and assert the printed MLIR matches a known-good snapshot.
//!
//! The reference text is the **MLIR generic form** since that's what our
//! Phase 2 printer emits. The Triton MLIR parser accepts generic form for
//! every op, so this same text can be fed to `triton_compile_mlir` once the
//! C ABI shim lands in Phase 1 (cross-phase work item).

use triton_ir::dialect::arith::CmpiPred;
use triton_ir::prelude::*;

const BLOCK: i64 = 1024;

fn build_vector_add() -> Module {
    let mut m = Module::new();

    let mut f = m.func("vector_add");
    let x_ptr = f.arg("x", Type::ptr(Type::f32()));
    let y_ptr = f.arg("y", Type::ptr(Type::f32()));
    let out_ptr = f.arg("out", Type::ptr(Type::f32()));
    let n = f.arg("n", Type::i32());

    // pid = tt.get_program_id(0)
    let pid = f.op_one(tt::get_program_id(0));

    // base = pid * BLOCK
    let block_const = f.op_one(arith::constant_i32(BLOCK as i32));
    let base = f.op_one(arith::muli(pid, block_const));

    // range = tt.make_range(0, BLOCK)  -> tensor<BLOCKxi32>
    let range = f.op_one(tt::make_range(0, BLOCK as i32));

    // base_v = tt.splat(base) -> tensor<BLOCKxi32>
    let base_v = f.op_one(tt::splat(base, vec![BLOCK]));

    // off = base_v + range
    let off = f.op_one(arith::addi(base_v, range));

    // mask = off < splat(n)
    let n_v = f.op_one(tt::splat(n, vec![BLOCK]));
    let mask = f.op_one(arith::cmpi(CmpiPred::Slt, off.clone(), n_v));

    // x_ptrs = splat(x_ptr) ; x_ptrs += off ; xv = load(x_ptrs, mask)
    let xp = f.op_one(tt::splat(x_ptr, vec![BLOCK]));
    let xp_off = f.op_one(tt::addptr(xp, off.clone()));
    let xv = f.op_one(tt::load(xp_off, Some(mask.clone())));

    // y_ptrs likewise
    let yp = f.op_one(tt::splat(y_ptr, vec![BLOCK]));
    let yp_off = f.op_one(tt::addptr(yp, off.clone()));
    let yv = f.op_one(tt::load(yp_off, Some(mask.clone())));

    // sum = xv + yv  (float add)
    let sum = f.op_one(arith::addf(xv, yv));

    // out_ptrs += off ; store(out_ptrs, sum, mask)
    let outp = f.op_one(tt::splat(out_ptr, vec![BLOCK]));
    let outp_off = f.op_one(tt::addptr(outp, off));
    f.op_void(tt::store(outp_off, sum, Some(mask)));

    f.op_void(tt::return_());
    f.finish();

    m
}

#[test]
fn vector_add_prints_generic_mlir() {
    let m = build_vector_add();
    let text = m.to_string();

    // Spot-checks: the MLIR text contains the structural pieces we expect.
    // We don't compare to a literal byte-for-byte string yet (that comes
    // when we add insta snapshot testing in Phase 2.9 polish).
    assert!(text.contains("module {"), "missing module header:\n{text}");
    // No `public` keyword: matches upstream Triton convention. The
    // `tt.func` parser does not accept a visibility prefix.
    assert!(
        text.contains("tt.func @vector_add("),
        "missing func header:\n{text}"
    );
    assert!(
        !text.contains("tt.func public"),
        "tt.func should not carry a visibility keyword:\n{text}"
    );
    assert!(text.contains("!tt.ptr<f32>"), "missing pointer type:\n{text}");
    assert!(text.contains("tensor<1024xi32>"), "missing tile-width tensor:\n{text}");
    assert!(text.contains("\"tt.get_program_id\""), "missing program_id:\n{text}");
    assert!(text.contains("\"tt.make_range\""), "missing make_range:\n{text}");
    assert!(text.contains("\"tt.splat\""), "missing splat:\n{text}");
    assert!(text.contains("\"tt.addptr\""), "missing addptr:\n{text}");
    assert!(text.contains("\"tt.load\""), "missing load:\n{text}");
    assert!(text.contains("\"tt.store\""), "missing store:\n{text}");
    assert!(text.contains("\"arith.addf\""), "missing addf:\n{text}");
    assert!(text.contains("\"arith.cmpi\""), "missing cmpi:\n{text}");
    assert!(text.contains("\"tt.return\""), "missing return:\n{text}");
}

#[test]
fn vector_add_print_for_eyeballing() {
    // Print the kernel so a human (or `cargo test -- --nocapture`) can see
    // the actual generated IR. Useful while iterating on the printer.
    let m = build_vector_add();
    let text = m.to_string();
    eprintln!("===== vector_add MLIR =====");
    eprintln!("{text}");
    eprintln!("===========================");
    assert!(text.lines().count() > 10);
}
