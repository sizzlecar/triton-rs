//! Phase 3.4 step 2: Rust `if cond { ... } [else { ... }]` translates to
//! `scf.if` inside `#[triton_kernel]` bodies. Two forms:
//!
//!   1. Statement form (no value yielded). Optional else.
//!   2. Expression form (`let v = if ... else ...`). Both branches required.

use triton_dsl::triton_kernel;

#[allow(dead_code)]
struct Ptr<T>(::std::marker::PhantomData<T>);

// ── Statement form, no else: early-return / guarded-store. ───────────
//
// Pattern lifted from vLLM's triton_unified_attention.py:197 — early
// return for blocks past the query length:
//   `if q_block_local_idx * BLOCK_Q >= cur_batch_query_len: return`
//
// Without scf.if support we'd be forced to express this via a mask
// tensor, which won't compose with paged-attention pruning where some
// program_ids should not run any kernel work at all.
#[triton_kernel]
fn early_return_demo<const BLOCK: usize>(x: Ptr<f32>, out: Ptr<f32>, n: i32, threshold: i32) {
    let pid = program_id(0);
    if pid < threshold {
        let off = pid * (BLOCK as i32) + make_range(0, BLOCK as i32);
        let n_v = splat_1d(n, BLOCK as i32);
        let mask = off < n_v;
        let xp = splat_1d(x, BLOCK as i32);
        let xp_off = addptr(xp, off);
        let v = load(xp_off, mask);

        let outp = splat_1d(out, BLOCK as i32);
        let outp_off = addptr(outp, off);
        store(outp_off, v, mask);
    }
}

#[test]
fn scf_if_statement_form_emits_guarded_store() {
    let text = early_return_demo::<1024>::mlir();
    eprintln!("===== early_return_demo MLIR =====\n{text}\n==================================");

    assert!(text.contains("\"scf.if\""), "missing scf.if op:\n{text}");
    // Two scf.yield terminators — one per region (then + empty else).
    assert!(
        text.matches("\"scf.yield\"").count() >= 2,
        "expected ≥2 scf.yield terminators (then + else):\n{text}"
    );
    // Statement form: signature must be `(i1) -> ()`.
    assert!(
        text.contains("(i1) -> ()"),
        "scf.if statement form should print `(i1) -> ()` signature:\n{text}"
    );
    // The store must live INSIDE the scf.if region. Find the scf.if op
    // start and check the next tt.store is between the opening `({` and
    // the closing `})`.
    let scf_if_pos = text.find("\"scf.if\"").expect("scf.if not found");
    let after = &text[scf_if_pos..];
    let store_offset = after
        .find("\"tt.store\"")
        .expect("tt.store not found after scf.if");
    let close_offset = after.find("})").expect("scf.if region close not found");
    assert!(
        store_offset < close_offset,
        "tt.store must appear inside the scf.if region:\nscf.if at {scf_if_pos}\nstore offset {store_offset}\nregion close offset {close_offset}\nMLIR:\n{text}"
    );
}

// ── Statement form with else: branch on parity, store from each branch. ─
#[triton_kernel]
fn dispatch_demo(out: Ptr<i32>) {
    let pid = program_id(0);
    let two = const_i32(2);
    let zero = const_i32(0);
    let parity = pid % two;
    if parity == zero {
        let v = const_i32(100);
        store(out, v);
    } else {
        let v = const_i32(200);
        store(out, v);
    }
}

#[test]
fn scf_if_statement_form_with_else_emits_both_branches() {
    let text = dispatch_demo::mlir();
    eprintln!("===== dispatch_demo MLIR =====\n{text}\n==============================");

    assert!(text.contains("\"scf.if\""), "missing scf.if:\n{text}");
    // Both branches store, so we expect two tt.store occurrences inside
    // the scf.if region.
    assert!(
        text.matches("\"tt.store\"").count() >= 2,
        "expected ≥2 tt.store (one per branch):\n{text}"
    );
    assert!(
        text.contains("(i1) -> ()"),
        "statement-form with else should still have `(i1) -> ()` signature:\n{text}"
    );
}

// ── Expression form: `let v = if cond { a } else { b };`. ─────────────
#[triton_kernel]
fn ternary_const(out: Ptr<i32>) {
    let pid = program_id(0);
    let zero = const_i32(0);
    let v = if pid == zero {
        const_i32(42)
    } else {
        const_i32(99)
    };
    store(out, v);
}

#[test]
fn scf_if_expression_form_yields_typed_result() {
    let text = ternary_const::mlir();
    eprintln!("===== ternary_const MLIR =====\n{text}\n==============================");

    assert!(text.contains("\"scf.if\""), "missing scf.if:\n{text}");
    // Both branches yield exactly one i32 — scf.if signature should be `(i1) -> i32`.
    assert!(
        text.contains("(i1) -> i32"),
        "scf.if expression form should print `(i1) -> i32` signature:\n{text}"
    );
    // The store after the if must consume the if's result — check
    // textual order: scf.if appears before tt.store.
    let if_pos = text.find("\"scf.if\"").expect("scf.if not found");
    let store_pos = text.find("\"tt.store\"").expect("tt.store not found");
    assert!(
        store_pos > if_pos,
        "tt.store should follow scf.if (consuming its result):\n{text}"
    );
}
