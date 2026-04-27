//! Dump a DSL kernel using statement-form `if cond { ... }` (no else)
//! to stdout. Demonstrates `scf.if` plumbing through the proc-macro:
//! the load + store land inside the scf.if then-region, so program_ids
//! whose pid >= threshold do nothing.
//!
//! Round-trip-validate against Triton 3.2.0 with `tools/validate_mlir.py`.

use triton_dsl::triton_kernel;

#[allow(dead_code)]
struct Ptr<T>(::std::marker::PhantomData<T>);

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

fn main() {
    print!("{}", early_return_demo::<1024>::mlir());
}
