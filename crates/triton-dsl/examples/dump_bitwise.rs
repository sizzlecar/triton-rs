//! Dump example exercising the new bitwise / shift ops.
//!
//! Pattern: GPTQ-style nibble unpack — `(packed >> shift) & mask`.
//! Cheaper than a real kernel but lets the IR builder + DSL codegen
//! integration get exercised end-to-end without a CUDA box.
//!
//! Run on Mac:
//!   cargo run --example dump_bitwise -p triton-dsl

use triton_dsl::triton_kernel;

#[allow(dead_code)]
struct Ptr<T>(::std::marker::PhantomData<T>);

/// Take an int32 packed word, unpack the i-th nibble (4 bits) as f32 via
/// `((packed >> shift) & 0xF) - 8` then store. Demonstrates:
///   - operator overloading: `>>`, `&`, `-`
///   - logical-right-shift named call `shr_u_i32` (separate from `>>`,
///     which is sign-extending — usually not what you want for unpack)
#[triton_kernel]
fn unpack_nibble_demo<const BLOCK: usize>(
    packed: Ptr<i32>,
    output: Ptr<f32>,
    n: i32,
    shift: i32,
) {
    let pid = program_id(0);
    let off = pid * (BLOCK as i32) + make_range(0, BLOCK as i32);
    let mask = off < n;
    let p = load(packed + off, mask);

    // Logical shift right + AND mask: ((p >> shift) & 0xF) — 4-bit nibble.
    let shift_v = splat_1d(shift, BLOCK as i64);
    let mask_v = splat_1d(const_i32(15), BLOCK as i64);
    let nib = bit_and(shr_u_i32(p, shift_v), mask_v);

    // Subtract zero-point (8 for symmetric INT4) and cast to f32.
    let eight = splat_1d(const_i32(8), BLOCK as i64);
    let centred = nib - eight;
    let f = to_f32(centred);

    store(output + off, f, mask);
}

/// Same logical pattern as `unpack_nibble_demo` but using Rust operator
/// overloads (`>>`, `&`, `^`, `|`, `<<`) instead of named calls.
/// Note: `>>` lowers to `arith.shrsi` (sign-extending). For nibble
/// unpack the trailing `& 0xF` strips the sign bits anyway, so both
/// shapes produce the same f32 result.
#[triton_kernel]
fn unpack_nibble_ops<const BLOCK: usize>(
    packed: Ptr<i32>,
    output: Ptr<f32>,
    n: i32,
    shift: i32,
) {
    let pid = program_id(0);
    let off = pid * (BLOCK as i32) + make_range(0, BLOCK as i32);
    let mask = off < n;
    let p = load(packed + off, mask);

    // Operator-overload form: shifts via `>>`, mask via `&`, subtract via `-`.
    let shift_v = splat_1d(shift, BLOCK as i64);
    let mask_v = splat_1d(const_i32(15), BLOCK as i64);
    let nib = (p >> shift_v) & mask_v;

    let eight = splat_1d(const_i32(8), BLOCK as i64);
    let centred = nib - eight;
    let f = to_f32(centred);

    store(output + off, f, mask);
}

fn main() {
    let mut args = std::env::args();
    args.next();
    match args.next().as_deref() {
        Some("ops") => print!("{}", unpack_nibble_ops::<128>::mlir()),
        _ => print!("{}", unpack_nibble_demo::<128>::mlir()),
    }
}
