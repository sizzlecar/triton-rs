//! `cargo run --example dump_all -p triton-kernels [-- KERNEL]`
//!
//! Dumps the MLIR for one or more shipped kernels, useful for piping
//! into `tools/mlir_to_cubin.py` to verify each compiles through Triton
//! end-to-end. Without an argument lists all known kernels.

use triton_kernels::prelude::*;

fn main() {
    let target: Option<String> = std::env::args().nth(1);

    macro_rules! all_kernels {
        ($($name:literal => $emit:expr),* $(,)?) => {{
            let kernels: &[(&str, fn() -> String)] = &[$(($name, || $emit)),*];
            kernels
        }};
    }

    let kernels = all_kernels![
        "vec_add_f32"                       => vec_add_f32::<1024>::mlir(),
        "residual_add_f32"                  => residual_add_f32::<1024>::mlir(),
        "residual_add_inplace_f32"          => residual_add_inplace_f32::<1024>::mlir(),
        "add_bias_f32"                      => add_bias_f32::<1024>::mlir(),
        "gelu_f32"                          => gelu_f32::<1024>::mlir(),
        "fused_silu_mul_f32"                => fused_silu_mul_f32::<1024>::mlir(),
        "rms_norm_f32"                      => rms_norm_f32::<1024>::mlir(),
        "layer_norm_f32"                    => layer_norm_f32::<1024>::mlir(),
        "fused_add_rms_norm_f32"            => fused_add_rms_norm_f32::<1024>::mlir(),
        "softmax_f32"                       => softmax_f32::<1024>::mlir(),
        "cross_entropy_forward_f32"         => cross_entropy_forward_f32::<1024>::mlir(),
        "embedding_lookup_f32"              => embedding_lookup_f32::<1024>::mlir(),
        "rope_q_f32"                        => rope_q_f32::<64>::mlir(),
        "rope_k_f32"                        => rope_k_f32::<64>::mlir(),
        "rope_full_f32"                     => rope_full_f32::<64>::mlir(),
        "kv_cache_append_f32"               => kv_cache_append_f32::<128>::mlir(),
        "split_qkv_f32"                     => split_qkv_f32::<128>::mlir(),
        "transpose_head_to_token_f32"       => transpose_head_to_token_f32::<128>::mlir(),
    ];

    match target {
        Some(name) => {
            for (n, emit) in kernels {
                if *n == name {
                    print!("{}", emit());
                    return;
                }
            }
            eprintln!("unknown kernel `{name}`. Available:");
            for (n, _) in kernels {
                eprintln!("  - {n}");
            }
            std::process::exit(2);
        }
        None => {
            eprintln!("triton-kernels — {} shipped kernels:", kernels.len());
            for (n, _) in kernels {
                eprintln!("  - {n}");
            }
            eprintln!();
            eprintln!("Run with -- KERNEL_NAME to dump that kernel's MLIR.");
        }
    }
}
