//! `cargo run --example dump_all -p triton-kernels [-- KERNEL [DTYPE]]`
//!
//! Dumps the MLIR for one or more shipped kernels, useful for piping
//! into `tools/mlir_to_cubin.py` to verify each compiles through Triton
//! end-to-end. Without an argument lists all known kernels.
//!
//! For the dtype-generic decode-attention kernels
//! (`decode_attention_typed`, `decode_attention_hm_typed`,
//! `paged_decode_attention_typed`) you can pass an optional second arg
//! `f32` (default) / `f16` / `bf16` to choose the instantiation.

use triton_kernels::prelude::*;

fn main() {
    let target: Option<String> = std::env::args().nth(1);
    let dtype: String = std::env::args().nth(2).unwrap_or_else(|| "f32".into());

    macro_rules! all_kernels {
        ($($name:literal => $emit:expr),* $(,)?) => {{
            let kernels: &[(&str, fn() -> String)] = &[$(($name, || $emit)),*];
            kernels
        }};
    }

    // Dtype-aware shortcut: for `decode_attention_typed` and friends we
    // dispatch through here so a CLI second-arg can pick the dtype
    // instantiation (matches the per-kernel `ferrum_*` example pattern).
    let dtype_kernel = |name: &str| -> Option<String> {
        match (name, dtype.as_str()) {
            ("decode_attention_typed", "f32") => Some(decode_attention_typed::<f32, 128, 32>::mlir()),
            ("decode_attention_typed", "f16") => Some(decode_attention_typed::<f16, 128, 32>::mlir()),
            ("decode_attention_hm_typed", "f32") => Some(decode_attention_hm_typed::<f32, 128, 32>::mlir()),
            ("decode_attention_hm_typed", "f16") => Some(decode_attention_hm_typed::<f16, 128, 32>::mlir()),
            ("paged_decode_attention_typed", "f32") => Some(paged_decode_attention_typed::<f32, 128, 32>::mlir()),
            ("paged_decode_attention_typed", "f16") => Some(paged_decode_attention_typed::<f16, 128, 32>::mlir()),
            _ => None,
        }
    };

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
        // HEAD_DIM=128 (typical Llama / Qwen size), BLOCK_KV=32 tile.
        // decode_attention_typed / decode_attention_hm_typed /
        // paged_decode_attention_typed are dtype-generic — the listed
        // names below default to f32 (use the CLI second arg to pick
        // f16 / bf16). Listed here so they show up in the kernel index.
        "decode_attention_typed"            => decode_attention_typed::<f32, 128, 32>::mlir(),
        "decode_attention_hm_typed"         => decode_attention_hm_typed::<f32, 128, 32>::mlir(),
        "batched_decode_attention_f32"      => batched_decode_attention_f32::<128, 32>::mlir(),
        "paged_decode_attention_typed"      => paged_decode_attention_typed::<f32, 128, 32>::mlir(),
        // f16 variants — ferrum's production path uses f16. Reduce ops
        // upcast to f32 internally to match Python Triton's accuracy.
        "vec_add_f16"                       => vec_add_f16::<1024>::mlir(),
        "residual_add_f16"                  => residual_add_f16::<1024>::mlir(),
        "residual_add_inplace_f16"          => residual_add_inplace_f16::<1024>::mlir(),
        "add_bias_f16"                      => add_bias_f16::<1024>::mlir(),
        "gelu_f16"                          => gelu_f16::<1024>::mlir(),
        "fused_silu_mul_f16"                => fused_silu_mul_f16::<1024>::mlir(),
        "fused_silu_mul_interleaved_f16"    => fused_silu_mul_interleaved_f16::<1024>::mlir(),
        "rms_norm_f16"                      => rms_norm_f16::<1024>::mlir(),
        "layer_norm_f16"                    => layer_norm_f16::<1024>::mlir(),
        "fused_add_rms_norm_f16"            => fused_add_rms_norm_f16::<1024>::mlir(),
        "softmax_f16"                       => softmax_f16::<1024>::mlir(),
        "embedding_lookup_f16"              => embedding_lookup_f16::<1024>::mlir(),
        // Flash decode (split-K) — 2 coordinated kernels.
        "flash_decode_attn_phase1_f32"      => flash_decode_attn_phase1_f32::<128, 32>::mlir(),
        "flash_decode_attn_phase2_f32"      => flash_decode_attn_phase2_f32::<128, 32>::mlir(),
        "batched_flash_decode_attn_phase1_f32"
                                            => batched_flash_decode_attn_phase1_f32::<128, 32>::mlir(),
        // flash_attn_full IS dtype-generic in source — same body emits
        // f32 or f16 depending on T. Internal compute always runs in f32
        // (Q/K/V upcast on load, downcast via `as_t::<T>` at store) so
        // the f16 instantiation works around NVPTX's lack of native f16
        // division / `math.exp` instructions. BLOCK_Q is 1 here as a
        // dump-only sanity check; real prefill runs use BLOCK_Q ≥ 32.
        "flash_attn_full_f32"               => flash_attn_full::<f32, 128, 1, 32>::mlir(),
        "flash_attn_full_f16"               => flash_attn_full::<f16, 128, 1, 32>::mlir(),
        // Unified prefill + decode with paged KV (vLLM-style). HEAD_DIM=128
        // (Llama / Qwen), BLOCK_Q=16 (works for both decode q_len=1 and
        // small prefill tiles), BLOCK_KV=32. Real decode prefers BLOCK_Q=1.
        "unified_attention_f32"             => unified_attention_f32::<128, 16, 32>::mlir(),
    ];

    match target {
        Some(name) => {
            // Dtype-aware kernels first (their default-table entry is f32;
            // the CLI's optional second arg picks the actual instantiation).
            if let Some(mlir) = dtype_kernel(&name) {
                print!("{}", mlir);
                return;
            }
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
