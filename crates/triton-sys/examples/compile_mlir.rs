//! Replacement for `tools/mlir_to_cubin.py` — compile an MLIR file to
//! cubin + metadata JSON via the Rust C ABI shim, no Python.
//!
//! Usage (mirrors the Python script CLI for drop-in replacement):
//!
//!   compile_mlir INPUT.mlir OUTPUT_DIR [--arch sm_89] [--num-warps 4] [--num-stages 3]
//!
//! Outputs (in OUTPUT_DIR):
//!   kernel.cubin    CUDA binary (load via cuModuleLoadData / cudarc)
//!   kernel.json     Metadata: { name, num_warps, num_stages, num_ctas, shared_mem, target_arch }
//!
//! Only built when the `compile-triton` feature is enabled — runs the
//! actual Triton pass pipeline via the linked C++ shim.

#[cfg(not(feature = "compile-triton"))]
fn main() {
    eprintln!(
        "compile_mlir: this example requires `--features compile-triton` \
         (Linux + CUDA only). Build with:\n  \
         cargo run -p triton-sys --features compile-triton --example compile_mlir -- ARGS"
    );
    std::process::exit(2);
}

#[cfg(feature = "compile-triton")]
fn main() -> std::process::ExitCode {
    use std::path::PathBuf;
    use triton_sys::{CompileOptions, Context};

    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        eprintln!(
            "usage: compile_mlir INPUT.mlir OUTPUT_DIR \
             [--arch sm_89] [--num-warps 4] [--num-stages 3]"
        );
        return 2.into();
    }

    let input = PathBuf::from(&args[0]);
    let output_dir = PathBuf::from(&args[1]);

    let mut opts = CompileOptions::default();
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--arch" => {
                opts.target_arch = args
                    .get(i + 1)
                    .cloned()
                    .unwrap_or_else(|| "sm_89".to_string());
                // accept both "89" (int) and "sm_89" forms
                if !opts.target_arch.starts_with("sm_") && opts.target_arch.parse::<u32>().is_ok() {
                    opts.target_arch = format!("sm_{}", opts.target_arch);
                }
                i += 2;
            }
            "--num-warps" => {
                opts.num_warps = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(4);
                i += 2;
            }
            "--num-stages" => {
                opts.num_stages = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(3);
                i += 2;
            }
            "--num-ctas" => {
                opts.num_ctas = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(1);
                i += 2;
            }
            other => {
                eprintln!("unknown arg `{other}`");
                return 2.into();
            }
        }
    }

    let mlir_text = match std::fs::read_to_string(&input) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("read {}: {e}", input.display());
            return 2.into();
        }
    };

    if let Err(e) = std::fs::create_dir_all(&output_dir) {
        eprintln!("mkdir {}: {e}", output_dir.display());
        return 2.into();
    }

    let ctx = match Context::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("triton context create: {e}");
            return 2.into();
        }
    };

    eprintln!(
        "# compiling {} -> {} (arch={}, num_warps={}, num_stages={})",
        input.display(),
        output_dir.display(),
        opts.target_arch,
        opts.num_warps,
        opts.num_stages,
    );

    let compiled = match ctx.compile(&mlir_text, &opts) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("triton compile failed: {e}");
            return 1.into();
        }
    };

    let cubin_path = output_dir.join("kernel.cubin");
    if let Err(e) = std::fs::write(&cubin_path, compiled.cubin()) {
        eprintln!("write {}: {e}", cubin_path.display());
        return 2.into();
    }
    eprintln!("# wrote {} ({} bytes)", cubin_path.display(), compiled.cubin().len());

    if !compiled.ptx_text().is_empty() {
        let ptx_path = output_dir.join("kernel.ptx");
        if let Err(e) = std::fs::write(&ptx_path, compiled.ptx_text()) {
            eprintln!("write {}: {e}", ptx_path.display());
            return 2.into();
        }
        eprintln!("# wrote {} ({} chars)", ptx_path.display(), compiled.ptx_text().len());
    }

    let meta_path = output_dir.join("kernel.json");
    if let Err(e) = std::fs::write(&meta_path, compiled.metadata_json()) {
        eprintln!("write {}: {e}", meta_path.display());
        return 2.into();
    }
    eprintln!("# wrote {}", meta_path.display());
    eprintln!("# metadata: {}", compiled.metadata_json());

    0.into()
}
