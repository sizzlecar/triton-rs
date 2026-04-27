//! Head-to-head benchmark for `fused_silu_mul_f32` — math-heavy
//! element-wise (each lane does `g / (1 + exp(-g)) * u`).
//!
//! Same shape/launch as bench_residual_add but with the SiLU host reference.

#[cfg(not(feature = "cuda"))]
fn main() { eprintln!("requires --features cuda"); std::process::exit(1); }

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
    use cudarc::nvrtc::Ptx;
    use std::time::Instant;

    let bench_root = std::env::args()
        .nth(1)
        .ok_or("usage: bench_silu_mul BENCH_ROOT_DIR")?;

    const N: usize = 32 * 1024 * 1024;
    const TRITON_BLOCK: u32 = 1024;
    const TRITON_NUM_WARPS: u32 = 4;
    const FERRUM_THREADS_PER_BLOCK: u32 = 256;
    const ITERS: usize = 200;
    const WARMUP: usize = 20;

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    // Inputs spread across a useful SiLU range.
    let host_gate: Vec<f32> = (0..N).map(|i| (i as f32 * 1e-5).sin() * 4.0).collect();
    let host_up:   Vec<f32> = (0..N).map(|i| (i as f32 * 1e-5).cos() * 2.0 + 1.0).collect();
    let dev_gate = stream.clone_htod(&host_gate)?;
    let dev_up   = stream.clone_htod(&host_up)?;
    let mut dev_out: cudarc::driver::CudaSlice<f32> = stream.alloc_zeros::<f32>(N)?;
    let n_arg: i32 = N as i32;

    let bytes_per_launch = (N * std::mem::size_of::<f32>() * 3) as f64;
    const PEAK_GBPS: f64 = 896.0;

    println!("fused_silu_mul_f32 head-to-head");
    println!("    N = {} elements, {:.1} MB / call", N, bytes_per_launch / 1e6);
    println!("    iters = {ITERS} (after {WARMUP} warmup)");
    println!();
    println!("{:<22} {:>10} {:>11} {:>10} {:>14}",
             "source", "us / call", "GB/s eff.", "% peak", "grid×block");

    let runs = [("triton-rs DSL", "dsl"), ("ferrum .cu (nvcc)", "cu"), ("Python @triton.jit", "py")];

    for (label, sub) in &runs {
        let ptx_path = format!("{}/{}/kernel.ptx", bench_root, sub);
        let json_path = format!("{}/{}/kernel.json", bench_root, sub);
        let ptx = Ptx::from(std::fs::read_to_string(&ptx_path)?);
        let meta: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&json_path)?)?;
        let kernel_name = meta["name"].as_str().ok_or("name missing")?.to_string();
        let compiled_via = meta["compiled_via"].as_str().unwrap_or("triton_mlir").to_string();
        let needs_scratch = compiled_via != "nvcc";
        let global_scratch_size = meta["global_scratch_size"].as_u64().unwrap_or(0) as usize;
        let profile_scratch_size = meta["profile_scratch_size"].as_u64().unwrap_or(0) as usize;
        let scratch: cudarc::driver::CudaSlice<u8> =
            stream.alloc_zeros::<u8>(global_scratch_size.max(1))?;
        let profile_scratch: cudarc::driver::CudaSlice<u8> =
            stream.alloc_zeros::<u8>(profile_scratch_size.max(1))?;

        let cfg = match compiled_via.as_str() {
            "nvcc" => LaunchConfig {
                grid_dim: (((N as u32) + FERRUM_THREADS_PER_BLOCK - 1) / FERRUM_THREADS_PER_BLOCK, 1, 1),
                block_dim: (FERRUM_THREADS_PER_BLOCK, 1, 1),
                shared_mem_bytes: 0,
            },
            _ => LaunchConfig {
                grid_dim: (((N as u32) + TRITON_BLOCK - 1) / TRITON_BLOCK, 1, 1),
                block_dim: (TRITON_NUM_WARPS * 32, 1, 1),
                shared_mem_bytes: 0,
            },
        };
        let grid_block_str = format!("{}×{}", cfg.grid_dim.0, cfg.block_dim.0);

        let module = ctx.load_module(ptx)?;
        let func = module.load_function(&kernel_name)?;

        for _ in 0..WARMUP {
            unsafe {
                let mut builder = stream.launch_builder(&func);
                builder
                    .arg(&dev_gate)
                    .arg(&dev_up)
                    .arg(&mut dev_out)
                    .arg(&n_arg);
                if needs_scratch {
                    builder.arg(&scratch).arg(&profile_scratch);
                }
                builder.launch(cfg)?;
            }
        }
        stream.synchronize()?;

        let t0 = Instant::now();
        for _ in 0..ITERS {
            unsafe {
                let mut builder = stream.launch_builder(&func);
                builder
                    .arg(&dev_gate)
                    .arg(&dev_up)
                    .arg(&mut dev_out)
                    .arg(&n_arg);
                if needs_scratch {
                    builder.arg(&scratch).arg(&profile_scratch);
                }
                builder.launch(cfg)?;
            }
        }
        stream.synchronize()?;
        let elapsed = t0.elapsed().as_secs_f64();

        let per_call = elapsed / ITERS as f64;
        let bw_gbps = bytes_per_launch / per_call / 1e9;
        let pct_peak = bw_gbps / PEAK_GBPS * 100.0;

        println!("{:<22} {:>10.1} {:>11.1} {:>9.1}% {:>14}",
                 label, per_call * 1e6, bw_gbps, pct_peak, grid_block_str);
    }

    let host_out = stream.clone_dtoh(&dev_out)?;
    let want = (host_gate[42] / (1.0 + (-host_gate[42]).exp())) * host_up[42];
    let err = (host_out[42] - want).abs();
    println!();
    println!("sanity: out[42] = {} want {} err {}", host_out[42], want, err);
    if err > 1e-4 {
        eprintln!("WARN: last benchmarked kernel diverges from host reference!");
        std::process::exit(2);
    }

    Ok(())
}
