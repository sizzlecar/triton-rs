//! Head-to-head benchmark for `rms_norm_f32` — exercises reduce + math
//! intrinsic (rsqrt) on a per-row basis. Different kernels have slightly
//! different launch shapes and arg lists, so we dispatch per-kernel.
//!
//! Same shape as bench_residual_add but with row-block launch + 2 inputs
//! (data + weight) + 1 output and an extra eps scalar.

#[cfg(not(feature = "cuda"))]
fn main() { eprintln!("requires --features cuda"); std::process::exit(1); }

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
    use cudarc::nvrtc::Ptx;
    use std::time::Instant;

    let bench_root = std::env::args()
        .nth(1)
        .ok_or("usage: bench_rms_norm BENCH_ROOT_DIR")?;

    // num_rows × row_size — pick big enough to overflow L2 (64 MB on RTX 5070 Ti)
    // 32K rows × 1024 cols × 4B = 128 MB input, 128 MB output = 256 MB touched.
    const ROWS: usize = 32 * 1024;
    const ROW_SIZE: usize = 1024; // == BLOCK in the Triton path
    const EPS: f32 = 1e-6;

    const TRITON_NUM_WARPS: u32 = 4;
    const FERRUM_THREADS_PER_BLOCK: u32 = 1024; // ferrum's rms_norm uses min(row_size, 1024)

    const ITERS: usize = 100;
    const WARMUP: usize = 10;

    let dev = CudaDevice::new(0)?;

    // input/weight/output: same set across all three so benchmarks compare
    // on identical memory state.
    let host_in: Vec<f32> = (0..ROWS * ROW_SIZE)
        .map(|i| ((i as f32) * 0.0173).sin())
        .collect();
    let host_w: Vec<f32> = (0..ROW_SIZE).map(|i| 1.0 + (i as f32) * 0.001).collect();

    let dev_in = dev.htod_copy(host_in.clone())?;
    let dev_w = dev.htod_copy(host_w.clone())?;
    let mut dev_out: cudarc::driver::CudaSlice<f32> = dev.alloc_zeros::<f32>(ROWS * ROW_SIZE)?;

    let row_size: i32 = ROW_SIZE as i32;
    let eps: f32 = EPS;
    let inv_n: f32 = 1.0 / (ROW_SIZE as f32);

    // bytes_per_launch: read input (RxC), read weight (C), write output (RxC).
    // Ignore the weight read (small) for cleaner GB/s numbers.
    let bytes_per_launch = (ROWS * ROW_SIZE * std::mem::size_of::<f32>() * 2) as f64;

    const PEAK_GBPS: f64 = 896.0;

    println!("rms_norm_f32 head-to-head");
    println!("    ROWS x ROW_SIZE = {} x {} = {:.1} MB input + output (no weight)",
             ROWS, ROW_SIZE, bytes_per_launch / 1e6);
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
        let kernel_name: &'static str = Box::leak(
            meta["name"].as_str().ok_or("name missing")?.to_string().into_boxed_str(),
        );
        let module_name: &'static str = Box::leak(format!("bench_rms_{}", sub).into_boxed_str());
        let compiled_via = meta["compiled_via"]
            .as_str()
            .unwrap_or("triton_mlir")
            .to_string();
        let global_scratch_size = meta["global_scratch_size"].as_u64().unwrap_or(0) as usize;
        let profile_scratch_size = meta["profile_scratch_size"].as_u64().unwrap_or(0) as usize;
        let scratch: cudarc::driver::CudaSlice<u8> =
            dev.alloc_zeros::<u8>(global_scratch_size.max(1))?;
        let profile_scratch: cudarc::driver::CudaSlice<u8> =
            dev.alloc_zeros::<u8>(profile_scratch_size.max(1))?;

        let cfg = match compiled_via.as_str() {
            "nvcc" => LaunchConfig {
                grid_dim: (ROWS as u32, 1, 1),
                block_dim: (FERRUM_THREADS_PER_BLOCK, 1, 1),
                shared_mem_bytes: 0,
            },
            _ => LaunchConfig {
                grid_dim: (ROWS as u32, 1, 1),
                block_dim: (TRITON_NUM_WARPS * 32, 1, 1),
                shared_mem_bytes: meta["shared_mem"].as_u64().unwrap_or(0) as u32,
            },
        };
        let grid_block_str = format!("{}×{}", cfg.grid_dim.0, cfg.block_dim.0);

        dev.load_ptx(ptx, module_name, &[kernel_name])?;
        let func = dev.get_func(module_name, kernel_name).ok_or("kernel not found")?;

        // Per-frontend arg list:
        //   ferrum: (input, weight, output, row_size, eps)
        //   triton-rs DSL + Python: (input, weight, output, row_size, inv_n, eps)
        macro_rules! launch_one {
            () => {
                if compiled_via == "nvcc" {
                    func.clone().launch(cfg, (&dev_in, &dev_w, &mut dev_out, row_size, eps))?
                } else {
                    // v3.6 — append global_scratch + profile_scratch to user args.
                    func.clone().launch(cfg, (&dev_in, &dev_w, &mut dev_out, row_size, inv_n, eps, &scratch, &profile_scratch))?
                }
            };
        }

        for _ in 0..WARMUP {
            unsafe { launch_one!(); }
        }
        dev.synchronize()?;

        let t0 = Instant::now();
        for _ in 0..ITERS {
            unsafe { launch_one!(); }
        }
        dev.synchronize()?;
        let elapsed = t0.elapsed().as_secs_f64();

        let per_call = elapsed / ITERS as f64;
        let bw_gbps = bytes_per_launch / per_call / 1e9;
        let pct_peak = bw_gbps / PEAK_GBPS * 100.0;

        println!("{:<22} {:>10.1} {:>11.1} {:>9.1}% {:>14}",
                 label, per_call * 1e6, bw_gbps, pct_peak, grid_block_str);
    }

    // Sanity check the last-benchmarked kernel.
    let host_out = dev.dtoh_sync_copy(&dev_out)?;
    // Compute host reference for row 0.
    let row = &host_in[..ROW_SIZE];
    let mean_sq: f32 = row.iter().map(|x| x * x).sum::<f32>() * inv_n;
    let inv_rms = (mean_sq + EPS).sqrt().recip();
    let want = row[7] * inv_rms * host_w[7];
    let err = (host_out[7] - want).abs();
    println!();
    println!("sanity row=0 col=7: out={} want={} err={}", host_out[7], want, err);
    if err > 1e-2 {
        eprintln!("WARN: output diverges from host reference!");
        std::process::exit(2);
    }

    Ok(())
}
