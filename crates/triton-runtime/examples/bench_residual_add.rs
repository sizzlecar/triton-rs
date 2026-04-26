//! Head-to-head benchmark: same `residual_add_f32(a, b, out, n)` kernel
//! compiled through three different paths, timed under per-kernel launch
//! shape on the same GPU. Reports per-kernel mean latency + effective
//! memory bandwidth.
//!
//! Two correctness pieces that the first cut got wrong:
//!   1. ferrum's hand-written .cu is **per-thread-per-element** (1 elt
//!      per thread); the Triton path is **per-block-tile** (BLOCK elts
//!      per block). We dispatch on `kernel.json.compiled_via` to pick the
//!      right grid for each so all three actually write all N elements.
//!   2. Buffers must exceed L2 cache (64 MB on RTX 5070 Ti) or we end up
//!      benchmarking L2 hit bandwidth instead of DRAM. We use N = 32 M
//!      f32 elements = 128 MB per buffer, 384 MB total — well above L2.

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("requires --features cuda");
    std::process::exit(1);
}

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
    use cudarc::nvrtc::Ptx;
    use std::time::Instant;

    let bench_root = std::env::args()
        .nth(1)
        .ok_or("usage: bench_residual_add BENCH_ROOT_DIR")?;

    // 32 M f32 elements per buffer = 128 MB; 384 MB total touched per
    // launch. RTX 5070 Ti L2 is 64 MB — this comfortably overflows it,
    // so we measure DRAM bandwidth, not L2 hit.
    const N: usize = 32 * 1024 * 1024;

    // Triton tile shape (must match how the DSL/Python paths were compiled).
    const TRITON_BLOCK: u32 = 1024;
    const TRITON_NUM_WARPS: u32 = 4;
    // Ferrum's .cu uses 1 element per thread; pick a common 256-thread
    // block to match conventional CUDA launches.
    const FERRUM_THREADS_PER_BLOCK: u32 = 256;

    const ITERS: usize = 200;
    const WARMUP: usize = 20;

    let dev = CudaDevice::new(0)?;

    let host_a: Vec<f32> = (0..N).map(|i| (i as f32).sin()).collect();
    let host_b: Vec<f32> = (0..N).map(|i| (i as f32).cos()).collect();
    let dev_a = dev.htod_copy(host_a.clone())?;
    let dev_b = dev.htod_copy(host_b.clone())?;
    let mut dev_out: cudarc::driver::CudaSlice<f32> = dev.alloc_zeros::<f32>(N)?;
    let n_arg: i32 = N as i32;

    let bytes_per_launch = (N * std::mem::size_of::<f32>() * 3) as f64;

    // Approx effective DRAM peak for RTX 5070 Ti GDDR7 256-bit @ 28 Gbps.
    const PEAK_GBPS: f64 = 896.0;

    println!("residual_add_f32 head-to-head");
    println!("    N = {} elements ({} MB per buffer, {:.1} MB touched per call)",
             N, N * 4 / 1_000_000, bytes_per_launch / 1e6);
    println!("    iters = {ITERS} (after {WARMUP} warmup)");
    println!();
    println!("{:<22} {:>10} {:>11} {:>10} {:>10}",
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
        let module_name: &'static str = Box::leak(format!("bench_{}", sub).into_boxed_str());
        let compiled_via = meta["compiled_via"]
            .as_str()
            .unwrap_or("triton_mlir")
            .to_string();
        // v3.6 implicit scratch args. Triton-compiled kernels (dsl, py)
        // get global_scratch + profile_scratch appended as the last 2
        // launch params; ferrum's hand-CUDA path doesn't.
        let needs_scratch = compiled_via != "nvcc";
        let global_scratch_size =
            meta["global_scratch_size"].as_u64().unwrap_or(0) as usize;
        let profile_scratch_size =
            meta["profile_scratch_size"].as_u64().unwrap_or(0) as usize;
        let scratch: cudarc::driver::CudaSlice<u8> =
            dev.alloc_zeros::<u8>(global_scratch_size.max(1))?;
        let profile_scratch: cudarc::driver::CudaSlice<u8> =
            dev.alloc_zeros::<u8>(profile_scratch_size.max(1))?;

        // Pick the launch shape that actually covers all N elements for
        // this kernel's compute model.
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
        let grid_block_str = format!(
            "{}×{}",
            cfg.grid_dim.0,
            cfg.block_dim.0
        );

        dev.load_ptx(ptx, module_name, &[kernel_name])?;
        let func = dev.get_func(module_name, kernel_name).ok_or("kernel not found")?;

        for _ in 0..WARMUP {
            unsafe {
                if needs_scratch {
                    func.clone().launch(cfg, (&dev_a, &dev_b, &mut dev_out, n_arg, &scratch, &profile_scratch))?;
                } else {
                    func.clone().launch(cfg, (&dev_a, &dev_b, &mut dev_out, n_arg))?;
                }
            }
        }
        dev.synchronize()?;

        let t0 = Instant::now();
        for _ in 0..ITERS {
            unsafe {
                if needs_scratch {
                    func.clone().launch(cfg, (&dev_a, &dev_b, &mut dev_out, n_arg, &scratch, &profile_scratch))?;
                } else {
                    func.clone().launch(cfg, (&dev_a, &dev_b, &mut dev_out, n_arg))?;
                }
            }
        }
        dev.synchronize()?;
        let elapsed = t0.elapsed().as_secs_f64();

        let per_call = elapsed / ITERS as f64;
        let bw_gbps = bytes_per_launch / per_call / 1e9;
        let pct_peak = bw_gbps / PEAK_GBPS * 100.0;

        println!("{:<22} {:>10.1} {:>11.1} {:>9.1}% {:>10}",
                 label, per_call * 1e6, bw_gbps, pct_peak, grid_block_str);
    }

    // Quick sanity check: spot-check the last result against the host
    // reference so we don't accidentally bench a no-op kernel.
    let host_out = dev.dtoh_sync_copy(&dev_out)?;
    let want = host_a[42] + host_b[42];
    let err = (host_out[42] - want).abs();
    println!();
    println!("sanity: out[42] = {} (host want {} err {})", host_out[42], want, err);
    if err > 1e-5 {
        eprintln!("WARN: last benchmarked kernel is NOT producing correct output!");
        std::process::exit(2);
    }

    Ok(())
}
