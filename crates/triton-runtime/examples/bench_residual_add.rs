//! Head-to-head benchmark: same `residual_add_f32(a, b, out, n)` kernel
//! compiled through three different paths, timed under identical launch
//! conditions on the same GPU. Reports per-kernel mean latency + effective
//! memory bandwidth.
//!
//! Build the three PTXes externally before running this binary:
//!
//! ```text
//! # 1. triton-rs DSL  →  MLIR  →  Triton  →  PTX
//! cargo run --example ferrum_residual_add -p triton-dsl --quiet -- f32 \\
//!     > /tmp/bench/dsl/kernel.mlir
//! python3 tools/mlir_to_cubin.py /tmp/bench/dsl/kernel.mlir /tmp/bench/dsl
//!
//! # 2. ferrum hand-written .cu  →  nvcc  →  PTX
//! KERNEL_NAME=residual_add_f32 bash tools/compile_cu.sh \\
//!     ../ferrum-infer-rs/crates/ferrum-kernels/kernels/residual_add.cu \\
//!     /tmp/bench/cu
//!
//! # 3. Python @triton.jit  →  Triton  →  PTX (same backend as #1)
//! python3 tools/compile_python_kernel.py \\
//!     tools/python_kernels/residual_add.py residual_add_f32 \\
//!     /tmp/bench/py \\
//!     --signature 'a_ptr=*fp32,b_ptr=*fp32,out_ptr=*fp32,n=i32' \\
//!     --constants 'BLOCK=1024'
//!
//! # 4. Run.
//! cargo run --example bench_residual_add -p triton-runtime \\
//!     --features cuda --release -- /tmp/bench
//! ```

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

    // Element count chosen large enough that launch overhead < kernel time.
    // 16 MiB / kernel = a, b, out = 48 MiB total = well above L2 cache.
    const N: usize = 4 * 1024 * 1024;
    const BLOCK: u32 = 1024; // tile width; must match how each kernel was compiled
    const NUM_WARPS: u32 = 4;
    const ITERS: usize = 200;
    const WARMUP: usize = 20;

    let dev = CudaDevice::new(0)?;

    // Allocate ONE set of buffers and reuse — keep memory state identical
    // across kernels.
    let host_a: Vec<f32> = (0..N).map(|i| (i as f32).sin()).collect();
    let host_b: Vec<f32> = (0..N).map(|i| (i as f32).cos()).collect();
    let dev_a = dev.htod_copy(host_a.clone())?;
    let dev_b = dev.htod_copy(host_b.clone())?;
    let mut dev_out: cudarc::driver::CudaSlice<f32> = dev.alloc_zeros::<f32>(N)?;
    let n_arg: i32 = N as i32;

    let cfg = LaunchConfig {
        grid_dim: (((N as u32) + BLOCK - 1) / BLOCK, 1, 1),
        block_dim: (NUM_WARPS * 32, 1, 1),
        shared_mem_bytes: 0,
    };

    let runs = [("triton-rs DSL", "dsl"), ("ferrum .cu (nvcc)", "cu"), ("Python @triton.jit", "py")];

    let bytes_per_launch = (N * std::mem::size_of::<f32>() * 3) as f64; // a + b + out

    println!("residual_add_f32 head-to-head — N={N}, BLOCK={BLOCK}, num_warps={NUM_WARPS}");
    println!("    bytes_per_launch = {:.2} MB", bytes_per_launch / 1e6);
    println!("    iters = {ITERS} (after {WARMUP} warmup)");
    println!();
    println!("{:<22} {:>10} {:>14} {:>14} {:>10}",
             "source", "us / call", "GB/s eff.", "GB/s peak%", "kernel");

    // RTX 5070 Ti theoretical peak: ~896 GB/s (rough estimate for 50-series)
    const PEAK_GBPS: f64 = 896.0;

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

        dev.load_ptx(ptx, module_name, &[kernel_name])?;
        let func = dev.get_func(module_name, kernel_name).ok_or("kernel not found")?;

        // Warmup.
        for _ in 0..WARMUP {
            unsafe { func.launch(cfg, (&dev_a, &dev_b, &mut dev_out, n_arg))?; }
        }
        dev.synchronize()?;

        // Time.
        let t0 = Instant::now();
        for _ in 0..ITERS {
            unsafe { func.launch(cfg, (&dev_a, &dev_b, &mut dev_out, n_arg))?; }
        }
        dev.synchronize()?;
        let elapsed = t0.elapsed().as_secs_f64();

        let per_call = elapsed / ITERS as f64;
        let bw_gbps = bytes_per_launch / per_call / 1e9;
        let pct_peak = bw_gbps / PEAK_GBPS * 100.0;

        println!("{:<22} {:>10.2} {:>14.1} {:>13.1}% {:>10}",
                 label, per_call * 1e6, bw_gbps, pct_peak, kernel_name);
    }

    Ok(())
}
