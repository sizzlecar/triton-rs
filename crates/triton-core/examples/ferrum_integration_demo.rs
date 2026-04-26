//! Worked example for downstream consumers (e.g. ferrum-infer-rs).
//!
//! Shows the full Rust-only workflow: write a kernel with `#[triton_kernel]`,
//! compile to cubin via `Context::compile`, load via cudarc, launch on GPU,
//! verify result. No Python at any stage.
//!
//! Build & run on a Linux + CUDA box:
//!
//! ```sh
//! export TRITON_LLVM_SYSPATH=$HOME/.cache/triton-rs/llvm/llvm-86b69c31-ubuntu-x64
//! export TRITON_LIBDEVICE_PATH=.../vendor/triton/third_party/nvidia/backend/lib/libdevice.10.bc
//! cargo run -p triton-core --features 'cuda compile-triton' \
//!     --release --example ferrum_integration_demo
//! ```
//!
//! Without `compile-triton` (e.g. on macOS): the program prints a friendly
//! "feature not enabled" message and exits 0 — useful to keep CI green on
//! non-CUDA hosts.

use triton_core::triton_kernel;

#[allow(dead_code)]
struct Ptr<T>(::std::marker::PhantomData<T>);

// 1. Author the kernel in Rust. The proc-macro emits both the runtime
//    launcher type AND a `::mlir() -> String` constructor.
#[triton_kernel]
fn vec_add_demo<const BLOCK: usize>(x: Ptr<f32>, y: Ptr<f32>, out: Ptr<f32>, n: i32) {
    let pid = program_id(0);
    let off = pid * (BLOCK as i32) + make_range(0, BLOCK as i32);
    let mask = off < n;
    let xv = load(x + off, mask);
    let yv = load(y + off, mask);
    store(out + off, xv + yv, mask);
}

#[cfg(not(all(feature = "compile-triton", feature = "cuda")))]
fn main() {
    eprintln!(
        "ferrum_integration_demo: needs both `compile-triton` and `cuda` features.\n  \
         cargo run -p triton-core --features 'cuda compile-triton' \\\n  \
             --release --example ferrum_integration_demo"
    );
}

#[cfg(all(feature = "compile-triton", feature = "cuda"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
    use cudarc::nvrtc::Ptx;
    use triton_core::sys::{CompileOptions, Context};

    // 2. Get the kernel's MLIR text. This is generated at compile time by
    //    the proc-macro — no I/O.
    const BLOCK: usize = 1024;
    let mlir = vec_add_demo::<BLOCK>::mlir();

    // 3. Compile via the Rust C ABI shim — same path the bench scripts use.
    //    Returns cubin bytes + PTX text + metadata JSON.
    let ctx = Context::new()?;
    let opts = CompileOptions {
        target_arch: "sm_89".into(),
        num_warps: 4,
        num_stages: 3,
        num_ctas: 1,
    };
    let compiled = ctx.compile(&mlir, &opts)?;
    println!("# kernel compiled: {} bytes cubin, metadata = {}",
             compiled.cubin().len(), compiled.metadata_json());

    // 4. Parse out the kernel name (the proc-macro doesn't promise the
    //    Rust function name = MLIR symbol name, so use the JSON).
    let kernel_name = parse_kernel_name(compiled.metadata_json());
    let kernel_static: &'static str = Box::leak(kernel_name.clone().into_boxed_str());
    let module_static: &'static str = "ferrum_demo";

    // 5. Load the PTX via cudarc 0.13 (which doesn't have load_cubin —
    //    uses the PTX text we got from the shim).
    let dev = CudaDevice::new(0)?;
    dev.load_ptx(Ptx::from(compiled.ptx_text().to_string()), module_static, &[kernel_static])?;
    let func = dev.get_func(module_static, kernel_static).ok_or("kernel not found")?;

    // 6. Allocate buffers + launch.
    const N: usize = 1 << 20; // 1M elements
    let host_x: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let host_y: Vec<f32> = (0..N).map(|i| (i * 2) as f32).collect();
    let dev_x = dev.htod_copy(host_x.clone())?;
    let dev_y = dev.htod_copy(host_y.clone())?;
    let mut dev_out: cudarc::driver::CudaSlice<f32> = dev.alloc_zeros::<f32>(N)?;

    let grid_x = ((N as u32) + (BLOCK as u32) - 1) / (BLOCK as u32);
    let cfg = LaunchConfig {
        grid_dim: (grid_x, 1, 1),
        block_dim: (4 * 32, 1, 1), // num_warps * 32
        shared_mem_bytes: 0,
    };
    unsafe {
        func.launch(cfg, (&dev_x, &dev_y, &mut dev_out, N as i32))?;
    }
    dev.synchronize()?;

    // 7. Verify.
    let host_out = dev.dtoh_sync_copy(&dev_out)?;
    let mut max_err = 0.0f32;
    for i in 0..N {
        let want = host_x[i] + host_y[i];
        max_err = max_err.max((host_out[i] - want).abs());
    }
    println!("# launched on GPU, N={N}, max_err={max_err}");
    if max_err > 0.0 {
        return Err(format!("BAD: max_err={max_err}").into());
    }
    println!("OK — first 5 results: {:?}", &host_out[..5]);
    Ok(())
}

#[cfg(all(feature = "compile-triton", feature = "cuda"))]
fn parse_kernel_name(json: &str) -> String {
    // Quick-and-dirty — the metadata JSON is hand-formatted by the shim,
    // schema is small enough that pulling serde_json in is unnecessary.
    let needle = "\"name\":\"";
    if let Some(start) = json.find(needle) {
        let rest = &json[start + needle.len()..];
        if let Some(end) = rest.find('"') {
            return rest[..end].to_string();
        }
    }
    "vec_add_demo".to_string()
}
