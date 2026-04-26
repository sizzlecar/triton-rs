//! GPU end-to-end runner for ferrum `gelu` (2-pointer + i32 signature).
//! Sibling of `run_vec_add` for 3-pointer kernels — picks up the same
//! kernel.ptx / kernel.json the Python compiler emits.

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("This example requires `--features cuda` and a working CUDA install.");
    std::process::exit(1);
}

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let ptx_path = args.next().ok_or("usage: run_gelu PTX_PATH METADATA_JSON_PATH")?;
    let meta_path = args.next().ok_or("usage: run_gelu PTX_PATH METADATA_JSON_PATH")?;

    let meta_text = std::fs::read_to_string(&meta_path)?;
    let meta: serde_json::Value = serde_json::from_str(&meta_text)?;
    let kernel_name: &'static str = Box::leak(
        meta["name"]
            .as_str()
            .ok_or("metadata.name missing")?
            .to_string()
            .into_boxed_str(),
    );
    let module_name: &'static str = "triton_kernel";
    let num_warps = meta["num_warps"].as_u64().unwrap_or(4) as u32;
    let shared_mem = meta["shared_mem"].as_u64().unwrap_or(0) as u32;
    // v3.6 implicit scratch args — see run_vec_add.rs.
    let global_scratch_size =
        meta["global_scratch_size"].as_u64().unwrap_or(0) as usize;
    let profile_scratch_size =
        meta["profile_scratch_size"].as_u64().unwrap_or(0) as usize;

    const BLOCK: u32 = 1024;
    const N: usize = 4096;

    use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
    use cudarc::nvrtc::Ptx;

    let dev = CudaDevice::new(0)?;
    let ptx_text = std::fs::read_to_string(&ptx_path)?;
    dev.load_ptx(Ptx::from(ptx_text), module_name, &[kernel_name])?;
    let func = dev.get_func(module_name, kernel_name).ok_or("kernel not found")?;

    // Spread inputs across a useful range for activation testing.
    let host_x: Vec<f32> = (0..N).map(|i| (i as f32 - N as f32 * 0.5) / 256.0).collect();
    let dev_x = dev.htod_copy(host_x.clone())?;
    let mut dev_out: cudarc::driver::CudaSlice<f32> = dev.alloc_zeros::<f32>(N)?;

    let grid_x = ((N as u32) + BLOCK - 1) / BLOCK;
    let scratch: cudarc::driver::CudaSlice<u8> =
        dev.alloc_zeros::<u8>(global_scratch_size.max(1))?;
    let profile_scratch: cudarc::driver::CudaSlice<u8> =
        dev.alloc_zeros::<u8>(profile_scratch_size.max(1))?;
    let cfg = LaunchConfig {
        grid_dim: (grid_x, 1, 1),
        block_dim: (num_warps * 32, 1, 1),
        shared_mem_bytes: shared_mem,
    };

    let n_arg: i32 = N as i32;
    unsafe {
        // 2-pointer + i32 calling convention.
        func.launch(cfg, (&dev_x, &mut dev_out, n_arg, &scratch, &profile_scratch))?;
    }
    dev.synchronize()?;

    let host_out = dev.dtoh_sync_copy(&dev_out)?;

    // GELU reference (PyTorch default, erf-based).
    let gelu = |x: f32| -> f32 {
        0.5 * x * (1.0 + libm::erff(x / std::f32::consts::SQRT_2))
    };

    let mut max_err = 0f32;
    let mut first_bad: Option<(usize, f32, f32)> = None;
    for i in 0..N {
        let expected = gelu(host_x[i]);
        let err = (host_out[i] - expected).abs();
        let tol = (1e-5_f32).max(expected.abs() * 1e-4);
        if err > max_err {
            max_err = err;
        }
        if err > tol && first_bad.is_none() {
            first_bad = Some((i, host_out[i], expected));
        }
    }

    println!("kernel={kernel_name} N={N} BLOCK={BLOCK} num_warps={num_warps} max_err={max_err}");
    if let Some((i, got, want)) = first_bad {
        eprintln!("MISMATCH at i={i} (x={}): got {got}, want {want}", host_x[i]);
        std::process::exit(1);
    }
    println!(
        "OK — first 8 (x, gelu(x)): {:?}",
        host_x.iter().zip(host_out.iter()).take(8).collect::<Vec<_>>()
    );
    Ok(())
}
