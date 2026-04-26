//! Runner for `residual_add_inplace_f32` (2-pointer + i32). `a += b`.
#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("requires --features cuda");
    std::process::exit(1);
}

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let ptx_path = args.next().ok_or("usage: PTX META")?;
    let meta_path = args.next().ok_or("usage: PTX META")?;
    let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&meta_path)?)?;
    let kernel_name: &'static str =
        Box::leak(meta["name"].as_str().ok_or("name")?.to_string().into_boxed_str());
    let module_name: &'static str = "triton_kernel";
    let num_warps = meta["num_warps"].as_u64().unwrap_or(4) as u32;
    let shared_mem = meta["shared_mem"].as_u64().unwrap_or(0) as u32;

    const BLOCK: u32 = 1024;
    const N: usize = 4096;

    use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
    use cudarc::nvrtc::Ptx;
    let dev = CudaDevice::new(0)?;
    dev.load_ptx(Ptx::from(std::fs::read_to_string(&ptx_path)?), module_name, &[kernel_name])?;
    let func = dev.get_func(module_name, kernel_name).ok_or("kernel missing")?;

    let host_a: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let host_b: Vec<f32> = (0..N).map(|i| (i * 2) as f32).collect();
    let mut dev_a = dev.htod_copy(host_a.clone())?;
    let dev_b = dev.htod_copy(host_b.clone())?;

    let cfg = LaunchConfig {
        grid_dim: (((N as u32) + BLOCK - 1) / BLOCK, 1, 1),
        block_dim: (num_warps * 32, 1, 1),
        shared_mem_bytes: shared_mem,
    };
    let n_arg: i32 = N as i32;
    unsafe { func.launch(cfg, (&mut dev_a, &dev_b, n_arg))?; }
    dev.synchronize()?;
    let out = dev.dtoh_sync_copy(&dev_a)?;

    let mut max_err = 0f32;
    for i in 0..N {
        let want = host_a[i] + host_b[i];
        let err = (out[i] - want).abs();
        if err > max_err { max_err = err; }
    }
    println!("kernel={kernel_name} N={N} max_err={max_err}");
    if max_err > 1e-5 { std::process::exit(1); }
    println!("OK first 8: {:?}", &out[..8]);
    Ok(())
}
