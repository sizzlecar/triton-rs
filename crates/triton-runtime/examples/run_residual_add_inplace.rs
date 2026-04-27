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
    let kernel_name = meta["name"].as_str().ok_or("name")?.to_string();
    let num_warps = meta["num_warps"].as_u64().unwrap_or(4) as u32;
    let shared_mem = meta["shared_mem"].as_u64().unwrap_or(0) as u32;
    // v3.6 implicit scratch args — see run_vec_add.rs.
    let global_scratch_size =
        meta["global_scratch_size"].as_u64().unwrap_or(0) as usize;
    let profile_scratch_size =
        meta["profile_scratch_size"].as_u64().unwrap_or(0) as usize;

    const BLOCK: u32 = 1024;
    const N: usize = 4096;

    use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
    use cudarc::nvrtc::Ptx;
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = ctx.load_module(Ptx::from(std::fs::read_to_string(&ptx_path)?))?;
    let func = module.load_function(&kernel_name)?;

    let host_a: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let host_b: Vec<f32> = (0..N).map(|i| (i * 2) as f32).collect();
    let mut dev_a = stream.clone_htod(&host_a)?;
    let dev_b = stream.clone_htod(&host_b)?;

    let scratch: cudarc::driver::CudaSlice<u8> =
        stream.alloc_zeros::<u8>(global_scratch_size.max(1))?;
    let profile_scratch: cudarc::driver::CudaSlice<u8> =
        stream.alloc_zeros::<u8>(profile_scratch_size.max(1))?;
    let cfg = LaunchConfig {
        grid_dim: (((N as u32) + BLOCK - 1) / BLOCK, 1, 1),
        block_dim: (num_warps * 32, 1, 1),
        shared_mem_bytes: shared_mem,
    };
    let n_arg: i32 = N as i32;
    unsafe {
        let mut builder = stream.launch_builder(&func);
        builder
            .arg(&mut dev_a)
            .arg(&dev_b)
            .arg(&n_arg)
            .arg(&scratch)
            .arg(&profile_scratch);
        builder.launch(cfg)?;
    }
    stream.synchronize()?;
    let out = stream.clone_dtoh(&dev_a)?;

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
