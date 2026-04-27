//! Runner for `rms_norm_f32` (3-pointer + i32 + 2 f32). One block per row.
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

    const ROWS: usize = 16;
    const ROW_SIZE: usize = 1024; // typical hidden_size, must be <= BLOCK in kernel
    const EPS: f32 = 1e-6;

    use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
    use cudarc::nvrtc::Ptx;
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = ctx.load_module(Ptx::from(std::fs::read_to_string(&ptx_path)?))?;
    let func = module.load_function(&kernel_name)?;

    let host_in: Vec<f32> = (0..ROWS * ROW_SIZE)
        .map(|i| ((i as f32) * 0.01).sin())
        .collect();
    let host_w: Vec<f32> = (0..ROW_SIZE).map(|i| 1.0 + (i as f32) * 0.001).collect();

    let dev_in = stream.clone_htod(&host_in)?;
    let dev_w  = stream.clone_htod(&host_w)?;
    let mut dev_out: cudarc::driver::CudaSlice<f32> = stream.alloc_zeros::<f32>(ROWS * ROW_SIZE)?;

    let scratch: cudarc::driver::CudaSlice<u8> =
        stream.alloc_zeros::<u8>(global_scratch_size.max(1))?;
    let profile_scratch: cudarc::driver::CudaSlice<u8> =
        stream.alloc_zeros::<u8>(profile_scratch_size.max(1))?;
    let cfg = LaunchConfig {
        grid_dim: (ROWS as u32, 1, 1),
        block_dim: (num_warps * 32, 1, 1),
        shared_mem_bytes: shared_mem,
    };
    let row_size: i32 = ROW_SIZE as i32;
    let inv_n: f32 = 1.0 / (ROW_SIZE as f32);
    let eps: f32 = EPS;
    unsafe {
        let mut builder = stream.launch_builder(&func);
        builder
            .arg(&dev_in)
            .arg(&dev_w)
            .arg(&mut dev_out)
            .arg(&row_size)
            .arg(&inv_n)
            .arg(&eps)
            .arg(&scratch)
            .arg(&profile_scratch);
        builder.launch(cfg)?;
    }
    stream.synchronize()?;
    let out = stream.clone_dtoh(&dev_out)?;

    // Host reference RMS norm.
    let mut max_err = 0f32;
    for r in 0..ROWS {
        let row = &host_in[r * ROW_SIZE..(r + 1) * ROW_SIZE];
        let mean_sq: f32 = row.iter().map(|x| x * x).sum::<f32>() / (ROW_SIZE as f32);
        let inv_rms = 1.0 / (mean_sq + EPS).sqrt();
        for c in 0..ROW_SIZE {
            let want = row[c] * inv_rms * host_w[c];
            let err = (out[r * ROW_SIZE + c] - want).abs();
            let tol = (1e-4_f32).max(want.abs() * 1e-4);
            if err > max_err { max_err = err; }
            if err > tol {
                eprintln!("MISMATCH r={r} c={c}: got {} want {}", out[r * ROW_SIZE + c], want);
                std::process::exit(1);
            }
        }
    }
    println!("kernel={kernel_name} ROWS={ROWS} ROW_SIZE={ROW_SIZE} max_err={max_err}");
    println!("OK row[0][0..4]={:?}", &out[..4]);
    Ok(())
}
