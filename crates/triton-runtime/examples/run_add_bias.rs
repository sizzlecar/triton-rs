//! Runner for `add_bias_f32` (2-pointer + 2-i32). 2D row-broadcast bias add.
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
    // v3.6 implicit scratch args — see run_vec_add.rs.
    let global_scratch_size =
        meta["global_scratch_size"].as_u64().unwrap_or(0) as usize;
    let profile_scratch_size =
        meta["profile_scratch_size"].as_u64().unwrap_or(0) as usize;

    const ROWS: usize = 32;
    const COLS: usize = 768; // typical Whisper hidden_size
    let n_total = ROWS * COLS;

    use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
    use cudarc::nvrtc::Ptx;
    let dev = CudaDevice::new(0)?;
    dev.load_ptx(Ptx::from(std::fs::read_to_string(&ptx_path)?), module_name, &[kernel_name])?;
    let func = dev.get_func(module_name, kernel_name).ok_or("kernel missing")?;

    let host_data: Vec<f32> = (0..n_total).map(|i| i as f32 * 0.001).collect();
    let host_bias: Vec<f32> = (0..COLS).map(|i| i as f32 * 0.01).collect();
    let mut dev_data = dev.htod_copy(host_data.clone())?;
    let dev_bias = dev.htod_copy(host_bias.clone())?;

    let scratch: cudarc::driver::CudaSlice<u8> =
        dev.alloc_zeros::<u8>(global_scratch_size.max(1))?;
    let profile_scratch: cudarc::driver::CudaSlice<u8> =
        dev.alloc_zeros::<u8>(profile_scratch_size.max(1))?;
    let cfg = LaunchConfig {
        grid_dim: (ROWS as u32, 1, 1),
        block_dim: (num_warps * 32, 1, 1),
        shared_mem_bytes: shared_mem,
    };
    let rows: i32 = ROWS as i32;
    let cols: i32 = COLS as i32;
    unsafe { func.launch(cfg, (&mut dev_data, &dev_bias, rows, cols, &scratch, &profile_scratch))?; }
    dev.synchronize()?;
    let out = dev.dtoh_sync_copy(&dev_data)?;

    let mut max_err = 0f32;
    for r in 0..ROWS {
        for c in 0..COLS {
            let want = host_data[r * COLS + c] + host_bias[c];
            let got = out[r * COLS + c];
            let err = (got - want).abs();
            if err > max_err { max_err = err; }
        }
    }
    println!("kernel={kernel_name} ROWS={ROWS} COLS={COLS} max_err={max_err}");
    if max_err > 1e-4 { std::process::exit(1); }
    println!("OK row[0][0..4]={:?} row[10][0..4]={:?}",
             &out[..4], &out[10*COLS..10*COLS+4]);
    Ok(())
}
