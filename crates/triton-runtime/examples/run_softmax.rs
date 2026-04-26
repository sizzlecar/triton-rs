//! Runner for `softmax_f32` (2-ptr + 2-i32). One block per row.
#[cfg(not(feature = "cuda"))]
fn main() { eprintln!("requires --features cuda"); std::process::exit(1); }

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

    const ROWS: usize = 8;
    const COLS: usize = 1024;

    use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
    use cudarc::nvrtc::Ptx;
    let dev = CudaDevice::new(0)?;
    dev.load_ptx(Ptx::from(std::fs::read_to_string(&ptx_path)?), module_name, &[kernel_name])?;
    let func = dev.get_func(module_name, kernel_name).ok_or("kernel missing")?;

    // Vary inputs across rows to stress the per-row reduce.
    let host_in: Vec<f32> = (0..ROWS * COLS)
        .map(|i| (((i % COLS) as f32) - 512.0) * 0.01 * ((i / COLS + 1) as f32))
        .collect();

    let dev_in = dev.htod_copy(host_in.clone())?;
    let mut dev_out: cudarc::driver::CudaSlice<f32> = dev.alloc_zeros::<f32>(ROWS * COLS)?;

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
    unsafe { func.launch(cfg, (&dev_in, &mut dev_out, rows, cols, &scratch, &profile_scratch))?; }
    dev.synchronize()?;
    let out = dev.dtoh_sync_copy(&dev_out)?;

    // Host softmax reference.
    let mut max_err = 0f32;
    for r in 0..ROWS {
        let row = &host_in[r * COLS..(r + 1) * COLS];
        let row_max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = row.iter().map(|&v| (v - row_max).exp()).collect();
        let sum_exp: f32 = exps.iter().sum();
        for c in 0..COLS {
            let want = exps[c] / sum_exp;
            let got = out[r * COLS + c];
            let err = (got - want).abs();
            let tol = (1e-5_f32).max(want.abs() * 1e-4);
            if err > max_err { max_err = err; }
            if err > tol {
                eprintln!("MISMATCH r={r} c={c}: got {} want {}", got, want);
                std::process::exit(1);
            }
        }
        // sanity: row sums to 1.
        let row_sum: f32 = (0..COLS).map(|c| out[r * COLS + c]).sum();
        if (row_sum - 1.0).abs() > 1e-4 {
            eprintln!("row {r} sum != 1: {row_sum}");
            std::process::exit(1);
        }
    }
    println!("kernel={kernel_name} ROWS={ROWS} COLS={COLS} max_err={max_err}");
    println!("OK row[0][:4]={:?}", &out[..4]);
    Ok(())
}
