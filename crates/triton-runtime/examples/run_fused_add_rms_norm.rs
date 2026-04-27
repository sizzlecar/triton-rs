//! Runner for `fused_add_rms_norm_f32` — verifies the fused output matches
//! the host-computed (residual_add → rms_norm → weight) sequence.
#[cfg(not(feature = "cuda"))]
fn main() { eprintln!("requires --features cuda"); std::process::exit(1); }

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
    const HIDDEN: usize = 1024;
    const EPS: f32 = 1e-6;

    use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
    use cudarc::nvrtc::Ptx;
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = ctx.load_module(Ptx::from(std::fs::read_to_string(&ptx_path)?))?;
    let func = module.load_function(&kernel_name)?;

    let host_in: Vec<f32>  = (0..ROWS * HIDDEN).map(|i| ((i as f32) * 0.013).sin()).collect();
    let host_res: Vec<f32> = (0..ROWS * HIDDEN).map(|i| ((i as f32) * 0.027).cos()).collect();
    let host_w: Vec<f32>   = (0..HIDDEN).map(|i| 1.0 + (i as f32) * 0.001).collect();

    let dev_in   = stream.clone_htod(&host_in)?;
    let dev_res  = stream.clone_htod(&host_res)?;
    let dev_w    = stream.clone_htod(&host_w)?;
    let mut dev_out: cudarc::driver::CudaSlice<f32> = stream.alloc_zeros::<f32>(ROWS * HIDDEN)?;
    let mut dev_res_out: cudarc::driver::CudaSlice<f32> = stream.alloc_zeros::<f32>(ROWS * HIDDEN)?;

    let scratch: cudarc::driver::CudaSlice<u8> =
        stream.alloc_zeros::<u8>(global_scratch_size.max(1))?;
    let profile_scratch: cudarc::driver::CudaSlice<u8> =
        stream.alloc_zeros::<u8>(profile_scratch_size.max(1))?;
    let cfg = LaunchConfig {
        grid_dim: (ROWS as u32, 1, 1),
        block_dim: (num_warps * 32, 1, 1),
        shared_mem_bytes: shared_mem,
    };
    let hidden: i32 = HIDDEN as i32;
    let inv_n: f32 = 1.0 / (HIDDEN as f32);
    unsafe {
        let mut builder = stream.launch_builder(&func);
        builder
            .arg(&dev_in)
            .arg(&dev_res)
            .arg(&dev_w)
            .arg(&mut dev_out)
            .arg(&mut dev_res_out)
            .arg(&hidden)
            .arg(&inv_n)
            .arg(&EPS)
            .arg(&scratch)
            .arg(&profile_scratch);
        builder.launch(cfg)?;
    }
    stream.synchronize()?;
    let out = stream.clone_dtoh(&dev_out)?;
    let res_out = stream.clone_dtoh(&dev_res_out)?;

    let mut max_err = 0f32;
    for r in 0..ROWS {
        let in_row = &host_in[r * HIDDEN..(r + 1) * HIDDEN];
        let res_row = &host_res[r * HIDDEN..(r + 1) * HIDDEN];
        let summed: Vec<f32> = in_row.iter().zip(res_row.iter()).map(|(a, b)| a + b).collect();
        let mean_sq: f32 = summed.iter().map(|x| x * x).sum::<f32>() * inv_n;
        let inv_rms = (mean_sq + EPS).sqrt().recip();

        for c in 0..HIDDEN {
            // residual_out
            let rsum_err = (res_out[r * HIDDEN + c] - summed[c]).abs();
            // output
            let want = summed[c] * inv_rms * host_w[c];
            let out_err = (out[r * HIDDEN + c] - want).abs();
            let tol = (2e-3_f32).max(want.abs() * 5e-3);
            if out_err > max_err { max_err = out_err; }
            if rsum_err > 1e-5 || out_err > tol {
                eprintln!("MISMATCH r={r} c={c}: out got {} want {} (rsum_err {})",
                          out[r * HIDDEN + c], want, rsum_err);
                std::process::exit(1);
            }
        }
    }
    println!("kernel={kernel_name} ROWS={ROWS} HIDDEN={HIDDEN} max_err={max_err}");
    println!("OK first 4 of out: {:?}", &out[..4]);
    Ok(())
}
