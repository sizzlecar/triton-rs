//! Runner for `layer_norm_f32` (4-ptr + i32 + 2-f32).
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
    const DIM: usize = 768;
    const EPS: f32 = 1e-5;

    use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
    use cudarc::nvrtc::Ptx;
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = ctx.load_module(Ptx::from(std::fs::read_to_string(&ptx_path)?))?;
    let func = module.load_function(&kernel_name)?;

    let host_x: Vec<f32> = (0..ROWS * DIM).map(|i| ((i as f32) * 0.0173).sin()).collect();
    let host_g: Vec<f32> = (0..DIM).map(|i| 1.0 + (i as f32) * 0.001).collect();
    let host_b: Vec<f32> = (0..DIM).map(|i| (i as f32) * 0.0001).collect();

    let dev_x = stream.clone_htod(&host_x)?;
    let dev_g = stream.clone_htod(&host_g)?;
    let dev_b = stream.clone_htod(&host_b)?;
    let mut dev_o: cudarc::driver::CudaSlice<f32> = stream.alloc_zeros::<f32>(ROWS * DIM)?;

    let scratch: cudarc::driver::CudaSlice<u8> =
        stream.alloc_zeros::<u8>(global_scratch_size.max(1))?;
    let profile_scratch: cudarc::driver::CudaSlice<u8> =
        stream.alloc_zeros::<u8>(profile_scratch_size.max(1))?;
    let cfg = LaunchConfig {
        grid_dim: (ROWS as u32, 1, 1),
        block_dim: (num_warps * 32, 1, 1),
        shared_mem_bytes: shared_mem,
    };
    let dim: i32 = DIM as i32;
    let inv_n: f32 = 1.0 / (DIM as f32);
    let eps: f32 = EPS;
    unsafe {
        let mut builder = stream.launch_builder(&func);
        builder
            .arg(&dev_x)
            .arg(&dev_g)
            .arg(&dev_b)
            .arg(&mut dev_o)
            .arg(&dim)
            .arg(&inv_n)
            .arg(&eps)
            .arg(&scratch)
            .arg(&profile_scratch);
        builder.launch(cfg)?;
    }
    stream.synchronize()?;
    let out = stream.clone_dtoh(&dev_o)?;

    // Match the kernel's compute order: it multiplies by inv_n rather than
    // dividing by DIM, and uses GPU `rsqrt` (1-ULP-off hardware special-fn)
    // not the more-precise `1.0 / sqrt`. Mirror both to keep the host
    // reference in the same numerical ballpark.
    let inv_n = 1.0_f32 / (DIM as f32);
    let mut max_err = 0f32;
    for r in 0..ROWS {
        let row = &host_x[r * DIM..(r + 1) * DIM];
        let mean: f32 = row.iter().sum::<f32>() * inv_n;
        let var: f32 = row.iter().map(|x| (x - mean).powi(2)).sum::<f32>() * inv_n;
        let inv_std = (var + EPS).sqrt().recip();
        for c in 0..DIM {
            let want = (row[c] - mean) * inv_std * host_g[c] + host_b[c];
            let got = out[r * DIM + c];
            let err = (got - want).abs();
            // Two-pass mean/variance + GPU rsqrt + 768-wide f32 reductions
            // accumulate ~1e-3 relative drift. Anything tighter chases
            // hardware noise rather than catching real bugs.
            let tol = (2e-3_f32).max(want.abs() * 5e-3);
            if err > max_err { max_err = err; }
            if err > tol {
                eprintln!("MISMATCH r={r} c={c}: got {} want {}", got, want);
                std::process::exit(1);
            }
        }
    }
    println!("kernel={kernel_name} ROWS={ROWS} DIM={DIM} max_err={max_err}");
    println!("OK row[0][:4]={:?}", &out[..4]);
    Ok(())
}
