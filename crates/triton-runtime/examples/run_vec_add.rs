//! End-to-end Phase-1-lite demo: load a Triton-compiled PTX, run our
//! canonical `vec_add` kernel on the GPU, and verify the result matches
//! the host-computed reference.
//!
//! Workflow on the GPU machine:
//!
//! ```text
//! # 1. Generate the kernel MLIR from the DSL.
//! cargo run --example dump_vec_add_generic -p triton-dsl --quiet \
//!     > /tmp/vec_add.mlir
//!
//! # 2. Compile MLIR -> cubin + ptx via Triton's Python pipeline.
//! python3 tools/mlir_to_cubin.py /tmp/vec_add.mlir /tmp/vec_add_out --arch 89
//!
//! # 3. Run on GPU.
//! cargo run --example run_vec_add -p triton-runtime --features cuda --release -- \
//!     /tmp/vec_add_out/kernel.ptx /tmp/vec_add_out/kernel.json
//! ```
//!
//! Only compiled when the `cuda` feature is enabled — the cudarc dep
//! requires the CUDA toolkit at build time.

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("This example requires `--features cuda` and a working CUDA install.");
    std::process::exit(1);
}

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let ptx_path = args
        .next()
        .ok_or("usage: run_vec_add PTX_PATH METADATA_JSON_PATH [--op add|silu_mul|mul]")?;
    let meta_path = args
        .next()
        .ok_or("usage: run_vec_add PTX_PATH METADATA_JSON_PATH [--op add|silu_mul|mul]")?;
    let op_kind = match args.next().as_deref() {
        None => "add".to_string(),
        Some("--op") => args.next().unwrap_or_else(|| "add".to_string()),
        Some(other) => return Err(format!("unexpected positional arg `{other}`").into()),
    };

    // 1. Load the metadata so we know the kernel name and launch shape.
    let meta_text = std::fs::read_to_string(&meta_path)?;
    let meta: serde_json::Value = serde_json::from_str(&meta_text)?;
    let kernel_name = meta["name"].as_str().ok_or("metadata.name missing")?.to_string();

    let num_warps = meta["num_warps"].as_u64().unwrap_or(4) as u32;
    let shared_mem = meta["shared_mem"].as_u64().unwrap_or(0) as u32;

    // v3.6 implicit kernel args: the lowering pipeline appends two extra
    // device-pointer slots (global_scratch + profile_scratch) after the
    // user-defined args. Sizes come from the metadata; for simple
    // element-wise kernels both are usually 0, but the pointer slots are
    // still mandatory or the launch SIGSEGVs at arg-marshalling time.
    let global_scratch_size =
        meta["global_scratch_size"].as_u64().unwrap_or(0) as usize;
    let profile_scratch_size =
        meta["profile_scratch_size"].as_u64().unwrap_or(0) as usize;

    // 2. Tile size is hard-coded to 1024 to match `dump_vec_add_generic`.
    const BLOCK: u32 = 1024;
    const N: usize = 4096; // total elements; gives a 4-block grid

    // 3. Initialise CUDA + load the PTX.
    use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
    use cudarc::nvrtc::Ptx;

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    let ptx_text = std::fs::read_to_string(&ptx_path)?;
    let ptx = Ptx::from(ptx_text);
    let module = ctx.load_module(ptx)?;
    let func = module.load_function(&kernel_name)?;

    // 4. Allocate + populate device buffers.
    let host_x: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let host_y: Vec<f32> = (0..N).map(|i| (i * 2) as f32).collect();

    let dev_x = stream.clone_htod(&host_x)?;
    let dev_y = stream.clone_htod(&host_y)?;
    let mut dev_out: cudarc::driver::CudaSlice<f32> = stream.alloc_zeros::<f32>(N)?;

    // v3.6 implicit-arg buffers. Allocate at least 1 byte even when the
    // metadata reports size 0 — cudarc's auto-marshaller needs a real
    // CudaSlice to produce a CUdeviceptr; the kernel just won't read it
    // when the lowering pass set the size to 0.
    let scratch: cudarc::driver::CudaSlice<u8> =
        stream.alloc_zeros::<u8>(global_scratch_size.max(1))?;
    let profile_scratch: cudarc::driver::CudaSlice<u8> =
        stream.alloc_zeros::<u8>(profile_scratch_size.max(1))?;

    // 5. Launch — Triton kernels expect (num_warps * 32) threads per block,
    //    and one block per BLOCK-element tile.
    let grid_x = ((N as u32) + BLOCK - 1) / BLOCK;
    let cfg = LaunchConfig {
        grid_dim: (grid_x, 1, 1),
        block_dim: (num_warps * 32, 1, 1),
        shared_mem_bytes: shared_mem,
    };

    let n_arg: i32 = N as i32;
    unsafe {
        // v3.6 calling convention: 3 user pointers + 1 i32 + global_scratch
        // device-ptr + profile_scratch device-ptr (the lowering pipeline
        // appends both unconditionally; v3.2 didn't).
        let mut builder = stream.launch_builder(&func);
        builder
            .arg(&dev_x)
            .arg(&dev_y)
            .arg(&mut dev_out)
            .arg(&n_arg)
            .arg(&scratch)
            .arg(&profile_scratch);
        builder.launch(cfg)?;
    }

    stream.synchronize()?;

    // 6. Copy back and verify against the host reference.
    let host_out = stream.clone_dtoh(&dev_out)?;

    let reference = |a: f32, b: f32| -> f32 {
        match op_kind.as_str() {
            "add" => a + b,
            "mul" => a * b,
            // SiLU(a) * b where SiLU(x) = x / (1 + exp(-x)).
            "silu_mul" => (a / (1.0 + (-a).exp())) * b,
            other => panic!("unknown --op `{other}` (supported: add, mul, silu_mul)"),
        }
    };

    let mut max_err = 0f32;
    let mut first_bad: Option<(usize, f32, f32)> = None;
    for i in 0..N {
        let expected = reference(host_x[i], host_y[i]);
        let err = (host_out[i] - expected).abs();
        // Allow a small relative tolerance for nonlinear ops (exp/log/etc.
        // round-tripping through float32). 1e-4 covers typical SiLU error.
        let tol = (1e-5_f32).max(expected.abs() * 1e-5);
        if err > max_err {
            max_err = err;
        }
        if err > tol && first_bad.is_none() {
            first_bad = Some((i, host_out[i], expected));
        }
    }

    println!(
        "kernel={kernel_name} op={op_kind} N={N} BLOCK={BLOCK} num_warps={num_warps} max_err={max_err}"
    );
    if let Some((i, got, want)) = first_bad {
        eprintln!("MISMATCH at i={i}: got {got}, want {want}");
        std::process::exit(1);
    }
    println!("OK — first 8 results: {:?}", &host_out[..8]);
    Ok(())
}
