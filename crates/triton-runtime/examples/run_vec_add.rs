//! End-to-end Phase-1-lite demo: load a Triton-compiled cubin, run our
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
//! # 2. Compile MLIR -> cubin via Triton's Python pipeline.
//! python3 tools/mlir_to_cubin.py /tmp/vec_add.mlir /tmp/vec_add_out --arch 89
//!
//! # 3. Run on GPU.
//! cargo run --example run_vec_add -p triton-runtime --features cuda --release -- \
//!     /tmp/vec_add_out/kernel.cubin /tmp/vec_add_out/kernel.json
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
    use std::sync::Arc;

    let mut args = std::env::args().skip(1);
    let cubin_path = args
        .next()
        .ok_or("usage: run_vec_add CUBIN_PATH METADATA_JSON_PATH")?;
    let meta_path = args
        .next()
        .ok_or("usage: run_vec_add CUBIN_PATH METADATA_JSON_PATH")?;

    // 1. Load the metadata so we know the kernel name and launch shape.
    let meta_text = std::fs::read_to_string(&meta_path)?;
    let meta: serde_json::Value = serde_json::from_str(&meta_text)?;
    let kernel_name = meta["name"].as_str().ok_or("metadata.name missing")?;
    let num_warps = meta["num_warps"].as_u64().unwrap_or(4) as u32;
    let shared_mem = meta["shared_mem"].as_u64().unwrap_or(0) as u32;

    // 2. Tile size is hard-coded to 1024 to match `dump_vec_add_generic`.
    //    Adjust this if compiling a different BLOCK.
    const BLOCK: u32 = 1024;
    const N: usize = 4096; // total elements; gives a 4-block grid

    // 3. Initialise CUDA + load the cubin.
    let ctx = cudarc::driver::CudaContext::new(0)?;
    let stream = ctx.default_stream();

    let cubin_bytes = std::fs::read(&cubin_path)?;
    let module = ctx.load_module(cubin_bytes.into())?;
    let func = module.load_function(kernel_name)?;

    // 4. Allocate + populate device buffers.
    let host_x: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let host_y: Vec<f32> = (0..N).map(|i| (i * 2) as f32).collect();

    let dev_x = stream.memcpy_stod(&host_x)?;
    let dev_y = stream.memcpy_stod(&host_y)?;
    let mut dev_out: cudarc::driver::CudaSlice<f32> = stream.alloc_zeros(N)?;

    // 5. Launch — Triton kernels expect (num_warps * 32) threads per block,
    //    and one block per BLOCK-element tile.
    let grid_x = ((N as u32) + BLOCK - 1) / BLOCK;
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (grid_x, 1, 1),
        block_dim: (num_warps * 32, 1, 1),
        shared_mem_bytes: shared_mem,
    };

    let n_arg: i32 = N as i32;
    unsafe {
        // Triton's calling convention for our vec_add: 3 pointers + 1 i32.
        // cudarc auto-marshals CudaSlice<T> as a device pointer.
        stream.launch_builder(&func)
            .arg(&dev_x)
            .arg(&dev_y)
            .arg(&mut dev_out)
            .arg(&n_arg)
            .launch(cfg)?;
    }

    stream.synchronize()?;

    // 6. Copy back and verify against the host reference.
    let mut host_out = vec![0f32; N];
    stream.memcpy_dtoh(&dev_out, &mut host_out)?;

    let mut max_err = 0f32;
    let mut first_bad: Option<(usize, f32, f32)> = None;
    for i in 0..N {
        let expected = host_x[i] + host_y[i];
        let err = (host_out[i] - expected).abs();
        if err > max_err {
            max_err = err;
        }
        if err > 1e-5 && first_bad.is_none() {
            first_bad = Some((i, host_out[i], expected));
        }
    }

    println!(
        "vec_add: N={N} BLOCK={BLOCK} num_warps={num_warps} max_err={max_err}"
    );
    if let Some((i, got, want)) = first_bad {
        eprintln!("MISMATCH at i={i}: got {got}, want {want}");
        std::process::exit(1);
    }
    println!("OK — first 8 results: {:?}", &host_out[..8]);
    drop(stream);
    drop(ctx);
    let _ = Arc::<()>::default;
    Ok(())
}
