// build.rs for triton-sys
//
// Behavior depends on the `compile-triton` cargo feature.
//
//   default (compile-triton OFF):
//     - No-op. Crate compiles as a Rust skeleton only.
//     - Allows `cargo check --workspace` to succeed on Mac / no-CUDA boxes.
//
//   compile-triton ON (Linux + CUDA only):
//     1. Ensure vendor/triton/ exists by running tools/fetch_vendor.sh
//        (shallow clone of v3.2.0; idempotent).
//     2. Resolve LLVM root: TRITON_LLVM_SYSPATH env var > $HOME/.cache/triton-rs/llvm/<rev>-<suffix>
//        Download Triton's pre-built LLVM tarball if missing.
//     3. cmake-build vendor/triton with the NVIDIA backend only and the
//        TRITON_BUILD_{PYTHON,TUTORIALS,UT,PROTON}=OFF flags.
//     4. Compile shim/triton_c.cpp via cc::Build, linking Triton + LLVM libs.
//     5. Run bindgen on shim/triton_c.h → OUT_DIR/bindings.rs.
//
// Phase 1A landed `tools/fetch_vendor.sh` and the SPIKE doc.
// Phase 1B (this file) wires steps 1-5; the actual `cmake build` of
// Triton will be invoked from a Linux+CUDA machine, not from Mac.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=shim/triton_c.h");
    println!("cargo:rerun-if-changed=shim/triton_c.cpp");

    if std::env::var_os("CARGO_FEATURE_COMPILE_TRITON").is_none() {
        // Skeleton-only build — nothing to compile.
        return;
    }

    #[cfg(feature = "compile-triton")]
    compile_triton::run();
}

#[cfg(feature = "compile-triton")]
mod compile_triton {
    use std::env;
    use std::path::PathBuf;
    use std::process::Command;

    /// Drive Phase 1B's cmake subbuild + Phase 1C's shim + Phase 1D's bindgen.
    pub fn run() {
        // Triton's pre-built LLVM tarballs only ship for Linux (and we
        // need CUDA + ptxas at runtime anyway). Refuse early with a
        // clear message rather than fail mid-download.
        let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
        if target_os != "linux" {
            panic!(
                "triton-sys: feature `compile-triton` is supported on Linux only \
                 (target_os={target_os}). \
                 Use the default feature set (no Triton C++ libs) on macOS / Windows; \
                 dev workflow is to push to a Linux+CUDA box (see ARCHITECTURE.md)."
            );
        }

        let crate_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
        let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
        let vendor_dir = crate_dir.join("vendor/triton");

        ensure_vendor(&crate_dir, &vendor_dir);
        let llvm_root = ensure_llvm(&vendor_dir);
        let triton_build = build_triton(&vendor_dir, &llvm_root, &out_dir);
        compile_shim(&crate_dir, &vendor_dir, &llvm_root, &triton_build);
        run_bindgen(&crate_dir, &out_dir);
    }

    /// Step 1 — invoke fetch_vendor.sh if vendor dir is missing.
    fn ensure_vendor(crate_dir: &PathBuf, vendor_dir: &PathBuf) {
        if vendor_dir.join(".git").exists() {
            return;
        }
        let script = crate_dir.join("tools/fetch_vendor.sh");
        eprintln!("triton-sys: vendor/triton missing — running {}", script.display());
        let status = Command::new("bash")
            .arg(&script)
            .status()
            .expect("failed to spawn fetch_vendor.sh");
        if !status.success() {
            panic!("fetch_vendor.sh failed (exit {:?})", status.code());
        }
    }

    /// Step 2 — locate (or download) Triton's pre-built LLVM tarball.
    /// Returns the LLVM root (containing `lib/`, `include/`, `lib/cmake/mlir/`).
    fn ensure_llvm(vendor_dir: &PathBuf) -> PathBuf {
        if let Ok(p) = env::var("TRITON_LLVM_SYSPATH") {
            return PathBuf::from(p);
        }

        let llvm_rev = std::fs::read_to_string(vendor_dir.join("cmake/llvm-hash.txt"))
            .expect("can't read cmake/llvm-hash.txt — vendor/triton may be incomplete");
        let llvm_rev: String = llvm_rev.trim().chars().take(8).collect();

        let suffix = match (env::var("CARGO_CFG_TARGET_OS").as_deref(), env::var("CARGO_CFG_TARGET_ARCH").as_deref()) {
            (Ok("linux"), Ok("x86_64")) => "ubuntu-x64",
            (Ok("linux"), Ok("aarch64")) => "ubuntu-arm64",
            (Ok("macos"), Ok("aarch64")) => "macos-arm64",
            (os, arch) => panic!("triton-sys: no Triton-LLVM tarball for {:?}-{:?}", os, arch),
        };

        let cache_dir = dirs_cache_dir().join("triton-rs/llvm");
        std::fs::create_dir_all(&cache_dir).unwrap();
        let llvm_dir_name = format!("llvm-{}-{}", llvm_rev, suffix);
        let llvm_dir = cache_dir.join(&llvm_dir_name);

        if llvm_dir.is_dir() {
            return llvm_dir;
        }

        let tarball = cache_dir.join(format!("{}.tar.gz", llvm_dir_name));
        if !tarball.exists() {
            let url = format!(
                "https://oaitriton.blob.core.windows.net/public/llvm-builds/{}.tar.gz",
                llvm_dir_name,
            );
            eprintln!("triton-sys: downloading {}", url);
            let status = Command::new("curl")
                .args(["-sSL", "-o"])
                .arg(&tarball)
                .arg(&url)
                .status()
                .expect("curl not found");
            if !status.success() {
                panic!("LLVM tarball download failed: {}", url);
            }
        }

        eprintln!("triton-sys: extracting {}", tarball.display());
        let status = Command::new("tar")
            .args(["xzf"])
            .arg(&tarball)
            .arg("-C")
            .arg(&cache_dir)
            .status()
            .expect("tar not found");
        if !status.success() {
            panic!("LLVM tarball extraction failed");
        }

        if !llvm_dir.is_dir() {
            panic!(
                "expected {} after extraction; tarball layout differs from assumption",
                llvm_dir.display()
            );
        }
        llvm_dir
    }

    /// Step 3 — drive Triton's cmake. Returns the cmake build root.
    ///
    /// Builds only the library targets we need to link the shim against
    /// (no `bin/` tools — `triton-opt`, `triton-lsp`, etc. are not part
    /// of our ABI surface and have known cross-platform compile issues
    /// in v3.2.0). The list comes from inspecting Triton's own
    /// `add_triton_library` calls; verified by `cmake --build . --target help`.
    fn build_triton(vendor_dir: &PathBuf, llvm_root: &PathBuf, out_dir: &PathBuf) -> PathBuf {
        const LIB_TARGETS: &[&str] = &[
            "TritonIR",
            "TritonAnalysis",
            "TritonTransforms",
            "TritonGPUIR",
            "TritonGPUTransforms",
            "TritonGPUToLLVM",
            "TritonToTritonGPU",
            "TritonLLVMIR",
            "TritonTools",
            "NVGPUIR",
            "NVGPUToLLVM",
            "TritonNvidiaGPUIR",
            "TritonNvidiaGPUTransforms",
            "TritonNVIDIAGPUToLLVM",
        ];

        let mut cfg = cmake::Config::new(vendor_dir);
        cfg.define("TRITON_BUILD_TUTORIALS", "OFF")
            .define("TRITON_BUILD_PYTHON_MODULE", "OFF")
            .define("TRITON_BUILD_PROTON", "OFF")
            .define("TRITON_BUILD_UT", "OFF")
            .define("TRITON_CODEGEN_BACKENDS", "nvidia")
            .define("CMAKE_BUILD_TYPE", "Release")
            .define("LLVM_LIBRARY_DIR", llvm_root.join("lib"))
            .define("LLVM_INCLUDE_DIRS", llvm_root.join("include"))
            .define("MLIR_DIR", llvm_root.join("lib/cmake/mlir"))
            .out_dir(out_dir.join("triton-build"));

        // cmake-rs runs `cmake --build PATH --config Release -- BUILD_ARGS`,
        // so `build_arg` lands at the underlying build tool (make on Linux,
        // not cmake itself). Pass target names directly as positional make
        // args (`make TARGET1 TARGET2 ...`) instead of `--target` which is
        // a cmake flag, not a make flag.
        for t in LIB_TARGETS {
            cfg.build_arg(*t);
        }
        // -k = keep-going: don't bail on first failed target so we get
        //      maximum coverage when bumping Triton versions.
        cfg.build_arg("-k");

        let dst = cfg.build();
        eprintln!("triton-sys: cmake build root = {}", dst.display());
        dst
    }

    /// Step 4 — compile our C++ shim. Phase 1C will fill in the
    /// shim/triton_c.cpp body; this build.rs just wires the cc::Build.
    fn compile_shim(
        crate_dir: &PathBuf,
        vendor_dir: &PathBuf,
        llvm_root: &PathBuf,
        triton_build: &PathBuf,
    ) {
        let mut build = cc::Build::new();
        build
            .cpp(true)
            .file(crate_dir.join("shim/triton_c.cpp"))
            .include(crate_dir.join("shim"))
            .include(vendor_dir.join("include"))
            .include(vendor_dir.join("third_party"))
            .include(triton_build.join("include")) // tablegen'd
            .include(llvm_root.join("include"))
            .flag("-std=c++17")
            .flag("-fPIC")
            .flag("-fno-exceptions") // match LLVM/MLIR convention
            .flag("-fno-rtti");
        build.compile("triton_c");

        // TODO(phase-1B): emit `cargo:rustc-link-search=native={}/lib`
        // for triton_build and llvm_root, plus the long list of
        // `cargo:rustc-link-lib=static=...` for each Triton/MLIR/LLVM
        // archive. Order matters; will be discovered in 1B by reading
        // ${triton_build}/lib/*.a.
    }

    /// Step 5 — bindgen the C header into Rust types.
    fn run_bindgen(crate_dir: &PathBuf, out_dir: &PathBuf) {
        let bindings = bindgen::Builder::default()
            .header(crate_dir.join("shim/triton_c.h").to_string_lossy())
            .allowlist_function("triton_.*")
            .allowlist_type("Triton.*")
            .layout_tests(false)
            .generate()
            .expect("bindgen failed for triton_c.h");
        bindings
            .write_to_file(out_dir.join("bindings.rs"))
            .expect("write bindings.rs");
    }

    fn dirs_cache_dir() -> PathBuf {
        env::var_os("XDG_CACHE_HOME")
            .map(PathBuf::from)
            .or_else(|| env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))
            .expect("no HOME or XDG_CACHE_HOME set")
    }
}
