// build.rs for triton-sys
//
// Phase 0 (current): no-op. The crate ships the Rust skeleton only so the
// workspace can `cargo check` on any machine without a CUDA / LLVM / cmake
// toolchain.
//
// Phase 1 will:
//   1. cmake-build the vendored Triton at `vendor/triton/` (pinned v3.6.0)
//   2. compile `shim/triton_c.cpp` against Triton's headers
//   3. emit `cargo:rustc-link-lib=triton_c`
//   4. run bindgen on `shim/triton_c.h` to produce `OUT_DIR/bindings.rs`
//
// Gated behind the `compile-triton` feature to keep the default build cheap.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=shim/triton_c.h");
    println!("cargo:rerun-if-changed=shim/triton_c.cpp");

    if std::env::var_os("CARGO_FEATURE_COMPILE_TRITON").is_none() {
        // Skeleton-only build — nothing to compile.
        return;
    }

    // TODO(phase-1): cmake build + bindgen.
    //
    // let dst = cmake::Config::new("vendor/triton")
    //     .define("TRITON_BUILD_PYTHON_MODULE", "OFF")
    //     .define("TRITON_BUILD_TUTORIALS", "OFF")
    //     .define("CMAKE_BUILD_TYPE", "Release")
    //     .build();
    // println!("cargo:rustc-link-search=native={}/lib", dst.display());
    //
    // cc::Build::new()
    //     .cpp(true)
    //     .file("shim/triton_c.cpp")
    //     .include("vendor/triton/include")
    //     .include(format!("{}/include", dst.display()))
    //     .flag("-std=c++17")
    //     .compile("triton_c");
    //
    // let bindings = bindgen::Builder::default()
    //     .header("shim/triton_c.h")
    //     .allowlist_function("triton_.*")
    //     .allowlist_type("Triton.*")
    //     .generate()
    //     .expect("bindgen failed for triton_c.h");
    // bindings
    //     .write_to_file(std::path::Path::new(&std::env::var("OUT_DIR").unwrap()).join("bindings.rs"))
    //     .expect("failed to write bindings");
    panic!("compile-triton feature is not yet implemented (planned for Phase 1)");
}
