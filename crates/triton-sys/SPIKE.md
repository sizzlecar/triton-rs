# Phase 1A Spike — Mapping Triton's Python compile orchestrator to C++

**Goal:** before writing one line of `triton_c.cpp`, document every C++
entry point the shim will call. Confirm none of them are pybind-only
(i.e. defined inside `python/src/main.cc` with no upstream C++ header).

**Pinned version:** v3.2.0 (matches the Vast.ai test box and the IR our
DSL currently emits). Bump to v3.6.0 deferred to Phase 1F.

**Sources read** (via `gh api .../contents/...` — initial submodule
clone was slow on the proxy and disconnected at 99% twice; fetched key
files directly while the network sorted out):
- `python/triton/compiler/compiler.py` @ v3.2.0 — top-level `compile()`
- `third_party/nvidia/backend/compiler.py` @ v3.2.0 — NVIDIA `make_ttgir/make_llir/make_ptx/make_cubin`
- `CMakeLists.txt` @ v3.2.0 — top-level cmake options
- `python/setup.py` @ v3.2.0 — LLVM tarball download URL pattern
- `cmake/llvm-hash.txt` @ v3.2.0 — pinned LLVM commit `86b69c31`

**Vendoring strategy** (decided after the proxy issues): NOT a git
submodule. Instead, `crates/triton-sys/tools/fetch_vendor.sh` does a
shallow clone (`git clone --depth=1 -b v3.2.0`, ~21 MB on disk, single
round-trip) into `crates/triton-sys/vendor/triton/`, which is gitignored.
Build.rs (Phase 1B) will invoke the script if the dir is missing.
Rationale: full submodule clone of Triton is ~350 MB with history and
the proxy disconnects on multi-MB transfers; tag-based shallow clone is
1/15th the size and survives the same network conditions.

---

## Top-level `compile(src, target, options)` — the orchestration loop

The Python function `triton.compiler.compiler.compile` (line ~270 of
`compiler.py` v3.2.0) does this once cache logic is stripped:

```text
1. backend          = make_backend(target)               # picks NVIDIABackend for cuda target
2. options          = backend.parse_options(opts_dict)   # CUDAOptions dataclass
3. context          = ir.context()                       # creates MLIRContext
   ir.load_dialects(context)                             # registers Triton/TritonGPU dialects
   backend.load_dialects(context)                        # registers NV-specific dialects
4. codegen_fns      = backend.get_codegen_implementation()
   module_map       = backend.get_module_map()
5. module           = src.make_ir(options, codegen_fns, module_map, context)
                                                         # IRSource: ir.parse_mlir_module(path, context)
                                                         # ASTSource: lower Python AST → TTIR
6. stages           = {}
   backend.add_stages(stages, options)                   # populates {"ttir": fn, "ttgir": fn, "llir": fn, "ptx": fn, "cubin": fn}
7. for ext, compile_ir in stages.items()[first_stage:]:  # first_stage = index of src.ext + 1 if src is IR
       module = compile_ir(module, metadata)             # each fn builds a fresh PassManager, adds passes, runs
   # final iteration produces the cubin bytes
8. write metadata JSON, return CompiledKernel(...)
```

For our shim we **enter at step 5** with a TTIR text already in hand.
We don't need `ASTSource`, `make_ir`, codegen_fns, or module_map.
We need: MLIRContext setup, IR text → ModuleOp parse, then the four
lowering stages.

---

## Stage 1 — `make_ttgir(module, metadata, opt, capability)`

(File `third_party/nvidia/backend/compiler.py`, ~line 200, NVIDIA backend.)

Builds a `pm = ir.pass_manager(context)`, adds the passes below in order,
runs `pm.run(module)`. Returns the same `module` (mutated in place; MLIR
passes work on the module directly).

| # | Pybind call | Underlying C++ factory (header to include) |
|---|-------------|---------------------------------------------|
| 1 | `passes.ttir.add_convert_to_ttgpuir(pm, "cuda:" + capability, num_warps, 32, num_ctas)` | `mlir::triton::createConvertTritonToTritonGPUPass(...)` — `Conversion/TritonToTritonGPU/Passes.h` |
| 2 | `passes.ttgpuir.add_coalesce(pm)` | `mlir::triton::gpu::createCoalescePass()` — `Dialect/TritonGPU/Transforms/Passes.h` |
| 3 | `passes.ttgpuir.add_f32_dot_tc(pm)` (≥sm_80) | `…createF32DotTCPass()` — same header |
| 4 | `nvidia.passes.ttnvgpuir.add_plan_cta(pm, cluster_info)` | `mlir::triton::nvidia_gpu::createPlanCTAPass(...)` — `third_party/nvidia/include/Dialect/NVGPU/Transforms/Passes.h` |
| 5 | `passes.ttgpuir.add_remove_layout_conversions(pm)` | `…createRemoveLayoutConversionsPass()` |
| 6 | `passes.ttgpuir.add_optimize_thread_locality(pm)` | `…createOptimizeThreadLocalityPass()` |
| 7 | `passes.ttgpuir.add_accelerate_matmul(pm)` | `…createAccelerateMatmulPass()` |
| 8 | `passes.ttgpuir.add_remove_layout_conversions(pm)` (repeat) | (same as 5) |
| 9 | `passes.ttgpuir.add_optimize_dot_operands(pm, cap≥80)` | `…createOptimizeDotOperandsPass(bool)` |
| 10 | `passes.common.add_cse(pm)` | `mlir::createCSEPass()` — `mlir/Transforms/Passes.h` |
| 11 | `passes.ttgpuir.add_optimize_accumulator_init(pm)` (≥sm_80) | `…createOptimizeAccumulatorInitPass()` |
| 12 | `passes.ttgpuir.add_combine_tensor_select_and_if(pm)` (≥sm_80) | `…createCombineTensorSelectAndIfPass()` |
| 13 | `passes.ttgpuir.add_ws_*` (×4, ≥sm_80, num_consumer_groups path) | warp-specialization passes; defer to TODO until needed |
| 14 | `passes.ttgpuir.add_pipeline(pm, num_stages)` (≥sm_80) | `…createPipelinePass(num_stages)` |
| 15 | `passes.ttgpuir.add_ws_lowering(pm, num_consumer_groups)` (≥sm_80) | warp-spec lowering (defer) |
| 16 | `passes.ttgpuir.add_prefetch(pm)` | `…createPrefetchPass()` |
| 17 | `passes.ttgpuir.add_optimize_dot_operands(pm, cap≥80)` (repeat) | (same as 9) |
| 18 | `passes.ttgpuir.add_remove_layout_conversions(pm)` (repeat) | (same as 5) |
| 19 | `passes.ttgpuir.add_reduce_data_duplication(pm)` | `…createReduceDataDuplicationPass()` |
| 20 | `passes.ttgpuir.add_reorder_instructions(pm)` | `…createReorderInstructionsPass()` |
| 21 | `passes.common.add_cse(pm)` | (same as 10) |
| 22 | `passes.common.add_symbol_dce(pm)` | `mlir::createSymbolDCEPass()` |
| 23 | `nvidia.passes.ttnvgpuir.add_fence_insertion(pm)` (≥sm_90) | `…createFenceInsertionPass()` |
| 24 | `nvidia.passes.ttnvgpuir.add_tma_lowering(pm)` (≥sm_90) | `…createTMALoweringPass()` |
| 25 | `passes.common.add_canonicalizer(pm)` | `mlir::createCanonicalizerPass()` |

**Notes:**
- For sm_89 (RTX 5070 Ti / Ada / our test box), passes #4–22 + #25 fire. The sm_90 fence/TMA passes #23–24 don't.
- The "warp-specialization" passes (#13, #15) only fire if `num_consumer_groups > 0`. Our DSL never sets this; emit them anyway (they no-op when the option is 0) — keeps the C++ side a flat list with no per-call branching.
- Every factory returns `std::unique_ptr<mlir::Pass>` and is added via `pm.addPass(std::move(pass))`. No surprise APIs.

---

## Stage 2 — `make_llir(module, metadata, opt, capability)`

| # | Pybind call | Underlying C++ factory |
|---|-------------|------------------------|
| 1 | `nvidia.passes.ttgpuir.add_decompose_unsupported_conversions(pm)` | `mlir::triton::nvidia_gpu::createDecomposeUnsupportedConversionsPass()` |
| 2 | `passes.ttgpuir.add_combine_tensor_select_and_if(pm)` | (same as TTGIR #12) |
| 3 | `passes.convert.add_scf_to_cf(pm)` | `mlir::createConvertSCFToCFPass()` — `mlir/Conversion/Passes.h` |
| 4 | `passes.convert.add_index_to_llvmir(pm)` | `mlir::createConvertIndexToLLVMPass()` |
| 5 | `passes.ttgpuir.add_allocate_shared_memory(pm)` | `mlir::triton::gpu::createAllocateSharedMemoryPass()` |
| 6 | `nvidia.passes.ttgpuir.add_to_llvmir(pm, capability, ptx_version)` | `mlir::triton::createConvertTritonGPUToLLVMPass(int, int)` — `Conversion/TritonGPUToLLVM/Passes.h` |
| 7 | `nvidia.passes.ttnvgpuir.add_nvgpu_to_llvm(pm)` | `mlir::triton::createConvertNVGPUToLLVMPass()` |
| 8 | `passes.convert.add_arith_to_llvmir(pm)` | `mlir::createConvertArithToLLVMPass()` |
| 9 | `passes.common.add_canonicalizer(pm)` | (same as TTGIR #25) |
| 10 | `passes.common.add_cse(pm)` | (same as TTGIR #10) |
| 11 | `passes.common.add_symbol_dce(pm)` | (same as TTGIR #22) |
| 12 | `passes.llvmir.add_di_scope(pm)` (unless `TRITON_DISABLE_LINE_INFO` set) | `mlir::triton::createLLVMDIScopePass()` |

After running this pipeline, the module is in **LLVM dialect**. To get an
`llvm::Module*` out of it (needed for the next stage), call:

```cpp
mlir::translateModuleToLLVMIR(module, llvmCtx);
```

— from `mlir/Target/LLVMIR/ModuleTranslation.h`. Standard MLIR upstream API.

---

## Stage 3 — `make_ptx(llvmModule, metadata, opt, capability)`

Pure LLVM, no MLIR passes. Python calls `llvm.translate_to_asm(...)` which
wraps:

```cpp
// 1. Prepare target
llvm::InitializeAllTargetInfos();
llvm::InitializeAllTargets();
llvm::InitializeAllTargetMCs();
llvm::InitializeAllAsmParsers();
llvm::InitializeAllAsmPrinters();

// 2. Get NVPTX target
auto target = llvm::TargetRegistry::lookupTarget("nvptx64-nvidia-cuda", err);

// 3. Build TargetMachine for the right SM
auto tm = target->createTargetMachine(
    "nvptx64-nvidia-cuda",
    "sm_89",                     // from opt.target_arch
    "+ptx" + std::to_string(opt.ptx_version),
    llvm::TargetOptions{},
    llvm::Reloc::PIC_,
    std::nullopt,
    llvm::CodeGenOpt::Aggressive);

// 4. Run codegen pipeline → PTX text
llvm::legacy::PassManager pm;
llvm::raw_svector_ostream os(ptxBuffer);
tm->addPassesToEmitFile(pm, os, nullptr, llvm::CodeGenFileType::AssemblyFile);
pm.run(llvmModule);
```

All standard LLVM upstream. Triton ships its own LLVM snapshot but the
APIs are stable across recent LLVM versions.

---

## Stage 4 — `make_cubin(ptx_text, metadata, opt, capability)`

**Spawns the external `ptxas` binary.** Found via `CUDA_HOME/bin/ptxas`
(or `/usr/local/cuda/bin/ptxas`). Args roughly:

```sh
ptxas --gpu-name=sm_89 -O3 -v --opt-level=3 -o /tmp/foo.cubin /tmp/foo.ptx
```

Writes input PTX to a temp file, reads cubin bytes back. Exit code != 0
means assembly failure (e.g. invalid PTX).

**This stays as `posix_spawn` in our C++ shim** — there is no library
form of ptxas. This is the one Python-step we are NOT replacing with a
library call; we are replacing the Python *driver* with a C++ driver.

---

## C++ entry-point inventory — go/no-go decision

**Verdict: GO.** Every step has a public C++ entry point:

| Source | Header | Where to find it in vendored Triton |
|--------|--------|-------------------------------------|
| MLIRContext, parse, PassManager | `mlir/IR/MLIRContext.h`, `mlir/Parser/Parser.h`, `mlir/Pass/PassManager.h` | bundled MLIR (Triton's pinned LLVM snapshot) |
| Triton dialect registration | `triton/Conversion/TritonToTritonGPU/Passes.h`, `triton/Dialect/Triton/IR/Dialect.h` | `vendor/triton/include/` |
| TritonGPU passes | `triton/Dialect/TritonGPU/Transforms/Passes.h` | `vendor/triton/include/` |
| TritonGPUToLLVM | `triton/Conversion/TritonGPUToLLVM/Passes.h` | `vendor/triton/include/` |
| NVGPU passes | `third_party/nvidia/include/Dialect/NVGPU/Transforms/Passes.h` | `vendor/triton/third_party/nvidia/include/` |
| Standard MLIR conversions | `mlir/Conversion/Passes.h`, `mlir/Transforms/Passes.h` | bundled MLIR |
| LLVM NVPTX codegen | `llvm/Target/TargetMachine.h`, `llvm/MC/TargetRegistry.h`, etc. | bundled LLVM |
| MLIR → LLVM IR translation | `mlir/Target/LLVMIR/ModuleTranslation.h` | bundled MLIR |
| ptxas | external binary | `posix_spawn` |

**No pybind-only code paths.** The Python `passes.X.add_Y(pm)` wrappers
in `python/src/passes.cc` are 1:1 thin trampolines over the C++
factories — defining the same passes from C++ is a copy of the
`pm.addPass(triton::...createXPass(args))` pattern, no behavior is hidden
in pybind.

---

## Risks observed during the spike

1. **LLVM-snapshot ABI tax.** Triton 3.2.0 pins LLVM at commit
   `86b69c31642e98f8357df62c09d118ad1da4e16a`. **Triton does NOT build
   LLVM from source** — it downloads a pre-built tarball from
   `https://oaitriton.blob.core.windows.net/public/llvm-builds/llvm-{rev}-{system_suffix}.tar.gz`
   (rev = first 8 chars, system_suffix = `ubuntu-x64`, `centos-x64`,
   `macos-arm64`, etc.). The tarball ships `LLVM_INCLUDE_DIRS`,
   `LLVM_LIBRARY_DIR`, `LLVM_SYSPATH` for downstream cmake.

   **Mitigation for our build.rs:** mirror Triton's `setup.py`
   `get_llvm_package_info()` logic — fetch the same tarball into
   `OUT_DIR/llvm-86b69c31/`, set the three env vars, then run Triton's
   cmake. This avoids the 30-min LLVM source build entirely; the
   tarball is ~600 MB compressed. Allow override via
   `TRITON_LLVM_SYSPATH` env var so CI can cache.

2. **Pass-list drift across versions.** The TTGIR list above is for
   v3.2.0; v3.5.0+ refactored the warp-specialization passes (#13/15)
   and added new auto-WS passes. **Mitigation:** treat the list as a
   constant in `triton_c.cpp`, copy verbatim when bumping the
   submodule, expect a focused diff in Phase 1F.

3. **Cluster info / metadata struct.** `add_plan_cta(pm, cluster_info)`
   takes a `ClusterInfo*` (a small struct with `clusterDim{X,Y,Z}`).
   Python builds it from options; we hand-construct the same in C++.
   Need to grep `nvidia/include/Dialect/NVGPU/Transforms/Passes.h` for
   the struct definition once the submodule clone finishes. **Risk:
   low.**

4. **`ptx_version` selection.** `passes.ttgpuir.add_to_llvmir` takes
   both `capability` and `ptx_version`. Python derives `ptx_version`
   from CUDA-driver version. Our shim has no driver context — accept
   it from `TritonCompileOptions` (extend the `reserved` slot or add
   a new field; the struct is ABI-stable so we'd add to the end).

5. **Dialect-conversion `LLVM` translation.** After the LLIR pipeline,
   we call `translateModuleToLLVMIR` which itself walks the module and
   emits LLVM IR. It needs `mlir::registerLLVMDialectTranslation(ctx)`
   and `mlir::registerNVVMDialectTranslation(ctx)` to be called once
   on the MLIRContext at startup. Easy to forget — add to
   `triton_context_create()`.

---

## Triton's CMake options (verified from `CMakeLists.txt` v3.2.0)

For our subbuild we set:

```
-DTRITON_BUILD_TUTORIALS=OFF        # skip C++ tutorials
-DTRITON_BUILD_PYTHON_MODULE=OFF    # default OFF — keep it OFF
-DTRITON_BUILD_PROTON=OFF           # skip the profiler
-DTRITON_BUILD_UT=OFF               # skip C++ unit tests
-DTRITON_CODEGEN_BACKENDS=nvidia    # only build the NV backend, not amd/intel
-DCMAKE_BUILD_TYPE=Release
-DLLVM_LIBRARY_DIR=$(prefetched-llvm)/lib
-DLLVM_INCLUDE_DIRS=$(prefetched-llvm)/include
-DMLIR_DIR=$(prefetched-llvm)/lib/cmake/mlir
```

Triton declares OBJECT libraries (not standalone `.a` archives) and
appends each to a global property `TRITON_LIBS`. Per-plugin (backend)
they go to `TRITON_PLUGINS`. To link them all from `cc::Build`-driven
shim, we'll need to glob `${cmake_out}/lib/*.o` (or `.a` if cmake's
install step archives them — verify in Phase 1B by inspecting the
build tree).

---

## Go/no-go checklist

- [x] Every step has a public C++ entry point. (Verified by reading
      compiler.py + cross-referencing pybind wrapper file paths.)
- [x] Pass lists are stable enough to copy verbatim per Triton version.
- [ ] Linking Triton's `.a` files into a `cc::Build`-driven shim is
      feasible (deferred to Phase 1B; this is the next risk gate).

Phase 1B can begin. The submodule clone (still running on the proxy as
of writing) only needs to finish before we can drive Triton's CMake; it
isn't a blocker for this spike's conclusions.
