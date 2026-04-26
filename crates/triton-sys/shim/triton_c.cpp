// triton_c.cpp — C ABI shim that drives Triton's MLIR pass pipeline
// from C++, eliminating the need for Python's triton.compiler.compile().
//
// Designed per ARCHITECTURE.md §2.1: input is MLIR text (Triton IR),
// output is cubin bytes + metadata JSON. Internally we orchestrate
// the same pass pipeline that the Python script `mlir_to_cubin.py`
// calls (see SPIKE.md for the full pass list).
//
// Status: skeleton landed in Phase 1B; pass-pipeline bodies are TODO
// (Phase 1C). Builds + links once Triton + LLVM are ready, then we
// fill in `triton_compile_mlir`'s body to drive the cmake'd Triton
// libs.

#include "triton_c.h"

#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

// Triton public C++ headers (resolved once vendor/triton/include is on
// the include path — driven by build.rs Phase 1B).
#include "triton/Conversion/TritonToTritonGPU/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Conversion/TritonGPUToLLVM/Passes.h"

// NVIDIA-specific (lives in the third_party/nvidia tree)
#include "third_party/nvidia/include/Dialect/NVGPU/IR/Dialect.h"
#include "third_party/nvidia/include/Dialect/NVGPU/Transforms/Passes.h"
#include "third_party/nvidia/include/NVGPUToLLVM/Passes.h"

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"

extern "C" {

// ── opaque context ───────────────────────────────────────────────
struct TritonContext {
    mlir::MLIRContext mlirCtx;
    bool initialized;
};

// ── lifecycle ────────────────────────────────────────────────────
TritonContext* triton_context_create(void) {
    auto* ctx = new TritonContext();
    mlir::DialectRegistry registry;
    registry.insert<mlir::triton::TritonDialect>();
    registry.insert<mlir::triton::gpu::TritonGPUDialect>();
    registry.insert<mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect>();
    registry.insert<mlir::triton::nvgpu::NVGPUDialect>();
    // Standard MLIR dialects we use post-lowering.
    mlir::registerLLVMDialectTranslation(registry);
    mlir::registerNVVMDialectTranslation(registry);
    ctx->mlirCtx.appendDialectRegistry(registry);
    ctx->mlirCtx.loadAllAvailableDialects();
    ctx->initialized = true;

    // One-shot LLVM target init (idempotent).
    static bool llvm_inited = false;
    if (!llvm_inited) {
        llvm::InitializeAllTargetInfos();
        llvm::InitializeAllTargets();
        llvm::InitializeAllTargetMCs();
        llvm::InitializeAllAsmParsers();
        llvm::InitializeAllAsmPrinters();
        llvm_inited = true;
    }
    return ctx;
}

void triton_context_destroy(TritonContext* ctx) {
    delete ctx;
}

const char* triton_get_version(void) {
    static const char* v = "v3.2.0";  // pinned in fetch_vendor.sh
    return v;
}

// ── result helpers ────────────────────────────────────────────────
static TritonResult* make_error_result(const std::string& msg) {
    auto* r = static_cast<TritonResult*>(std::malloc(sizeof(TritonResult)));
    r->status = 1;
    r->binary_data = nullptr;
    r->binary_size = 0;
    r->metadata_json = nullptr;
    char* err = static_cast<char*>(std::malloc(msg.size() + 1));
    std::memcpy(err, msg.c_str(), msg.size() + 1);
    r->error_message = err;
    return r;
}

void triton_result_destroy(TritonResult* result) {
    if (!result) return;
    std::free(reinterpret_cast<void*>(const_cast<char*>(result->error_message)));
    std::free(reinterpret_cast<void*>(const_cast<char*>(result->metadata_json)));
    std::free(result->binary_data);
    std::free(result);
}

// ── core entry point ─────────────────────────────────────────────
TritonResult* triton_compile_mlir(
    TritonContext* ctx,
    const char* mlir_text,
    const TritonCompileOptions* opts)
{
    if (!ctx || !mlir_text || !opts) {
        return make_error_result("triton_compile_mlir: null argument");
    }

    try {
        // Step 1 — parse text → ModuleOp.
        // TODO(phase-1C): mlir::parseSourceString<mlir::ModuleOp>(mlir_text, &ctx->mlirCtx)
        //                 + diagnostic capture.

        // Step 2 — make_ttgir pass pipeline (~25 passes, see SPIKE.md).
        // TODO(phase-1C): pm.addPass(mlir::triton::createConvertTritonToTritonGPUPass(
        //                     /*target=*/std::string("cuda:") + opts->target_arch,
        //                     opts->num_warps, /*threadsPerWarp=*/32, opts->num_ctas));
        //                 ... etc per SPIKE.md table.

        // Step 3 — make_llir pass pipeline (~12 passes).
        // TODO(phase-1C): scf-to-cf, allocate-shared-memory,
        //                 ConvertTritonGPUToLLVM, NVGPUToLLVM, etc.

        // Step 4 — translateModuleToLLVMIR + NVPTX TargetMachine codegen → PTX text.
        // TODO(phase-1C).

        // Step 5 — posix_spawn ptxas to assemble PTX → cubin bytes.
        // TODO(phase-1C).

        // Step 6 — build metadata JSON (name, num_warps, shared_mem, target_arch).
        // TODO(phase-1C).

        // Placeholder result while Phase 1C is being filled in.
        return make_error_result(
            "triton_compile_mlir: pass pipeline not yet implemented (Phase 1C)");
    } catch (const std::exception& e) {
        return make_error_result(std::string("triton_compile_mlir threw: ") + e.what());
    } catch (...) {
        return make_error_result("triton_compile_mlir threw an unknown exception");
    }
}

}  // extern "C"
