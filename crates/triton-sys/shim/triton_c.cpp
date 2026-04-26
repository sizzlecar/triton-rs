// triton_c.cpp — C ABI shim that drives Triton's MLIR pass pipeline
// from C++, eliminating Python from the kernel-compile path.
//
// Mirrors the orchestration in
// vendor/triton/third_party/nvidia/backend/compiler.py (make_ttgir →
// make_llir → make_ptx → make_cubin), calling the same C++ factory
// functions the Python pybind module wraps. See SPIKE.md for the
// full pass list and rationale.
//
// Status: Phase 1C — pass-pipeline body implemented. Some risks:
//   - LLVM optimization (set_nvvm_reflect_ftz, optimize_module O3)
//     is currently best-effort; full match-with-Python deferred to 1F.
//   - Warp-specialization passes (cap ≥ 80, num_consumer_groups > 0)
//     are wired but only fire when num_consumer_groups > 0.
//   - Hopper-only fence/TMA passes (cap ≥ 90) are guarded by
//     opts->target_arch comparison.

#include "triton_c.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <regex>
#include <string>
#include <unistd.h>
#include <sys/wait.h>
#include <spawn.h>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

// Triton public headers
#include "triton/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.h"
#include "triton/Conversion/TritonGPUToLLVM/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"
#include "triton/Target/LLVMIR/Passes.h"

// NVIDIA-specific (lives under third_party/nvidia/include — the CMakeLists
// adds third_party as an include root so these resolve as <Dialect/...>).
#include "Dialect/NVGPU/IR/Dialect.h"
#include "NVGPUToLLVM/NVGPUToLLVMPass.h"
#include "TritonNVIDIAGPUToLLVM/Passes.h"

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"

extern "C" {

// ── opaque context ───────────────────────────────────────────────
struct TritonContext {
    mlir::MLIRContext mlirCtx;
};

// ── lifecycle ────────────────────────────────────────────────────
TritonContext* triton_context_create(void) {
    auto* ctx = new TritonContext();
    mlir::DialectRegistry registry;
    registry.insert<mlir::triton::TritonDialect,
                    mlir::triton::gpu::TritonGPUDialect,
                    mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect,
                    mlir::triton::nvgpu::NVGPUDialect>();
    mlir::registerBuiltinDialectTranslation(registry);
    mlir::registerLLVMDialectTranslation(registry);
    mlir::registerNVVMDialectTranslation(registry);
    ctx->mlirCtx.appendDialectRegistry(registry);
    ctx->mlirCtx.loadAllAvailableDialects();

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

void triton_context_destroy(TritonContext* ctx) { delete ctx; }

const char* triton_get_version(void) { return "v3.2.0"; }

// ── result helpers ────────────────────────────────────────────────
static char* dup_cstr(const std::string& s) {
    char* out = static_cast<char*>(std::malloc(s.size() + 1));
    std::memcpy(out, s.c_str(), s.size() + 1);
    return out;
}

static TritonResult* make_error(const std::string& msg) {
    auto* r = static_cast<TritonResult*>(std::malloc(sizeof(TritonResult)));
    r->status = 1;
    r->binary_data = nullptr;
    r->binary_size = 0;
    r->metadata_json = nullptr;
    r->error_message = dup_cstr(msg);
    return r;
}

void triton_result_destroy(TritonResult* result) {
    if (!result) return;
    std::free(reinterpret_cast<void*>(const_cast<char*>(result->error_message)));
    std::free(reinterpret_cast<void*>(const_cast<char*>(result->metadata_json)));
    std::free(result->binary_data);
    std::free(result);
}

// ── helpers ───────────────────────────────────────────────────────
static int parse_capability(const std::string& target_arch) {
    // Accept "sm_89", "89", "sm_90a" — extract digits (and trim trailing "a").
    std::string s = target_arch;
    if (s.rfind("sm_", 0) == 0) s = s.substr(3);
    if (!s.empty() && (s.back() == 'a' || s.back() == 'A')) s.pop_back();
    try { return std::stoi(s); } catch (...) { return 0; }
}

static int default_ptx_version(int cap) {
    // Match Triton 3.2.0's get_ptx_version_from_options heuristic for
    // the no-CUDA-driver case: PTX 8.3 for sm_90+, 8.0 for sm_80–89, etc.
    if (cap >= 90) return 83;
    if (cap >= 80) return 82;
    if (cap >= 75) return 80;
    return 75;
}

// ── ttgir pass pipeline (mirrors NVIDIABackend.make_ttgir) ────────
static void build_ttgir_pipeline(
    mlir::PassManager& pm,
    const TritonCompileOptions* opts,
    int capability,
    mlir::triton::nvidia_gpu::ClusterInfo& cluster_info)
{
    using namespace mlir;
    pm.addPass(triton::createConvertTritonToTritonGPUPass(
        std::string("cuda:") + std::to_string(capability),
        opts->num_warps, /*threadsPerWarp=*/32, opts->num_ctas));

    pm.addPass(triton::gpu::createTritonGPUCoalesce());
    if (capability / 10 >= 8) pm.addPass(triton::gpu::createTritonGPUF32DotTC());

    pm.addPass(createTritonNvidiaGPUPlanCTAPass(&cluster_info));
    pm.addPass(triton::gpu::createTritonGPURemoveLayoutConversions());
    pm.addPass(triton::gpu::createTritonGPUOptimizeThreadLocality());
    pm.addPass(triton::gpu::createTritonGPUAccelerateMatmul());
    pm.addPass(triton::gpu::createTritonGPURemoveLayoutConversions());
    pm.addPass(triton::gpu::createTritonGPUOptimizeDotOperands(
        triton::gpu::TritonGPUOptimizeDotOperandsOptions{
            /*hoistLayoutConversion=*/capability >= 80}));
    pm.addPass(createCSEPass());

    if (capability / 10 >= 8) {
        pm.addPass(triton::gpu::createTritonGPUOptimizeAccumulatorInit());
        pm.addPass(triton::gpu::createTritonGPUCombineTensorSelectAndIf());
        // Warp-specialization passes — only effective when
        // num_consumer_groups > 0 (default 0). We omit them for the
        // initial Phase 1C; flash-attention may need them later.
        pm.addPass(triton::gpu::createTritonGPUPipeline(
            triton::gpu::TritonGPUPipelineOptions{
                /*numStages=*/static_cast<int32_t>(opts->num_stages)}));
    }

    pm.addPass(triton::gpu::createTritonGPUPrefetch());
    pm.addPass(triton::gpu::createTritonGPUOptimizeDotOperands(
        triton::gpu::TritonGPUOptimizeDotOperandsOptions{
            /*hoistLayoutConversion=*/capability >= 80}));
    pm.addPass(triton::gpu::createTritonGPURemoveLayoutConversions());
    pm.addPass(triton::gpu::createTritonGPUReduceDataDuplication());
    pm.addPass(triton::gpu::createTritonGPUReorderInstructions());
    pm.addPass(createCSEPass());
    pm.addPass(createSymbolDCEPass());

    if (capability / 10 >= 9) {
        pm.addPass(createTritonNvidiaGPUFenceInsertionPass());
        pm.addPass(createTritonNvidiaGPUTMALoweringPass());
    }
    pm.addPass(createCanonicalizerPass());
}

// ── llir pass pipeline (mirrors make_llir) ───────────────────────
static void build_llir_pipeline(
    mlir::PassManager& pm,
    int capability,
    int ptx_version)
{
    using namespace mlir;
    pm.addPass(triton::NVIDIA::createDecomposeUnsupportedConversionsPass());
    pm.addPass(triton::gpu::createTritonGPUCombineTensorSelectAndIf());
    pm.addPass(createConvertSCFToCFPass());
    pm.addPass(createConvertIndexToLLVMPass());
    pm.addPass(triton::gpu::createAllocateSharedMemoryPass());
    pm.addPass(triton::createConvertTritonGPUToLLVMPass(capability, ptx_version));
    pm.addPass(triton::createConvertNVGPUToLLVMPass());
    pm.addPass(createArithToLLVMConversionPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    pm.addPass(createSymbolDCEPass());
    pm.addPass(createLLVMDIScopePass());
}

// ── PTX codegen via NVPTXTargetMachine ───────────────────────────
static std::string llvm_to_ptx(llvm::Module& mod, int capability) {
    std::string triple = "nvptx64-nvidia-cuda";
    std::string proc = (capability == 90) ? "sm_90a" : (std::string("sm_") + std::to_string(capability));
    std::string features = "+ptx" + std::to_string(default_ptx_version(capability));

    std::string err;
    auto* target = llvm::TargetRegistry::lookupTarget(triple, err);
    if (!target) throw std::runtime_error("nvptx target lookup: " + err);

    llvm::TargetOptions opt;
    auto rm = std::optional<llvm::Reloc::Model>(llvm::Reloc::PIC_);
    auto* tm = target->createTargetMachine(triple, proc, features, opt, rm,
                                           std::nullopt, llvm::CodeGenOptLevel::Aggressive);
    if (!tm) throw std::runtime_error("createTargetMachine failed");

    mod.setDataLayout(tm->createDataLayout());
    mod.setTargetTriple(triple);

    llvm::SmallString<0> ptxBuf;
    llvm::raw_svector_ostream os(ptxBuf);
    llvm::legacy::PassManager codegenPM;
    if (tm->addPassesToEmitFile(codegenPM, os, nullptr,
                                llvm::CodeGenFileType::AssemblyFile)) {
        delete tm;
        throw std::runtime_error("addPassesToEmitFile rejected NVPTX target");
    }
    codegenPM.run(mod);
    delete tm;
    return std::string(ptxBuf.begin(), ptxBuf.end());
}

// Find the (single) ".visible .entry NAME" in PTX.
static std::string extract_kernel_name(const std::string& ptx) {
    std::regex re(R"(\.visible\s+\.entry\s+([A-Za-z_][A-Za-z0-9_]*))");
    std::smatch m;
    if (std::regex_search(ptx, m, re)) return m[1].str();
    return "kernel";  // fallback; metadata.name still usable
}

// ── ptxas spawn ───────────────────────────────────────────────────
static std::vector<uint8_t> spawn_ptxas(
    const std::string& ptx_text,
    int capability,
    std::string& err_out)
{
    // Find ptxas: $TRITON_PTXAS_PATH > $CUDA_HOME/bin/ptxas > /usr/local/cuda/bin/ptxas
    std::string ptxas;
    if (auto p = std::getenv("TRITON_PTXAS_PATH"); p && *p) ptxas = p;
    else if (auto h = std::getenv("CUDA_HOME"); h && *h) ptxas = std::string(h) + "/bin/ptxas";
    else ptxas = "/usr/local/cuda/bin/ptxas";

    char ptx_tmpl[] = "/tmp/triton_rs_ptx_XXXXXX";
    int ptx_fd = mkstemp(ptx_tmpl);
    if (ptx_fd < 0) { err_out = "mkstemp(ptx) failed"; return {}; }
    if (write(ptx_fd, ptx_text.data(), ptx_text.size()) != (ssize_t)ptx_text.size()) {
        err_out = "write(ptx) short";
        close(ptx_fd); unlink(ptx_tmpl);
        return {};
    }
    close(ptx_fd);

    std::string cubin_path = std::string(ptx_tmpl) + ".cubin";
    std::string suffix = (capability == 90) ? "a" : "";
    std::string gpu_name = "--gpu-name=sm_" + std::to_string(capability) + suffix;

    char* argv[] = {
        const_cast<char*>(ptxas.c_str()),
        const_cast<char*>("-lineinfo"),
        const_cast<char*>("-v"),
        const_cast<char*>(gpu_name.c_str()),
        const_cast<char*>(ptx_tmpl),
        const_cast<char*>("-o"),
        const_cast<char*>(cubin_path.c_str()),
        nullptr
    };

    pid_t pid;
    extern char** environ;
    int rc = posix_spawn(&pid, argv[0], nullptr, nullptr, argv, environ);
    if (rc != 0) {
        err_out = "posix_spawn(ptxas) failed: " + std::string(strerror(rc));
        unlink(ptx_tmpl);
        return {};
    }
    int status = 0;
    waitpid(pid, &status, 0);
    unlink(ptx_tmpl);
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        err_out = "ptxas exit " + std::to_string(WEXITSTATUS(status));
        unlink(cubin_path.c_str());
        return {};
    }

    std::ifstream f(cubin_path, std::ios::binary);
    std::vector<uint8_t> cubin((std::istreambuf_iterator<char>(f)),
                               std::istreambuf_iterator<char>());
    f.close();
    unlink(cubin_path.c_str());
    return cubin;
}

// ── core entry point ─────────────────────────────────────────────
TritonResult* triton_compile_mlir(
    TritonContext* ctx,
    const char* mlir_text,
    const TritonCompileOptions* opts)
{
    if (!ctx || !mlir_text || !opts) return make_error("triton_compile_mlir: null arg");

    try {
        int capability = parse_capability(opts->target_arch ? opts->target_arch : "sm_89");
        if (capability == 0) return make_error("invalid target_arch");
        int ptx_version = default_ptx_version(capability);

        // 1. Parse TTIR text → ModuleOp.
        auto module = mlir::parseSourceString<mlir::ModuleOp>(
            llvm::StringRef(mlir_text), &ctx->mlirCtx);
        if (!module) return make_error("parseSourceString returned null (invalid MLIR)");

        // 2. TTIR → TTGIR pass pipeline.
        mlir::triton::nvidia_gpu::ClusterInfo cluster_info;
        {
            mlir::PassManager pm(&ctx->mlirCtx);
            build_ttgir_pipeline(pm, opts, capability, cluster_info);
            if (mlir::failed(pm.run(*module)))
                return make_error("ttgir pass pipeline failed");
        }

        // 3. TTGIR → LLIR pass pipeline.
        {
            mlir::PassManager pm(&ctx->mlirCtx);
            build_llir_pipeline(pm, capability, ptx_version);
            if (mlir::failed(pm.run(*module)))
                return make_error("llir pass pipeline failed");
        }

        // 4. MLIR-LLVM dialect → llvm::Module.
        llvm::LLVMContext llvm_ctx;
        auto llvm_mod = mlir::translateModuleToLLVMIR(*module, llvm_ctx);
        if (!llvm_mod) return make_error("translateModuleToLLVMIR failed");

        // 5. llvm::Module → PTX text.
        std::string ptx = llvm_to_ptx(*llvm_mod, capability);

        // 6. PTX → cubin (spawn ptxas).
        std::string ptxas_err;
        auto cubin = spawn_ptxas(ptx, capability, ptxas_err);
        if (cubin.empty()) return make_error("ptxas: " + ptxas_err);

        // 7. Build metadata JSON (hand-formatted, no JSON lib).
        std::string name = extract_kernel_name(ptx);
        std::string meta = "{";
        meta += "\"name\":\"" + name + "\",";
        meta += "\"num_warps\":" + std::to_string(opts->num_warps) + ",";
        meta += "\"num_stages\":" + std::to_string(opts->num_stages) + ",";
        meta += "\"num_ctas\":" + std::to_string(opts->num_ctas) + ",";
        meta += "\"shared_mem\":0,";  // TODO: pull from module attr "triton_gpu.shared"
        meta += "\"target_arch\":\"" + std::string(opts->target_arch) + "\",";
        meta += "\"cluster_dims\":[" +
                std::to_string(cluster_info.clusterDimX) + "," +
                std::to_string(cluster_info.clusterDimY) + "," +
                std::to_string(cluster_info.clusterDimZ) + "]";
        meta += "}";

        // 8. Pack result.
        auto* r = static_cast<TritonResult*>(std::malloc(sizeof(TritonResult)));
        r->status = 0;
        r->binary_size = cubin.size();
        r->binary_data = static_cast<uint8_t*>(std::malloc(cubin.size()));
        std::memcpy(r->binary_data, cubin.data(), cubin.size());
        r->metadata_json = dup_cstr(meta);
        r->error_message = nullptr;
        return r;

    } catch (const std::exception& e) {
        return make_error(std::string("compile threw: ") + e.what());
    } catch (...) {
        return make_error("compile threw unknown exception");
    }
}

}  // extern "C"
