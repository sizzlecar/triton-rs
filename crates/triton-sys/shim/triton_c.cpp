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
#include "triton/Conversion/TritonToTritonGPU/Passes.h"  // renamed from TritonToTritonGPUPass.h in v3.6
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
#include "NVGPUToLLVM/Passes.h"            // declares mlir::triton::createConvertNVGPUToLLVM in v3.6
#include "NVGPUToLLVM/NVGPUToLLVMPass.h"
#include "TritonNVIDIAGPUToLLVM/Passes.h"

#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Triple.h"

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

const char* triton_get_version(void) { return "v3.6.0"; }

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
    r->ptx_text = nullptr;
    return r;
}

void triton_result_destroy(TritonResult* result) {
    if (!result) return;
    std::free(reinterpret_cast<void*>(const_cast<char*>(result->error_message)));
    std::free(reinterpret_cast<void*>(const_cast<char*>(result->metadata_json)));
    std::free(reinterpret_cast<void*>(const_cast<char*>(result->ptx_text)));
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

// ── ttgir pass pipeline (mirrors v3.6.0 NVIDIABackend.make_ttgir,
//     simplified for sm_8x; Hopper / Blackwell-specific passes skipped) ──
static void build_ttgir_pipeline(
    mlir::PassManager& pm,
    const TritonCompileOptions* opts,
    int capability)
{
    using namespace mlir;
    // 1. TTIR → TTGIR conversion. v3.6 takes the Options struct directly
    //    via brace-init (5 fields, see Passes.td).
    pm.addPass(triton::createConvertTritonToTritonGPU(
        triton::ConvertTritonToTritonGPUOptions{
            /*target=*/std::string("cuda:") + std::to_string(capability),
            /*numWarps=*/opts->num_warps,
            /*threadsPerWarp=*/32,
            /*numCTAs=*/opts->num_ctas,
            /*enableSourceRemat=*/false}));

    pm.addPass(triton::gpu::createTritonGPUCoalesce());
    // v3.6 takes Options struct.
    if (capability / 10 >= 8)
        pm.addPass(triton::gpu::createTritonGPUF32DotTC(
            triton::gpu::TritonGPUF32DotTCOptions{/*useTF32=*/false}));

    // v3.6 sig change: PlanCTAPass no longer takes ClusterInfo*.
    pm.addPass(triton::nvidia_gpu::createTritonNvidiaGPUPlanCTAPass());
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
        // v3.6 sig change: createTritonGPUPipeline takes 2-field Options
        // (numStages, dumpIntermediateSteps).
        pm.addPass(triton::gpu::createTritonGPUPipeline(
            triton::gpu::TritonGPUPipelineOptions{
                /*numStages=*/static_cast<int32_t>(opts->num_stages),
                /*dumpIntermediateSteps=*/false}));
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
        // v3.6 names: TritonGPUFenceInsertion (no Pass suffix) lives in
        // nvidia_gpu namespace; TMALoweringPass keeps the suffix.
        pm.addPass(triton::nvidia_gpu::createTritonGPUFenceInsertion());
        pm.addPass(triton::nvidia_gpu::createTritonNvidiaGPUTMALoweringPass());
    }
    pm.addPass(createCanonicalizerPass());
}

// ── llir pass pipeline (mirrors v3.6.0 make_llir, simplified) ────
static void build_llir_pipeline(
    mlir::PassManager& pm,
    int capability,
    int ptx_version)
{
    using namespace mlir;
    // v3.6: NVIDIA::createDecomposeUnsupportedConversionsPass dropped —
    // covered by other passes now.
    pm.addPass(triton::gpu::createTritonGPUCombineTensorSelectAndIf());
    pm.addPass(createSCFToControlFlowPass());
    pm.addPass(createConvertIndexToLLVMPass());
    pm.addPass(triton::gpu::createAllocateSharedMemory());
    pm.addPass(triton::createConvertTritonGPUToLLVMPass(capability, ptx_version));
    pm.addPass(triton::createConvertNVGPUToLLVM());
    pm.addPass(createArithToLLVMConversionPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    pm.addPass(createSymbolDCEPass());
    pm.addPass(createLLVMDIScope());
}

// Link libdevice.10.bc into the module so calls to __nv_rsqrtf etc.
// resolve. Same pattern as Python's `link_extern_libs` (NV side).
// Returns empty error string on success.
static std::string link_libdevice(llvm::Module& dst, const std::string& path) {
    llvm::SMDiagnostic err;
    std::unique_ptr<llvm::Module> libMod =
        llvm::parseIRFile(path, err, dst.getContext());
    if (!libMod) {
        return "parseIRFile(libdevice) failed: " + err.getMessage().str();
    }
    libMod->setTargetTriple(dst.getTargetTriple());
    libMod->setDataLayout(dst.getDataLayout());
    llvm::Linker linker(dst);
    if (linker.linkInModule(std::move(libMod),
                            llvm::Linker::Flags::LinkOnlyNeeded)) {
        return "Linker::linkInModule(libdevice) failed";
    }
    return "";
}

// Run LLVM's new-PM optimization pipeline at O3 with NVPTX target info,
// matching what Python's `optimize_module(mod, OptimizationLevel::O3, ...)`
// does. This closes the bench gap vs the Python compile path: vec_add /
// silu_mul / residual_add were ~25% slower without it.
static void optimize_module_O3(llvm::Module& mod, llvm::TargetMachine& tm) {
    using namespace llvm;
    LoopAnalysisManager lam;
    FunctionAnalysisManager fam;
    CGSCCAnalysisManager cgam;
    ModuleAnalysisManager mam;

    PipelineTuningOptions tune;
    tune.LoopUnrolling = true;
    tune.LoopVectorization = true;
    tune.SLPVectorization = true;

    PassBuilder pb(&tm, tune);
    pb.registerModuleAnalyses(mam);
    pb.registerCGSCCAnalyses(cgam);
    pb.registerFunctionAnalyses(fam);
    pb.registerLoopAnalyses(lam);
    pb.crossRegisterProxies(lam, fam, cgam, mam);

    ModulePassManager mpm = pb.buildPerModuleDefaultPipeline(OptimizationLevel::O3);
    mpm.run(mod, mam);
}

// Set the "nvvm-reflect-ftz" function attribute on every defined function
// — this enables the fast-math-flags-aware paths in libdevice (rsqrt etc).
// Matches Triton's `set_nvvm_reflect_ftz`.
static void set_nvvm_reflect_ftz(llvm::Module& mod) {
    for (auto& f : mod) {
        if (!f.isDeclaration()) {
            f.addFnAttr("nvvm-reflect-ftz", "1");
        }
    }
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
    auto* tm = target->createTargetMachine(llvm::Triple(triple), proc, features, opt, rm,
                                           std::nullopt, llvm::CodeGenOptLevel::Aggressive);
    if (!tm) throw std::runtime_error("createTargetMachine failed");

    mod.setDataLayout(tm->createDataLayout());
    mod.setTargetTriple(llvm::Triple(triple));

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

// Strip the ", debug" / "debug, " modifier from PTX `.target` lines.
// Triton's compiler.py does this verbatim with the comment "Remove the
// debug flag that prevents ptxas from optimizing the code". Without it,
// ptxas treats the kernel as a debug build and disables most opts —
// observed as a 13-14% perf gap vs Python on math-heavy benches.
// We also keep emitting the LLVM DI-scope pass for source-line info,
// which ptxas picks up via `-lineinfo` instead.
static std::string strip_debug_target(std::string ptx) {
    static const std::regex re_with_comma_after(R"(,\s*debug)");
    static const std::regex re_with_comma_before(R"(debug,\s*)");
    ptx = std::regex_replace(ptx, re_with_comma_after, "");
    ptx = std::regex_replace(ptx, re_with_comma_before, "");
    return ptx;
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

        // 2. TTIR → TTGIR pass pipeline. (v3.6 dropped the ClusterInfo
        // arg from PlanCTAPass; cluster dims are now in module attrs.)
        {
            mlir::PassManager pm(&ctx->mlirCtx);
            build_ttgir_pipeline(pm, opts, capability);
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

        // 4a. Build a NVPTXTargetMachine for datalayout + optimizer.
        std::string triple = "nvptx64-nvidia-cuda";
        std::string proc = (capability == 90) ? "sm_90a" : (std::string("sm_") + std::to_string(capability));
        std::string features = "+ptx" + std::to_string(default_ptx_version(capability));
        std::string err;
        auto* nvtarget = llvm::TargetRegistry::lookupTarget(triple, err);
        if (!nvtarget) return make_error("nvptx target lookup: " + err);
        std::unique_ptr<llvm::TargetMachine> tm{nvtarget->createTargetMachine(
            llvm::Triple(triple), proc, features, llvm::TargetOptions{},
            std::optional<llvm::Reloc::Model>(llvm::Reloc::PIC_), std::nullopt,
            llvm::CodeGenOptLevel::Aggressive)};
        if (!tm) return make_error("createTargetMachine failed");
        llvm_mod->setDataLayout(tm->createDataLayout());
        llvm_mod->setTargetTriple(llvm::Triple(triple));

        // 4b. Link libdevice (resolves __nv_rsqrtf etc.).
        const char* libdevice_env = std::getenv("TRITON_LIBDEVICE_PATH");
        std::string libdevice = libdevice_env ? libdevice_env :
            "crates/triton-sys/vendor/triton/third_party/nvidia/backend/lib/libdevice.10.bc";
        std::string link_err = link_libdevice(*llvm_mod, libdevice);
        if (!link_err.empty()) return make_error(link_err);

        // 4c. set ftz, then optimize at O3 (closes ~25% perf gap).
        set_nvvm_reflect_ftz(*llvm_mod);
        optimize_module_O3(*llvm_mod, *tm);

        // 5. llvm::Module → PTX text.
        std::string ptx = strip_debug_target(llvm_to_ptx(*llvm_mod, capability));

        // 6. PTX → cubin (spawn ptxas).
        std::string ptxas_err;
        auto cubin = spawn_ptxas(ptx, capability, ptxas_err);
        if (cubin.empty()) return make_error("ptxas: " + ptxas_err);

        // 7. Build metadata JSON (hand-formatted, no JSON lib).
        std::string name = extract_kernel_name(ptx);
        // shared_mem from the module's "triton_gpu.shared" int attr (set
        // by createAllocateSharedMemoryPass during the LLIR pipeline).
        int64_t shared_mem = 0;
        if (auto attr = module->getOperation()->getAttrOfType<mlir::IntegerAttr>(
                "triton_gpu.shared")) {
            shared_mem = attr.getInt();
        }
        std::string meta = "{";
        meta += "\"name\":\"" + name + "\",";
        meta += "\"num_warps\":" + std::to_string(opts->num_warps) + ",";
        meta += "\"num_stages\":" + std::to_string(opts->num_stages) + ",";
        meta += "\"num_ctas\":" + std::to_string(opts->num_ctas) + ",";
        meta += "\"shared_mem\":" + std::to_string(shared_mem) + ",";
        meta += "\"target_arch\":\"" + std::string(opts->target_arch) + "\",";
        // Cluster dims live as a module attribute in v3.6 (set by
        // PlanCTAPass) — for now hardcode (1,1,1) which is the default
        // for non-Hopper. Phase-2: extract via mod->getAttr("ttg.cluster-dims").
        meta += "\"cluster_dims\":[1,1,1]";
        meta += "}";

        // 8. Pack result.
        auto* r = static_cast<TritonResult*>(std::malloc(sizeof(TritonResult)));
        r->status = 0;
        r->binary_size = cubin.size();
        r->binary_data = static_cast<uint8_t*>(std::malloc(cubin.size()));
        std::memcpy(r->binary_data, cubin.data(), cubin.size());
        r->metadata_json = dup_cstr(meta);
        r->error_message = nullptr;
        r->ptx_text = dup_cstr(ptx);
        return r;

    } catch (const std::exception& e) {
        return make_error(std::string("compile threw: ") + e.what());
    } catch (...) {
        return make_error("compile threw unknown exception");
    }
}

}  // extern "C"
