/* triton_c.h — Stable C ABI surface for the Triton MLIR backend.
 *
 * Design rules (see ARCHITECTURE.md §2.3):
 *   - 5 entry points, no more.
 *   - No C++ types in the signatures.
 *   - Errors carried in TritonResult, never thrown.
 *   - Forward struct decls only; bodies live in triton_c.cpp.
 */

#ifndef TRITON_C_H
#define TRITON_C_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

/* Opaque handle to a Triton compilation context (MLIRContext + dialect
 * registration + pass pipeline). One per host process is typically enough. */
typedef struct TritonContext TritonContext;

/* Output of a successful or failed compile. */
typedef struct TritonResult {
    /* 0 = success, non-zero = error. error_message is set when non-zero. */
    int      status;

    /* Compiled binary (cubin/hsaco/spirv). NULL on error. */
    uint8_t* binary_data;
    size_t   binary_size;

    /* JSON-encoded kernel metadata (name, args, shared_mem, num_warps, ...).
     * NULL on error. */
    const char* metadata_json;

    /* UTF-8 error message, NULL on success. Owned by the result. */
    const char* error_message;
} TritonResult;

/* Compile options. ABI-stable: never reorder, only append. */
typedef struct TritonCompileOptions {
    /* "sm_80", "sm_89", "sm_90a", "gfx942", "xelp", ... */
    const char* target_arch;

    int num_warps;     /* 0 = backend default */
    int num_stages;    /* 0 = backend default */
    int num_ctas;      /* 0 = backend default */

    /* Reserved for forward compatibility. Must be zero-filled. */
    uint64_t reserved[8];
} TritonCompileOptions;

/* === The 5 entry points === */

TritonContext* triton_context_create(void);

void triton_context_destroy(TritonContext* ctx);

/* Compile MLIR text → backend binary + metadata JSON.
 * Caller must free with triton_result_destroy. */
TritonResult* triton_compile_mlir(
    TritonContext* ctx,
    const char* mlir_text,
    const TritonCompileOptions* opts);

void triton_result_destroy(TritonResult* result);

/* Returns the underlying Triton library version string ("v3.6.0").
 * Caller must NOT free. Lifetime = process. */
const char* triton_get_version(void);

#ifdef __cplusplus
}
#endif

#endif /* TRITON_C_H */
