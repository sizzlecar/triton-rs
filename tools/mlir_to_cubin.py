#!/usr/bin/env python3
"""Compile a Triton IR (.mlir) file into a CUDA cubin via Triton's internal
MLIR pipeline. Bridges triton-ir's text output to a runnable GPU artifact
without needing the C ABI shim — used for Phase-1-lite end-to-end demos.

Usage:
  mlir_to_cubin.py INPUT.mlir OUTPUT_DIR [--arch 89] [--num-warps 4]

Outputs (in OUTPUT_DIR):
  kernel.cubin       CUDA binary, ready to load with cuModuleLoadData / cudarc
  kernel.json        { name, num_warps, shared_mem, target_arch }

The MLIR file MUST be at the Triton IR level (uses `tt.func`, `tt.load`,
`tt.dot`, ... on plain `tensor<...>` types — not yet `triton_gpu` layout-
encoded). Our triton-ir crate emits exactly this form.

Exit codes:
  0 = compiled successfully
  1 = compile failed (Triton parser or pass pipeline rejected the IR)
  2 = environment problem (Triton import / API mismatch)
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("output_dir")
    p.add_argument("--arch", type=int, default=89,
                   help="compute capability * 10 (e.g. 89 for sm_89). "
                        "Use 89 (Ada Lovelace) as a safe default — PTX is "
                        "forward-compatible to newer GPUs via driver JIT.")
    p.add_argument("--num-warps", type=int, default=4)
    p.add_argument("--num-stages", type=int, default=3)
    args = p.parse_args()

    try:
        from triton.compiler import compile
        from triton.backends.compiler import GPUTarget
        from triton.backends.nvidia.compiler import CUDAOptions
    except Exception as e:
        print(f"environment: cannot import triton compiler internals: {e}",
              file=sys.stderr)
        return 2

    if not os.path.exists(args.input):
        print(f"input file not found: {args.input}", file=sys.stderr)
        return 2

    os.makedirs(args.output_dir, exist_ok=True)

    # IRSource picks the right pass entry-point off the file extension.
    # `.ttir` = Triton IR (high level — what triton-ir emits).
    # `.ttgir` = TritonGPU IR (already lowered).
    # `.mlir` is rejected — IRSource doesn't recognise that extension.
    # Copy to a `.ttir` if needed so callers can hand us anything.
    input_path = args.input
    if not input_path.endswith((".ttir", ".ttgir")):
        import shutil
        ttir_path = os.path.join(args.output_dir, "kernel.ttir")
        shutil.copy(input_path, ttir_path)
        input_path = ttir_path
        print(f"# copied {args.input} -> {ttir_path} for IRSource", file=sys.stderr)

    target = GPUTarget("cuda", args.arch, 32)
    opts = CUDAOptions(
        num_warps=args.num_warps,
        num_stages=args.num_stages,
        # Empty defaults for the rest; CUDAOptions handles the missing fields.
    )

    print(f"# compiling {args.input} -> sm_{args.arch}", file=sys.stderr)
    try:
        # `compile()` expects either ASTSource (Python kernel) or a file
        # path string; if given a string, it constructs the IRSource
        # itself. Passing our own IRSource instance is rejected by an
        # internal assertion in 3.2.0.
        result = compile(input_path, target=target, options=opts)
    except Exception as e:
        print(f"triton compile failed: {type(e).__name__}: {e}",
              file=sys.stderr)
        return 1

    asm = result.asm
    if "cubin" not in asm:
        print(f"triton compile produced no cubin (got: {list(asm.keys())})",
              file=sys.stderr)
        return 1

    cubin_bytes: bytes = asm["cubin"]
    cubin_path = os.path.join(args.output_dir, "kernel.cubin")
    with open(cubin_path, "wb") as f:
        f.write(cubin_bytes)

    # Also dump the PTX text so cudarc 0.13 (which only ships `load_ptx`,
    # no `load_cubin`) can pick the kernel up. cuModuleLoadData under the
    # hood will JIT-compile this PTX to native at load time.
    ptx_text = asm.get("ptx")
    if ptx_text is not None:
        ptx_path = os.path.join(args.output_dir, "kernel.ptx")
        with open(ptx_path, "w") as f:
            if isinstance(ptx_text, bytes):
                f.write(ptx_text.decode("utf-8", errors="replace"))
            else:
                f.write(ptx_text)
        print(f"# wrote {ptx_path} ({len(ptx_text)} chars)", file=sys.stderr)

    md = result.metadata
    # `metadata` is a dataclass / namedtuple; access common fields defensively
    # since attribute names have shifted across Triton versions.
    def get(field, default=None):
        return getattr(md, field, default)

    meta_out = {
        "name": get("name") or get("function_name") or "kernel",
        "num_warps": get("num_warps", args.num_warps),
        "num_stages": get("num_stages", args.num_stages),
        "num_ctas": get("num_ctas", 1),
        "shared_mem": get("shared", get("shared_mem", 0)),
        "target_arch": f"sm_{args.arch}",
        "cubin_bytes": len(cubin_bytes),
    }
    meta_path = os.path.join(args.output_dir, "kernel.json")
    with open(meta_path, "w") as f:
        json.dump(meta_out, f, indent=2)

    print(f"# wrote {cubin_path} ({len(cubin_bytes)} bytes)", file=sys.stderr)
    print(f"# wrote {meta_path}", file=sys.stderr)
    print(f"# kernel name: {meta_out['name']}", file=sys.stderr)
    print(f"# num_warps: {meta_out['num_warps']}", file=sys.stderr)
    print(f"# shared_mem: {meta_out['shared_mem']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
