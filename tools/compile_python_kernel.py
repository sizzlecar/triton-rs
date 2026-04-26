#!/usr/bin/env python3
"""Compile a Python @triton.jit kernel to PTX/cubin via Triton's ASTSource path.

This is the head-to-head comparator for our DSL pipeline (which uses
IRSource on the MLIR text emitted by triton-ir). For a benchmark to be
fair, both paths should hit the same Triton MLIR pass pipeline with the
same options — `--num-warps 4 --num-stages 3 --block 1024`.

Usage:
  compile_python_kernel.py PY_FILE FN_NAME OUT_DIR \\
      --signature 'a_ptr=*fp32,b_ptr=*fp32,out_ptr=*fp32,n=i32' \\
      --constants 'BLOCK=1024' \\
      --arch 89 [--num-warps 4] [--num-stages 3]

Outputs (same shape as mlir_to_cubin.py): kernel.cubin / kernel.ptx /
kernel.json.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys


def parse_kv(s: str) -> dict[str, str]:
    if not s:
        return {}
    out = {}
    for part in s.split(","):
        k, _, v = part.partition("=")
        out[k.strip()] = v.strip()
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("py_file")
    p.add_argument("fn_name")
    p.add_argument("output_dir")
    p.add_argument("--signature", required=True,
                   help="comma-separated arg=type, e.g. a_ptr=*fp32,n=i32")
    p.add_argument("--constants", default="",
                   help="comma-separated name=value for tl.constexpr params")
    p.add_argument("--arch", type=int, default=89)
    p.add_argument("--num-warps", type=int, default=4)
    p.add_argument("--num-stages", type=int, default=3)
    args = p.parse_args()

    try:
        from triton.compiler import compile, ASTSource
        from triton.backends.compiler import GPUTarget
    except Exception as e:
        print(f"environment: cannot import triton: {e}", file=sys.stderr)
        return 2

    spec = importlib.util.spec_from_file_location("user_kernel", args.py_file)
    if spec is None or spec.loader is None:
        print(f"cannot load {args.py_file}", file=sys.stderr)
        return 2
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = getattr(mod, args.fn_name)

    signature = parse_kv(args.signature)
    constants_str = parse_kv(args.constants)
    constants = {k: int(v) if v.lstrip("-").isdigit() else v
                 for k, v in constants_str.items()}

    src = ASTSource(fn=fn, signature=signature, constants=constants)
    target = GPUTarget("cuda", args.arch, 32)
    opts = {"num_warps": args.num_warps, "num_stages": args.num_stages}

    print(f"# compiling {args.py_file}::{args.fn_name} -> sm_{args.arch}",
          file=sys.stderr)
    try:
        result = compile(src, target=target, options=opts)
    except Exception as e:
        print(f"triton compile failed: {type(e).__name__}: {e}",
              file=sys.stderr)
        return 1

    os.makedirs(args.output_dir, exist_ok=True)

    cubin_bytes: bytes = result.asm["cubin"]
    with open(os.path.join(args.output_dir, "kernel.cubin"), "wb") as f:
        f.write(cubin_bytes)

    ptx_text = result.asm.get("ptx")
    if ptx_text is not None:
        with open(os.path.join(args.output_dir, "kernel.ptx"), "w") as f:
            if isinstance(ptx_text, bytes):
                f.write(ptx_text.decode("utf-8", errors="replace"))
            else:
                f.write(ptx_text)

    md = result.metadata
    def get(field, default=None):
        return getattr(md, field, default)
    meta_out = {
        "name": get("name") or args.fn_name,
        "num_warps": get("num_warps", args.num_warps),
        "num_stages": get("num_stages", args.num_stages),
        "shared_mem": get("shared", 0),
        "target_arch": f"sm_{args.arch}",
        "cubin_bytes": len(cubin_bytes),
        "compiled_via": "python_triton_ast",
    }
    with open(os.path.join(args.output_dir, "kernel.json"), "w") as f:
        json.dump(meta_out, f, indent=2)

    print(f"# wrote {args.output_dir}/kernel.{{ptx,cubin,json}}", file=sys.stderr)
    print(f"# kernel name: {meta_out['name']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
