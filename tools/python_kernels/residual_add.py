"""Reference Python @triton.jit implementation of residual_add_f32.

This kernel is the head-to-head comparator for the triton-rs DSL port —
both should compile through Triton's same MLIR pipeline with the same
launch shape (BLOCK=1024, num_warps=4) and produce ~identical PTX.

Compiled standalone via tools/compile_python_kernel.py.
"""

import triton
import triton.language as tl


@triton.jit
def residual_add_f32(a_ptr, b_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    mask = off < n
    av = tl.load(a_ptr + off, mask=mask)
    bv = tl.load(b_ptr + off, mask=mask)
    tl.store(out_ptr + off, av + bv, mask=mask)
