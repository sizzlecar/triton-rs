"""Reference Python @triton.jit implementation of fused_silu_mul_f32.

Same signature as the triton-rs DSL port and ferrum's hand-written .cu:
  (gate, up, output, n).
"""

import triton
import triton.language as tl


@triton.jit
def fused_silu_mul_f32(gate_ptr, up_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    mask = off < n
    g = tl.load(gate_ptr + off, mask=mask)
    u = tl.load(up_ptr + off, mask=mask)
    silu_g = g / (1.0 + tl.exp(-g))
    tl.store(out_ptr + off, silu_g * u, mask=mask)
