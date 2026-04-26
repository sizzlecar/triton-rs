"""Reference Python @triton.jit implementation of rms_norm_f32.

Matches the triton-rs DSL port's 6-arg signature for direct head-to-head:
  (input, weight, output, row_size: i32, inv_n: f32, eps: f32)

The caller precomputes `inv_n = 1.0 / row_size` so we don't need an
int->float cast inside the kernel — keeps the comparison apples-to-apples.
ferrum's hand-written .cu uses a 5-arg signature with the cast inline;
the bench's runner dispatches arg lists per kernel.
"""

import triton
import triton.language as tl


@triton.jit
def rms_norm_f32(input_ptr, weight_ptr, output_ptr,
                 row_size, inv_n, eps,
                 BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < row_size
    in_off = row * row_size + cols

    x = tl.load(input_ptr + in_off, mask=mask, other=0.0)
    sum_sq = tl.sum(x * x, axis=0)
    mean = sum_sq * inv_n
    inv_rms = 1.0 / tl.sqrt(mean + eps)

    w = tl.load(weight_ptr + cols, mask=mask)
    out = x * inv_rms * w
    tl.store(output_ptr + in_off, out, mask=mask)
