# tools/ — IR validation harness

Round-trip validation rig that proves our `triton-ir`-emitted MLIR is
accepted by an upstream Triton install. Used during Phase 2 to flush out
generic-form attribute requirements (e.g. `operandSegmentSizes`) and
dialect parser quirks (e.g. `tt.func` rejecting visibility keywords)
**before** we have the C ABI shim from Phase 1.

## Files

- `validate_mlir.py` — drives Triton's MLIR Python binding to parse a
  `.mlir` file and round-trip print it. Reports parse errors with file
  location.
- `vector_add.mlir` — output of `cargo run --example dump_vector_add
  -p triton-ir`. Regenerate with that command after IR changes.
- `vector_add.roundtrip.mlir` — what Triton emits after parsing our
  input. Useful as a "this is what canonical Triton MLIR looks like for
  the same kernel" reference, and to spot what attributes Triton elides
  vs. retains in custom form.

## How to run

Triton has no macOS pip wheel, so validation runs inside a Linux Docker
container.

```bash
# from triton-rs/ root
cargo run --example dump_vector_add -p triton-ir --quiet > tools/vector_add.mlir

docker run --rm --platform linux/amd64 \
  -v "$(pwd)/tools":/work -w /work \
  python:3.11-slim \
  bash -c "pip install --quiet triton==3.2.0 && python validate_mlir.py vector_add.mlir"
```

Exit codes from `validate_mlir.py`:

| code | meaning |
|---|---|
| 0 | parse + round-trip succeeded |
| 1 | parse failed — our IR has a real problem |
| 2 | environment problem (Triton import / API mismatch) |

## Fixes flushed out by this harness so far

1. **`tt.func public @name` → `tt.func @name`** — Triton's `tt.func`
   parser does not accept a visibility keyword. Upstream `test/Triton/`
   has 23 occurrences of `tt.func @`, zero of `tt.func public`.

2. **Variadic ops need `operandSegmentSizes`** — `tt.load` and
   `tt.store` have multiple variadic operand groups (ptr / mask /
   other for load; ptr / value / mask for store). In MLIR generic
   form this attribute is **mandatory** — without it the parser
   errors with "operand count (N) does not match the total size (0)
   specified in attribute 'operandSegmentSizes'".

   Encoded as `array<i32: A, B, C>` (a `DenseI32ArrayAttr`).

## Notes

- Triton 3.2.0 is the latest PyPI release (December 2024). The
  vendored backend in Phase 1 will pin v3.6.0 (a later GitHub-only
  release). The MLIR text format is stable enough that generated IR
  should be cross-compatible.
- This harness validates **parse + verify**, not codegen. To validate
  that the IR also lowers to PTX/cubin we need either GPU-equipped CI
  or the C ABI shim plus a CUDA driver.
