#!/usr/bin/env python3
"""Round-trip validate a triton-ir-generated MLIR module against a real Triton install.

Reads MLIR text from a file, hands it to Triton's MLIR Python bindings to
parse + verify. On success, prints the round-tripped IR (with attribute
canonicalization applied by the parser, which surfaces any subtle text-format
mismatches our printer might have).

Exit codes:
  0 = parse + verify succeeded
  1 = parse or verify failed (validation found a real problem in our IR)
  2 = environment problem (Triton not importable, API mismatch, etc.)
"""

from __future__ import annotations

import sys
import traceback


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: validate_mlir.py FILE.mlir", file=sys.stderr)
        return 2

    path = sys.argv[1]
    try:
        text = open(path).read()
    except OSError as e:
        print(f"cannot read {path}: {e}", file=sys.stderr)
        return 2

    try:
        import triton  # noqa: F401
        from triton._C.libtriton import ir
    except Exception as e:
        print(f"environment: cannot import triton: {e}", file=sys.stderr)
        return 2

    print(f"# triton version: {triton.__version__}", file=sys.stderr)
    print(f"# input bytes:    {len(text)}", file=sys.stderr)
    print(f"# input lines:    {text.count(chr(10))}", file=sys.stderr)

    # Construct a context with all Triton-relevant dialects loaded.
    ctx = ir.context()
    try:
        ir.load_dialects(ctx)
    except AttributeError:
        # Older API: dialects auto-loaded inside parse functions.
        pass

    # Try the standard module parse entry. Different Triton versions expose
    # either parse_mlir_module(path, ctx) or parse_module(text, ctx).
    parse_attempts = [
        ("ir.parse_mlir_module(path, ctx)", lambda: ir.parse_mlir_module(path, ctx)),
        ("ir.parse_module(text, ctx)",       lambda: ir.parse_module(text, ctx)),
    ]

    last_err: Exception | None = None
    for label, fn in parse_attempts:
        if not _api_available(ir, label):
            continue
        try:
            mod = fn()
            print(f"# PARSE OK via {label}", file=sys.stderr)
            try:
                roundtrip = mod.str() if hasattr(mod, "str") else str(mod)
                print("# ===== round-trip MLIR =====")
                print(roundtrip)
                print("# ===========================")
            except Exception as e:
                print(f"# warning: round-trip print failed: {e}", file=sys.stderr)
            return 0
        except Exception as e:
            last_err = e
            print(f"# {label} -> FAILED:", file=sys.stderr)
            print(f"# {type(e).__name__}: {e}", file=sys.stderr)

    if last_err is None:
        print("# environment: no parse entry point found in this Triton build", file=sys.stderr)
        print(f"# available ir attributes: {sorted(a for a in dir(ir) if not a.startswith('_'))}",
              file=sys.stderr)
        return 2

    traceback.print_exception(type(last_err), last_err, last_err.__traceback__)
    return 1


def _api_available(ir_module, label: str) -> bool:
    name = label.split("(")[0].split(".")[-1]
    return hasattr(ir_module, name)


if __name__ == "__main__":
    sys.exit(main())
