#!/bin/bash
# Compile a ferrum-style .cu kernel to PTX via nvcc.
#
# Usage: compile_cu.sh INPUT.cu OUTPUT_DIR [--arch 89]
#
# Writes OUTPUT_DIR/kernel.ptx and OUTPUT_DIR/kernel.json. The JSON's
# "name" field defaults to the basename — set $KERNEL_NAME to override.
set -euo pipefail

INPUT="${1:?usage: compile_cu.sh INPUT.cu OUTPUT_DIR [--arch N]}"
OUT="${2:?usage: compile_cu.sh INPUT.cu OUTPUT_DIR [--arch N]}"
ARCH=89
if [[ "${3:-}" == "--arch" ]]; then
    ARCH="$4"
fi

mkdir -p "$OUT"

# Use sm_$ARCH; nvcc emits PTX targeting that arch. The driver JIT-compiles
# at load time on a different SM (e.g. sm_120 Blackwell) — same forward-
# compatibility story we use for the Triton path.
nvcc -ptx -arch="sm_${ARCH}" --use_fast_math \
    -o "$OUT/kernel.ptx" "$INPUT"

KERNEL_NAME="${KERNEL_NAME:-$(basename "$INPUT" .cu)}"
cat > "$OUT/kernel.json" <<JSON
{
  "name": "${KERNEL_NAME}",
  "num_warps": 4,
  "num_stages": 0,
  "shared_mem": 0,
  "target_arch": "sm_${ARCH}",
  "compiled_via": "nvcc"
}
JSON

echo "# wrote $OUT/kernel.{ptx,json}  arch=sm_${ARCH}  kernel=${KERNEL_NAME}"
