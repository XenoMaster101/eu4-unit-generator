#!/usr/bin/env bash
# Root step (SPEC M21): glibc >= 2.41 declares sinpi/cospi (2.41) and rsqrt (2.43) noexcept; CUDA 12.9's
# crt/math_functions.h declares the same functions without it, so every .cu that includes <cmath> fails
# ("exception specification is incompatible"). Adds noexcept (true) to those six declarations (the fix
# distributions ship). Idempotent; keeps math_functions.h.orig.
#   sudo bash cuda_glibc_patch.sh [/usr/local/cuda-12.9]
set -euo pipefail
H=${1:-/usr/local/cuda-12.9}/include/crt/math_functions.h
[ -f "$H.orig" ] || cp "$H" "$H.orig"
sed -i -E 's/^(extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ (double|float) +(rsqrtf?|sinpif?|cospif?)\((double|float) x\));$/\1 noexcept (true);/' "$H"
grep -cE '(rsqrtf?|sinpif?|cospif?)\((double|float) x\) noexcept \(true\);' "$H"
