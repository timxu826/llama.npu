#ifndef HMX_UTILS_H
#define HMX_UTILS_H

#include <hexagon_types.h>
#include <stddef.h>
#include <stdint.h>

// HMX is available on V75+
#if defined(__HMX__) || defined(HEXAGON_ARCH_75) || defined(HEXAGON_ARCH_79) || defined(HEXAGON_ARCH_81)
#include "hmx-hexagon-protos.h"
#endif

#define HMX_FP16_TILE_N_ROWS 32
#define HMX_FP16_TILE_N_COLS 32
#define HMX_FP16_TILE_N_ELMS 1024
#define HMX_FP16_TILE_SIZE   2048

#define HMX_UB_TILE_N_ROWS 32
#define HMX_UB_TILE_N_COLS 32
#define HMX_UB_TILE_N_ELMS 1024
#define HMX_UB_TILE_SIZE   1024

#define HMX_INLINE_ALWAYS inline __attribute__((unused, always_inline))

// HMX is available on V75+
#if defined(__HMX__) || defined(HEXAGON_ARCH_75) || defined(HEXAGON_ARCH_79) || defined(HEXAGON_ARCH_81)

static HMX_INLINE_ALWAYS void hmx_set_output_scales(const void *scales) {
  asm volatile("bias = mxmem2(%0)" ::"r"(scales));
}

static HMX_INLINE_ALWAYS void hmx_init_column_scales(void *out_scales, HVX_Vector v_scale) {
  HVX_Vector *pv = (HVX_Vector *) out_scales;
  *pv++ = v_scale;
  *pv   = Q6_V_vzero();
}

static HMX_INLINE_ALWAYS void hmx_load_tiles_fp16(const __fp16 *row_tiles, const __fp16 *col_tiles, size_t n_tiles) {
  size_t limit = n_tiles * HMX_FP16_TILE_SIZE - 1;
  asm volatile(
    "{ activation.hf = mxmem(%0, %1):deep\n"
    "weight.hf = mxmem(%2, %3) }\n" ::"r"(row_tiles),
    "r"(limit), "r"(col_tiles), "r"(limit)
    : "memory");
}

static HMX_INLINE_ALWAYS void hmx_load_tiles_ub(const uint8_t *row_tiles, const uint8_t *col_tiles, size_t n_tiles) {
  size_t limit = n_tiles * HMX_UB_TILE_SIZE - 1;
  asm volatile(
    "{ activation.ub = mxmem(%0, %1):deep\n"
    "weight.ubit = mxmem(%2, %3) }\n" ::"r"(row_tiles),
    "r"(limit), "r"(col_tiles), "r"(limit)
    : "memory");
}

static HMX_INLINE_ALWAYS void hmx_consume_accumulator_fp16(__fp16 *out) {
  asm volatile(
    "cvt.hf = acc(%0)\n"
    "mxmem(%1, %2) = cvt\n" ::"r"(2),
    "r"(out), "r"(0)
    : "memory");
}

static HMX_INLINE_ALWAYS void hmx_consume_accumulator_ub(uint8_t *out) {
  asm volatile(
    "cvt.ub = acc(%0)\n"
    "mxmem(%1, %2) = cvt\n" ::"r"(2),
    "r"(out), "r"(0)
    : "memory");
}

static HMX_INLINE_ALWAYS void hmx_dot_fp16(__fp16 *out, const __fp16 *row_tiles, const __fp16 *col_tiles,
                                           size_t n_tiles) {
  hmx_load_tiles_fp16(row_tiles, col_tiles, n_tiles);
  hmx_consume_accumulator_fp16(out);
}

#endif /* __HMX__ */

#endif /* HMX_UTILS_H */
