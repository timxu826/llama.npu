// HMX Matrix Multiplication Operations
// Adapted from htp-ops-lib for integration with llama.cpp's HVX backend

#include "hmx-matmul-ops.h"
#include "hmx-mgr.h"

#include <hexagon_protos.h>
#include <hexagon_types.h>
#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef HTP_DEBUG
#    define FARF_ALWAYS 1
#endif
#include "HAP_farf.h"

// #ifdef __HMX__
// HMX is available - include full implementation
#include "hmx-dma.h"
#include "hvx-convert.h"
#include "hmx-utils.h"

#define VLEN 128

#define WEIGHT_AREA_SIZE     (1 * 1024 * 1024)
#define ACTIVATION_AREA_SIZE (1 * 1024 * 1024)
#define OUTPUT_AREA_SIZE     (1 * 1024 * 1024)

static inline size_t hmx_ceil_div(size_t num, size_t den) {
    return (num + den - 1) / den;
}

static inline size_t hmx_align_down(size_t v, size_t align) {
    return (v / align) * align;
}

static inline size_t hmx_smin(size_t a, size_t b) {
    return a < b ? a : b;
}

static inline int hmx_is_aligned(const void *ptr, size_t align) {
    return ((size_t)ptr & (align - 1)) == 0;
}

static inline uint8_t *hmx_vtcm_seq_alloc(uint8_t **vtcm_ptr, size_t size) {
    uint8_t *p = *vtcm_ptr;
    *vtcm_ptr += size;
    return p;
}

static void hmx_find_chunk_size(size_t x_max, size_t y_max, size_t xy_max,
                                size_t x_unit, size_t y_unit,
                                size_t *x_out, size_t *y_out) {
    int64_t best_xy = 0;
    size_t best_x = 0, best_y = 0;

    for (size_t x = x_max; x > 0; x -= x_unit) {
        size_t y = hmx_smin(hmx_align_down(xy_max / x, y_unit), y_max);
        int64_t xy = (int64_t)x * (int64_t)y;
        if (best_xy < xy) {
            best_xy = xy;
            best_x = x;
            best_y = y;
        }
    }
    *x_out = best_x;
    *y_out = best_y;
}

// Transfer activation chunk from FP32 (DDR) to FP16 tiled format (VTCM)
static void hmx_transfer_activation_fp32_to_fp16(__fp16 *restrict vtcm_dst,
                                                  const float *restrict src,
                                                  int n_rows, int k_block, int k_stride) {
    assert(k_block % HMX_FP16_TILE_N_COLS == 0);

    for (int r = 0; r < n_rows; r += 2) {
        int prefetch_row_idx = r + 2;
        if (prefetch_row_idx < n_rows) {
            const float *prefetch_addr = src + prefetch_row_idx * k_stride;
            Q6_l2fetch_AP((void *)prefetch_addr, Q6_R_combine_RlRl(k_stride * sizeof(float), k_block * sizeof(float)));
        }

        int r0 = r / HMX_FP16_TILE_N_ROWS;
        int r1 = r % HMX_FP16_TILE_N_ROWS;

        const bool next_row_valid = (r + 1) < n_rows;

        const HVX_Vector *pv_in0 = (const HVX_Vector *)(src + (r + 0) * k_stride);
        const HVX_Vector *pv_in1 = (const HVX_Vector *)(src + (r + 1) * k_stride);

        for (int c = 0; c < k_block; c += 32) {
            HVX_Vector v0 = *pv_in0++;
            HVX_Vector v1 = next_row_valid ? *pv_in1++ : Q6_V_vzero();

            HVX_Vector v_out = hvx_my_wsf_to_vhf(v1, v0);

            int c0 = c / HMX_FP16_TILE_N_COLS;
            int tile_idx = r0 * (k_block / HMX_FP16_TILE_N_COLS) + c0;

            HVX_Vector *tile = (HVX_Vector *)(vtcm_dst + tile_idx * HMX_FP16_TILE_N_ELMS);
            tile[r1 / 2] = v_out;
        }
    }
}

// Transfer FP16 weight chunk to VTCM using DMA
static void hmx_transfer_weight_fp16(__fp16 *vtcm_dst, const __fp16 *src, int n_cols, int k) {
    size_t size = n_cols * k * sizeof(__fp16);
    hmx_dma_load_sync(vtcm_dst, src, size);
}

// Core HMX dot product computation
static void hmx_core_dot_fp16(__fp16 *output, const __fp16 *activation,
                              const __fp16 *weight, const __fp16 *scales,
                              int n_row_tiles, int n_col_tiles, int n_dot_tiles) {
    hmx_unit_acquire();
    FARF(ALWAYS, "HMX core dot: n_row_tiles=%d n_col_tiles=%d n_dot_tiles=%d",
         n_row_tiles, n_col_tiles, n_dot_tiles);

    asm volatile("mxclracc.hf");
    FARF(ALWAYS, "HMX core dot: cleared accumulator");
    hmx_set_output_scales(scales);
    FARF(ALWAYS, "HMX core dot: set output scales");

    for (int r = 0; r < n_row_tiles; ++r) {
        for (int c = 0; c < n_col_tiles; ++c) {
            const __fp16 *row_tiles = activation + r * n_dot_tiles * HMX_FP16_TILE_N_ELMS;
            const __fp16 *col_tiles = weight + c * n_dot_tiles * HMX_FP16_TILE_N_ELMS;

            for (int kk = 0; kk < n_dot_tiles; kk += 32) {
                int offset = kk * HMX_FP16_TILE_N_ELMS;
                size_t n_tiles = hmx_smin(n_dot_tiles - kk, 32);
                hmx_load_tiles_fp16(row_tiles + offset, col_tiles + offset, n_tiles);
                FARF(ALWAYS, "HMX core dot: loaded tiles kk=%d n_tiles=%zu", kk, n_tiles);
            }

            __fp16 *out_tile = output + (r * n_col_tiles + c) * HMX_FP16_TILE_N_ELMS;
            hmx_consume_accumulator_fp16(out_tile);
            FARF(ALWAYS, "HMX core dot: consumed accumulator for tile (%d, %d)", r, c);
        }
    }

    hmx_unit_release();
}

// Transfer output from FP16 tiled format (VTCM) to FP32 (DDR)
static void hmx_transfer_output_fp16_to_fp32(float *restrict dst,
                                             const __fp16 *restrict vtcm_src,
                                             int n_rows, int n_cols, int n) {
    assert(n_cols % HMX_FP16_TILE_N_COLS == 0);
    const int n_col_tiles = n_cols / HMX_FP16_TILE_N_COLS;

    for (int r = 0; r < n_rows; r += 2) {
        int r0 = r / HMX_FP16_TILE_N_ROWS;
        int r1 = r % HMX_FP16_TILE_N_ROWS;

        for (int c = 0; c < n_cols; c += HMX_FP16_TILE_N_COLS) {
            int c0 = c / HMX_FP16_TILE_N_COLS;

            const __fp16 *tile = vtcm_src + (r0 * n_col_tiles + c0) * HMX_FP16_TILE_N_ELMS;
            HVX_Vector v_src = ((const HVX_Vector *)tile)[r1 / 2];

            HVX_VectorPair vp = hvx_my_vhf_to_wsf(v_src);

            HVX_Vector *pv_out0 = (HVX_Vector *)(dst + (r * n + c + 0));
            HVX_Vector *pv_out1 = (HVX_Vector *)(dst + (r * n + c + n));

            *pv_out0 = Q6_V_lo_W(vp);
            if (r + 1 < n_rows) {
                *pv_out1 = Q6_V_hi_W(vp);
            }
        }
    }
}

// Main HMX matmul function for FP16 weights
// dst[m,n] = activation[m,k] * weight[k,n]^T
// Weight is pre-permuted to [n,k] in tiled layout
int hmx_matmul_fp16_weight(float *restrict dst, const float *restrict activation,
                           const __fp16 *restrict weight, int m, int k, int n,
                           uint8_t *vtcm_base, size_t vtcm_size) {
    (void)vtcm_size;

    if (!dst || !activation || !weight || !m || !n || !k) {
        FARF(ALWAYS, "HMX matmul: invalid parameters");
        return -1;
    }
    if (k % 32 != 0 || n % 32 != 0) {
        FARF(ALWAYS, "HMX matmul: k(%d) and n(%d) must be multiples of 32", k, n);
        return -1;
    }
    if (!hmx_is_aligned(dst, VLEN) || !hmx_is_aligned(activation, VLEN) ||
        !hmx_is_aligned(weight, VLEN)) {
        FARF(ALWAYS, "HMX matmul: buffers must be 128-byte aligned");
        return -1;
    }
    if (!hmx_is_available()) {
        FARF(ALWAYS, "HMX matmul: HMX not available");
        return -1;
    }

    // Allocate VTCM areas
    uint8_t *vtcm_ptr = vtcm_base;
    __fp16 *vtcm_weight = (__fp16 *)hmx_vtcm_seq_alloc(&vtcm_ptr, WEIGHT_AREA_SIZE);
    __fp16 *vtcm_activation = (__fp16 *)hmx_vtcm_seq_alloc(&vtcm_ptr, ACTIVATION_AREA_SIZE);
    __fp16 *vtcm_output = (__fp16 *)hmx_vtcm_seq_alloc(&vtcm_ptr, OUTPUT_AREA_SIZE);
    __fp16 *vtcm_scales = (__fp16 *)hmx_vtcm_seq_alloc(&vtcm_ptr, 256);

    // Initialize column scales to 1.0 (FP16: 0x3c00)
    hmx_init_column_scales(vtcm_scales, Q6_V_vsplat_R(0x3c00));

    // Calculate chunk sizes
    size_t vec_dot_size = k * sizeof(__fp16);
    size_t m_chunk_max_n_rows = hmx_align_down(ACTIVATION_AREA_SIZE / vec_dot_size, HMX_FP16_TILE_N_ROWS);
    size_t n_chunk_max_n_cols = hmx_align_down(WEIGHT_AREA_SIZE / vec_dot_size, HMX_FP16_TILE_N_COLS);

    size_t m_chunk_n_rows = 0, n_chunk_n_cols = 0;
    hmx_find_chunk_size(m_chunk_max_n_rows, n_chunk_max_n_cols,
                        OUTPUT_AREA_SIZE / sizeof(__fp16),
                        HMX_FP16_TILE_N_ROWS, HMX_FP16_TILE_N_COLS,
                        &m_chunk_n_rows, &n_chunk_n_cols);

    FARF(ALWAYS, "computed chunk size: %d, %d", m_chunk_n_rows, n_chunk_n_cols);
    if (m_chunk_n_rows == 0 || n_chunk_n_cols == 0) {
        FARF(ALWAYS, "HMX matmul: cannot find valid chunk size");
        return -1;
    }

    FARF(ALWAYS, "HMX matmul: m=%d k=%d n=%d chunk_m=%zu chunk_n=%zu",
         m, k, n, m_chunk_n_rows, n_chunk_n_cols);

    // Enable HMX execution for this thread
    hmx_manager_enable_execution();

    // Main matmul loop
    for (size_t mr = 0; mr < (size_t)m; mr += m_chunk_n_rows) {
        size_t n_rows = hmx_smin((size_t)m - mr, m_chunk_n_rows);

        // Transfer activation chunk to VTCM
        const float *activation_chunk = activation + mr * k;
        hmx_transfer_activation_fp32_to_fp16(vtcm_activation, activation_chunk, n_rows, k, k);

        FARF(ALWAYS, "transfer activation ok, mr = %d, n_rows = %d", mr, n_rows);
        for (size_t nc = 0; nc < (size_t)n; nc += n_chunk_n_cols) {
            size_t n_cols = hmx_smin((size_t)n - nc, n_chunk_n_cols);

            // Transfer weight chunk to VTCM
            const __fp16 *weight_chunk = weight + nc * k;
            hmx_transfer_weight_fp16(vtcm_weight, weight_chunk, n_cols, k);
            FARF(ALWAYS, "transfer weight ok, nc = %d, n_cols = %d", nc, n_cols);

            // Compute matmul
            const int n_row_tiles = hmx_ceil_div(n_rows, HMX_FP16_TILE_N_ROWS);
            const int n_col_tiles = hmx_ceil_div(n_cols, HMX_FP16_TILE_N_COLS);
            FARF(ALWAYS, "core hmx_ceil_div ok, (%d, %d) tiles", n_rows, n_cols);
            hmx_core_dot_fp16(vtcm_output, vtcm_activation, vtcm_weight,
                              vtcm_scales, n_row_tiles, n_col_tiles, k / 32);
            FARF(ALWAYS, "core compute ok, (%d, %d) tiles", n_row_tiles, n_col_tiles);

            // Transfer output to DDR
            float *output = dst + (mr * n + nc);
            hmx_transfer_output_fp16_to_fp32(output, vtcm_output, n_rows, n_cols, n);
            FARF(ALWAYS, "transfer output ok, (%d, %d)", mr, nc);
        }
    }

    hmx_manager_disable_execution();

    return 0;
}

// #else /* !__HMX__ */

// // HMX not available on this target - provide stub that returns -1 to trigger fallback
// int hmx_matmul_fp16_weight(float *restrict dst, const float *restrict activation,
//                            const __fp16 *restrict weight, int m, int k, int n,
//                            uint8_t *vtcm_base, size_t vtcm_size) {
//     (void)dst; (void)activation; (void)weight;
//     (void)m; (void)k; (void)n;
//     (void)vtcm_base; (void)vtcm_size;

//     FARF(ALWAYS, "HMX matmul: HMX not supported on this target");
//     return -1;
// }

// #endif /* __HMX__ */
