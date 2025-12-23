#ifndef HMX_MATMUL_OPS_H
#define HMX_MATMUL_OPS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// HMX Matrix Multiplication for FP16 weights
// dst[m,n] = activation[m,k] * weight[k,n]^T
// Weight must be pre-permuted to [n,k] in HMX tiled layout
// Returns 0 on success, -1 on failure
int hmx_matmul_fp16_weight(float *restrict dst, const float *restrict activation,
                           const __fp16 *restrict weight, int m, int k, int n,
                           uint8_t *vtcm_base, size_t vtcm_size);

#ifdef __cplusplus
}
#endif

#endif // HMX_MATMUL_OPS_H
