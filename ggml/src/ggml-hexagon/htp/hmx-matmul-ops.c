// HMX Matrix Multiplication Operations
// Adapted from htp-ops-lib for integration with llama.cpp's HVX backend

#include "hmx-matmul-ops.h"
#include "hmx-mgr.h"

#include <stddef.h>
#include <stdint.h>

#ifdef HTP_DEBUG
#    define FARF_HIGH 1
#endif
#include "HAP_farf.h"

// Stub implementation - HMX matmul will fall back to HVX for now
// Full HMX implementation requires proper HMX tile operations which 
// depend on target-specific intrinsics

int hmx_matmul_fp16_weight(float *restrict dst, const float *restrict activation,
                           const __fp16 *restrict weight, int m, int k, int n,
                           uint8_t *vtcm_base, size_t vtcm_size) {
    (void)dst; (void)activation; (void)weight;
    (void)m; (void)k; (void)n;
    (void)vtcm_base; (void)vtcm_size;
    
    // Return -1 to signal fallback to HVX implementation
    // Full HMX implementation can be added later
    FARF(HIGH, "HMX matmul not yet implemented, use HVX fallback");
    return -1;
}
