// HMX ops extension for ggml-hexagon
// HMX is now integrated into the HVX backend - uses same dspqueue with HTP_OP_HMX_* opcodes
#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include <memory>

#ifdef __cplusplus
extern "C" {
#endif

// Check if a buffer type is hexagon rpcmem
bool ggml_backend_buft_is_hexagon_rpcmem(ggml_backend_buffer_type_t buft);

// Check if a hexagon buffer is already mapped to DSP
bool ggml_backend_hexagon_buffer_is_mapped(ggml_backend_buffer_t buffer);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus

// Global HMX ops context (singleton)
// HMX is now integrated into HVX backend - no separate library loading needed
struct ggml_hexagon_context {
    bool   ops_backend_initialized  = false;
    bool   hmx_ops_enabled          = false;

    ggml_hexagon_context();
    ~ggml_hexagon_context();

    static ggml_hexagon_context * instance();
};

// Global accessor for HMX context
inline ggml_hexagon_context * ggml_hexagon_get_context() {
    return ggml_hexagon_context::instance();
}

#endif
