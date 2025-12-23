#include "ggml-hexagon-hmx.h"

#include <cstdlib>
#include <cstdio>
#include <mutex>
#include <cstring>

// HMX context implementation
// HMX now uses the same dspqueue as HVX - no separate library loading needed.
// HMX operations are sent via dspqueue with HTP_OP_HMX_* op codes.
// The DSP-side htp library handles HMX initialization and operation dispatch.

ggml_hexagon_context::ggml_hexagon_context() {
    fprintf(stderr, "Initializing HMX ops support for Hexagon backend...\n");
    
    // Check if HMX ops should be enabled via environment variable
    const char * enable_hmx = std::getenv("GGML_HEXAGON_ENABLE_HMX");
    if (enable_hmx && (strcmp(enable_hmx, "1") == 0 || strcmp(enable_hmx, "true") == 0)) {
        hmx_ops_enabled = true;
        fprintf(stderr, "HMX ops enabled via GGML_HEXAGON_ENABLE_HMX environment variable\n");
    } else {
        hmx_ops_enabled = false;
        fprintf(stderr, "HMX ops disabled. Set GGML_HEXAGON_ENABLE_HMX=1 to enable.\n");
    }

    // HMX is now integrated into the HVX backend's DSP library (libggml-htp-vXX.so)
    // No separate library loading or session management needed.
    // The DSP side will automatically initialize HMX when available.
    ops_backend_initialized = hmx_ops_enabled;
}

ggml_hexagon_context::~ggml_hexagon_context() {
    // HMX cleanup is handled by the HVX backend's DSP library
    // No separate cleanup needed on host side
    ops_backend_initialized = false;
}

ggml_hexagon_context * ggml_hexagon_context::instance() {
    static std::unique_ptr<ggml_hexagon_context> ctx_ptr;
    static std::once_flag                         ctx_once_flag;

    std::call_once(ctx_once_flag, [&] {
        auto * ctx = new ggml_hexagon_context;
        ctx_ptr.reset(ctx);
    });
    return ctx_ptr.get();
}

// Buffer type check for rpcmem buffers
extern "C" {
bool ggml_backend_buft_is_hexagon_rpcmem(ggml_backend_buffer_type_t buft) {
    // For now, assume all hexagon buffers can be used with rpcmem
    // This may need refinement based on actual buffer type
    if (buft == nullptr) {
        return false;
    }
    const char * name = ggml_backend_buft_name(buft);
    return name != nullptr && 
           (strstr(name, "HTP") != nullptr ||      // Hexagon HTP buffers (HTP0, HTP1, HTP0-REPACK, etc.)
            strstr(name, "Hexagon") != nullptr ||  // Alternative naming
            strstr(name, "RPCMEM") != nullptr);    // Explicit rpcmem naming
}
}
