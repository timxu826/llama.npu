// HMX ops extension for ggml-hexagon
#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "rpcmem_mapper.h"
#include <memory>

#ifdef __cplusplus
extern "C" {
#endif

// Check if a buffer type is hexagon rpcmem
bool ggml_backend_buft_is_hexagon_rpcmem(ggml_backend_buffer_type_t buft);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus

static const char * HTP_OPS_DL_PATH = "libhtp_ops.so";

// Global HMX ops context (singleton)
struct ggml_hexagon_context {
    static constexpr size_t max_msg_size = 4096;

    // HMX ops backend library
    void * ops_dl_handle            = nullptr;
    bool   ops_backend_initialized  = false;
    void * ops_msg_chan             = nullptr;
    int    msg_chan_fd              = -1;
    bool   hmx_ops_enabled          = false;

    // shared rpcmem mapper for HMX ops
    std::unique_ptr<RpcMemMapper> hmx_mapper;

    ggml_hexagon_context();
    ~ggml_hexagon_context();

    int init_message_channel();

    static ggml_hexagon_context * instance();
};

// Global accessor for HMX context
inline ggml_hexagon_context * ggml_hexagon_get_context() {
    return ggml_hexagon_context::instance();
}

#endif
