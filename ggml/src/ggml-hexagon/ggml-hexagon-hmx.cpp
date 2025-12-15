#include "ggml-hexagon-hmx.h"
#include "dsprpc_interface.h"
#include "htp-ops.h"
#include "message.h"

#include <dlfcn.h>
#include <cstdlib>
#include <cstdio>
#include <mutex>

// HMX context implementation
ggml_hexagon_context::ggml_hexagon_context() {
    fprintf(stderr, "Initializing HMX ops support for Hexagon backend...\n");

    // Check if HMX ops should be enabled
    const char * enable_hmx = std::getenv("GGML_HEXAGON_ENABLE_HMX");
    if (enable_hmx && (strcmp(enable_hmx, "1") == 0 || strcmp(enable_hmx, "true") == 0)) {
        hmx_ops_enabled = true;
    } else {
        fprintf(stderr, "HMX ops disabled. Set GGML_HEXAGON_ENABLE_HMX=1 to enable.\n");
        return;
    }

    // Initialize rpcmem
    rpcmem_init();

    // Initialize rpcmem mapper (3GB max, defer unmap)
    hmx_mapper = std::make_unique<RpcMemMapper>(3 * 1024UL * 1024 * 1024, true);

    // Try to load HMX ops library
    ops_dl_handle = dlopen(HTP_OPS_DL_PATH, RTLD_LAZY | RTLD_LOCAL);
    if (ops_dl_handle != nullptr) {
        using open_session_fn_type = int(int, int);
        using init_htp_ops_fn_type = void();

        auto open_session = reinterpret_cast<open_session_fn_type *>(dlsym(ops_dl_handle, "open_dsp_session"));
        auto init_htp_ops = reinterpret_cast<init_htp_ops_fn_type *>(dlsym(ops_dl_handle, "init_htp_backend"));
        
        if (open_session && init_htp_ops) {
            int err = open_session(CDSP_DOMAIN_ID, 1);
            if (err == 0) {
                init_htp_ops();

                if (init_message_channel() == 0) {
                    ops_backend_initialized = true;
                    fprintf(stderr, "HMX ops backend initialized successfully!\n");
                } else {
                    fprintf(stderr, "Failed to initialize HMX message channel\n");
                }
            } else {
                fprintf(stderr, "Failed to open remote session on Hexagon NPU (0x%x)\n", err);
            }
        } else {
            fprintf(stderr, "Failed to find required symbols in HMX ops library\n");
        }
    } else {
        fprintf(stderr, "Cannot load HMX ops backend library (%s), all OPs will use HVX implementation\n", HTP_OPS_DL_PATH);
        fprintf(stderr, "dlerror: %s\n", dlerror());
    }
}

ggml_hexagon_context::~ggml_hexagon_context() {
    if (ops_dl_handle) {
        if (ops_backend_initialized) {
            using close_session_fn = void();

            auto close_session = reinterpret_cast<close_session_fn *>(dlsym(ops_dl_handle, "close_dsp_session"));
            if (close_session) {
                close_session();
            }
            
            // release message channel
            if (ops_msg_chan && msg_chan_fd >= 0) {
                fastrpc_munmap(CDSP_DOMAIN_ID, msg_chan_fd, ops_msg_chan, max_msg_size);
                rpcmem_free(ops_msg_chan);
            }
            ops_backend_initialized = false;
        }

        dlclose(ops_dl_handle);
    }

    rpcmem_deinit();
}

int ggml_hexagon_context::init_message_channel() {
    using create_msg_channel_fn_type = int(int, unsigned int);

    auto create_msg_channel =
        reinterpret_cast<create_msg_channel_fn_type *>(dlsym(ops_dl_handle, "create_htp_message_channel"));
    if (!create_msg_channel) {
        return -1;
    }

    ops_msg_chan = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_FLAG_UNCACHED, max_msg_size);
    if (!ops_msg_chan) {
        return -1;
    }

    msg_chan_fd = rpcmem_to_fd(ops_msg_chan);
    if (msg_chan_fd < 0) {
        return -1;
    }

    int err = fastrpc_mmap(CDSP_DOMAIN_ID, msg_chan_fd, ops_msg_chan, 0, max_msg_size, FASTRPC_MAP_FD);
    if (err) {
        return -1;
    }

    return create_msg_channel(msg_chan_fd, max_msg_size);
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
    return buft != nullptr && 
           (strstr(ggml_backend_buft_name(buft), "Hexagon") != nullptr ||
            strstr(ggml_backend_buft_name(buft), "RPCMEM") != nullptr);
}
}
