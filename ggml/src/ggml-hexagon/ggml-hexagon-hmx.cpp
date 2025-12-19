#include "ggml-hexagon-hmx.h"
#include "dsprpc_interface.h"
#include "htp-ops.h"
#include "message.h"

#include <dlfcn.h>
#include <cstdlib>
#include <cstdio>
#include <mutex>
#include <atomic>
#include <dspqueue.h>
#include <AEEStdErr.h>

// HMX dspqueue response handling (non-static for access from htp-ops.cc)
std::atomic<int> hmx_pending_ops{0};
std::atomic<int> hmx_last_result{0};

// External function to get HVX session handle (defined in ggml-hexagon.cpp)
extern "C" uint64_t ggml_hexagon_get_session_handle();

// HMX dspqueue packet callback - called when DSP sends response
// Signature must match dspqueue_callback_t: void (*)(dspqueue_t, AEEResult, void*)
static void hmx_dspqueue_packet_callback(dspqueue_t queue, AEEResult status, void * context) {
    (void)status;
    auto * ctx = static_cast<ggml_hexagon_context *>(context);
    
    // Read response from queue
    uint8_t rsp_buf[64];
    uint32_t rsp_size = 0;
    uint32_t flags = 0;
    struct dspqueue_buffer bufs[4];
    uint32_t n_bufs = 4;
    
    int err = dspqueue_read_noblock(queue, &flags, 4, &n_bufs, bufs, sizeof(rsp_buf), &rsp_size, rsp_buf);
    if (err == 0 && rsp_size >= 4) {
        hmx_last_result.store(*reinterpret_cast<int32_t*>(rsp_buf));
    }
    
    hmx_pending_ops.fetch_sub(1);
}

// HMX dspqueue error callback
static void hmx_dspqueue_error_callback(dspqueue_t queue, AEEResult error, void * context) {
    (void)queue;
    (void)context;
    fprintf(stderr, "HMX dspqueue error: 0x%x\n", error);
    hmx_pending_ops.store(0);  // Reset pending ops on error
}

// HMX context implementation
ggml_hexagon_context::ggml_hexagon_context() {
    fprintf(stderr, "Initializing HMX ops support for Hexagon backend...\n");
    hmx_ops_enabled = false;
    return;  // FIXME: 完全跳过所有初始化

    // Check if HMX ops should be enabled
    const char * enable_hmx = std::getenv("GGML_HEXAGON_ENABLE_HMX");
    if (enable_hmx && (strcmp(enable_hmx, "1") == 0 || strcmp(enable_hmx, "true") == 0)) {
        hmx_ops_enabled = true;
    } else {
        fprintf(stderr, "HMX ops disabled. Set GGML_HEXAGON_ENABLE_HMX=1 to enable.\n");
        return;
    }

    // Initialize rpcmem mapper (3GB max, defer unmap)
    // Note: rpcmem_init is already called by HVX backend
    hmx_mapper = std::make_unique<RpcMemMapper>(3 * 1024UL * 1024 * 1024, true);

    // Try to load HMX ops library
    ops_dl_handle = dlopen(HTP_OPS_DL_PATH, RTLD_LAZY | RTLD_LOCAL);
    if (ops_dl_handle != nullptr) {
        using set_existing_session_fn_type = int(uint64_t);
        using init_htp_ops_fn_type = void();
        using create_hmx_dspqueue_fn_type = int(int, void*, void*, void*);

        auto set_existing_session = reinterpret_cast<set_existing_session_fn_type *>(dlsym(ops_dl_handle, "set_existing_session"));
        auto init_htp_ops = reinterpret_cast<init_htp_ops_fn_type *>(dlsym(ops_dl_handle, "init_htp_backend"));
        auto create_hmx_dspqueue = reinterpret_cast<create_hmx_dspqueue_fn_type *>(dlsym(ops_dl_handle, "create_hmx_dspqueue"));
        
        if (set_existing_session && init_htp_ops && create_hmx_dspqueue) {
            // Reuse HVX's existing session instead of opening a new one
            // This avoids conflicts between two FastRPC sessions to the same DSP domain
            uint64_t hvx_handle = ggml_hexagon_get_session_handle();
            
            if (hvx_handle == 0) {
                fprintf(stderr, "HMX: HVX session not available yet, deferring initialization\n");
                return;
            }
            
            int err = set_existing_session(hvx_handle);
            if (err == 0) {
                fprintf(stderr, "HMX: Using existing HVX session handle 0x%lx\n", (unsigned long)hvx_handle);
                // init_htp_ops();

                // Create dspqueue for HMX communication
                // err = create_hmx_dspqueue(
                //     CDSP_DOMAIN_ID,
                //     reinterpret_cast<void*>(hmx_dspqueue_packet_callback),
                //     reinterpret_cast<void*>(hmx_dspqueue_error_callback),
                //     this
                // );
                
                if (err == 0) {
                    ops_backend_initialized = true;
                    fprintf(stderr, "HMX ops backend initialized successfully with dspqueue!\n");
                } else {
                    fprintf(stderr, "Failed to create HMX dspqueue\n");
                }
            } else {
                fprintf(stderr, "Failed to set existing HVX session (0x%x)\n", err);
            }
        } else {
            fprintf(stderr, "Failed to find required symbols in HMX ops library\n");
            fprintf(stderr, "  set_existing_session=%p init_htp_backend=%p create_hmx_dspqueue=%p\n", 
                    (void*)set_existing_session, (void*)init_htp_ops, (void*)create_hmx_dspqueue);
        }
    } else {
        fprintf(stderr, "Cannot load HMX ops backend library (%s), all OPs will use HVX implementation\n", HTP_OPS_DL_PATH);
        fprintf(stderr, "dlerror: %s\n", dlerror());
    }
}

ggml_hexagon_context::~ggml_hexagon_context() {
    if (ops_dl_handle) {
        if (ops_backend_initialized) {
            // Close HMX dspqueue
            using close_hmx_dspqueue_fn = void();
            auto close_hmx_dspqueue = reinterpret_cast<close_hmx_dspqueue_fn *>(dlsym(ops_dl_handle, "close_hmx_dspqueue"));
            if (close_hmx_dspqueue) {
                close_hmx_dspqueue();
            }

            // NOTE: Don't close DSP session - we're reusing HVX's session
            
            ops_backend_initialized = false;
        }

        dlclose(ops_dl_handle);
    }
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
