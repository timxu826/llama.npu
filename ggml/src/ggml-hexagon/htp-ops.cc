#include "htp-ops.h"

#include <dlfcn.h>
#include <unistd.h>

#include <atomic>
#include <cstring>
#include <vector>

#include "ggml-backend-impl.h"
#include "ggml-hexagon.h"
#include "ggml-hexagon-hmx.h"

////// Special headers intended for CPU-NPU communication. Keep them in sync with ops backend.
#include "message.h"
#include "op_reg.h"
#include "dsprpc_interface.h"
#include "rpcmem_mapper.h"

// External atomic variables for HMX dspqueue response handling
extern std::atomic<int> hmx_pending_ops;
extern std::atomic<int> hmx_last_result;

namespace {

auto get_all_rpcmem_mappings(const ggml_tensor * dst, ggml_hexagon_context * ctx) {
    std::vector<std::pair<int, ssize_t>> mappings;
    
    auto is_rpcmem = [](const ggml_tensor * t) -> bool {
        return t && t->buffer && ggml_backend_buft_is_hexagon_rpcmem(t->buffer->buft);
    };
    
    if (is_rpcmem(dst) && ctx->hmx_mapper) {
        mappings.push_back(ctx->hmx_mapper->get_tensor_mapping(dst));
    }
    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        auto * src = dst->src[i];
        if (is_rpcmem(src) && ctx->hmx_mapper) {
            mappings.push_back(ctx->hmx_mapper->get_tensor_mapping(src));
        }
    }
    return mappings;
}

template <typename T> void write_buf(uint8_t *& p, const T & v) {
    *reinterpret_cast<T *>(p) = v;
    p += sizeof(v);
}

void write_buf(uint8_t *& p, void * src, size_t size) {
    std::memcpy((void *) p, src, size);
    p += size;
}

uint8_t param_buf[4096];  // TODO: better implementation

}  // namespace

extern "C" {

bool htp_ops_support_op(const struct ggml_tensor * dst) {
    auto * ctx = ggml_hexagon_get_context();
    if (!ctx || !ctx->hmx_ops_enabled) {
        return false;
    }
    if (!ctx->ops_backend_initialized) {
        return false;
    }

    void * ops_dl_handle = ctx->ops_dl_handle;
    GGML_ASSERT(ops_dl_handle);

    switch (dst->op) {
        case GGML_OP_RMS_NORM:
            return false;

            if (dst->type == GGML_TYPE_F32 && dst->src[0]->type == GGML_TYPE_F32) {
                // NOTE: RPC version is mainly for testing
                return dlsym(ops_dl_handle, "htp_ops_rpc_rms_norm_f32") != nullptr;
            }
            return false;
        case GGML_OP_MUL_MAT:
            {
                auto * weight     = dst->src[0];
                auto * activation = dst->src[1];

                size_t k = weight->ne[0];
                size_t n = weight->ne[1];

                bool shape_ok = k % 32 == 0 && n % 32 == 0 && ggml_nrows(dst) == dst->ne[1] &&
                                ggml_nrows(activation) == activation->ne[1];

                // FP16 weight
                if (dst->type == GGML_TYPE_F32 && weight->type == GGML_TYPE_F16 && activation->type == GGML_TYPE_F32) {
                    return shape_ok;
                }
                // (repacked) Q4_0 weight
                if (dst->type == GGML_TYPE_F32 && weight->type == GGML_TYPE_Q4_0 && activation->type == GGML_TYPE_F32) {
                    return shape_ok;
                }
                // (repacked) Q8_0 weight
                if (dst->type == GGML_TYPE_F32 && weight->type == GGML_TYPE_Q8_0 && activation->type == GGML_TYPE_F32) {
                    return shape_ok;
                }
                // (repacked) IQ4_NL weight
                if (dst->type == GGML_TYPE_F32 && weight->type == GGML_TYPE_IQ4_NL &&
                    activation->type == GGML_TYPE_F32) {
                    return shape_ok;
                }
                fprintf(stderr, "unsupported matmul: dst: %s weight: %s act: %s\n", ggml_type_name(dst->type),
                        ggml_type_name(weight->type), ggml_type_name(activation->type));
                return false;
            }
        case GGML_OP_FLASH_ATTN_EXT:
            {
                float scale         = *reinterpret_cast<const float *>(&dst->op_params[0]);
                float max_bias      = *reinterpret_cast<const float *>(&dst->op_params[1]);
                float logit_softcap = *reinterpret_cast<const float *>(&dst->op_params[2]);

                auto * q    = dst->src[0];
                auto * k    = dst->src[1];
                auto * v    = dst->src[2];
                auto * mask = dst->src[3];

                return dst->type == GGML_TYPE_F32 && q->type == GGML_TYPE_F32 && k->type == GGML_TYPE_F16 &&
                       v->type == GGML_TYPE_F16 && mask->type == GGML_TYPE_F16 && max_bias == 0 && logit_softcap == 0;
            }
        default:
            return false;
    }
}

int htp_ops_compute_op(struct ggml_tensor * dst) {
    prepare_tensor_rpcmem_mapping(dst);

    auto * ctx           = ggml_hexagon_get_context();
    void * ops_dl_handle = ctx->ops_dl_handle;
    GGML_ASSERT(ops_dl_handle);

    constexpr bool prefer_rpc = false;

    int op_index  = -1;
    int args_size = 0;  // strictly 32 bits

    switch (dst->op) {
        case GGML_OP_RMS_NORM:
            {
                auto mappings = get_all_rpcmem_mappings(dst, ctx);
                GGML_ASSERT(mappings.size() == 2);

                auto [dst_fd, dst_offset] = mappings[0];
                auto [src_fd, src_offset] = mappings[1];

                if (prefer_rpc) {
                    using fn_type = int(int, int, int, int, int, int);

                    auto op_fn = reinterpret_cast<fn_type *>(dlsym(ops_dl_handle, "htp_ops_rpc_rms_norm_f32"));
                    GGML_ASSERT(op_fn);

                    return op_fn(dst_fd, dst_offset, src_fd, src_offset, dst->ne[0], ggml_nrows(dst));
                }

                RmsNormF32Params params{
                    .dst = { dst_fd, (int32_t) dst_offset },
                    .src = { src_fd, (int32_t) src_offset },
                    .ne0 = (int32_t) dst->ne[0],
                    .ne1 = (int32_t) ggml_nrows(dst),
                };
                *reinterpret_cast<RmsNormF32Params *>(param_buf) = params;

                op_index  = HTP_OPS_RMS_NORM_F32;
                args_size = sizeof(RmsNormF32Params);
            }
            break;

        case GGML_OP_MUL_MAT:
            {
                auto * weight     = dst->src[0];
                auto * activation = dst->src[1];

                auto mappings = get_all_rpcmem_mappings(dst, ctx);
                GGML_ASSERT(mappings.size() == 3);

                auto [output_fd, output_offset]         = mappings[0];
                auto [weight_fd, weight_offset]         = mappings[1];
                auto [activation_fd, activation_offset] = mappings[2];

                int m = ggml_nrows(activation);
                int k = weight->ne[0];
                int n = weight->ne[1];

                MatMulParams params{
                    .output     = { output_fd,     (int32_t) output_offset     },
                    .activation = { activation_fd, (int32_t) activation_offset },
                    .weight     = { weight_fd,     (int32_t) weight_offset     },
                    .m          = m,
                    .k          = k,
                    .n          = n,
                };
                *reinterpret_cast<MatMulParams *>(param_buf) = params;

                args_size = sizeof(MatMulParams);

                if (dst->type == GGML_TYPE_F32 && weight->type == GGML_TYPE_F16 && activation->type == GGML_TYPE_F32) {
                    if (prefer_rpc) {
                        using fn_type = int(int, int, int, int, int, int, int, int, int);

                        auto op_fn =
                            reinterpret_cast<fn_type *>(dlsym(ops_dl_handle, "htp_ops_rpc_mat_mul_permuted_w16a32"));
                        GGML_ASSERT(op_fn);

                        return op_fn(output_fd, output_offset, activation_fd, activation_offset, weight_fd,
                                     weight_offset, m, k, n);
                    }

                    op_index = HTP_OPS_MAT_MUL_PERMUTED_W16A32;
                } else if (dst->type == GGML_TYPE_F32 && weight->type == GGML_TYPE_Q4_0 &&
                           activation->type == GGML_TYPE_F32) {
                    op_index = HTP_OPS_MAT_MUL_PERMUTED_W4D16A32;
                } else if (dst->type == GGML_TYPE_F32 && weight->type == GGML_TYPE_Q8_0 &&
                           activation->type == GGML_TYPE_F32) {
                    op_index = HTP_OPS_MAT_MUL_PERMUTED_W8D16A32;
                } else if (dst->type == GGML_TYPE_F32 && weight->type == GGML_TYPE_IQ4_NL &&
                           activation->type == GGML_TYPE_F32) {
                    op_index = HTP_OPS_MAT_MUL_PERMUTED_W4D16A32_IQ4_NL;
                } else {
                    GGML_ASSERT(false && "not implemented");
                }
            }
            break;

        case GGML_OP_FLASH_ATTN_EXT:
            {
                auto * q    = dst->src[0];
                auto * k    = dst->src[1];
                auto * v    = dst->src[2];
                auto * mask = dst->src[3];

                auto mappings = get_all_rpcmem_mappings(dst, ctx);
                GGML_ASSERT(mappings.size() == 5);

                auto [o_fd, o_offset]       = mappings[0];
                auto [q_fd, q_offset]       = mappings[1];
                auto [k_fd, k_offset]       = mappings[2];
                auto [v_fd, v_offset]       = mappings[3];
                auto [mask_fd, mask_offset] = mappings[4];

                int head_dim   = q->ne[0];
                int qo_len     = q->ne[1];
                int kv_len     = k->ne[1];
                int n_heads    = q->ne[2];
                int n_kv_heads = k->ne[2];

                FlashAttnParams params{
                    .o          = { o_fd,    (int32_t) o_offset    },
                    .q          = { q_fd,    (int32_t) q_offset    },
                    .k          = { k_fd,    (int32_t) k_offset    },
                    .v          = { v_fd,    (int32_t) v_offset    },
                    .mask       = { mask_fd, (int32_t) mask_offset },
                    .qo_len     = qo_len,
                    .kv_len     = kv_len,
                    .n_heads    = n_heads,
                    .n_kv_heads = n_kv_heads,
                    .head_dim   = head_dim,
                };
                *reinterpret_cast<FlashAttnParams *>(param_buf) = params;

                op_index  = HTP_OPS_FLASH_ATTN_QO_F32_KV_F16;
                args_size = sizeof(FlashAttnParams);
            }
            break;

        default:
            break;
    }

    // Build packet for dspqueue
    // Packet format: HmxPacketHeader + RequestHeader + OpComputeRequest + params
    
    struct HmxPacketHeader {
        int32_t n_reqs;
        int32_t reserved;
    } __attribute__((packed));
    
    uint8_t packet_buf[4096];
    uint8_t * p = packet_buf;
    
    // Write packet header
    HmxPacketHeader pkt_hdr{
        .n_reqs = 1,
        .reserved = 0,
    };
    write_buf(p, pkt_hdr);
    
    // Write request header
    RequestHeader req_hdr{
        .state = 0,
        .type  = REQUEST_TYPE_OP_COMPUTE,
    };
    write_buf(p, req_hdr);
    
    // Write op compute request
    OpComputeRequest op_req{
        .op = (uint32_t) op_index,
    };
    write_buf(p, op_req);
    
    // Write op parameters
    write_buf(p, param_buf, args_size);
    
    size_t packet_size = p - packet_buf;
    
    // Get dspqueue handle from library
    using get_hmx_dspqueue_fn_type = void*();
    auto get_hmx_dspqueue = reinterpret_cast<get_hmx_dspqueue_fn_type *>(
        dlsym(ctx->ops_dl_handle, "get_hmx_dspqueue"));
    
    if (!get_hmx_dspqueue) {
        fprintf(stderr, "HMX: get_hmx_dspqueue not found\n");
        return -1;
    }
    
    void * queue = get_hmx_dspqueue();
    if (!queue) {
        fprintf(stderr, "HMX: dspqueue not initialized\n");
        return -1;
    }
    
    // Get dspqueue_write function
    using dspqueue_write_fn_type = int(void*, uint32_t, uint32_t, void*, uint32_t, const uint8_t*, uint32_t);
    auto dspqueue_write = reinterpret_cast<dspqueue_write_fn_type *>(
        dlsym(ctx->ops_dl_handle, "dspqueue_write"));
    
    if (!dspqueue_write) {
        fprintf(stderr, "HMX: dspqueue_write not found\n");
        return -1;
    }
    
    // Increment pending ops counter
    hmx_pending_ops.fetch_add(1);
    
    // Send request via dspqueue
    int err = dspqueue_write(queue, 0, 0, nullptr, packet_size, packet_buf, 1000000);  // 1 second timeout
    if (err != 0) {
        fprintf(stderr, "HMX: dspqueue_write failed: 0x%x\n", err);
        hmx_pending_ops.fetch_sub(1);
        return -1;
    }
    
    // Wait for response (poll pending ops counter)
    while (hmx_pending_ops.load() > 0) {
        usleep(1);
    }
    
    // Handle pending unmap requests
    int n_unmap_fds = ctx->hmx_mapper->get_pending_unmap_reqs().size();
    if (n_unmap_fds > 0) {
        ctx->hmx_mapper->unmap_all_pending_buffers();
    }
    
    return hmx_last_result.load();
}
}
