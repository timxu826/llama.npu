#include "htp-ops.h"

#include "ggml-backend-impl.h"
#include "ggml-hexagon.h"

// HMX ops are now integrated into the HVX backend via dspqueue with HTP_OP_HMX_* opcodes.
// This file provides stub implementations - actual HMX dispatching happens through
// the normal HVX path in ggml-hexagon.cpp using HTP_OP_HMX_MUL_MAT etc.
//
// The htp_ops_support_op() function returns false to indicate that the old
// separate HMX library path is not used. HMX operations are handled by HVX backend.

extern "C" {

bool htp_ops_support_op(const struct ggml_tensor * dst) {
    // HMX ops are now dispatched through HVX backend's dspqueue
    // This function returns false to indicate old path is not used
    (void)dst;
    return false;
}

int htp_ops_compute_op(struct ggml_tensor * dst) {
    // HMX ops are now dispatched through HVX backend's dspqueue
    // This function should not be called
    (void)dst;
    return -1;
}

}
