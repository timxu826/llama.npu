#include "hmx-mgr.h"

#include <HAP_compute_res.h>
#include <stdatomic.h>
#include <stdbool.h>

#ifdef HTP_DEBUG
#    define FARF_HIGH 1
#endif
#include "HAP_farf.h"

static unsigned int hmx_mgr_ctx_id = 0;
static atomic_int hmx_available = 0;

int hmx_manager_setup(void) {
    compute_res_attr_t req;
    HAP_compute_res_attr_init(&req);
    HAP_compute_res_attr_set_hmx_param(&req, 1);

    hmx_mgr_ctx_id = HAP_compute_res_acquire(&req, 10000);  // 10ms timeout
    if (hmx_mgr_ctx_id == 0) {
        FARF(HIGH, "hmx_manager_setup: HAP_compute_res_acquire failed - HMX not available");
        atomic_store(&hmx_available, 0);
        return -1;
    }

    atomic_store(&hmx_available, 1);
    FARF(HIGH, "hmx_manager_setup: HMX resources acquired successfully (ctx_id=%u)", hmx_mgr_ctx_id);
    return 0;
}

void hmx_manager_reset(void) {
    if (hmx_mgr_ctx_id) {
        HAP_compute_res_release(hmx_mgr_ctx_id);
        hmx_mgr_ctx_id = 0;
    }
    atomic_store(&hmx_available, 0);
}

void hmx_manager_enable_execution(void) {
    if (!hmx_mgr_ctx_id) {
        return;
    }
    // HMX lock is handled by HAP_compute_res_acquire
}

void hmx_manager_disable_execution(void) {
    if (!hmx_mgr_ctx_id) {
        return;
    }
    // HMX unlock is handled by HAP_compute_res_release
}

void hmx_unit_acquire(void) {
    // Simple spinlock not needed for single-threaded HMX usage
}

void hmx_unit_release(void) {
    // Simple spinlock not needed for single-threaded HMX usage
}

int hmx_is_available(void) {
    return atomic_load(&hmx_available);
}
