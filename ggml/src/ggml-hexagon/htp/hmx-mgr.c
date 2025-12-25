#include "hmx-mgr.h"

#include <HAP_compute_res.h>
#include <stdatomic.h>
#include <stdbool.h>

#ifdef HTP_DEBUG
#    define FARF_HIGH 1
#endif
#include "HAP_farf.h"

static unsigned int hmx_mgr_ctx_id = 0;
static int hmx_mgr_spin_lock;
static atomic_int hmx_available = 0;

int hmx_manager_setup(void) {
    // compute_res_attr_t req;
    // HAP_compute_res_attr_init(&req);
    // HAP_compute_res_attr_set_hmx_param(&req, 1);

    // hmx_mgr_ctx_id = HAP_compute_res_acquire(&req, 10000);  // 10ms timeout
    // if (hmx_mgr_ctx_id == 0) {
    //     FARF(ALWAYS, "hmx_manager_setup: HAP_compute_res_acquire failed - HMX not available");
    //     atomic_store(&hmx_available, 0);
    //     return -1;
    // }

    // atomic_store(&hmx_available, 1);
    // FARF(ALWAYS, "hmx_manager_setup: HMX resources acquired successfully (ctx_id=%u)", hmx_mgr_ctx_id);
    // return 0;


    // FIXME: only for debug use. Remove this simulation code when HAP_compute_res_acquire is functional.
    // For the purpose of this example, we will simulate successful acquisition
    hmx_mgr_ctx_id = 1; // Simulated context ID
    atomic_store(&hmx_available, 1);
    FARF(ALWAYS, "hmx_manager_setup: HMX resources acquired successfully (ctx_id=%u)", hmx_mgr_ctx_id);
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
    // enable current thread to timeshare HMX unit
    int err = HAP_compute_res_hmx_lock2(hmx_mgr_ctx_id, HAP_COMPUTE_RES_HMX_SHARED);
    if (err) {
        FARF(ALWAYS, "HAP_compute_res_hmx_lock2 failed with return code 0x%x", err);
    }
    FARF(ALWAYS, "HAP_compute_res_hmx_lock2 called");
    if (!hmx_mgr_ctx_id) {
        return;
    }
    // HMX lock is handled by HAP_compute_res_acquire
}

void hmx_manager_disable_execution(void) {
    if (!hmx_mgr_ctx_id) {
        return;
    }
    HAP_compute_res_hmx_unlock2(hmx_mgr_ctx_id, HAP_COMPUTE_RES_HMX_SHARED);
    FARF(ALWAYS, "HAP_compute_res_hmx_unlock2 called");
    // HMX unlock is handled by HAP_compute_res_release
}

void hmx_unit_acquire(void) {
    int *lock_ptr = &hmx_mgr_spin_lock;
    asm volatile(
        "1:  r0 = memw_locked(%0)     \n"
        "    p0 = cmp.eq(r0, #0)      \n"
        "    if (!p0) jump 2f         \n"
        "    memw_locked(%0, p0) = %0 \n"
        "    if (p0) jump 3f          \n"
        "2:  pause(#8)                \n"
        "    jump 1b                  \n"
        "3:"
        : "+r"(lock_ptr)::"p0", "r0");
    FARF(ALWAYS, "HMX unit acquired by thread");
}

void hmx_unit_release(void) {
    *(volatile int *) &hmx_mgr_spin_lock = 0;
}

int hmx_is_available(void) {
    return atomic_load(&hmx_available);
}
