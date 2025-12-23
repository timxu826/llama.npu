#ifndef HMX_MGR_H
#define HMX_MGR_H

#ifdef __cplusplus
extern "C" {
#endif

// HMX manager - handles HMX resource acquisition and release
int  hmx_manager_setup(void);
void hmx_manager_reset(void);

// Enable/disable HMX execution for current thread
void hmx_manager_enable_execution(void);
void hmx_manager_disable_execution(void);

// Spinlock for exclusive HMX unit access
void hmx_unit_acquire(void);
void hmx_unit_release(void);

// Check if HMX is available
int hmx_is_available(void);

#ifdef __cplusplus
}
#endif

#endif /* HMX_MGR_H */
