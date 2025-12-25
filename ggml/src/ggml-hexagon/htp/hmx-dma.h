#ifndef HMX_DMA_H
#define HMX_DMA_H

#include <hexagon_protos.h>
#include <stdbool.h>
#include <stdint.h>

#define DM0_STATUS_MASK  3
#define DM0_STATUS_IDLE  0
#define DM0_STATUS_RUN   1
#define DM0_STATUS_ERROR 2

#define DMA_DESC_DSTATE_PENDING 0
#define DMA_DESC_DSTATE_DONE    1

#define DMA_DESC_TYPE_1D 0
#define DMA_DESC_TYPE_2D 1

struct hmx_dma_desc_1d {
    uint32_t next;

    union {
        struct {
            unsigned length     : 24;
            unsigned type       : 2;
            unsigned dst_dlbc   : 1;
            unsigned src_dlbc   : 1;
            unsigned dst_bypass : 1;
            unsigned src_bypass : 1;
            unsigned ordered    : 1;
            unsigned dstate     : 1;
        } __attribute__((packed));

        uint32_t dstate_order_bypass_type_length;
    };

    uint32_t src;
    uint32_t dst;
} __attribute__((packed));

typedef struct hmx_dma_desc_1d hmx_dma_desc_1d_t;

static inline void hmx_dmstart(void *next) {
    asm volatile("release(%0):at" ::"r"(next));
    Q6_dmstart_A(next);
}

static inline uint32_t hmx_dmpoll(void) {
    return Q6_R_dmpoll();
}

static inline uint32_t hmx_dmwait(void) {
    return Q6_R_dmwait();
}

static inline bool hmx_dma_wait_for_idle(void) {
    uint32_t dm0_status = hmx_dmwait() & DM0_STATUS_MASK;
    return (dm0_status == DM0_STATUS_IDLE);
}

static inline int hmx_dma_submit_one(hmx_dma_desc_1d_t *desc) {
    if (!desc) {
        return -1;
    }

    uint32_t dm0_status = hmx_dmpoll() & DM0_STATUS_MASK;
    if (dm0_status != DM0_STATUS_IDLE) {
        return 1;
    }

    hmx_dmstart(desc);
    hmx_dmpoll();
    return 0;
}

static inline int hmx_dma_load_sync(void *vtcm_dst, const void *src, size_t size) {
    hmx_dma_wait_for_idle();

    hmx_dma_desc_1d_t desc __attribute__((aligned(64)));
    desc.next       = 0;
    desc.length     = size;
    desc.type       = DMA_DESC_TYPE_1D;
    desc.src_bypass = 1;
    desc.dst_bypass = 0;
    desc.ordered    = 1;
    desc.dstate     = DMA_DESC_DSTATE_PENDING;
    desc.src        = (uint32_t)(uintptr_t)src;
    desc.dst        = (uint32_t)(uintptr_t)vtcm_dst;

    hmx_dma_submit_one(&desc);
    hmx_dma_wait_for_idle();

    return 0;
}

#endif /* HMX_DMA_H */
