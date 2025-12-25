#ifndef HVX_CONVERT_H
#define HVX_CONVERT_H

#include <hexagon_types.h>

#define HVX_INLINE_ALWAYS inline __attribute__((unused, always_inline))

// Convert two FP32 vectors to one interleaved FP16 vector
// v0: [a0, a1, ..., a31] (FP32)
// v1: [b0, b1, ..., b31] (FP32)
// out: [a0, b0, a1, b1, ..., a31, b31] (FP16)
static HVX_INLINE_ALWAYS HVX_Vector hvx_my_wsf_to_vhf(HVX_Vector v1, HVX_Vector v0) {
    const HVX_Vector v_zero = Q6_V_vzero();

    HVX_Vector v0_qf32 = Q6_Vqf32_vadd_VsfVsf(v0, v_zero);
    HVX_Vector v1_qf32 = Q6_Vqf32_vadd_VsfVsf(v1, v_zero);

    return Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(v1_qf32, v0_qf32));
}

// Convert QF16 vector to two QF32 vectors
static HVX_INLINE_ALWAYS HVX_VectorPair hvx_my_vqf16_to_wqf32(HVX_Vector v_src) {
    const HVX_Vector v_lo_mask = Q6_V_vsplat_R(0x0000ffff);
    const HVX_Vector v_hi_mask = Q6_V_vsplat_R(0xffff0000);
    const HVX_Vector v_shift16 = Q6_V_vsplat_R(16);

    // extract packed exp & mantissa
    HVX_Vector exp_comp = Q6_V_vand_VV(v_src, Q6_Vh_vsplat_R(0x1f));    // exp component: low 5 bits
    HVX_Vector mantissa = Q6_V_vand_VV(v_src, Q6_Vh_vsplat_R(0xffe0));  // mantissa: bits 5~15

    // Convert qf16 biased exponent to qf32 biased exponent
    // new exp = exp + (127 (qf32 bias) - 15 (qf16 bias)) = 112
    exp_comp = Q6_Vh_vadd_VhVh(exp_comp, Q6_Vh_vsplat_R(112));

    // unpack exp
    HVX_Vector exp_comp0 = Q6_V_vand_VV(exp_comp, v_lo_mask);
    HVX_Vector exp_comp1 = Q6_Vw_vlsr_VwVw(exp_comp, v_shift16);

    // unpack mantissa + convert qf16 mantissa to qf32 mantissa
    HVX_Vector mantissa0 = Q6_Vw_vasl_VwVw(mantissa, v_shift16);
    HVX_Vector mantissa1 = Q6_V_vand_VV(mantissa, v_hi_mask);

    // merge qf32 exp + mantissa
    HVX_Vector v0_qf32 = Q6_Vw_vadd_VwVw(mantissa0, exp_comp0);
    HVX_Vector v1_qf32 = Q6_Vw_vadd_VwVw(mantissa1, exp_comp1);

    return Q6_W_vcombine_VV(v1_qf32, v0_qf32);
}

// Convert QF16 vector to two FP32 vectors
static HVX_INLINE_ALWAYS HVX_VectorPair hvx_my_vqf16_to_wsf(HVX_Vector v_src) {
    HVX_VectorPair vp = hvx_my_vqf16_to_wqf32(v_src);

    HVX_Vector v0_sf = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(vp));
    HVX_Vector v1_sf = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(vp));
    return Q6_W_vcombine_VV(v1_sf, v0_sf);
}

// Convert FP16 to QF16
static HVX_INLINE_ALWAYS HVX_Vector hvx_my_vhf_to_vqf16(HVX_Vector vx) {
    return Q6_Vqf16_vadd_VhfVhf(vx, Q6_V_vzero());
}

// Convert FP16 vector to two FP32 vectors
// vx: [a0, b0, a1, b1, ..., a31, b31] (FP16)
// out.lo: [a0, a1, ..., a31] (FP32)
// out.hi: [b0, b1, ..., b31] (FP32)
static HVX_INLINE_ALWAYS HVX_VectorPair hvx_my_vhf_to_wsf(HVX_Vector vx) {
    HVX_Vector v_src = hvx_my_vhf_to_vqf16(vx);
    return hvx_my_vqf16_to_wsf(v_src);
}

// Convert FP16 vector to two QF32 vectors
static HVX_INLINE_ALWAYS HVX_VectorPair hvx_my_vhf_to_wqf32(HVX_Vector vx) {
    HVX_Vector v_src = hvx_my_vhf_to_vqf16(vx);
    return hvx_my_vqf16_to_wqf32(v_src);
}

#endif /* HVX_CONVERT_H */
