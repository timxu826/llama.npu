/******************************************************************************/
/*   (c) 2020 Qualcomm Innovation Center, Inc. All rights reserved.           */
/*   HMX intrinsics header - extracted from Hexagon SDK                       */
/******************************************************************************/

#ifndef _HMX_HEXAGON_PROTOS_H_
#define _HMX_HEXAGON_PROTOS_H_ 1

// HMX is available on V75+
#if defined(__HMX__) || defined(HEXAGON_ARCH_75) || defined(HEXAGON_ARCH_79) || defined(HEXAGON_ARCH_81)

#define Q6_acc_mxshl_acc __builtin_HEXAGON_M8_mxaccshl
#define Q6_mxclracc __builtin_HEXAGON_M8_mxclracc
#define Q6_mxclracc_hf __builtin_HEXAGON_M8_mxclracc_hf

#define Q6_mxmem_AR_before_hf __builtin_HEXAGON_M8_mxcvtl_sat_hf
#define Q6_mxmem_AR_before_retain_hf __builtin_HEXAGON_M8_mxcvtl_sat_hf_r
#define Q6_mxmem_AR_after_hf __builtin_HEXAGON_M8_mxcvtr_sat_hf
#define Q6_mxmem_AR_after_retain_hf __builtin_HEXAGON_M8_mxcvtr_sat_hf_r

#define Q6_mxmem_AR_before_cm_sat_ub __builtin_HEXAGON_M8_mxcvtl_dm_sat_ub
#define Q6_mxmem_AR_after_cm_sat_ub __builtin_HEXAGON_M8_mxcvtr_dm_sat_ub

#define Q6_bias_mxmem2_A __builtin_HEXAGON_M8_mxmem2_bias
#define Q6_mxmem2_bias_A __builtin_HEXAGON_M8_mxmem2_st_bias
#define Q6_bias_mxmem_A __builtin_HEXAGON_M8_mxmem_bias
#define Q6_mxmem_bias_A __builtin_HEXAGON_M8_mxmem_st_bias

#define Q6_activation_hf_mxmem_RR __builtin_HEXAGON_M8_mxmem_blk_sm_act_hf
#define Q6_activation_ub_mxmem_RR __builtin_HEXAGON_M8_mxmem_blk_sm_act_ub
#define Q6_activation_ub_mxmem_RR_cm __builtin_HEXAGON_M8_mxmem_blk_dm_act_ub
#define Q6_activation_hf_mxmem_RR_deep __builtin_HEXAGON_M8_mxmem_sm_act_hf
#define Q6_activation_ub_mxmem_RR_deep __builtin_HEXAGON_M8_mxmem_sm_act_ub
#define Q6_activation_ub_mxmem_RR_deep_cm __builtin_HEXAGON_M8_mxmem_dm_act_ub

#define Q6_weight_hf_mxmem_RR __builtin_HEXAGON_M8_mxmem_wei_hf
#define Q6_weight_b_mxmem_RR __builtin_HEXAGON_M8_mxmem_wei_b
#define Q6_weight_ubit_mxmem_RR __builtin_HEXAGON_M8_mxmem_wei_b1
#define Q6_weight_hf_mxmem_RR_after __builtin_HEXAGON_M8_mxmema_wei_hf
#define Q6_weight_hf_mxmem_RR_deep __builtin_HEXAGON_M8_mxmemdp_wei_hf
#define Q6_weight_ubit_mxmem_RR_deep __builtin_HEXAGON_M8_mxmemdp_wei_b1

#endif /* __HMX__ */

#endif /* _HMX_HEXAGON_PROTOS_H_ */
