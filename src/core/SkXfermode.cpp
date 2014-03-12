
/*
 * Copyright 2006 The Android Open Source Project
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */


#include "SkXfermode.h"
#include "SkColorPriv.h"
#include "SkFlattenableBuffers.h"
#include "SkMathPriv.h"
#include "SkString.h"

SK_DEFINE_INST_COUNT(SkXfermode)
#if defined(__ARM_HAVE_NEON)
#include <arm_neon.h>


#define NEON_A (SK_A32_SHIFT / 8)
#define NEON_R (SK_R32_SHIFT / 8)
#define NEON_G (SK_G32_SHIFT / 8)
#define NEON_B (SK_B32_SHIFT / 8)
#endif

#define SkAlphaMulAlpha(a, b)   SkMulDiv255Round(a, b)

#if 0
// idea for higher precision blends in xfer procs (and slightly faster)
// see DstATop as a probable caller
static U8CPU mulmuldiv255round(U8CPU a, U8CPU b, U8CPU c, U8CPU d) {
    SkASSERT(a <= 255);
    SkASSERT(b <= 255);
    SkASSERT(c <= 255);
    SkASSERT(d <= 255);
    unsigned prod = SkMulS16(a, b) + SkMulS16(c, d) + 128;
    unsigned result = (prod + (prod >> 8)) >> 8;
    SkASSERT(result <= 255);
    return result;
}
#endif
#if defined(__ARM_HAVE_NEON)
static inline uint16x8_t SkAlpha255To256_neon8(uint8x8_t alpha) {
    return vaddw_u8(vdupq_n_u16(1), alpha);
}

static inline uint8x8_t SkAlphaMul_neon8(uint8x8_t color, uint16x8_t scale) {
    return vshrn_n_u16(vmovl_u8(color) * scale, 8);
}

static inline uint8x8x4_t SkAlphaMulQ_neon8(uint8x8x4_t color, uint16x8_t scale) {
    uint8x8x4_t ret;

    ret.val[NEON_A] = SkAlphaMul_neon8(color.val[NEON_A], scale);
    ret.val[NEON_R] = SkAlphaMul_neon8(color.val[NEON_R], scale);
    ret.val[NEON_G] = SkAlphaMul_neon8(color.val[NEON_G], scale);
    ret.val[NEON_B] = SkAlphaMul_neon8(color.val[NEON_B], scale);

    return ret;
}

/* This function expands 8 pixels from RGB565 (R, G, B from high to low) to
 * SkPMColor (all possible configurations supported) in the exact same way as
 * SkPixel16ToPixel32.
 */
static inline uint8x8x4_t SkPixel16ToPixel32_neon8(uint16x8_t vsrc) {

    uint8x8x4_t ret;
    uint8x8_t vr, vg, vb;

    vr = vmovn_u16(vshrq_n_u16(vsrc, SK_R16_SHIFT));
    vg = vmovn_u16(vshrq_n_u16(vshlq_n_u16(vsrc, SK_R16_BITS), SK_R16_BITS + SK_B16_BITS));
    vb = vmovn_u16(vsrc & vdupq_n_u16(SK_B16_MASK));

    ret.val[NEON_A] = vdup_n_u8(0xFF);
    ret.val[NEON_R] = vshl_n_u8(vr, 8 - SK_R16_BITS) | vshr_n_u8(vr, 2 * SK_R16_BITS - 8);
    ret.val[NEON_G] = vshl_n_u8(vg, 8 - SK_G16_BITS) | vshr_n_u8(vg, 2 * SK_G16_BITS - 8);
    ret.val[NEON_B] = vshl_n_u8(vb, 8 - SK_B16_BITS) | vshr_n_u8(vb, 2 * SK_B16_BITS - 8);

    return ret;
}

/* This function packs 8 pixels from SkPMColor (all possible configurations
 * supported) to RGB565 (R, G, B from high to low) in the exact same way as
 * SkPixel32ToPixel16.
 */
static inline uint16x8_t SkPixel32ToPixel16_neon8(uint8x8x4_t vsrc) {

    uint16x8_t ret;

    ret = vshll_n_u8(vsrc.val[NEON_R], 8);
    ret = vsriq_n_u16(ret, vshll_n_u8(vsrc.val[NEON_G], 8), SK_R16_BITS);
    ret = vsriq_n_u16(ret, vshll_n_u8(vsrc.val[NEON_B], 8), SK_R16_BITS + SK_G16_BITS);

    return ret;
}
static inline uint8x8_t SkDiv255Round_neon8_32_8(int32x4_t p1, int32x4_t p2) {
    uint16x8_t tmp;

    tmp = vcombine_u16(vmovn_u32(vreinterpretq_u32_s32(p1)),
                       vmovn_u32(vreinterpretq_u32_s32(p2)));

    tmp += vdupq_n_u16(128);
    tmp += vshrq_n_u16(tmp, 8);

    return vshrn_n_u16(tmp, 8);
}
static inline uint8x8_t clamp_div255round_simd8_32(int32x4_t val1, int32x4_t val2) {
    uint8x8_t ret;
    uint32x4_t cmp1, cmp2;
    uint16x8_t cmp16;
    uint8x8_t cmp8, cmp8_1;

    // Test if <= 0
    cmp1 = vcleq_s32(val1, vdupq_n_s32(0));
    cmp2 = vcleq_s32(val2, vdupq_n_s32(0));
    cmp16 = vcombine_u16(vmovn_u32(cmp1), vmovn_u32(cmp2));
    cmp8_1 = vmovn_u16(cmp16);

    // Init to zero
    ret = vdup_n_u8(0);

    // Test if >= 255*255
    cmp1 = vcgeq_s32(val1, vdupq_n_s32(255*255));
    cmp2 = vcgeq_s32(val2, vdupq_n_s32(255*255));
    cmp16 = vcombine_u16(vmovn_u32(cmp1), vmovn_u32(cmp2));
    cmp8 = vmovn_u16(cmp16);

    // Insert 255 where true
    ret = vbsl_u8(cmp8, vdup_n_u8(255), ret);

    // Calc SkDiv255Round
    uint8x8_t div = SkDiv255Round_neon8_32_8(val1, val2);

    // Insert where false and previous test false
    cmp8 = cmp8 | cmp8_1;
    ret = vbsl_u8(cmp8, ret, div);

    // Return the final combination
    return ret;
}
static inline uint8x8_t SkAlphaMulAlpha_neon8(uint8x8_t color, uint8x8_t alpha) {
    uint16x8_t tmp;
    uint8x8_t ret;

    tmp = vmull_u8(color, alpha);
    tmp = vaddq_u16(tmp, vdupq_n_u16(128));
    tmp = vaddq_u16(tmp, vshrq_n_u16(tmp, 8));

    ret = vshrn_n_u16(tmp, 8);

    return ret;
}
static inline uint8x8_t blendfunc_multiply_color(uint8x8_t sc, uint8x8_t dc,
                                                 uint8x8_t sa, uint8x8_t da) {
    uint32x4_t val1, val2;
    uint16x8_t scdc, t1, t2;

    t1 = vmull_u8(sc, vdup_n_u8(255) - da);
    t2 = vmull_u8(dc, vdup_n_u8(255) - sa);
    scdc = vmull_u8(sc, dc);

    val1 = vaddl_u16(vget_low_u16(t1), vget_low_u16(t2));
    val2 = vaddl_u16(vget_high_u16(t1), vget_high_u16(t2));

    val1 = vaddw_u16(val1, vget_low_u16(scdc));
    val2 = vaddw_u16(val2, vget_high_u16(scdc));

    return clamp_div255round_simd8_32(
                vreinterpretq_s32_u32(val1), vreinterpretq_s32_u32(val2));
}

#endif

static inline unsigned saturated_add(unsigned a, unsigned b) {
    SkASSERT(a <= 255);
    SkASSERT(b <= 255);
    unsigned sum = a + b;
    if (sum > 255) {
        sum = 255;
    }
    return sum;
}

static inline int clamp_signed_byte(int n) {
    if (n < 0) {
        n = 0;
    } else if (n > 255) {
        n = 255;
    }
    return n;
}

static inline int clamp_div255round(int prod) {
    if (prod <= 0) {
        return 0;
    } else if (prod >= 255*255) {
        return 255;
    } else {
        return SkDiv255Round(prod);
    }
}

static inline int clamp_max(int value, int max) {
    if (value > max) {
        value = max;
    }
    return value;
}

///////////////////////////////////////////////////////////////////////////////

//  kClear_Mode,    //!< [0, 0]
static SkPMColor clear_modeproc(SkPMColor src, SkPMColor dst) {
    return 0;
}

//  kSrc_Mode,      //!< [Sa, Sc]
static SkPMColor src_modeproc(SkPMColor src, SkPMColor dst) {
    return src;
}

//  kDst_Mode,      //!< [Da, Dc]
static SkPMColor dst_modeproc(SkPMColor src, SkPMColor dst) {
    return dst;
}

//  kSrcOver_Mode,  //!< [Sa + Da - Sa*Da, Sc + (1 - Sa)*Dc]
static SkPMColor srcover_modeproc(SkPMColor src, SkPMColor dst) {
#if 0
    // this is the old, more-correct way, but it doesn't guarantee that dst==255
    // will always stay opaque
    return src + SkAlphaMulQ(dst, SkAlpha255To256(255 - SkGetPackedA32(src)));
#else
    // this is slightly faster, but more importantly guarantees that dst==255
    // will always stay opaque
    return src + SkAlphaMulQ(dst, 256 - SkGetPackedA32(src));
#endif
}

//  kDstOver_Mode,  //!< [Sa + Da - Sa*Da, Dc + (1 - Da)*Sc]
static SkPMColor dstover_modeproc(SkPMColor src, SkPMColor dst) {
    // this is the reverse of srcover, just flipping src and dst
    // see srcover's comment about the 256 for opaqueness guarantees
#if defined(__ARM_HAVE_NEON)
    uint32_t result[1]={0};
    uint8x8x4_t src_neon = vld4_u8((uint8_t*)&src);//uint32_t * -> uint8_t* -> uint8x8x4_t
    uint8x8x4_t dst_neon = vld4_u8((uint8_t*)&dst);
    uint8x8x4_t ret;
    uint16x8_t scale;
    scale = vsubw_u8(vdupq_n_u16(256), dst_neon.val[NEON_A]);
    ret.val[NEON_A] = dst_neon.val[NEON_A] + SkAlphaMul_neon8(src_neon.val[NEON_A], scale);
    ret.val[NEON_R] = dst_neon.val[NEON_R] + SkAlphaMul_neon8(src_neon.val[NEON_R], scale);
    ret.val[NEON_G] = dst_neon.val[NEON_G] + SkAlphaMul_neon8(src_neon.val[NEON_G], scale);
    ret.val[NEON_B] = dst_neon.val[NEON_B] + SkAlphaMul_neon8(src_neon.val[NEON_B], scale);
    vst4_u8((uint8_t *)result,ret);//uint8x8x4_t -> uint8_t * -> uint32_t *
    return (*result);
#else
	return dst + SkAlphaMulQ(src, 256 - SkGetPackedA32(dst));
#endif
}

//  kSrcIn_Mode,    //!< [Sa * Da, Sc * Da]
static SkPMColor srcin_modeproc(SkPMColor src, SkPMColor dst) {
#if defined(__ARM_HAVE_NEON)
	uint32_t result[1]={0};
	uint8x8x4_t src_neon = vld4_u8((uint8_t*)&src);//uint32_t * -> uint8_t* -> uint8x8x4_t
	uint8x8x4_t dst_neon = vld4_u8((uint8_t*)&dst);
	uint8x8x4_t ret;
    uint16x8_t scale;
    scale = SkAlpha255To256_neon8(dst_neon.val[NEON_A]);
    ret.val[NEON_A] =SkAlphaMul_neon8(src_neon.val[NEON_A], scale);
    ret.val[NEON_R] =SkAlphaMul_neon8(src_neon.val[NEON_R], scale);
    ret.val[NEON_G] =SkAlphaMul_neon8(src_neon.val[NEON_G], scale);
    ret.val[NEON_B] =SkAlphaMul_neon8(src_neon.val[NEON_B], scale);
    vst4_u8((uint8_t *)result,ret);//uint8x8x4_t -> uint8_t * -> uint32_t *
    return (*result);
#else
	return SkAlphaMulQ(src, SkAlpha255To256(SkGetPackedA32(dst)));
#endif
}

//  kDstIn_Mode,    //!< [Sa * Da, Sa * Dc]
static SkPMColor dstin_modeproc(SkPMColor src, SkPMColor dst) {
#if defined(__ARM_HAVE_NEON)
    uint32_t result[1]={0};
    uint8x8x4_t src_neon = vld4_u8((uint8_t*)&src);//uint32_t * -> uint8_t* -> uint8x8x4_t
    uint8x8x4_t dst_neon = vld4_u8((uint8_t*)&dst);
    uint8x8x4_t ret;
    uint16x8_t scale;

    scale = SkAlpha255To256_neon8(src_neon.val[NEON_A]);
    ret = SkAlphaMulQ_neon8(dst_neon, scale);
	
    vst4_u8((uint8_t *)result,ret);//uint8x8x4_t -> uint8_t * -> uint32_t *
    return (*result);
#else
    return SkAlphaMulQ(dst, SkAlpha255To256(SkGetPackedA32(src)));
#endif
}

//  kSrcOut_Mode,   //!< [Sa * (1 - Da), Sc * (1 - Da)]
static SkPMColor srcout_modeproc(SkPMColor src, SkPMColor dst) {
#if defined(__ARM_HAVE_NEON)
    uint32_t result[1]={0};
    uint8x8x4_t src_neon = vld4_u8((uint8_t*)&src);//uint32_t * -> uint8_t* -> uint8x8x4_t
    uint8x8x4_t dst_neon = vld4_u8((uint8_t*)&dst);
    uint8x8x4_t ret;
    uint16x8_t scale = vsubw_u8(vdupq_n_u16(256), dst_neon.val[NEON_A]);
    ret = SkAlphaMulQ_neon8(src_neon, scale);
    vst4_u8((uint8_t *)result,ret);//uint8x8x4_t -> uint8_t * -> uint32_t *
    return (*result);
#else
	return SkAlphaMulQ(src, SkAlpha255To256(255 - SkGetPackedA32(dst)));
#endif
}

//  kDstOut_Mode,   //!< [Da * (1 - Sa), Dc * (1 - Sa)]
static SkPMColor dstout_modeproc(SkPMColor src, SkPMColor dst) {
#if defined(__ARM_HAVE_NEON)
    uint32_t result[1]={0};
    uint8x8x4_t src_neon = vld4_u8((uint8_t*)&src);//uint32_t * -> uint8_t* -> uint8x8x4_t
    uint8x8x4_t dst_neon = vld4_u8((uint8_t*)&dst);
    uint8x8x4_t ret;
    uint16x8_t scale = vsubw_u8(vdupq_n_u16(256), src_neon.val[NEON_A]);
    ret = SkAlphaMulQ_neon8(dst_neon, scale);
    vst4_u8((uint8_t *)result,ret);//uint8x8x4_t -> uint8_t * -> uint32_t *
    return (*result);
#else
	return SkAlphaMulQ(dst, SkAlpha255To256(255 - SkGetPackedA32(src)));
#endif
}

//  kSrcATop_Mode,  //!< [Da, Sc * Da + (1 - Sa) * Dc]
static SkPMColor srcatop_modeproc(SkPMColor src, SkPMColor dst) {
#if defined(__ARM_HAVE_NEON)
    uint32_t result[1]={0};
    uint8x8x4_t src_neon = vld4_u8((uint8_t*)&src);//uint32_t * -> uint8_t* -> uint8x8x4_t
    uint8x8x4_t dst_neon = vld4_u8((uint8_t*)&dst);
    uint8x8x4_t ret;
    uint8x8_t isa;
	
    isa = vsub_u8(vdup_n_u8(255), src_neon.val[NEON_A]);
    ret.val[NEON_A] = dst_neon.val[NEON_A];
    ret.val[NEON_R] = SkAlphaMulAlpha_neon8(src_neon.val[NEON_R], dst_neon.val[NEON_A])
                      + SkAlphaMulAlpha_neon8(dst_neon.val[NEON_R], isa);
    ret.val[NEON_G] = SkAlphaMulAlpha_neon8(src_neon.val[NEON_G], dst_neon.val[NEON_A])
                      + SkAlphaMulAlpha_neon8(dst_neon.val[NEON_G], isa);
    ret.val[NEON_B] = SkAlphaMulAlpha_neon8(src_neon.val[NEON_B], dst_neon.val[NEON_A])
                      + SkAlphaMulAlpha_neon8(dst_neon.val[NEON_B], isa);
    vst4_u8((uint8_t *)result,ret);//uint8x8x4_t -> uint8_t * -> uint32_t *
    return (*result);
#else
    unsigned sa = SkGetPackedA32(src);
    unsigned da = SkGetPackedA32(dst);
    unsigned isa = 255 - sa;

    return SkPackARGB32(da,
                        SkAlphaMulAlpha(da, SkGetPackedR32(src)) +
                            SkAlphaMulAlpha(isa, SkGetPackedR32(dst)),
                        SkAlphaMulAlpha(da, SkGetPackedG32(src)) +
                            SkAlphaMulAlpha(isa, SkGetPackedG32(dst)),
                        SkAlphaMulAlpha(da, SkGetPackedB32(src)) +
                            SkAlphaMulAlpha(isa, SkGetPackedB32(dst)));
#endif
}

//  kDstATop_Mode,  //!< [Sa, Sa * Dc + Sc * (1 - Da)]
static SkPMColor dstatop_modeproc(SkPMColor src, SkPMColor dst) {
#if defined(__ARM_HAVE_NEON)
    uint32_t result[1]={0};
    uint8x8x4_t src_neon = vld4_u8((uint8_t*)&src);//uint32_t * -> uint8_t* -> uint8x8x4_t
    uint8x8x4_t dst_neon = vld4_u8((uint8_t*)&dst);
    uint8x8x4_t ret;
    uint8x8_t isa;
 
	isa = vsub_u8(vdup_n_u8(255), dst_neon.val[NEON_A]);

    ret.val[NEON_A] = src_neon.val[NEON_A];
    ret.val[NEON_R] = SkAlphaMulAlpha_neon8(src_neon.val[NEON_R], dst_neon.val[NEON_A])
                      + SkAlphaMulAlpha_neon8(dst_neon.val[NEON_R], isa);
    ret.val[NEON_G] = SkAlphaMulAlpha_neon8(src_neon.val[NEON_G], dst_neon.val[NEON_A])
                      + SkAlphaMulAlpha_neon8(dst_neon.val[NEON_G], isa);
    ret.val[NEON_B] = SkAlphaMulAlpha_neon8(src_neon.val[NEON_B], dst_neon.val[NEON_A])
                      + SkAlphaMulAlpha_neon8(dst_neon.val[NEON_B], isa);
    vst4_u8((uint8_t *)result,ret);//uint8x8x4_t -> uint8_t * -> uint32_t *
    return (*result);
#else
    unsigned sa = SkGetPackedA32(src);
    unsigned da = SkGetPackedA32(dst);
    unsigned ida = 255 - da;

    return SkPackARGB32(sa,
                        SkAlphaMulAlpha(ida, SkGetPackedR32(src)) +
                            SkAlphaMulAlpha(sa, SkGetPackedR32(dst)),
                        SkAlphaMulAlpha(ida, SkGetPackedG32(src)) +
                            SkAlphaMulAlpha(sa, SkGetPackedG32(dst)),
                        SkAlphaMulAlpha(ida, SkGetPackedB32(src)) +
                            SkAlphaMulAlpha(sa, SkGetPackedB32(dst)));
#endif
}

//  kXor_Mode   [Sa + Da - 2 * Sa * Da, Sc * (1 - Da) + (1 - Sa) * Dc]
static SkPMColor xor_modeproc(SkPMColor src, SkPMColor dst) {
#if defined(__ARM_HAVE_NEON)
    uint32_t result[1]={0};
    uint8x8x4_t src_neon = vld4_u8((uint8_t*)&src);//uint32_t * -> uint8_t* -> uint8x8x4_t
    uint8x8x4_t dst_neon = vld4_u8((uint8_t*)&dst);
    uint8x8x4_t ret;
    uint8x8_t isa, ida;
    uint16x8_t tmp_wide, tmp_wide2;
 
    isa = vsub_u8(vdup_n_u8(255), src_neon.val[NEON_A]);
    ida = vsub_u8(vdup_n_u8(255), dst_neon.val[NEON_A]);

    // First calc alpha
    tmp_wide = vmovl_u8(src_neon.val[NEON_A]);
    tmp_wide = vaddw_u8(tmp_wide, dst_neon.val[NEON_A]);
    tmp_wide2 = vshll_n_u8(SkAlphaMulAlpha_neon8(src_neon.val[NEON_A], dst_neon.val[NEON_A]), 1);
    tmp_wide = vsubq_u16(tmp_wide, tmp_wide2);
    ret.val[NEON_A] = vmovn_u16(tmp_wide);

    // Then colors
    ret.val[NEON_R] = SkAlphaMulAlpha_neon8(src_neon.val[NEON_R], ida)
                      + SkAlphaMulAlpha_neon8(dst_neon.val[NEON_R], isa);
    ret.val[NEON_G] = SkAlphaMulAlpha_neon8(src_neon.val[NEON_G], ida)
                      + SkAlphaMulAlpha_neon8(dst_neon.val[NEON_G], isa);
    ret.val[NEON_B] = SkAlphaMulAlpha_neon8(src_neon.val[NEON_B], ida)
                      + SkAlphaMulAlpha_neon8(dst_neon.val[NEON_B], isa);
    vst4_u8((uint8_t *)result,ret);//uint8x8x4_t -> uint8_t * -> uint32_t *
    return (*result);
#else
    unsigned sa = SkGetPackedA32(src);
    unsigned da = SkGetPackedA32(dst);
    unsigned isa = 255 - sa;
    unsigned ida = 255 - da;

    return SkPackARGB32(sa + da - (SkAlphaMulAlpha(sa, da) << 1),
                        SkAlphaMulAlpha(ida, SkGetPackedR32(src)) +
                            SkAlphaMulAlpha(isa, SkGetPackedR32(dst)),
                        SkAlphaMulAlpha(ida, SkGetPackedG32(src)) +
                            SkAlphaMulAlpha(isa, SkGetPackedG32(dst)),
                        SkAlphaMulAlpha(ida, SkGetPackedB32(src)) +
                            SkAlphaMulAlpha(isa, SkGetPackedB32(dst)));
#endif
}

///////////////////////////////////////////////////////////////////////////////

// kPlus_Mode
static SkPMColor plus_modeproc(SkPMColor src, SkPMColor dst) {
#if defined(__ARM_HAVE_NEON)
    uint32_t result[1]={0};
    uint8x8x4_t src_neon = vld4_u8((uint8_t*)&src);//uint32_t * -> uint8_t* -> uint8x8x4_t
    uint8x8x4_t dst_neon = vld4_u8((uint8_t*)&dst);
    uint8x8x4_t ret;

    ret.val[NEON_A] = vqadd_u8(src_neon.val[NEON_A], dst_neon.val[NEON_A]);
    ret.val[NEON_R] = vqadd_u8(src_neon.val[NEON_R], dst_neon.val[NEON_R]);
    ret.val[NEON_G] = vqadd_u8(src_neon.val[NEON_G], dst_neon.val[NEON_G]);
    ret.val[NEON_B] = vqadd_u8(src_neon.val[NEON_B], dst_neon.val[NEON_B]);
    vst4_u8((uint8_t *)result,ret);//uint8x8x4_t -> uint8_t * -> uint32_t *
    return (*result);
#else
    unsigned b = saturated_add(SkGetPackedB32(src), SkGetPackedB32(dst));
    unsigned g = saturated_add(SkGetPackedG32(src), SkGetPackedG32(dst));
    unsigned r = saturated_add(SkGetPackedR32(src), SkGetPackedR32(dst));
    unsigned a = saturated_add(SkGetPackedA32(src), SkGetPackedA32(dst));
    return SkPackARGB32(a, r, g, b);
#endif
}

// kModulate_Mode
static SkPMColor modulate_modeproc(SkPMColor src, SkPMColor dst) {
    int a = SkAlphaMulAlpha(SkGetPackedA32(src), SkGetPackedA32(dst));
    int r = SkAlphaMulAlpha(SkGetPackedR32(src), SkGetPackedR32(dst));
    int g = SkAlphaMulAlpha(SkGetPackedG32(src), SkGetPackedG32(dst));
    int b = SkAlphaMulAlpha(SkGetPackedB32(src), SkGetPackedB32(dst));
    return SkPackARGB32(a, r, g, b);
}
#if defined(__ARM_HAVE_NEON)
static inline uint16x8_t SkAlphaMulAlpha_neon8_16(uint8x8_t color, uint8x8_t alpha) {
    uint16x8_t ret;

    ret = vmull_u8(color, alpha);
    ret = vaddq_u16(ret, vdupq_n_u16(128));
    ret = vaddq_u16(ret, vshrq_n_u16(ret, 8));

    ret = vshrq_n_u16(ret, 8);

    return ret;
}
#endif
#if defined(__ARM_HAVE_NEON)
static inline uint8x8_t srcover_color(uint8x8_t a, uint8x8_t b) {
    uint16x8_t tmp;

    tmp = vaddl_u8(a, b);
    tmp -= SkAlphaMulAlpha_neon8_16(a, b);

    return vmovn_u16(tmp);
}
#endif
static inline int srcover_byte(int a, int b) {
    return a + b - SkAlphaMulAlpha(a, b);
}
// kScreen_Mode
static SkPMColor screen_modeproc(SkPMColor src, SkPMColor dst) {
#if defined(__ARM_HAVE_NEON)
    uint32_t result[1]={0};
    uint8x8x4_t src_neon = vld4_u8((uint8_t*)&src);//uint32_t * -> uint8_t* -> uint8x8x4_t
    uint8x8x4_t dst_neon = vld4_u8((uint8_t*)&dst);
    uint8x8x4_t ret;

    ret.val[NEON_A] = srcover_color(src_neon.val[NEON_A], dst_neon.val[NEON_A]);
    ret.val[NEON_R] = srcover_color(src_neon.val[NEON_R], dst_neon.val[NEON_R]);
    ret.val[NEON_G] = srcover_color(src_neon.val[NEON_G], dst_neon.val[NEON_G]);
    ret.val[NEON_B] = srcover_color(src_neon.val[NEON_B], dst_neon.val[NEON_B]);
    vst4_u8((uint8_t *)result,ret);//uint8x8x4_t -> uint8_t * -> uint32_t *
    return (*result);
#else
    int a = srcover_byte(SkGetPackedA32(src), SkGetPackedA32(dst));
    int r = srcover_byte(SkGetPackedR32(src), SkGetPackedR32(dst));
    int g = srcover_byte(SkGetPackedG32(src), SkGetPackedG32(dst));
    int b = srcover_byte(SkGetPackedB32(src), SkGetPackedB32(dst));
    return SkPackARGB32(a, r, g, b);
#endif
}

// kOverlay_Mode
#if defined(__ARM_HAVE_NEON)
template <bool overlay>
static inline uint8x8_t overlay_hardlight_color(uint8x8_t sc, uint8x8_t dc,
                                               uint8x8_t sa, uint8x8_t da) {
    /*
     * In the end we're gonna use (rc + tmp) with a different rc
     * coming from an alternative.
     * The whole value (rc + tmp) can always be expressed as
     * VAL = COM - SUB in the if case
     * VAL = COM + SUB - sa*da in the else case
     *
     * with COM = 255 * (sc + dc)
     * and  SUB = sc*da + dc*sa - 2*dc*sc
     */

    // Prepare common subexpressions
    uint16x8_t const255 = vdupq_n_u16(255);
    uint16x8_t sc_plus_dc = vaddl_u8(sc, dc);
    uint16x8_t scda = vmull_u8(sc, da);
    uint16x8_t dcsa = vmull_u8(dc, sa);
    uint16x8_t sada = vmull_u8(sa, da);

    // Prepare non common subexpressions
    uint16x8_t dc2, sc2;
    uint32x4_t scdc2_1, scdc2_2;
    if (overlay) {
        dc2 = vshll_n_u8(dc, 1);
        scdc2_1 = vmull_u16(vget_low_u16(dc2), vget_low_u16(vmovl_u8(sc)));
        scdc2_2 = vmull_u16(vget_high_u16(dc2), vget_high_u16(vmovl_u8(sc)));
    } else {
        sc2 = vshll_n_u8(sc, 1);
        scdc2_1 = vmull_u16(vget_low_u16(sc2), vget_low_u16(vmovl_u8(dc)));
        scdc2_2 = vmull_u16(vget_high_u16(sc2), vget_high_u16(vmovl_u8(dc)));
    }

    // Calc COM
    int32x4_t com1, com2;
    com1 = vreinterpretq_s32_u32(
                vmull_u16(vget_low_u16(const255), vget_low_u16(sc_plus_dc)));
    com2 = vreinterpretq_s32_u32(
                vmull_u16(vget_high_u16(const255), vget_high_u16(sc_plus_dc)));

    // Calc SUB
    int32x4_t sub1, sub2;
    sub1 = vreinterpretq_s32_u32(vaddl_u16(vget_low_u16(scda), vget_low_u16(dcsa)));
    sub2 = vreinterpretq_s32_u32(vaddl_u16(vget_high_u16(scda), vget_high_u16(dcsa)));
    sub1 = vsubq_s32(sub1, vreinterpretq_s32_u32(scdc2_1));
    sub2 = vsubq_s32(sub2, vreinterpretq_s32_u32(scdc2_2));

    // Compare 2*dc <= da
    uint16x8_t cmp;

    if (overlay) {
        cmp = vcleq_u16(dc2, vmovl_u8(da));
    } else {
        cmp = vcleq_u16(sc2, vmovl_u8(sa));
    }

    // Prepare variables
    int32x4_t val1_1, val1_2;
    int32x4_t val2_1, val2_2;
    uint32x4_t cmp1, cmp2;

    cmp1 = vmovl_u16(vget_low_u16(cmp));
    cmp1 |= vshlq_n_u32(cmp1, 16);
    cmp2 = vmovl_u16(vget_high_u16(cmp));
    cmp2 |= vshlq_n_u32(cmp2, 16);

    // Calc COM - SUB
    val1_1 = com1 - sub1;
    val1_2 = com2 - sub2;

    // Calc COM + SUB - sa*da
    val2_1 = com1 + sub1;
    val2_2 = com2 + sub2;

    val2_1 = vsubq_s32(val2_1, vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(sada))));
    val2_2 = vsubq_s32(val2_2, vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(sada))));

    // Insert where needed
    val1_1 = vbslq_s32(cmp1, val1_1, val2_1);
    val1_2 = vbslq_s32(cmp2, val1_2, val2_2);

    // Call the clamp_div255round function
    return clamp_div255round_simd8_32(val1_1, val1_2);
}
#endif
#if defined(__ARM_HAVE_NEON)
static inline uint8x8_t overlay_color(uint8x8_t sc, uint8x8_t dc,
                                      uint8x8_t sa, uint8x8_t da) {
    return overlay_hardlight_color<true>(sc, dc, sa, da);
}
#endif
static inline int overlay_byte(int sc, int dc, int sa, int da) {
    int tmp = sc * (255 - da) + dc * (255 - sa);
    int rc;
    if (2 * dc <= da) {
        rc = 2 * sc * dc;
    } else {
        rc = sa * da - 2 * (da - dc) * (sa - sc);
    }
    return clamp_div255round(rc + tmp);
}
static SkPMColor overlay_modeproc(SkPMColor src, SkPMColor dst) {
#if defined(__ARM_HAVE_NEON)
    uint32_t result[1]={0};
    uint8x8x4_t src_neon = vld4_u8((uint8_t*)&src);//uint32_t * -> uint8_t* -> uint8x8x4_t
    uint8x8x4_t dst_neon = vld4_u8((uint8_t*)&dst);
    uint8x8x4_t ret;

    ret.val[NEON_A] = srcover_color(src_neon.val[NEON_A], dst_neon.val[NEON_A]);
    ret.val[NEON_R] = overlay_color(src_neon.val[NEON_R], dst_neon.val[NEON_R],
                                    src_neon.val[NEON_A], dst_neon.val[NEON_A]);
    ret.val[NEON_G] = overlay_color(src_neon.val[NEON_G], dst_neon.val[NEON_G],
                                    src_neon.val[NEON_A], dst_neon.val[NEON_A]);
    ret.val[NEON_B] = overlay_color(src_neon.val[NEON_B], dst_neon.val[NEON_B],
    	 							src_neon.val[NEON_A], dst_neon.val[NEON_A]);
    vst4_u8((uint8_t *)result,ret);//uint8x8x4_t -> uint8_t * -> uint32_t *
    return (*result);                      
#else 
    int sa = SkGetPackedA32(src);
    int da = SkGetPackedA32(dst);
    int a = srcover_byte(sa, da);
    int r = overlay_byte(SkGetPackedR32(src), SkGetPackedR32(dst), sa, da);
    int g = overlay_byte(SkGetPackedG32(src), SkGetPackedG32(dst), sa, da);
    int b = overlay_byte(SkGetPackedB32(src), SkGetPackedB32(dst), sa, da);
    return SkPackARGB32(a, r, g, b);
#endif
}

// kDarken_Mode
#if defined(__ARM_HAVE_NEON)
static inline uint16x8_t SkDiv255Round_neon8_16_16(uint16x8_t prod) {
    prod += vdupq_n_u16(128);
    prod += vshrq_n_u16(prod, 8);

    return vshrq_n_u16(prod, 8);
}
template <bool lighten>
static inline uint8x8_t lighten_darken_color(uint8x8_t sc, uint8x8_t dc,
                                             uint8x8_t sa, uint8x8_t da) {
    uint16x8_t sd, ds, cmp, tmp, tmp2;

    // Prepare
    sd = vmull_u8(sc, da);
    ds = vmull_u8(dc, sa);

    // Do test
    if (lighten) {
        cmp = vcgtq_u16(sd, ds);
    } else {
        cmp = vcltq_u16(sd, ds);
    }

    // Assign if
    tmp = vaddl_u8(sc, dc);
    tmp2 = tmp;
    tmp -= SkDiv255Round_neon8_16_16(ds);

    // Calc else
    tmp2 -= SkDiv255Round_neon8_16_16(sd);

    // Insert where needed
    tmp = vbslq_u16(cmp, tmp, tmp2);

    return vmovn_u16(tmp);
}
static inline uint8x8_t darken_color(uint8x8_t sc, uint8x8_t dc,
                                     uint8x8_t sa, uint8x8_t da) {
    return lighten_darken_color<false>(sc, dc, sa, da);
}
static inline uint8x8_t lighten_color(uint8x8_t sc, uint8x8_t dc,
                                      uint8x8_t sa, uint8x8_t da) {
    return lighten_darken_color<true>(sc, dc, sa, da);
}
#endif
static inline int darken_byte(int sc, int dc, int sa, int da) {
    int sd = sc * da;
    int ds = dc * sa;
    if (sd < ds) {
        // srcover
        return sc + dc - SkDiv255Round(ds);
    } else {
        // dstover
        return dc + sc - SkDiv255Round(sd);
    }
}
static SkPMColor darken_modeproc(SkPMColor src, SkPMColor dst) {
#if defined(__ARM_HAVE_NEON)
    uint32_t result[1]={0};
    uint8x8x4_t src_neon = vld4_u8((uint8_t*)&src);//uint32_t * -> uint8_t* -> uint8x8x4_t
    uint8x8x4_t dst_neon = vld4_u8((uint8_t*)&dst);
    uint8x8x4_t ret;

    ret.val[NEON_A] = srcover_color(src_neon.val[NEON_A], dst_neon.val[NEON_A]);
    ret.val[NEON_R] = darken_color(src_neon.val[NEON_R], dst_neon.val[NEON_R],
                                      src_neon.val[NEON_A], dst_neon.val[NEON_A]);
    ret.val[NEON_G] = darken_color(src_neon.val[NEON_G], dst_neon.val[NEON_G],
                                      src_neon.val[NEON_A], dst_neon.val[NEON_A]);
    ret.val[NEON_B] = darken_color(src_neon.val[NEON_B], dst_neon.val[NEON_B],
                                      src_neon.val[NEON_A], dst_neon.val[NEON_A]);
    vst4_u8((uint8_t *)result,ret);//uint8x8x4_t -> uint8_t * -> uint32_t *
    return (*result);    
#else
    int sa = SkGetPackedA32(src);
    int da = SkGetPackedA32(dst);
    int a = srcover_byte(sa, da);
    int r = darken_byte(SkGetPackedR32(src), SkGetPackedR32(dst), sa, da);
    int g = darken_byte(SkGetPackedG32(src), SkGetPackedG32(dst), sa, da);
    int b = darken_byte(SkGetPackedB32(src), SkGetPackedB32(dst), sa, da);
    return SkPackARGB32(a, r, g, b);
#endif
}

// kLighten_Mode
static inline int lighten_byte(int sc, int dc, int sa, int da) {
    int sd = sc * da;
    int ds = dc * sa;
    if (sd > ds) {
        // srcover
        return sc + dc - SkDiv255Round(ds);
    } else {
        // dstover
        return dc + sc - SkDiv255Round(sd);
    }
}
static SkPMColor lighten_modeproc(SkPMColor src, SkPMColor dst) {
#if defined(__ARM_HAVE_NEON)
    uint32_t result[1]={0};
    uint8x8x4_t src_neon = vld4_u8((uint8_t*)&src);//uint32_t * -> uint8_t* -> uint8x8x4_t
    uint8x8x4_t dst_neon = vld4_u8((uint8_t*)&dst);
    uint8x8x4_t ret;

    ret.val[NEON_A] = srcover_color(src_neon.val[NEON_A], dst_neon.val[NEON_A]);
    ret.val[NEON_R] = lighten_color(src_neon.val[NEON_R], dst_neon.val[NEON_R],
                                      src_neon.val[NEON_A], dst_neon.val[NEON_A]);
    ret.val[NEON_G] = lighten_color(src_neon.val[NEON_G], dst_neon.val[NEON_G],
                                      src_neon.val[NEON_A], dst_neon.val[NEON_A]);
    ret.val[NEON_B] = lighten_color(src_neon.val[NEON_B], dst_neon.val[NEON_B],
                                      src_neon.val[NEON_A], dst_neon.val[NEON_A]);
    vst4_u8((uint8_t *)result,ret);//uint8x8x4_t -> uint8_t * -> uint32_t *
    return (*result);   
#else
    int sa = SkGetPackedA32(src);
    int da = SkGetPackedA32(dst);
    int a = srcover_byte(sa, da);
    int r = lighten_byte(SkGetPackedR32(src), SkGetPackedR32(dst), sa, da);
    int g = lighten_byte(SkGetPackedG32(src), SkGetPackedG32(dst), sa, da);
    int b = lighten_byte(SkGetPackedB32(src), SkGetPackedB32(dst), sa, da);
    return SkPackARGB32(a, r, g, b);
#endif
}

// kColorDodge_Mode
static inline int colordodge_byte(int sc, int dc, int sa, int da) {
    int diff = sa - sc;
    int rc;
    if (0 == dc) {
        return SkAlphaMulAlpha(sc, 255 - da);
    } else if (0 == diff) {
        rc = sa * da + sc * (255 - da) + dc * (255 - sa);
    } else {
        diff = dc * sa / diff;
        rc = sa * ((da < diff) ? da : diff) + sc * (255 - da) + dc * (255 - sa);
    }
    return clamp_div255round(rc);
}
static SkPMColor colordodge_modeproc(SkPMColor src, SkPMColor dst) {
    int sa = SkGetPackedA32(src);
    int da = SkGetPackedA32(dst);
    int a = srcover_byte(sa, da);
    int r = colordodge_byte(SkGetPackedR32(src), SkGetPackedR32(dst), sa, da);
    int g = colordodge_byte(SkGetPackedG32(src), SkGetPackedG32(dst), sa, da);
    int b = colordodge_byte(SkGetPackedB32(src), SkGetPackedB32(dst), sa, da);
    return SkPackARGB32(a, r, g, b);
}

// kColorBurn_Mode
static inline int colorburn_byte(int sc, int dc, int sa, int da) {
    int rc;
    if (dc == da) {
        rc = sa * da + sc * (255 - da) + dc * (255 - sa);
    } else if (0 == sc) {
        return SkAlphaMulAlpha(dc, 255 - sa);
    } else {
        int tmp = (da - dc) * sa / sc;
        rc = sa * (da - ((da < tmp) ? da : tmp))
            + sc * (255 - da) + dc * (255 - sa);
    }
    return clamp_div255round(rc);
}
static SkPMColor colorburn_modeproc(SkPMColor src, SkPMColor dst) {
    int sa = SkGetPackedA32(src);
    int da = SkGetPackedA32(dst);
    int a = srcover_byte(sa, da);
    int r = colorburn_byte(SkGetPackedR32(src), SkGetPackedR32(dst), sa, da);
    int g = colorburn_byte(SkGetPackedG32(src), SkGetPackedG32(dst), sa, da);
    int b = colorburn_byte(SkGetPackedB32(src), SkGetPackedB32(dst), sa, da);
    return SkPackARGB32(a, r, g, b);
}

// kHardLight_Mode
static inline int hardlight_byte(int sc, int dc, int sa, int da) {
    int rc;
    if (2 * sc <= sa) {
        rc = 2 * sc * dc;
    } else {
        rc = sa * da - 2 * (da - dc) * (sa - sc);
    }
    return clamp_div255round(rc + sc * (255 - da) + dc * (255 - sa));
}
#if defined(__ARM_HAVE_NEON)
static inline uint8x8_t hardlight_color(uint8x8_t sc, uint8x8_t dc,
                                        uint8x8_t sa, uint8x8_t da) {
    return overlay_hardlight_color<false>(sc, dc, sa, da);
}
#endif
static SkPMColor hardlight_modeproc(SkPMColor src, SkPMColor dst) {
#if defined(__ARM_HAVE_NEON)
    uint32_t result[1]={0};
    uint8x8x4_t src_neon = vld4_u8((uint8_t*)&src);//uint32_t * -> uint8_t* -> uint8x8x4_t
    uint8x8x4_t dst_neon = vld4_u8((uint8_t*)&dst);
    uint8x8x4_t ret;

    ret.val[NEON_A] = srcover_color(src_neon.val[NEON_A], dst_neon.val[NEON_A]);
    ret.val[NEON_R] = hardlight_color(src_neon.val[NEON_R], dst_neon.val[NEON_R],
                                      src_neon.val[NEON_A], dst_neon.val[NEON_A]);
    ret.val[NEON_G] = hardlight_color(src_neon.val[NEON_G], dst_neon.val[NEON_G],
                                      src_neon.val[NEON_A], dst_neon.val[NEON_A]);
    ret.val[NEON_B] = hardlight_color(src_neon.val[NEON_B], dst_neon.val[NEON_B],
                                      src_neon.val[NEON_A], dst_neon.val[NEON_A]);
    vst4_u8((uint8_t *)result,ret);//uint8x8x4_t -> uint8_t * -> uint32_t *
    return (*result);    
#else 
    int sa = SkGetPackedA32(src);
    int da = SkGetPackedA32(dst);
    int a = srcover_byte(sa, da);
    int r = hardlight_byte(SkGetPackedR32(src), SkGetPackedR32(dst), sa, da);
    int g = hardlight_byte(SkGetPackedG32(src), SkGetPackedG32(dst), sa, da);
    int b = hardlight_byte(SkGetPackedB32(src), SkGetPackedB32(dst), sa, da);
    return SkPackARGB32(a, r, g, b);
#endif
}

// returns 255 * sqrt(n/255)
static U8CPU sqrt_unit_byte(U8CPU n) {
    return SkSqrtBits(n, 15+4);
}

// kSoftLight_Mode
static inline int softlight_byte(int sc, int dc, int sa, int da) {
    int m = da ? dc * 256 / da : 0;
    int rc;
    if (2 * sc <= sa) {
        rc = dc * (sa + ((2 * sc - sa) * (256 - m) >> 8));
    } else if (4 * dc <= da) {
        int tmp = (4 * m * (4 * m + 256) * (m - 256) >> 16) + 7 * m;
        rc = dc * sa + (da * (2 * sc - sa) * tmp >> 8);
    } else {
        int tmp = sqrt_unit_byte(m) - m;
        rc = dc * sa + (da * (2 * sc - sa) * tmp >> 8);
    }
    return clamp_div255round(rc + sc * (255 - da) + dc * (255 - sa));
}
static SkPMColor softlight_modeproc(SkPMColor src, SkPMColor dst) {
    int sa = SkGetPackedA32(src);
    int da = SkGetPackedA32(dst);
    int a = srcover_byte(sa, da);
    int r = softlight_byte(SkGetPackedR32(src), SkGetPackedR32(dst), sa, da);
    int g = softlight_byte(SkGetPackedG32(src), SkGetPackedG32(dst), sa, da);
    int b = softlight_byte(SkGetPackedB32(src), SkGetPackedB32(dst), sa, da);
    return SkPackARGB32(a, r, g, b);
}

// kDifference_Mode
static inline int difference_byte(int sc, int dc, int sa, int da) {
    int tmp = SkMin32(sc * da, dc * sa);
    return clamp_signed_byte(sc + dc - 2 * SkDiv255Round(tmp));
}
#if defined(__ARM_HAVE_NEON)
static inline uint8x8_t difference_color(uint8x8_t sc, uint8x8_t dc,
                                         uint8x8_t sa, uint8x8_t da) {
    uint16x8_t sd, ds, tmp;
    int16x8_t val;

    sd = vmull_u8(sc, da);
    ds = vmull_u8(dc, sa);

    tmp = vminq_u16(sd, ds);
    tmp = SkDiv255Round_neon8_16_16(tmp);
    tmp = vshlq_n_u16(tmp, 1);

    val = vreinterpretq_s16_u16(vaddl_u8(sc, dc));

    val -= vreinterpretq_s16_u16(tmp);

    val = vmaxq_s16(val, vdupq_n_s16(0));
    val = vminq_s16(val, vdupq_n_s16(255));

    return vmovn_u16(vreinterpretq_u16_s16(val));
}
#endif
static SkPMColor difference_modeproc(SkPMColor src, SkPMColor dst) {
#if defined(__ARM_HAVE_NEON)
    uint32_t result[1]={0};
    uint8x8x4_t src_neon = vld4_u8((uint8_t*)&src);//uint32_t * -> uint8_t* -> uint8x8x4_t
    uint8x8x4_t dst_neon = vld4_u8((uint8_t*)&dst);
    uint8x8x4_t ret;

    ret.val[NEON_A] = srcover_color(src_neon.val[NEON_A], dst_neon.val[NEON_A]);
    ret.val[NEON_R] = difference_color(src_neon.val[NEON_R], dst_neon.val[NEON_R],
                                      src_neon.val[NEON_A], dst_neon.val[NEON_A]);
    ret.val[NEON_G] = difference_color(src_neon.val[NEON_G], dst_neon.val[NEON_G],
                                      src_neon.val[NEON_A], dst_neon.val[NEON_A]);
    ret.val[NEON_B] = difference_color(src_neon.val[NEON_B], dst_neon.val[NEON_B],
                                      src_neon.val[NEON_A], dst_neon.val[NEON_A]);
    vst4_u8((uint8_t *)result,ret);//uint8x8x4_t -> uint8_t * -> uint32_t *
    return (*result);  
#else
    int sa = SkGetPackedA32(src);
    int da = SkGetPackedA32(dst);
    int a = srcover_byte(sa, da);
    int r = difference_byte(SkGetPackedR32(src), SkGetPackedR32(dst), sa, da);
    int g = difference_byte(SkGetPackedG32(src), SkGetPackedG32(dst), sa, da);
    int b = difference_byte(SkGetPackedB32(src), SkGetPackedB32(dst), sa, da);
    return SkPackARGB32(a, r, g, b);
#endif
}

// kExclusion_Mode


static inline int exclusion_byte(int sc, int dc, int sa, int da) {
    // this equations is wacky, wait for SVG to confirm it
    int r = sc * da + dc * sa - 2 * sc * dc + sc * (255 - da) + dc * (255 - sa);
    return clamp_div255round(r);
}
#if defined(__ARM_HAVE_NEON)
static inline uint8x8_t exclusion_color(uint8x8_t sc, uint8x8_t dc,
                                        uint8x8_t sa, uint8x8_t da) {
    /* The equation can be simplified to 255(sc + dc) - 2 * sc * dc */

    uint16x8_t sc_plus_dc, scdc, const255;
    int32x4_t term1_1, term1_2, term2_1, term2_2;

    /* Calc (sc + dc) and (sc * dc) */
    sc_plus_dc = vaddl_u8(sc, dc);
    scdc = vmull_u8(sc, dc);

    /* Prepare constants */
    const255 = vdupq_n_u16(255);

    /* Calc the first term */
    term1_1 = vreinterpretq_s32_u32(
                vmull_u16(vget_low_u16(const255), vget_low_u16(sc_plus_dc)));
    term1_2 = vreinterpretq_s32_u32(
                vmull_u16(vget_high_u16(const255), vget_high_u16(sc_plus_dc)));

    /* Calc the second term */
    term2_1 = vreinterpretq_s32_u32(vshll_n_u16(vget_low_u16(scdc), 1));
    term2_2 = vreinterpretq_s32_u32(vshll_n_u16(vget_high_u16(scdc), 1));

    return clamp_div255round_simd8_32(term1_1 - term2_1, term1_2 - term2_2);
}
#endif
static SkPMColor exclusion_modeproc(SkPMColor src, SkPMColor dst) {
#if defined(__ARM_HAVE_NEON)
    uint32_t result[1]={0};
    uint8x8x4_t src_neon = vld4_u8((uint8_t*)&src);//uint32_t * -> uint8_t* -> uint8x8x4_t
    uint8x8x4_t dst_neon = vld4_u8((uint8_t*)&dst);
    uint8x8x4_t ret;

    ret.val[NEON_A] = srcover_color(src_neon.val[NEON_A], dst_neon.val[NEON_A]);
    ret.val[NEON_R] = exclusion_color(src_neon.val[NEON_R], dst_neon.val[NEON_R],
                                      src_neon.val[NEON_A], dst_neon.val[NEON_A]);
    ret.val[NEON_G] = exclusion_color(src_neon.val[NEON_G], dst_neon.val[NEON_G],
                                      src_neon.val[NEON_A], dst_neon.val[NEON_A]);
    ret.val[NEON_B] = exclusion_color(src_neon.val[NEON_B], dst_neon.val[NEON_B],
                                      src_neon.val[NEON_A], dst_neon.val[NEON_A]);
    vst4_u8((uint8_t *)result,ret);//uint8x8x4_t -> uint8_t * -> uint32_t *
    return (*result);    
#else
    int sa = SkGetPackedA32(src);
    int da = SkGetPackedA32(dst);
    int a = srcover_byte(sa, da);
    int r = exclusion_byte(SkGetPackedR32(src), SkGetPackedR32(dst), sa, da);
    int g = exclusion_byte(SkGetPackedG32(src), SkGetPackedG32(dst), sa, da);
    int b = exclusion_byte(SkGetPackedB32(src), SkGetPackedB32(dst), sa, da);
    return SkPackARGB32(a, r, g, b);
#endif
}
// kMultiply_Mode
static SkPMColor multiply_modeproc(SkPMColor src, SkPMColor dst) {
#if defined(__ARM_HAVE_NEON)
    uint32_t result[1]={0};
    uint8x8x4_t src_neon = vld4_u8((uint8_t*)&src);//uint32_t * -> uint8_t* -> uint8x8x4_t
    uint8x8x4_t dst_neon = vld4_u8((uint8_t*)&dst);
    uint8x8x4_t ret;

    ret.val[NEON_A] = srcover_color(src_neon.val[NEON_A], dst_neon.val[NEON_A]);
    ret.val[NEON_R] = blendfunc_multiply_color(src_neon.val[NEON_R], dst_neon.val[NEON_R],
                                               src_neon.val[NEON_A], dst_neon.val[NEON_A]);
    ret.val[NEON_G] = blendfunc_multiply_color(src_neon.val[NEON_G], dst_neon.val[NEON_G],
                                               src_neon.val[NEON_A], dst_neon.val[NEON_A]);
    ret.val[NEON_B] = blendfunc_multiply_color(src_neon.val[NEON_B], dst_neon.val[NEON_B],
                                               src_neon.val[NEON_A], dst_neon.val[NEON_A]);    
    vst4_u8((uint8_t *)result,ret);//uint8x8x4_t -> uint8_t * -> uint32_t *
    return (*result); 
#else
    int a = SkAlphaMulAlpha(SkGetPackedA32(src), SkGetPackedA32(dst));
    int r = SkAlphaMulAlpha(SkGetPackedR32(src), SkGetPackedR32(dst));
    int g = SkAlphaMulAlpha(SkGetPackedG32(src), SkGetPackedG32(dst));
    int b = SkAlphaMulAlpha(SkGetPackedB32(src), SkGetPackedB32(dst));
    return SkPackARGB32(a, r, g, b);
#endif
}

// The CSS compositing spec introduces the following formulas:
// (See https://dvcs.w3.org/hg/FXTF/rawfile/tip/compositing/index.html#blendingnonseparable)
// SkComputeLuminance is similar to this formula but it uses the new definition from Rec. 709
// while PDF and CG uses the one from Rec. Rec. 601
// See http://www.glennchan.info/articles/technical/hd-versus-sd-color-space/hd-versus-sd-color-space.htm
static inline int Lum(int r, int g, int b)
{
    return SkDiv255Round(r * 77 + g * 150 + b * 28);
}

static inline int min2(int a, int b) { return a < b ? a : b; }
static inline int max2(int a, int b) { return a > b ? a : b; }
#define minimum(a, b, c) min2(min2(a, b), c)
#define maximum(a, b, c) max2(max2(a, b), c)

static inline int Sat(int r, int g, int b) {
    return maximum(r, g, b) - minimum(r, g, b);
}

static inline void setSaturationComponents(int* Cmin, int* Cmid, int* Cmax, int s) {
    if(*Cmax > *Cmin) {
        *Cmid =  SkMulDiv(*Cmid - *Cmin, s, *Cmax - *Cmin);
        *Cmax = s;
    } else {
        *Cmax = 0;
        *Cmid = 0;
    }

    *Cmin = 0;
}

static inline void SetSat(int* r, int* g, int* b, int s) {
    if(*r <= *g) {
        if(*g <= *b) {
            setSaturationComponents(r, g, b, s);
        } else if(*r <= *b) {
            setSaturationComponents(r, b, g, s);
        } else {
            setSaturationComponents(b, r, g, s);
        }
    } else if(*r <= *b) {
        setSaturationComponents(g, r, b, s);
    } else if(*g <= *b) {
        setSaturationComponents(g, b, r, s);
    } else {
        setSaturationComponents(b, g, r, s);
    }
}

static inline void clipColor(int* r, int* g, int* b, int a) {
    int L = Lum(*r, *g, *b);
    int n = minimum(*r, *g, *b);
    int x = maximum(*r, *g, *b);
    if(n < 0) {
       *r = L + SkMulDiv(*r - L, L, L - n);
       *g = L + SkMulDiv(*g - L, L, L - n);
       *b = L + SkMulDiv(*b - L, L, L - n);
    }

    if (x > a) {
       *r = L + SkMulDiv(*r - L, a - L, x - L);
       *g = L + SkMulDiv(*g - L, a - L, x - L);
       *b = L + SkMulDiv(*b - L, a - L, x - L);
    }
}

static inline void SetLum(int* r, int* g, int* b, int a, int l) {
  int d = l - Lum(*r, *g, *b);
  *r +=  d;
  *g +=  d;
  *b +=  d;

  clipColor(r, g, b, a);
}

// non-separable blend modes are done in non-premultiplied alpha
#define  blendfunc_nonsep_byte(sc, dc, sa, da, blendval) \
  clamp_div255round(sc * (255 - da) +  dc * (255 - sa) + blendval)

// kHue_Mode
// B(Cb, Cs) = SetLum(SetSat(Cs, Sat(Cb)), Lum(Cb))
// Create a color with the hue of the source color and the saturation and luminosity of the backdrop color.
static SkPMColor hue_modeproc(SkPMColor src, SkPMColor dst) {
    int sr = SkGetPackedR32(src);
    int sg = SkGetPackedG32(src);
    int sb = SkGetPackedB32(src);
    int sa = SkGetPackedA32(src);

    int dr = SkGetPackedR32(dst);
    int dg = SkGetPackedG32(dst);
    int db = SkGetPackedB32(dst);
    int da = SkGetPackedA32(dst);
    int Sr, Sg, Sb;

    if(sa && da) {
        Sr = sr * sa;
        Sg = sg * sa;
        Sb = sb * sa;
        SetSat(&Sr, &Sg, &Sb, Sat(dr, dg, db) * sa);
        SetLum(&Sr, &Sg, &Sb, sa * da, Lum(dr, dg, db) * sa);
    } else {
        Sr = 0;
        Sg = 0;
        Sb = 0;
    }

    int a = srcover_byte(sa, da);
    int r = blendfunc_nonsep_byte(sr, dr, sa, da, Sr);
    int g = blendfunc_nonsep_byte(sg, dg, sa, da, Sg);
    int b = blendfunc_nonsep_byte(sb, db, sa, da, Sb);
    return SkPackARGB32(a, r, g, b);
}

// kSaturation_Mode
// B(Cb, Cs) = SetLum(SetSat(Cb, Sat(Cs)), Lum(Cb))
// Create a color with the saturation of the source color and the hue and luminosity of the backdrop color.
static SkPMColor saturation_modeproc(SkPMColor src, SkPMColor dst) {
    int sr = SkGetPackedR32(src);
    int sg = SkGetPackedG32(src);
    int sb = SkGetPackedB32(src);
    int sa = SkGetPackedA32(src);

    int dr = SkGetPackedR32(dst);
    int dg = SkGetPackedG32(dst);
    int db = SkGetPackedB32(dst);
    int da = SkGetPackedA32(dst);
    int Dr, Dg, Db;

    if(sa && da) {
        Dr = dr * sa;
        Dg = dg * sa;
        Db = db * sa;
        SetSat(&Dr, &Dg, &Db, Sat(sr, sg, sb) * da);
        SetLum(&Dr, &Dg, &Db, sa * da, Lum(dr, dg, db) * sa);
    } else {
        Dr = 0;
        Dg = 0;
        Db = 0;
    }

    int a = srcover_byte(sa, da);
    int r = blendfunc_nonsep_byte(sr, dr, sa, da, Dr);
    int g = blendfunc_nonsep_byte(sg, dg, sa, da, Dg);
    int b = blendfunc_nonsep_byte(sb, db, sa, da, Db);
    return SkPackARGB32(a, r, g, b);
}

// kColor_Mode
// B(Cb, Cs) = SetLum(Cs, Lum(Cb))
// Create a color with the hue and saturation of the source color and the luminosity of the backdrop color.
static SkPMColor color_modeproc(SkPMColor src, SkPMColor dst) {
    int sr = SkGetPackedR32(src);
    int sg = SkGetPackedG32(src);
    int sb = SkGetPackedB32(src);
    int sa = SkGetPackedA32(src);

    int dr = SkGetPackedR32(dst);
    int dg = SkGetPackedG32(dst);
    int db = SkGetPackedB32(dst);
    int da = SkGetPackedA32(dst);
    int Sr, Sg, Sb;

    if(sa && da) {
        Sr = sr * da;
        Sg = sg * da;
        Sb = sb * da;
        SetLum(&Sr, &Sg, &Sb, sa * da, Lum(dr, dg, db) * sa);
    } else {
        Sr = 0;
        Sg = 0;
        Sb = 0;
    }

    int a = srcover_byte(sa, da);
    int r = blendfunc_nonsep_byte(sr, dr, sa, da, Sr);
    int g = blendfunc_nonsep_byte(sg, dg, sa, da, Sg);
    int b = blendfunc_nonsep_byte(sb, db, sa, da, Sb);
    return SkPackARGB32(a, r, g, b);
}

// kLuminosity_Mode
// B(Cb, Cs) = SetLum(Cb, Lum(Cs))
// Create a color with the luminosity of the source color and the hue and saturation of the backdrop color.
static SkPMColor luminosity_modeproc(SkPMColor src, SkPMColor dst) {
    int sr = SkGetPackedR32(src);
    int sg = SkGetPackedG32(src);
    int sb = SkGetPackedB32(src);
    int sa = SkGetPackedA32(src);

    int dr = SkGetPackedR32(dst);
    int dg = SkGetPackedG32(dst);
    int db = SkGetPackedB32(dst);
    int da = SkGetPackedA32(dst);
    int Dr, Dg, Db;

    if(sa && da) {
        Dr = dr * sa;
        Dg = dg * sa;
        Db = db * sa;
        SetLum(&Dr, &Dg, &Db, sa * da, Lum(sr, sg, sb) * da);
    } else {
        Dr = 0;
        Dg = 0;
        Db = 0;
    }

    int a = srcover_byte(sa, da);
    int r = blendfunc_nonsep_byte(sr, dr, sa, da, Dr);
    int g = blendfunc_nonsep_byte(sg, dg, sa, da, Dg);
    int b = blendfunc_nonsep_byte(sb, db, sa, da, Db);
    return SkPackARGB32(a, r, g, b);
}


struct ProcCoeff {
    SkXfermodeProc      fProc;
    SkXfermode::Coeff   fSC;
    SkXfermode::Coeff   fDC;
};

#define CANNOT_USE_COEFF    SkXfermode::Coeff(-1)

static const ProcCoeff gProcCoeffs[] = {
    { clear_modeproc,   SkXfermode::kZero_Coeff,    SkXfermode::kZero_Coeff },
    { src_modeproc,     SkXfermode::kOne_Coeff,     SkXfermode::kZero_Coeff },
    { dst_modeproc,     SkXfermode::kZero_Coeff,    SkXfermode::kOne_Coeff },
    { srcover_modeproc, SkXfermode::kOne_Coeff,     SkXfermode::kISA_Coeff },
    { dstover_modeproc, SkXfermode::kIDA_Coeff,     SkXfermode::kOne_Coeff },
    { srcin_modeproc,   SkXfermode::kDA_Coeff,      SkXfermode::kZero_Coeff },
    { dstin_modeproc,   SkXfermode::kZero_Coeff,    SkXfermode::kSA_Coeff },
    { srcout_modeproc,  SkXfermode::kIDA_Coeff,     SkXfermode::kZero_Coeff },
    { dstout_modeproc,  SkXfermode::kZero_Coeff,    SkXfermode::kISA_Coeff },
    { srcatop_modeproc, SkXfermode::kDA_Coeff,      SkXfermode::kISA_Coeff },
    { dstatop_modeproc, SkXfermode::kIDA_Coeff,     SkXfermode::kSA_Coeff },
    { xor_modeproc,     SkXfermode::kIDA_Coeff,     SkXfermode::kISA_Coeff },

    { plus_modeproc,    SkXfermode::kOne_Coeff,     SkXfermode::kOne_Coeff },
    { modulate_modeproc,SkXfermode::kZero_Coeff,    SkXfermode::kSC_Coeff },
    { screen_modeproc,  SkXfermode::kOne_Coeff,     SkXfermode::kISC_Coeff },
    { overlay_modeproc,     CANNOT_USE_COEFF,       CANNOT_USE_COEFF },
    { darken_modeproc,      CANNOT_USE_COEFF,       CANNOT_USE_COEFF },
    { lighten_modeproc,     CANNOT_USE_COEFF,       CANNOT_USE_COEFF },
    { colordodge_modeproc,  CANNOT_USE_COEFF,       CANNOT_USE_COEFF },
    { colorburn_modeproc,   CANNOT_USE_COEFF,       CANNOT_USE_COEFF },
    { hardlight_modeproc,   CANNOT_USE_COEFF,       CANNOT_USE_COEFF },
    { softlight_modeproc,   CANNOT_USE_COEFF,       CANNOT_USE_COEFF },
    { difference_modeproc,  CANNOT_USE_COEFF,       CANNOT_USE_COEFF },
    { exclusion_modeproc,   CANNOT_USE_COEFF,       CANNOT_USE_COEFF },
    { multiply_modeproc,    CANNOT_USE_COEFF,       CANNOT_USE_COEFF },
    { hue_modeproc,         CANNOT_USE_COEFF,       CANNOT_USE_COEFF },
    { saturation_modeproc,  CANNOT_USE_COEFF,       CANNOT_USE_COEFF },
    { color_modeproc,       CANNOT_USE_COEFF,       CANNOT_USE_COEFF },
    { luminosity_modeproc,  CANNOT_USE_COEFF,       CANNOT_USE_COEFF },
};

///////////////////////////////////////////////////////////////////////////////

bool SkXfermode::asCoeff(Coeff* src, Coeff* dst) const {
    return false;
}

bool SkXfermode::asMode(Mode* mode) const {
    return false;
}

bool SkXfermode::asNewEffectOrCoeff(GrContext*, GrEffectRef**, Coeff* src, Coeff* dst, GrTexture*) const {
    return this->asCoeff(src, dst);
}

bool SkXfermode::AsNewEffectOrCoeff(SkXfermode* xfermode,
                                    GrContext* context,
                                    GrEffectRef** effect,
                                    Coeff* src,
                                    Coeff* dst,
                                    GrTexture* background) {
    if (NULL == xfermode) {
        return ModeAsCoeff(kSrcOver_Mode, src, dst);
    } else {
        return xfermode->asNewEffectOrCoeff(context, effect, src, dst, background);
    }
}

SkPMColor SkXfermode::xferColor(SkPMColor src, SkPMColor dst) const{
    // no-op. subclasses should override this
    return dst;
}

typedef uint8x8x4_t (*SkXfermodeProcSIMD)(uint8x8x4_t src, uint8x8x4_t dst);
void SkXfermode::xfer32(SkPMColor* SK_RESTRICT dst,
                        const SkPMColor* SK_RESTRICT src, int count,
                        const SkAlpha* SK_RESTRICT aa) const {
    SkASSERT(dst && src && count >= 0);

    if (NULL == aa) {
        for (int i = count - 1; i >= 0; --i) {
            dst[i] = this->xferColor(src[i], dst[i]);
        }
    } else {
        for (int i = count - 1; i >= 0; --i) {
            unsigned a = aa[i];
            if (0 != a) {
                SkPMColor dstC = dst[i];
                SkPMColor C = this->xferColor(src[i], dstC);
                if (0xFF != a) {
                    C = SkFourByteInterp(C, dstC, a);
                }
                dst[i] = C;
            }
        }
    }
}

void SkXfermode::xfer16(uint16_t* dst,
                        const SkPMColor* SK_RESTRICT src, int count,
                        const SkAlpha* SK_RESTRICT aa) const {
    SkASSERT(dst && src && count >= 0);

    if (NULL == aa) {
        for (int i = count - 1; i >= 0; --i) {
            SkPMColor dstC = SkPixel16ToPixel32(dst[i]);
            dst[i] = SkPixel32ToPixel16_ToU16(this->xferColor(src[i], dstC));
        }
    } else {
        for (int i = count - 1; i >= 0; --i) {
            unsigned a = aa[i];
            if (0 != a) {
                SkPMColor dstC = SkPixel16ToPixel32(dst[i]);
                SkPMColor C = this->xferColor(src[i], dstC);
                if (0xFF != a) {
                    C = SkFourByteInterp(C, dstC, a);
                }
                dst[i] = SkPixel32ToPixel16_ToU16(C);
            }
        }
    }
}

void SkXfermode::xferA8(SkAlpha* SK_RESTRICT dst,
                        const SkPMColor src[], int count,
                        const SkAlpha* SK_RESTRICT aa) const {
    SkASSERT(dst && src && count >= 0);

    if (NULL == aa) {
        for (int i = count - 1; i >= 0; --i) {
            SkPMColor res = this->xferColor(src[i], (dst[i] << SK_A32_SHIFT));
            dst[i] = SkToU8(SkGetPackedA32(res));
        }
    } else {
        for (int i = count - 1; i >= 0; --i) {
            unsigned a = aa[i];
            if (0 != a) {
                SkAlpha dstA = dst[i];
                unsigned A = SkGetPackedA32(this->xferColor(src[i],
                                            (SkPMColor)(dstA << SK_A32_SHIFT)));
                if (0xFF != a) {
                    A = SkAlphaBlend(A, dstA, SkAlpha255To256(a));
                }
                dst[i] = SkToU8(A);
            }
        }
    }
}

///////////////////////////////////////////////////////////////////////////////

void SkProcXfermode::xfer32(SkPMColor* SK_RESTRICT dst,
                            const SkPMColor* SK_RESTRICT src, int count,
                            const SkAlpha* SK_RESTRICT aa) const {
    SkASSERT(dst && src && count >= 0);

    SkXfermodeProc proc = fProc;

    if (NULL != proc) {
        if (NULL == aa) {
            for (int i = count - 1; i >= 0; --i) {
                dst[i] = proc(src[i], dst[i]);
            }
        } else {
            for (int i = count - 1; i >= 0; --i) {
                unsigned a = aa[i];
                if (0 != a) {
                    SkPMColor dstC = dst[i];
                    SkPMColor C = proc(src[i], dstC);
                    if (a != 0xFF) {
                        C = SkFourByteInterp(C, dstC, a);
                    }
                    dst[i] = C;
                }
            }
        }
    }
}

void SkProcXfermode::xfer16(uint16_t* SK_RESTRICT dst,
                            const SkPMColor* SK_RESTRICT src, int count,
                            const SkAlpha* SK_RESTRICT aa) const {
    SkASSERT(dst && src && count >= 0);

    SkXfermodeProc proc = fProc;

    if (NULL != proc) {
        if (NULL == aa) {
            for (int i = count - 1; i >= 0; --i) {
                SkPMColor dstC = SkPixel16ToPixel32(dst[i]);
                dst[i] = SkPixel32ToPixel16_ToU16(proc(src[i], dstC));
            }
        } else {
            for (int i = count - 1; i >= 0; --i) {
                unsigned a = aa[i];
                if (0 != a) {
                    SkPMColor dstC = SkPixel16ToPixel32(dst[i]);
                    SkPMColor C = proc(src[i], dstC);
                    if (0xFF != a) {
                        C = SkFourByteInterp(C, dstC, a);
                    }
                    dst[i] = SkPixel32ToPixel16_ToU16(C);
                }
            }
        }
    }
}

void SkProcXfermode::xferA8(SkAlpha* SK_RESTRICT dst,
                            const SkPMColor* SK_RESTRICT src, int count,
                            const SkAlpha* SK_RESTRICT aa) const {
    SkASSERT(dst && src && count >= 0);

    SkXfermodeProc proc = fProc;

    if (NULL != proc) {
        if (NULL == aa) {
            for (int i = count - 1; i >= 0; --i) {
                SkPMColor res = proc(src[i], dst[i] << SK_A32_SHIFT);
                dst[i] = SkToU8(SkGetPackedA32(res));
            }
        } else {
            for (int i = count - 1; i >= 0; --i) {
                unsigned a = aa[i];
                if (0 != a) {
                    SkAlpha dstA = dst[i];
                    SkPMColor res = proc(src[i], dstA << SK_A32_SHIFT);
                    unsigned A = SkGetPackedA32(res);
                    if (0xFF != a) {
                        A = SkAlphaBlend(A, dstA, SkAlpha255To256(a));
                    }
                    dst[i] = SkToU8(A);
                }
            }
        }
    }
}

SkProcXfermode::SkProcXfermode(SkFlattenableReadBuffer& buffer)
        : SkXfermode(buffer) {
    fProc = NULL;
    if (!buffer.isCrossProcess()) {
        fProc = (SkXfermodeProc)buffer.readFunctionPtr();
    }
}

void SkProcXfermode::flatten(SkFlattenableWriteBuffer& buffer) const {
    this->INHERITED::flatten(buffer);
    if (!buffer.isCrossProcess()) {
        buffer.writeFunctionPtr((void*)fProc);
    }
}

#ifdef SK_DEVELOPER
void SkProcXfermode::toString(SkString* str) const {
    str->appendf("SkProcXfermode: %p", fProc);
}
#endif

//////////////////////////////////////////////////////////////////////////////

#if SK_SUPPORT_GPU

#include "GrEffect.h"
#include "GrEffectUnitTest.h"
#include "GrTBackendEffectFactory.h"
#include "gl/GrGLEffect.h"
#include "gl/GrGLEffectMatrix.h"

/**
 * GrEffect that implements the all the separable xfer modes that cannot be expressed as Coeffs.
 */
class XferEffect : public GrEffect {
public:
    static bool IsSupportedMode(SkXfermode::Mode mode) {
        return mode > SkXfermode::kLastCoeffMode && mode <= SkXfermode::kLastMode;
    }

    static GrEffectRef* Create(SkXfermode::Mode mode, GrTexture* background) {
        if (!IsSupportedMode(mode)) {
            return NULL;
        } else {
            AutoEffectUnref effect(SkNEW_ARGS(XferEffect, (mode, background)));
            return CreateEffectRef(effect);
        }
    }

    virtual void getConstantColorComponents(GrColor* color,
                                            uint32_t* validFlags) const SK_OVERRIDE {
        *validFlags = 0;
    }

    virtual const GrBackendEffectFactory& getFactory() const SK_OVERRIDE {
        return GrTBackendEffectFactory<XferEffect>::getInstance();
    }

    static const char* Name() { return "XferEffect"; }

    SkXfermode::Mode mode() const { return fMode; }
    const GrTextureAccess&  backgroundAccess() const { return fBackgroundAccess; }

    class GLEffect : public GrGLEffect {
    public:
        GLEffect(const GrBackendEffectFactory& factory, const GrDrawEffect&)
            : GrGLEffect(factory )
            , fBackgroundEffectMatrix(kCoordsType) {
        }
        virtual void emitCode(GrGLShaderBuilder* builder,
                              const GrDrawEffect& drawEffect,
                              EffectKey key,
                              const char* outputColor,
                              const char* inputColor,
                              const TextureSamplerArray& samplers) SK_OVERRIDE {
            SkXfermode::Mode mode = drawEffect.castEffect<XferEffect>().mode();
            const GrTexture* backgroundTex = drawEffect.castEffect<XferEffect>().backgroundAccess().getTexture();
            const char* dstColor;
            if (backgroundTex) {
                const char* bgCoords;
                GrSLType bgCoordsType = fBackgroundEffectMatrix.emitCode(builder, key, &bgCoords, NULL, "BG");
                dstColor = "bgColor";
                builder->fsCodeAppendf("\t\tvec4 %s = ", dstColor);
                builder->appendTextureLookup(GrGLShaderBuilder::kFragment_ShaderType,
                                             samplers[0],
                                             bgCoords,
                                             bgCoordsType);
                builder->fsCodeAppendf(";\n");
            } else {
                dstColor = builder->dstColor();
            }
            GrAssert(NULL != dstColor);

            // We don't try to optimize for this case at all
            if (NULL == inputColor) {
                builder->fsCodeAppendf("\t\tconst vec4 ones = %s;\n", GrGLSLOnesVecf(4));
                inputColor = "ones";
            }
            builder->fsCodeAppendf("\t\t// SkXfermode::Mode: %s\n", SkXfermode::ModeName(mode));

            // These all perform src-over on the alpha channel.
            builder->fsCodeAppendf("\t\t%s.a = %s.a + (1.0 - %s.a) * %s.a;\n",
                                    outputColor, inputColor, inputColor, dstColor);

            switch (mode) {
                case SkXfermode::kOverlay_Mode:
                    // Overlay is Hard-Light with the src and dst reversed
                    HardLight(builder, outputColor, dstColor, inputColor);
                    break;
                case SkXfermode::kDarken_Mode:
                    builder->fsCodeAppendf("\t\t%s.rgb = min((1.0 - %s.a) * %s.rgb + %s.rgb, "
                                                            "(1.0 - %s.a) * %s.rgb + %s.rgb);\n",
                                            outputColor,
                                            inputColor, dstColor, inputColor,
                                            dstColor, inputColor, dstColor);
                    break;
                case SkXfermode::kLighten_Mode:
                    builder->fsCodeAppendf("\t\t%s.rgb = max((1.0 - %s.a) * %s.rgb + %s.rgb, "
                                                            "(1.0 - %s.a) * %s.rgb + %s.rgb);\n",
                                            outputColor,
                                            inputColor, dstColor, inputColor,
                                            dstColor, inputColor, dstColor);
                    break;
                case SkXfermode::kColorDodge_Mode:
                    ColorDodgeComponent(builder, outputColor, inputColor, dstColor, 'r');
                    ColorDodgeComponent(builder, outputColor, inputColor, dstColor, 'g');
                    ColorDodgeComponent(builder, outputColor, inputColor, dstColor, 'b');
                    break;
                case SkXfermode::kColorBurn_Mode:
                    ColorBurnComponent(builder, outputColor, inputColor, dstColor, 'r');
                    ColorBurnComponent(builder, outputColor, inputColor, dstColor, 'g');
                    ColorBurnComponent(builder, outputColor, inputColor, dstColor, 'b');
                    break;
                case SkXfermode::kHardLight_Mode:
                    HardLight(builder, outputColor, inputColor, dstColor);
                    break;
                case SkXfermode::kSoftLight_Mode:
                    builder->fsCodeAppendf("\t\tif (0.0 == %s.a) {\n", dstColor);
                    builder->fsCodeAppendf("\t\t\t%s.rgba = %s;\n", outputColor, inputColor);
                    builder->fsCodeAppendf("\t\t} else {\n");
                    SoftLightComponentPosDstAlpha(builder, outputColor, inputColor, dstColor, 'r');
                    SoftLightComponentPosDstAlpha(builder, outputColor, inputColor, dstColor, 'g');
                    SoftLightComponentPosDstAlpha(builder, outputColor, inputColor, dstColor, 'b');
                    builder->fsCodeAppendf("\t\t}\n");
                    break;
                case SkXfermode::kDifference_Mode:
                    builder->fsCodeAppendf("\t\t%s.rgb = %s.rgb + %s.rgb -"
                                                       "2.0 * min(%s.rgb * %s.a, %s.rgb * %s.a);\n",
                                           outputColor, inputColor, dstColor, inputColor, dstColor,
                                           dstColor, inputColor);
                    break;
                case SkXfermode::kExclusion_Mode:
                    builder->fsCodeAppendf("\t\t%s.rgb = %s.rgb + %s.rgb - "
                                                        "2.0 * %s.rgb * %s.rgb;\n",
                                           outputColor, dstColor, inputColor, dstColor, inputColor);
                    break;
                case SkXfermode::kMultiply_Mode:
                    builder->fsCodeAppendf("\t\t%s.rgb = (1.0 - %s.a) * %s.rgb + "
                                                        "(1.0 - %s.a) * %s.rgb + "
                                                         "%s.rgb * %s.rgb;\n",
                                           outputColor, inputColor, dstColor, dstColor, inputColor,
                                           inputColor, dstColor);
                    break;
                case SkXfermode::kHue_Mode: {
                    //  SetLum(SetSat(S * Da, Sat(D * Sa)), Sa*Da, D*Sa) + (1 - Sa) * D + (1 - Da) * S
                    SkString setSat, setLum;
                    AddSatFunction(builder, &setSat);
                    AddLumFunction(builder, &setLum);
                    builder->fsCodeAppendf("\t\tvec4 dstSrcAlpha = %s * %s.a;\n",
                                           dstColor, inputColor);
                    builder->fsCodeAppendf("\t\t%s.rgb = %s(%s(%s.rgb * %s.a, dstSrcAlpha.rgb), dstSrcAlpha.a, dstSrcAlpha.rgb);\n",
                                           outputColor, setLum.c_str(), setSat.c_str(), inputColor,
                                           dstColor);
                    builder->fsCodeAppendf("\t\t%s.rgb += (1.0 - %s.a) * %s.rgb + (1.0 - %s.a) * %s.rgb;\n",
                                           outputColor, inputColor, dstColor, dstColor, inputColor);
                    break;
                }
                case SkXfermode::kSaturation_Mode: {
                    // SetLum(SetSat(D * Sa, Sat(S * Da)), Sa*Da, D*Sa)) + (1 - Sa) * D + (1 - Da) * S
                    SkString setSat, setLum;
                    AddSatFunction(builder, &setSat);
                    AddLumFunction(builder, &setLum);
                    builder->fsCodeAppendf("\t\tvec4 dstSrcAlpha = %s * %s.a;\n",
                                           dstColor, inputColor);
                    builder->fsCodeAppendf("\t\t%s.rgb = %s(%s(dstSrcAlpha.rgb, %s.rgb * %s.a), dstSrcAlpha.a, dstSrcAlpha.rgb);\n",
                                           outputColor, setLum.c_str(), setSat.c_str(), inputColor,
                                           dstColor);
                    builder->fsCodeAppendf("\t\t%s.rgb += (1.0 - %s.a) * %s.rgb + (1.0 - %s.a) * %s.rgb;\n",
                                           outputColor, inputColor, dstColor, dstColor, inputColor);
                    break;
                }
                case SkXfermode::kColor_Mode: {
                    //  SetLum(S * Da, Sa* Da, D * Sa) + (1 - Sa) * D + (1 - Da) * S
                    SkString setLum;
                    AddLumFunction(builder, &setLum);
                    builder->fsCodeAppendf("\t\tvec4 srcDstAlpha = %s * %s.a;\n",
                                           inputColor, dstColor);
                    builder->fsCodeAppendf("\t\t%s.rgb = %s(srcDstAlpha.rgb, srcDstAlpha.a, %s.rgb * %s.a);\n",
                                           outputColor, setLum.c_str(), dstColor, inputColor);
                    builder->fsCodeAppendf("\t\t%s.rgb += (1.0 - %s.a) * %s.rgb + (1.0 - %s.a) * %s.rgb;\n",
                                           outputColor, inputColor, dstColor, dstColor, inputColor);
                    break;
                }
                case SkXfermode::kLuminosity_Mode: {
                    //  SetLum(D * Sa, Sa* Da, S * Da) + (1 - Sa) * D + (1 - Da) * S
                    SkString setLum;
                    AddLumFunction(builder, &setLum);
                    builder->fsCodeAppendf("\t\tvec4 srcDstAlpha = %s * %s.a;\n",
                                           inputColor, dstColor);
                    builder->fsCodeAppendf("\t\t%s.rgb = %s(%s.rgb * %s.a, srcDstAlpha.a, srcDstAlpha.rgb);\n",
                                           outputColor, setLum.c_str(), dstColor, inputColor);
                    builder->fsCodeAppendf("\t\t%s.rgb += (1.0 - %s.a) * %s.rgb + (1.0 - %s.a) * %s.rgb;\n",
                                           outputColor, inputColor, dstColor, dstColor, inputColor);
                    break;
                }
                default:
                    GrCrash("Unknown XferEffect mode.");
                    break;
            }
        }

        static inline EffectKey GenKey(const GrDrawEffect& drawEffect, const GrGLCaps&) {
            const XferEffect& xfer = drawEffect.castEffect<XferEffect>();
            GrTexture* bgTex = xfer.backgroundAccess().getTexture();
            EffectKey bgKey = 0;
            if (bgTex) {
                bgKey = GrGLEffectMatrix::GenKey(GrEffect::MakeDivByTextureWHMatrix(bgTex),
                                                 drawEffect,
                                                 GLEffect::kCoordsType,
                                                 bgTex);
            }
            EffectKey modeKey = xfer.mode() << GrGLEffectMatrix::kKeyBits;
            return modeKey | bgKey;
        }

        virtual void setData(const GrGLUniformManager& uman, const GrDrawEffect& drawEffect) SK_OVERRIDE {
            const XferEffect& xfer = drawEffect.castEffect<XferEffect>();
            GrTexture* bgTex = xfer.backgroundAccess().getTexture();
            if (bgTex) {
                fBackgroundEffectMatrix.setData(uman,
                                                GrEffect::MakeDivByTextureWHMatrix(bgTex),
                                                drawEffect,
                                                bgTex);
            }
        }

    private:
        static void HardLight(GrGLShaderBuilder* builder,
                              const char* final,
                              const char* src,
                              const char* dst) {
            static const char kComponents[] = {'r', 'g', 'b'};
            for (size_t i = 0; i < SK_ARRAY_COUNT(kComponents); ++i) {
                char component = kComponents[i];
                builder->fsCodeAppendf("\t\tif (2.0 * %s.%c <= %s.a) {\n", src, component, src);
                builder->fsCodeAppendf("\t\t\t%s.%c = 2.0 * %s.%c * %s.%c;\n", final, component, src, component, dst, component);
                builder->fsCodeAppend("\t\t} else {\n");
                builder->fsCodeAppendf("\t\t\t%s.%c = %s.a * %s.a - 2.0 * (%s.a - %s.%c) * (%s.a - %s.%c);\n",
                                       final, component, src, dst, dst, dst, component, src, src, component);
                builder->fsCodeAppend("\t\t}\n");
            }
            builder->fsCodeAppendf("\t\t%s.rgb += %s.rgb * (1.0 - %s.a) + %s.rgb * (1.0 - %s.a);\n",
                                   final, src, dst, dst, src);
        }

        // Does one component of color-dodge
        static void ColorDodgeComponent(GrGLShaderBuilder* builder,
                                        const char* final,
                                        const char* src,
                                        const char* dst,
                                        const char component) {
            builder->fsCodeAppendf("\t\tif (0.0 == %s.%c) {\n", dst, component);
            builder->fsCodeAppendf("\t\t\t%s.%c = %s.%c * (1.0 - %s.a);\n",
                                   final, component, src, component, dst);
            builder->fsCodeAppend("\t\t} else {\n");
            builder->fsCodeAppendf("\t\t\tfloat d = %s.a - %s.%c;\n", src, src, component);
            builder->fsCodeAppend("\t\t\tif (0.0 == d) {\n");
            builder->fsCodeAppendf("\t\t\t\t%s.%c = %s.a * %s.a + %s.%c * (1.0 - %s.a) + %s.%c * (1.0 - %s.a);\n",
                                   final, component, src, dst, src, component, dst, dst, component,
                                   src);
            builder->fsCodeAppend("\t\t\t} else {\n");
            builder->fsCodeAppendf("\t\t\t\td = min(%s.a, %s.%c * %s.a / d);\n",
                                   dst, dst, component, src);
            builder->fsCodeAppendf("\t\t\t\t%s.%c = d * %s.a + %s.%c * (1.0 - %s.a) + %s.%c * (1.0 - %s.a);\n",
                                   final, component, src, src, component, dst, dst, component, src);
            builder->fsCodeAppend("\t\t\t}\n");
            builder->fsCodeAppend("\t\t}\n");
        }

        // Does one component of color-burn
        static void ColorBurnComponent(GrGLShaderBuilder* builder,
                                       const char* final,
                                       const char* src,
                                       const char* dst,
                                       const char component) {
            builder->fsCodeAppendf("\t\tif (%s.a == %s.%c) {\n", dst, dst, component);
            builder->fsCodeAppendf("\t\t\t%s.%c = %s.a * %s.a + %s.%c * (1.0 - %s.a) + %s.%c * (1.0 - %s.a);\n",
                                   final, component, src, dst, src, component, dst, dst, component,
                                   src);
            builder->fsCodeAppendf("\t\t} else if (0.0 == %s.%c) {\n", src, component);
            builder->fsCodeAppendf("\t\t\t%s.%c = %s.%c * (1.0 - %s.a);\n",
                                   final, component, dst, component, src);
            builder->fsCodeAppend("\t\t} else {\n");
            builder->fsCodeAppendf("\t\t\tfloat d = max(0.0, %s.a - (%s.a - %s.%c) * %s.a / %s.%c);\n",
                                   dst, dst, dst, component, src, src, component);
            builder->fsCodeAppendf("\t\t\t%s.%c = %s.a * d + %s.%c * (1.0 - %s.a) + %s.%c * (1.0 - %s.a);\n",
                                   final, component, src, src, component, dst, dst, component, src);
            builder->fsCodeAppend("\t\t}\n");
        }

        // Does one component of soft-light. Caller should have already checked that dst alpha > 0.
        static void SoftLightComponentPosDstAlpha(GrGLShaderBuilder* builder,
                                                  const char* final,
                                                  const char* src,
                                                  const char* dst,
                                                  const char component) {
            // if (2S < Sa)
            builder->fsCodeAppendf("\t\t\tif (2.0 * %s.%c <= %s.a) {\n", src, component, src);
            // (D^2 (Sa-2 S))/Da+(1-Da) S+D (-Sa+2 S+1)
            builder->fsCodeAppendf("\t\t\t\t%s.%c = (%s.%c*%s.%c*(%s.a - 2.0*%s.%c)) / %s.a + (1.0 - %s.a) * %s.%c + %s.%c*(-%s.a + 2.0*%s.%c + 1.0);\n",
                                   final, component, dst, component, dst, component, src, src,
                                   component, dst, dst, src, component, dst, component, src, src,
                                   component);
            // else if (4D < Da)
            builder->fsCodeAppendf("\t\t\t} else if (4.0 * %s.%c <= %s.a) {\n",
                                   dst, component, dst);
            builder->fsCodeAppendf("\t\t\t\tfloat DSqd = %s.%c * %s.%c;\n",
                                   dst, component, dst, component);
            builder->fsCodeAppendf("\t\t\t\tfloat DCub = DSqd * %s.%c;\n", dst, component);
            builder->fsCodeAppendf("\t\t\t\tfloat DaSqd = %s.a * %s.a;\n", dst, dst);
            builder->fsCodeAppendf("\t\t\t\tfloat DaCub = DaSqd * %s.a;\n", dst);
            // (Da^3 (-S)+Da^2 (S-D (3 Sa-6 S-1))+12 Da D^2 (Sa-2 S)-16 D^3 (Sa-2 S))/Da^2
            builder->fsCodeAppendf("\t\t\t\t%s.%c = (-DaCub*%s.%c + DaSqd*(%s.%c - %s.%c * (3.0*%s.a - 6.0*%s.%c - 1.0)) + 12.0*%s.a*DSqd*(%s.a - 2.0*%s.%c) - 16.0*DCub * (%s.a - 2.0*%s.%c)) / DaSqd;\n",
                                   final, component, src, component, src, component, dst, component,
                                   src, src, component, dst, src, src, component, src, src,
                                   component);
            builder->fsCodeAppendf("\t\t\t} else {\n");
            // -sqrt(Da * D) (Sa-2 S)-Da S+D (Sa-2 S+1)+S
            builder->fsCodeAppendf("\t\t\t\t%s.%c = -sqrt(%s.a*%s.%c)*(%s.a - 2.0*%s.%c) - %s.a*%s.%c + %s.%c*(%s.a - 2.0*%s.%c + 1.0) + %s.%c;\n",
                                    final, component, dst, dst, component, src, src, component, dst,
                                    src, component, dst, component, src, src, component, src,
                                    component);
            builder->fsCodeAppendf("\t\t\t}\n");
        }

        // Adds a function that takes two colors and an alpha as input. It produces a color with the
        // hue and saturation of the first color, the luminosity of the second color, and the input
        // alpha. It has this signature:
        //      vec3 set_luminance(vec3 hueSatColor, float alpha, vec3 lumColor).
        static void AddLumFunction(GrGLShaderBuilder* builder, SkString* setLumFunction) {
            // Emit a helper that gets the luminance of a color.
            SkString getFunction;
            GrGLShaderVar getLumArgs[] = {
                GrGLShaderVar("color", kVec3f_GrSLType),
            };
            SkString getLumBody("\treturn dot(vec3(0.3, 0.59, 0.11), color);\n");
            builder->emitFunction(GrGLShaderBuilder::kFragment_ShaderType,
                                  kFloat_GrSLType,
                                  "luminance",
                                   SK_ARRAY_COUNT(getLumArgs), getLumArgs,
                                   getLumBody.c_str(),
                                   &getFunction);

            // Emit the set luminance function.
            GrGLShaderVar setLumArgs[] = {
                GrGLShaderVar("hueSat", kVec3f_GrSLType),
                GrGLShaderVar("alpha", kFloat_GrSLType),
                GrGLShaderVar("lumColor", kVec3f_GrSLType),
            };
            SkString setLumBody;
            setLumBody.printf("\tfloat diff = %s(lumColor - hueSat);\n", getFunction.c_str());
            setLumBody.append("\tvec3 outColor = hueSat + diff;\n");
            setLumBody.appendf("\tfloat outLum = %s(outColor);\n", getFunction.c_str());
            setLumBody.append("\tfloat minComp = min(min(outColor.r, outColor.g), outColor.b);\n"
                              "\tfloat maxComp = max(max(outColor.r, outColor.g), outColor.b);\n"
                              "\tif (minComp < 0.0) {\n"
                              "\t\toutColor = outLum + ((outColor - vec3(outLum, outLum, outLum)) * outLum) / (outLum - minComp);\n"
                              "\t}\n"
                              "\tif (maxComp > alpha) {\n"
                              "\t\toutColor = outLum + ((outColor - vec3(outLum, outLum, outLum)) * (alpha - outLum)) / (maxComp - outLum);\n"
                              "\t}\n"
                              "\treturn outColor;\n");
            builder->emitFunction(GrGLShaderBuilder::kFragment_ShaderType,
                        kVec3f_GrSLType,
                        "set_luminance",
                        SK_ARRAY_COUNT(setLumArgs), setLumArgs,
                        setLumBody.c_str(),
                        setLumFunction);
        }

        // Adds a function that creates a color with the hue and luminosity of one input color and
        // the saturation of another color. It will have this signature:
        //      float set_saturation(vec3 hueLumColor, vec3 satColor)
        static void AddSatFunction(GrGLShaderBuilder* builder, SkString* setSatFunction) {
            // Emit a helper that gets the saturation of a color
            SkString getFunction;
            GrGLShaderVar getSatArgs[] = { GrGLShaderVar("color", kVec3f_GrSLType) };
            SkString getSatBody;
            getSatBody.printf("\treturn max(max(color.r, color.g), color.b) - "
                              "min(min(color.r, color.g), color.b);\n");
            builder->emitFunction(GrGLShaderBuilder::kFragment_ShaderType,
                                  kFloat_GrSLType,
                                  "saturation",
                                  SK_ARRAY_COUNT(getSatArgs), getSatArgs,
                                  getSatBody.c_str(),
                                  &getFunction);

            // Emit a helper that sets the saturation given sorted input channels. This used
            // to use inout params for min, mid, and max components but that seems to cause
            // problems on PowerVR drivers. So instead it returns a vec3 where r, g ,b are the
            // adjusted min, mid, and max inputs, respectively.
            SkString helperFunction;
            GrGLShaderVar helperArgs[] = {
                GrGLShaderVar("minComp", kFloat_GrSLType),
                GrGLShaderVar("midComp", kFloat_GrSLType),
                GrGLShaderVar("maxComp", kFloat_GrSLType),
                GrGLShaderVar("sat", kFloat_GrSLType),
            };
            static const char kHelperBody[] = "\tif (minComp < maxComp) {\n"
                                              "\t\tvec3 result;\n"
                                              "\t\tresult.r = 0.0;\n"
                                              "\t\tresult.g = sat * (midComp - minComp) / (maxComp - minComp);\n"
                                              "\t\tresult.b = sat;\n"
                                              "\t\treturn result;\n"
                                              "\t} else {\n"
                                              "\t\treturn vec3(0, 0, 0);\n"
                                              "\t}\n";
            builder->emitFunction(GrGLShaderBuilder::kFragment_ShaderType,
                                  kVec3f_GrSLType,
                                  "set_saturation_helper",
                                  SK_ARRAY_COUNT(helperArgs), helperArgs,
                                  kHelperBody,
                                  &helperFunction);

            GrGLShaderVar setSatArgs[] = {
                GrGLShaderVar("hueLumColor", kVec3f_GrSLType),
                GrGLShaderVar("satColor", kVec3f_GrSLType),
            };
            const char* helpFunc = helperFunction.c_str();
            SkString setSatBody;
            setSatBody.appendf("\tfloat sat = %s(satColor);\n"
                               "\tif (hueLumColor.r <= hueLumColor.g) {\n"
                               "\t\tif (hueLumColor.g <= hueLumColor.b) {\n"
                               "\t\t\thueLumColor.rgb = %s(hueLumColor.r, hueLumColor.g, hueLumColor.b, sat);\n"
                               "\t\t} else if (hueLumColor.r <= hueLumColor.b) {\n"
                               "\t\t\thueLumColor.rbg = %s(hueLumColor.r, hueLumColor.b, hueLumColor.g, sat);\n"
                               "\t\t} else {\n"
                               "\t\t\thueLumColor.brg = %s(hueLumColor.b, hueLumColor.r, hueLumColor.g, sat);\n"
                               "\t\t}\n"
                               "\t} else if (hueLumColor.r <= hueLumColor.b) {\n"
                               "\t\thueLumColor.grb = %s(hueLumColor.g, hueLumColor.r, hueLumColor.b, sat);\n"
                               "\t} else if (hueLumColor.g <= hueLumColor.b) {\n"
                               "\t\thueLumColor.gbr = %s(hueLumColor.g, hueLumColor.b, hueLumColor.r, sat);\n"
                               "\t} else {\n"
                               "\t\thueLumColor.bgr = %s(hueLumColor.b, hueLumColor.g, hueLumColor.r, sat);\n"
                               "\t}\n"
                               "\treturn hueLumColor;\n",
                               getFunction.c_str(), helpFunc, helpFunc, helpFunc, helpFunc,
                               helpFunc, helpFunc);
            builder->emitFunction(GrGLShaderBuilder::kFragment_ShaderType,
                                  kVec3f_GrSLType,
                                  "set_saturation",
                                  SK_ARRAY_COUNT(setSatArgs), setSatArgs,
                                  setSatBody.c_str(),
                                  setSatFunction);

        }

        static const GrEffect::CoordsType kCoordsType = GrEffect::kLocal_CoordsType;
        GrGLEffectMatrix   fBackgroundEffectMatrix;
        typedef GrGLEffect INHERITED;
    };

    GR_DECLARE_EFFECT_TEST;

private:
    XferEffect(SkXfermode::Mode mode, GrTexture* background)
        : fMode(mode) {
        if (background) {
            fBackgroundAccess.reset(background);
            this->addTextureAccess(&fBackgroundAccess);
        } else {
            this->setWillReadDstColor();
        }
    }
    virtual bool onIsEqual(const GrEffect& other) const SK_OVERRIDE {
        const XferEffect& s = CastEffect<XferEffect>(other);
        return fMode == s.fMode &&
               fBackgroundAccess.getTexture() == s.fBackgroundAccess.getTexture();
    }

    SkXfermode::Mode fMode;
    GrTextureAccess  fBackgroundAccess;

    typedef GrEffect INHERITED;
};

GR_DEFINE_EFFECT_TEST(XferEffect);
GrEffectRef* XferEffect::TestCreate(SkMWCRandom* rand,
                                    GrContext*,
                                    const GrDrawTargetCaps&,
                                    GrTexture*[]) {
    int mode = rand->nextRangeU(SkXfermode::kLastCoeffMode + 1, SkXfermode::kLastSeparableMode);

    static AutoEffectUnref gEffect(SkNEW_ARGS(XferEffect, (static_cast<SkXfermode::Mode>(mode), NULL)));
    return CreateEffectRef(gEffect);
}

#endif

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

class SkProcCoeffXfermode : public SkProcXfermode {
public:
    SkProcCoeffXfermode(const ProcCoeff& rec, Mode mode)
            : INHERITED(rec.fProc) {
        fMode = mode;
        // these may be valid, or may be CANNOT_USE_COEFF
        fSrcCoeff = rec.fSC;
        fDstCoeff = rec.fDC;
    }

    virtual bool asMode(Mode* mode) const SK_OVERRIDE {
        if (mode) {
            *mode = fMode;
        }
        return true;
    }

    virtual bool asCoeff(Coeff* sc, Coeff* dc) const SK_OVERRIDE {
        if (CANNOT_USE_COEFF == fSrcCoeff) {
            return false;
        }

        if (sc) {
            *sc = fSrcCoeff;
        }
        if (dc) {
            *dc = fDstCoeff;
        }
        return true;
    }

#if SK_SUPPORT_GPU
    virtual bool asNewEffectOrCoeff(GrContext*,
                                    GrEffectRef** effect,
                                    Coeff* src,
                                    Coeff* dst,
                                    GrTexture* background) const SK_OVERRIDE {
        if (this->asCoeff(src, dst)) {
            return true;
        }
        if (XferEffect::IsSupportedMode(fMode)) {
            if (NULL != effect) {
                *effect = XferEffect::Create(fMode, background);
                SkASSERT(NULL != *effect);
            }
            return true;
        }
        return false;
    }
#endif

    SK_DEVELOPER_TO_STRING()
    SK_DECLARE_PUBLIC_FLATTENABLE_DESERIALIZATION_PROCS(SkProcCoeffXfermode)

protected:
    SkProcCoeffXfermode(SkFlattenableReadBuffer& buffer) : INHERITED(buffer) {
        fMode = (SkXfermode::Mode)buffer.read32();

        const ProcCoeff& rec = gProcCoeffs[fMode];
        // these may be valid, or may be CANNOT_USE_COEFF
        fSrcCoeff = rec.fSC;
        fDstCoeff = rec.fDC;
        // now update our function-ptr in the super class
        this->INHERITED::setProc(rec.fProc);
    }

    virtual void flatten(SkFlattenableWriteBuffer& buffer) const SK_OVERRIDE {
        this->INHERITED::flatten(buffer);
        buffer.write32(fMode);
    }

private:
    Mode    fMode;
    Coeff   fSrcCoeff, fDstCoeff;

    typedef SkProcXfermode INHERITED;
};

const char* SkXfermode::ModeName(Mode mode) {
    SkASSERT((unsigned) mode <= (unsigned)kLastMode);
    const char* gModeStrings[] = {
        "Clear", "Src", "Dst", "SrcOver", "DstOver", "SrcIn", "DstIn",
        "SrcOut", "DstOut", "SrcATop", "DstATop", "Xor", "Plus",
        "Modulate", "Screen", "Overlay", "Darken", "Lighten", "ColorDodge",
        "ColorBurn", "HardLight", "SoftLight", "Difference", "Exclusion",
        "Multiply", "Hue", "Saturation", "Color",  "Luminosity"
    };
    return gModeStrings[mode];
    SK_COMPILE_ASSERT(SK_ARRAY_COUNT(gModeStrings) == kLastMode + 1, mode_count);
}

#ifdef SK_DEVELOPER
void SkProcCoeffXfermode::toString(SkString* str) const {
    str->append("SkProcCoeffXfermode: ");

    str->append("mode: ");
    str->append(ModeName(fMode));

    static const char* gCoeffStrings[kCoeffCount] = {
        "Zero", "One", "SC", "ISC", "DC", "IDC", "SA", "ISA", "DA", "IDA"
    };

    str->append(" src: ");
    if (CANNOT_USE_COEFF == fSrcCoeff) {
        str->append("can't use");
    } else {
        str->append(gCoeffStrings[fSrcCoeff]);
    }

    str->append(" dst: ");
    if (CANNOT_USE_COEFF == fDstCoeff) {
        str->append("can't use");
    } else {
        str->append(gCoeffStrings[fDstCoeff]);
    }
}
#endif

///////////////////////////////////////////////////////////////////////////////

class SkClearXfermode : public SkProcCoeffXfermode {
public:
    SkClearXfermode(const ProcCoeff& rec) : SkProcCoeffXfermode(rec, kClear_Mode) {}

    virtual void xfer32(SkPMColor*, const SkPMColor*, int, const SkAlpha*) const SK_OVERRIDE;
    virtual void xferA8(SkAlpha*, const SkPMColor*, int, const SkAlpha*) const SK_OVERRIDE;

    SK_DEVELOPER_TO_STRING()
    SK_DECLARE_PUBLIC_FLATTENABLE_DESERIALIZATION_PROCS(SkClearXfermode)

private:
    SkClearXfermode(SkFlattenableReadBuffer& buffer)
        : SkProcCoeffXfermode(buffer) {}

    typedef SkProcCoeffXfermode INHERITED;
};

void SkClearXfermode::xfer32(SkPMColor* SK_RESTRICT dst,
                             const SkPMColor* SK_RESTRICT, int count,
                             const SkAlpha* SK_RESTRICT aa) const {
    SkASSERT(dst && count >= 0);

    if (NULL == aa) {
        memset(dst, 0, count << 2);
    } else {
        for (int i = count - 1; i >= 0; --i) {
            unsigned a = aa[i];
            if (0xFF == a) {
                dst[i] = 0;
            } else if (a != 0) {
                dst[i] = SkAlphaMulQ(dst[i], SkAlpha255To256(255 - a));
            }
        }
    }
}
void SkClearXfermode::xferA8(SkAlpha* SK_RESTRICT dst,
                             const SkPMColor* SK_RESTRICT, int count,
                             const SkAlpha* SK_RESTRICT aa) const {
    SkASSERT(dst && count >= 0);

    if (NULL == aa) {
        memset(dst, 0, count);
    } else {
        for (int i = count - 1; i >= 0; --i) {
            unsigned a = aa[i];
            if (0xFF == a) {
                dst[i] = 0;
            } else if (0 != a) {
                dst[i] = SkAlphaMulAlpha(dst[i], 255 - a);
            }
        }
    }
}

#ifdef SK_DEVELOPER
void SkClearXfermode::toString(SkString* str) const {
    this->INHERITED::toString(str);
}
#endif

///////////////////////////////////////////////////////////////////////////////

class SkSrcXfermode : public SkProcCoeffXfermode {
public:
    SkSrcXfermode(const ProcCoeff& rec) : SkProcCoeffXfermode(rec, kSrc_Mode) {}

    virtual void xfer32(SkPMColor*, const SkPMColor*, int, const SkAlpha*) const SK_OVERRIDE;
    virtual void xferA8(SkAlpha*, const SkPMColor*, int, const SkAlpha*) const SK_OVERRIDE;

    SK_DEVELOPER_TO_STRING()
    SK_DECLARE_PUBLIC_FLATTENABLE_DESERIALIZATION_PROCS(SkSrcXfermode)

private:
    SkSrcXfermode(SkFlattenableReadBuffer& buffer)
        : SkProcCoeffXfermode(buffer) {}

    typedef SkProcCoeffXfermode INHERITED;
};

void SkSrcXfermode::xfer32(SkPMColor* SK_RESTRICT dst,
                           const SkPMColor* SK_RESTRICT src, int count,
                           const SkAlpha* SK_RESTRICT aa) const {
    SkASSERT(dst && src && count >= 0);

    if (NULL == aa) {
        memcpy(dst, src, count << 2);
    } else {
        for (int i = count - 1; i >= 0; --i) {
            unsigned a = aa[i];
            if (a == 0xFF) {
                dst[i] = src[i];
            } else if (a != 0) {
                dst[i] = SkFourByteInterp(src[i], dst[i], a);
            }
        }
    }
}

void SkSrcXfermode::xferA8(SkAlpha* SK_RESTRICT dst,
                           const SkPMColor* SK_RESTRICT src, int count,
                           const SkAlpha* SK_RESTRICT aa) const {
    SkASSERT(dst && src && count >= 0);

    if (NULL == aa) {
        for (int i = count - 1; i >= 0; --i) {
            dst[i] = SkToU8(SkGetPackedA32(src[i]));
        }
    } else {
        for (int i = count - 1; i >= 0; --i) {
            unsigned a = aa[i];
            if (0 != a) {
                unsigned srcA = SkGetPackedA32(src[i]);
                if (a == 0xFF) {
                    dst[i] = SkToU8(srcA);
                } else {
                    dst[i] = SkToU8(SkAlphaBlend(srcA, dst[i], a));
                }
            }
        }
    }
}
#ifdef SK_DEVELOPER
void SkSrcXfermode::toString(SkString* str) const {
    this->INHERITED::toString(str);
}
#endif

///////////////////////////////////////////////////////////////////////////////

class SkDstInXfermode : public SkProcCoeffXfermode {
public:
    SkDstInXfermode(const ProcCoeff& rec) : SkProcCoeffXfermode(rec, kDstIn_Mode) {}

    virtual void xfer32(SkPMColor*, const SkPMColor*, int, const SkAlpha*) const SK_OVERRIDE;

    SK_DEVELOPER_TO_STRING()
    SK_DECLARE_PUBLIC_FLATTENABLE_DESERIALIZATION_PROCS(SkDstInXfermode)

private:
    SkDstInXfermode(SkFlattenableReadBuffer& buffer) : INHERITED(buffer) {}

    typedef SkProcCoeffXfermode INHERITED;
};

void SkDstInXfermode::xfer32(SkPMColor* SK_RESTRICT dst,
                             const SkPMColor* SK_RESTRICT src, int count,
                             const SkAlpha* SK_RESTRICT aa) const {
    SkASSERT(dst && src);

    if (count <= 0) {
        return;
    }
    if (NULL != aa) {
        return this->INHERITED::xfer32(dst, src, count, aa);
    }

    do {
        unsigned a = SkGetPackedA32(*src);
        *dst = SkAlphaMulQ(*dst, SkAlpha255To256(a));
        dst++;
        src++;
    } while (--count != 0);
}

#ifdef SK_DEVELOPER
void SkDstInXfermode::toString(SkString* str) const {
    this->INHERITED::toString(str);
}
#endif

///////////////////////////////////////////////////////////////////////////////

class SkDstOutXfermode : public SkProcCoeffXfermode {
public:
    SkDstOutXfermode(const ProcCoeff& rec) : SkProcCoeffXfermode(rec, kDstOut_Mode) {}

    virtual void xfer32(SkPMColor*, const SkPMColor*, int, const SkAlpha*) const SK_OVERRIDE;

    SK_DEVELOPER_TO_STRING()
    SK_DECLARE_PUBLIC_FLATTENABLE_DESERIALIZATION_PROCS(SkDstOutXfermode)

private:
    SkDstOutXfermode(SkFlattenableReadBuffer& buffer)
        : INHERITED(buffer) {}

    typedef SkProcCoeffXfermode INHERITED;
};

void SkDstOutXfermode::xfer32(SkPMColor* SK_RESTRICT dst,
                              const SkPMColor* SK_RESTRICT src, int count,
                              const SkAlpha* SK_RESTRICT aa) const {
    SkASSERT(dst && src);

    if (count <= 0) {
        return;
    }
    if (NULL != aa) {
        return this->INHERITED::xfer32(dst, src, count, aa);
    }

    do {
        unsigned a = SkGetPackedA32(*src);
        *dst = SkAlphaMulQ(*dst, SkAlpha255To256(255 - a));
        dst++;
        src++;
    } while (--count != 0);
}

#ifdef SK_DEVELOPER
void SkDstOutXfermode::toString(SkString* str) const {
    this->INHERITED::toString(str);
}
#endif

///////////////////////////////////////////////////////////////////////////////

SkXfermode* SkXfermode::Create(Mode mode) {
    SkASSERT(SK_ARRAY_COUNT(gProcCoeffs) == kModeCount);
    SkASSERT((unsigned)mode < kModeCount);

    const ProcCoeff& rec = gProcCoeffs[mode];

    switch (mode) {
        case kClear_Mode:
            return SkNEW_ARGS(SkClearXfermode, (rec));
        case kSrc_Mode:
            return SkNEW_ARGS(SkSrcXfermode, (rec));
        case kSrcOver_Mode:
            return NULL;
        case kDstIn_Mode:
            return SkNEW_ARGS(SkDstInXfermode, (rec));
        case kDstOut_Mode:
            return SkNEW_ARGS(SkDstOutXfermode, (rec));
        default:
            return SkNEW_ARGS(SkProcCoeffXfermode, (rec, mode));
    }
}

SkXfermodeProc SkXfermode::GetProc(Mode mode) {
    SkXfermodeProc  proc = NULL;
    if ((unsigned)mode < kModeCount) {
        proc = gProcCoeffs[mode].fProc;
    }
    return proc;
}

bool SkXfermode::ModeAsCoeff(Mode mode, Coeff* src, Coeff* dst) {
    SkASSERT(SK_ARRAY_COUNT(gProcCoeffs) == kModeCount);

    if ((unsigned)mode >= (unsigned)kModeCount) {
        // illegal mode parameter
        return false;
    }

    const ProcCoeff& rec = gProcCoeffs[mode];

    if (CANNOT_USE_COEFF == rec.fSC) {
        return false;
    }

    SkASSERT(CANNOT_USE_COEFF != rec.fDC);
    if (src) {
        *src = rec.fSC;
    }
    if (dst) {
        *dst = rec.fDC;
    }
    return true;
}

bool SkXfermode::AsMode(const SkXfermode* xfer, Mode* mode) {
    if (NULL == xfer) {
        if (mode) {
            *mode = kSrcOver_Mode;
        }
        return true;
    }
    return xfer->asMode(mode);
}

bool SkXfermode::AsCoeff(const SkXfermode* xfer, Coeff* src, Coeff* dst) {
    if (NULL == xfer) {
        return ModeAsCoeff(kSrcOver_Mode, src, dst);
    }
    return xfer->asCoeff(src, dst);
}

bool SkXfermode::IsMode(const SkXfermode* xfer, Mode mode) {
    // if xfer==null then the mode is srcover
    Mode m = kSrcOver_Mode;
    if (xfer && !xfer->asMode(&m)) {
        return false;
    }
    return mode == m;
}

///////////////////////////////////////////////////////////////////////////////
//////////// 16bit xfermode procs

#ifdef SK_DEBUG
static bool require_255(SkPMColor src) { return SkGetPackedA32(src) == 0xFF; }
static bool require_0(SkPMColor src) { return SkGetPackedA32(src) == 0; }
#endif

static uint16_t src_modeproc16_255(SkPMColor src, uint16_t dst) {
    SkASSERT(require_255(src));
    return SkPixel32ToPixel16(src);
}

static uint16_t dst_modeproc16(SkPMColor src, uint16_t dst) {
    return dst;
}

static uint16_t srcover_modeproc16_0(SkPMColor src, uint16_t dst) {
    SkASSERT(require_0(src));
    return dst;
}

static uint16_t srcover_modeproc16_255(SkPMColor src, uint16_t dst) {
    SkASSERT(require_255(src));
    return SkPixel32ToPixel16(src);
}

static uint16_t dstover_modeproc16_0(SkPMColor src, uint16_t dst) {
    SkASSERT(require_0(src));
    return dst;
}

static uint16_t dstover_modeproc16_255(SkPMColor src, uint16_t dst) {
    SkASSERT(require_255(src));
    return dst;
}

static uint16_t srcin_modeproc16_255(SkPMColor src, uint16_t dst) {
    SkASSERT(require_255(src));
    return SkPixel32ToPixel16(src);
}

static uint16_t dstin_modeproc16_255(SkPMColor src, uint16_t dst) {
    SkASSERT(require_255(src));
    return dst;
}

static uint16_t dstout_modeproc16_0(SkPMColor src, uint16_t dst) {
    SkASSERT(require_0(src));
    return dst;
}

static uint16_t srcatop_modeproc16(SkPMColor src, uint16_t dst) {
    unsigned isa = 255 - SkGetPackedA32(src);

    return SkPackRGB16(
           SkPacked32ToR16(src) + SkAlphaMulAlpha(SkGetPackedR16(dst), isa),
           SkPacked32ToG16(src) + SkAlphaMulAlpha(SkGetPackedG16(dst), isa),
           SkPacked32ToB16(src) + SkAlphaMulAlpha(SkGetPackedB16(dst), isa));
}

static uint16_t srcatop_modeproc16_0(SkPMColor src, uint16_t dst) {
    SkASSERT(require_0(src));
    return dst;
}

static uint16_t srcatop_modeproc16_255(SkPMColor src, uint16_t dst) {
    SkASSERT(require_255(src));
    return SkPixel32ToPixel16(src);
}

static uint16_t dstatop_modeproc16_255(SkPMColor src, uint16_t dst) {
    SkASSERT(require_255(src));
    return dst;
}

/*********
    darken and lighten boil down to this.

    darken  = (1 - Sa) * Dc + min(Sc, Dc)
    lighten = (1 - Sa) * Dc + max(Sc, Dc)

    if (Sa == 0) these become
        darken  = Dc + min(0, Dc) = 0
        lighten = Dc + max(0, Dc) = Dc

    if (Sa == 1) these become
        darken  = min(Sc, Dc)
        lighten = max(Sc, Dc)
*/

static uint16_t darken_modeproc16_0(SkPMColor src, uint16_t dst) {
    SkASSERT(require_0(src));
    return 0;
}

static uint16_t darken_modeproc16_255(SkPMColor src, uint16_t dst) {
    SkASSERT(require_255(src));
    unsigned r = SkFastMin32(SkPacked32ToR16(src), SkGetPackedR16(dst));
    unsigned g = SkFastMin32(SkPacked32ToG16(src), SkGetPackedG16(dst));
    unsigned b = SkFastMin32(SkPacked32ToB16(src), SkGetPackedB16(dst));
    return SkPackRGB16(r, g, b);
}

static uint16_t lighten_modeproc16_0(SkPMColor src, uint16_t dst) {
    SkASSERT(require_0(src));
    return dst;
}

static uint16_t lighten_modeproc16_255(SkPMColor src, uint16_t dst) {
    SkASSERT(require_255(src));
    unsigned r = SkMax32(SkPacked32ToR16(src), SkGetPackedR16(dst));
    unsigned g = SkMax32(SkPacked32ToG16(src), SkGetPackedG16(dst));
    unsigned b = SkMax32(SkPacked32ToB16(src), SkGetPackedB16(dst));
    return SkPackRGB16(r, g, b);
}

struct Proc16Rec {
    SkXfermodeProc16    fProc16_0;
    SkXfermodeProc16    fProc16_255;
    SkXfermodeProc16    fProc16_General;
};

static const Proc16Rec gModeProcs16[] = {
    { NULL,                 NULL,                   NULL            }, // CLEAR
    { NULL,                 src_modeproc16_255,     NULL            },
    { dst_modeproc16,       dst_modeproc16,         dst_modeproc16  },
    { srcover_modeproc16_0, srcover_modeproc16_255, NULL            },
    { dstover_modeproc16_0, dstover_modeproc16_255, NULL            },
    { NULL,                 srcin_modeproc16_255,   NULL            },
    { NULL,                 dstin_modeproc16_255,   NULL            },
    { NULL,                 NULL,                   NULL            },// SRC_OUT
    { dstout_modeproc16_0,  NULL,                   NULL            },
    { srcatop_modeproc16_0, srcatop_modeproc16_255, srcatop_modeproc16  },
    { NULL,                 dstatop_modeproc16_255, NULL            },
    { NULL,                 NULL,                   NULL            }, // XOR

    { NULL,                 NULL,                   NULL            }, // plus
    { NULL,                 NULL,                   NULL            }, // modulate
    { NULL,                 NULL,                   NULL            }, // screen
    { NULL,                 NULL,                   NULL            }, // overlay
    { darken_modeproc16_0,  darken_modeproc16_255,  NULL            }, // darken
    { lighten_modeproc16_0, lighten_modeproc16_255, NULL            }, // lighten
    { NULL,                 NULL,                   NULL            }, // colordodge
    { NULL,                 NULL,                   NULL            }, // colorburn
    { NULL,                 NULL,                   NULL            }, // hardlight
    { NULL,                 NULL,                   NULL            }, // softlight
    { NULL,                 NULL,                   NULL            }, // difference
    { NULL,                 NULL,                   NULL            }, // exclusion
    { NULL,                 NULL,                   NULL            }, // multiply
    { NULL,                 NULL,                   NULL            }, // hue
    { NULL,                 NULL,                   NULL            }, // saturation
    { NULL,                 NULL,                   NULL            }, // color
    { NULL,                 NULL,                   NULL            }, // luminosity
};

SkXfermodeProc16 SkXfermode::GetProc16(Mode mode, SkColor srcColor) {
    SkXfermodeProc16  proc16 = NULL;
    if ((unsigned)mode < kModeCount) {
        const Proc16Rec& rec = gModeProcs16[mode];
        unsigned a = SkColorGetA(srcColor);

        if (0 == a) {
            proc16 = rec.fProc16_0;
        } else if (255 == a) {
            proc16 = rec.fProc16_255;
        } else {
            proc16 = rec.fProc16_General;
        }
    }
    return proc16;
}

SK_DEFINE_FLATTENABLE_REGISTRAR_GROUP_START(SkXfermode)
    SK_DEFINE_FLATTENABLE_REGISTRAR_ENTRY(SkProcCoeffXfermode)
    SK_DEFINE_FLATTENABLE_REGISTRAR_ENTRY(SkClearXfermode)
    SK_DEFINE_FLATTENABLE_REGISTRAR_ENTRY(SkSrcXfermode)
    SK_DEFINE_FLATTENABLE_REGISTRAR_ENTRY(SkDstInXfermode)
    SK_DEFINE_FLATTENABLE_REGISTRAR_ENTRY(SkDstOutXfermode)
SK_DEFINE_FLATTENABLE_REGISTRAR_GROUP_END
