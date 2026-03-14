#include <arm_neon.h>
#include <cmath>
#include <algorithm>

#include "rknpu2-quantization.h"

namespace rknpu2_quantization {

// --- Conversion from FP32 ---

void convert_fp16_to_fp32(const uint16_t * src, float * dst, size_t n_elements) {
    size_t i = 0;
#ifdef __ARM_NEON
    for (; i + 3 < n_elements; i += 4) {
        float16x4_t f16_vec = (float16x4_t)vld1_u16(src + i);
        float32x4_t f32_vec = vcvt_f32_f16(f16_vec);
        vst1q_f32(dst + i, f32_vec);
    }
#endif
    for (; i < n_elements; ++i) {
        dst[i] = GGML_FP16_TO_FP32(src[i]);
    }
}

void convert_fp32_to_fp16(const float * src, uint16_t * dst, size_t n_elements) {
    size_t i = 0;
#ifdef __ARM_NEON
    for (; i + 7 < n_elements; i += 8) {
        float32x4_t f32_vec_0 = vld1q_f32(src + i);
        float32x4_t f32_vec_1 = vld1q_f32(src + i + 4);
        float16x8_t f16_vec = vcombine_f16(vcvt_f16_f32(f32_vec_0), vcvt_f16_f32(f32_vec_1));
        vst1q_u16(dst + i, (uint16x8_t)f16_vec);
    }
#endif
    for (; i < n_elements; ++i) {
        dst[i] = GGML_FP32_TO_FP16(src[i]);
    }
}

void quantize_fp32_to_int8(const float * src, int8_t * dst, size_t n_elements, float scale) {
    const float iscale = (scale == 0.0f) ? 0.0f : 1.0f / scale;
    size_t i = 0;
#ifdef __ARM_NEON
    float32x4_t iscale_vec = vdupq_n_f32(iscale);
    for (; i + 3 < n_elements; i += 4) {
        float32x4_t f32_vec = vld1q_f32(src + i);
        f32_vec = vmulq_f32(f32_vec, iscale_vec);
        // Round to nearest integer
        int32x4_t i32_vec = vcvtaq_s32_f32(f32_vec);
        // Narrow to 16-bit, then 8-bit
        int16x4_t i16_vec = vqmovn_s32(i32_vec);
        int8x8_t i8_vec = vqmovn_s16(vcombine_s16(i16_vec, i16_vec)); // Duplicate to fill 8 lanes
        vst1_lane_s8(dst + i + 0, i8_vec, 0);
        vst1_lane_s8(dst + i + 1, i8_vec, 1);
        vst1_lane_s8(dst + i + 2, i8_vec, 2);
        vst1_lane_s8(dst + i + 3, i8_vec, 3);
    }
#endif
    for (; i < n_elements; ++i) {
        dst[i] = (int8_t)roundf(src[i] * iscale);
    }
}

void quantize_fp16_to_int8(const uint16_t * src, int8_t * dst, size_t n_elements, float scale) {
    const float iscale = (scale == 0.0f) ? 0.0f : 1.0f / scale;
    size_t i = 0;
#ifdef __ARM_NEON
    float32x4_t iscale_vec = vdupq_n_f32(iscale);
    for (; i + 3 < n_elements; i += 4) {
        float16x4_t f16_vec = (float16x4_t)vld1_u16(src + i);
        float32x4_t f32_vec = vcvt_f32_f16(f16_vec);
        f32_vec = vmulq_f32(f32_vec, iscale_vec);
        // Round to nearest integer
        int32x4_t i32_vec = vcvtaq_s32_f32(f32_vec);
        // Narrow
        int16x4_t i16_vec = vqmovn_s32(i32_vec);
        int8x8_t i8_vec = vqmovn_s16(vcombine_s16(i16_vec, i16_vec));
        vst1_lane_s8(dst + i + 0, i8_vec, 0);
        vst1_lane_s8(dst + i + 1, i8_vec, 1);
        vst1_lane_s8(dst + i + 2, i8_vec, 2);
        vst1_lane_s8(dst + i + 3, i8_vec, 3);
    }
#endif
    for (; i < n_elements; ++i) {
        dst[i] = (int8_t)roundf(GGML_FP16_TO_FP32(src[i]) * iscale);
    }
}

float calculate_amax_fp32(const float * src, size_t n_elements) {
    float amax = 0.0f;
    size_t i = 0;
#ifdef __ARM_NEON
    float32x4_t amax_vec = vdupq_n_f32(0.0f);
    for (; i + 3 < n_elements; i += 4) {
        float32x4_t f32_vec = vld1q_f32(src + i);
        amax_vec = vmaxq_f32(amax_vec, vabsq_f32(f32_vec));
    }
    amax = std::max({vgetq_lane_f32(amax_vec, 0), vgetq_lane_f32(amax_vec, 1), vgetq_lane_f32(amax_vec, 2), vgetq_lane_f32(amax_vec, 3)});
#endif
    for (; i < n_elements; ++i) {
        amax = std::max(amax, std::abs(src[i]));
    }
    return amax;
}

float calculate_amax_fp16(const uint16_t * src, size_t n_elements) {
    float amax = 0.0f;
    size_t i = 0;
#ifdef __ARM_NEON
    float32x4_t amax_vec = vdupq_n_f32(0.0f);
    for (; i + 3 < n_elements; i += 4) {
        float16x4_t f16_vec = (float16x4_t)vld1_u16(src + i);
        float32x4_t f32_vec = vcvt_f32_f16(f16_vec);
        amax_vec = vmaxq_f32(amax_vec, vabsq_f32(f32_vec));
    }
    amax = std::max({vgetq_lane_f32(amax_vec, 0), vgetq_lane_f32(amax_vec, 1), vgetq_lane_f32(amax_vec, 2), vgetq_lane_f32(amax_vec, 3)});
#endif
    for (; i < n_elements; ++i) {
        amax = std::max(amax, std::abs(GGML_FP16_TO_FP32(src[i])));
    }
    return amax;
}

void quantize_fp32_to_int4_packed(const float * src, uint8_t * dst, size_t n_elements, float scale) {
    const float iscale = (scale == 0.0f) ? 0.0f : 1.0f / scale;
    for (size_t i = 0; i < n_elements / 2; ++i) {
        float v0_f = src[i * 2 + 0] * iscale;
        float v1_f = src[i * 2 + 1] * iscale;

        int8_t v0_i = std::max((int8_t)-7, std::min((int8_t)7, (int8_t)roundf(v0_f)));
        int8_t v1_i = std::max((int8_t)-7, std::min((int8_t)7, (int8_t)roundf(v1_f)));

        dst[i] = ((uint8_t)v0_i & 0x0F) | (((uint8_t)v1_i & 0x0F) << 4);
    }
}


// --- Dequantization to FP32 ---

void dequantize_int16_to_fp32(const int16_t * src, float * dst, size_t n_elements, float scale) {
    size_t i = 0;
#ifdef __ARM_NEON
    const float32x4_t scale_vec = vdupq_n_f32(scale);
    for (; i + 3 < n_elements; i += 4) {
        int16x4_t i16_vec = vld1_s16(src + i);
        int32x4_t i32_vec = vmovl_s16(i16_vec);
        float32x4_t f32_vec = vcvtq_f32_s32(i32_vec);
        f32_vec = vmulq_f32(f32_vec, scale_vec);
        vst1q_f32(dst + i, f32_vec);
    }
#endif
    for (; i < n_elements; ++i) {
        dst[i] = (float)src[i] * scale;
    }
}

void dequantize_int32_to_fp32(const int32_t * src, float * dst, size_t n_elements, float scale) {
    size_t i = 0;
#ifdef __ARM_NEON
    const float32x4_t scale_vec = vdupq_n_f32(scale);
    for (; i + 3 < n_elements; i += 4) {
        int32x4_t i32_vec = vld1q_s32(src + i);
        float32x4_t f32_vec = vcvtq_f32_s32(i32_vec);
        f32_vec = vmulq_f32(f32_vec, scale_vec);
        vst1q_f32(dst + i, f32_vec);
    }
#endif
    for (; i < n_elements; ++i) {
        dst[i] = (float)src[i] * scale;
    }
}

} // namespace rknpu2_quantization