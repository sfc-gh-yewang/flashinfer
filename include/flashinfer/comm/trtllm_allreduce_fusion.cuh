// vLLM AllReduce Fusion (optimized one-shot path with float4 mailbox)
// - One-shot Lamport path: 128-bit vectorized ops (float4), volatile polling, 32b -0 sentinel.
// - Two-shot path: keeps generic vec_t-based implementation.
// - Launcher uses TRT-like occupancy heuristic; barrier table size increased.
//
// File: include/flashinfer/comm/trtllm_allreduce_fusion.cuh

#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#if CUDA_VERSION >= 12080
#include <cuda_fp4.h>
#endif

#include <cuda/std/optional>
#include <array>
#include <algorithm>
#include <tuple>
#include <type_traits>

#include "../exception.h"
#include "../fp4_layout.cuh"
#include "../logging.h"
#include "../utils.cuh"
#include "../vec_dtypes.cuh"

namespace flashinfer {

namespace trtllm_allreduce_fusion {

using flashinfer::QuantizationSFLayout;

namespace details {

static constexpr int CVT_FP4_ELTS_PER_THREAD = 8;
static constexpr int CVT_FP4_SF_VEC_SIZE = 16;
static constexpr int kBytesPerAccess = 16;
static constexpr int kOneShotMaxToken = 128;
// Increased to avoid flag aliasing if gridDim.x grows.
static constexpr int kBarrierFlagCount = 1024;

}  // namespace details

// ============================================================================
// maths utils (mostly unchanged)
// ============================================================================
namespace maths {
template <typename T_OUT, typename T_IN>
__device__ inline T_OUT cuda_cast(T_IN val) { return val; }

template <> __device__ inline float2 cuda_cast<float2, int2>(int2 val) { return make_float2(val.x, val.y); }
template <> __device__ inline float2 cuda_cast<float2, float>(float val) { return make_float2(val, val); }
template <> __device__ inline float2 cuda_cast<float2, half2>(half2 val) { return __half22float2(val); }
template <> __device__ inline half2  cuda_cast<half2,  float2>(float2 val) { return __float22half2_rn(val); }
template <> __device__ inline half2  cuda_cast<half2,  float>(float val)   { return __float2half2_rn(val); }
template <> __device__ inline half2  cuda_cast<half2,  half>(half val)     { return __half2half2(val); }

template <> __device__ inline int8_t  cuda_cast<int8_t,  half>(half val) {
  union { int8_t int8[2]; int16_t int16; };
  union { half fp16; int16_t int16_in; };
  fp16 = val;
  asm volatile("cvt.rni.sat.s8.f16 %0, %1;" : "=h"(int16) : "h"(int16_in));
  return int8[0];
}
template <> __device__ inline int16_t cuda_cast<int16_t, half2>(half2 val) {
  union { int8_t int8[2]; int16_t int16; };
  int8[0] = cuda_cast<int8_t>(val.x);
  int8[1] = cuda_cast<int8_t>(val.y);
  return int16;
}
template <> __device__ inline int8_t  cuda_cast<int8_t,  float>(float val) {
  union { int8_t int8[2]; int16_t int16; };
  asm volatile("cvt.rni.sat.s8.f32 %0, %1;" : "=h"(int16) : "f"(val));
  return int8[0];
}
template <> __device__ inline int16_t cuda_cast<int16_t, float2>(float2 val) {
  union { int8_t int8[2]; int16_t int16; };
  int8[0] = cuda_cast<int8_t>(val.x);
  int8[1] = cuda_cast<int8_t>(val.y);
  return int16;
}
template <> __device__ inline half2  cuda_cast<half2,  int16_t>(int16_t val) {
  union { int8_t int8[2]; int16_t int16; };
  int16 = val;
  return make_half2(int8[0], int8[1]);
}
template <> __device__ inline float2 cuda_cast<float2, int16_t>(int16_t val) {
  union { int8_t int8[2]; int16_t int16; };
  int16 = val;
  return make_float2(int8[0], int8[1]);
}

template <> __device__ inline __nv_bfloat16 cuda_cast(int32_t val) { return static_cast<float>(val); }
template <> __device__ inline __nv_bfloat16 cuda_cast(int8_t  val) { return static_cast<float>(val); }
template <> __device__ inline int8_t        cuda_cast(__nv_bfloat16 val) { return static_cast<float>(val); }
template <> __device__ inline float         cuda_cast<float, __nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }

inline __device__ float2 bf1622float2(const __nv_bfloat162 val) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  float2 f_val;
  f_val.x = __low2float(val);
  f_val.y = __high2float(val);
  return f_val;
#else
  return __bfloat1622float2(val);
#endif
}
template <> __device__ inline float2 cuda_cast<float2, __nv_bfloat162>(__nv_bfloat162 val) { return bf1622float2(val); }
template <> __device__ inline half   cuda_cast<half,   __nv_bfloat16>(__nv_bfloat16 val) { return __float2half(__bfloat162float(val)); }

inline __device__ int16_t bf1622int16(__nv_bfloat162 val) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  float2 f_val;
  f_val.x = max(min(__low2float(val), 127.f), -128.f);
  f_val.y = max(min(__high2float(val), 127.f), -128.f);
  union { int8_t int8[2]; int16_t int16; };
  int8[0] = static_cast<int8_t>(static_cast<short>(f_val.x));
  int8[1] = static_cast<int8_t>(static_cast<short>(f_val.y));
  return int16;
#else
  val = __hmin2(val, make_bfloat162(127., 127.));
  val = __hmax2(val, make_bfloat162(-128., -128.));
  union { int8_t int8[2]; int16_t int16; };
  int8[0] = static_cast<int8_t>(static_cast<short>(val.x));
  int8[1] = static_cast<int8_t>(static_cast<short>(val.y));
  return int16;
#endif
}
template <> __device__ inline int16_t       cuda_cast<int16_t, __nv_bfloat162>(__nv_bfloat162 v) { return bf1622int16(v); }
template <> __device__ inline __nv_bfloat16 cuda_cast<__nv_bfloat16, float>(float v) { return __float2bfloat16(v); }
template <> __device__ inline __nv_bfloat16 cuda_cast<__nv_bfloat16, half>(half v)   { return __float2bfloat16(__half2float(v)); }

inline __device__ __nv_bfloat162 bf162bf162(const __nv_bfloat16 val) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  __nv_bfloat162 v2; v2.x = val; v2.y = val; return v2;
#else
  return __bfloat162bfloat162(val);
#endif
}
template <> __device__ inline __nv_bfloat162 cuda_cast<__nv_bfloat162, __nv_bfloat16>(__nv_bfloat16 v) { return bf162bf162(v); }
template <> __device__ inline __nv_bfloat162 cuda_cast<__nv_bfloat162, float>(float v) { return __float22bfloat162_rn(make_float2(v, v)); }
inline __device__ __nv_bfloat162 float22bf162(const float2 val) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  return __floats2bfloat162_rn(val.x, val.y);
#else
  return __float22bfloat162_rn(val);
#endif
}
template <> __device__ inline __nv_bfloat162 cuda_cast<__nv_bfloat162, float2>(float2 v) { return float22bf162(v); }
template <> __device__ inline __nv_bfloat162 cuda_cast<__nv_bfloat162, int16_t>(int16_t val) {
  union { int8_t int8[2]; int16_t int16; };
  int16 = val;
  __nv_bfloat162 res;
  res.x = cuda_cast<__nv_bfloat16>(int8[0]);
  res.y = cuda_cast<__nv_bfloat16>(int8[1]);
  return res;
}
template <> __device__ inline __nv_bfloat162 cuda_cast<__nv_bfloat162, half2>(half2 v) { return float22bf162(__half22float2(v)); }

// Abs
template <typename T> __device__ inline T cuda_abs(T val) { assert(false); return {}; }
template <> __device__ inline float         cuda_abs(float v) { return fabs(v); }
template <> __device__ inline float2        cuda_abs(float2 v){ return make_float2(fabs(v.x), fabs(v.y)); }
template <> __device__ inline half          cuda_abs(half v)  { return __habs(v); }
template <> __device__ inline half2         cuda_abs(half2 v) { return __habs2(v); }
#if __CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__)
template <> __device__ inline __nv_bfloat16  cuda_abs(__nv_bfloat16 v)  { return __habs(v); }
template <> __device__ inline __nv_bfloat162 cuda_abs(__nv_bfloat162 v) { return __habs2(v); }
#endif

// Max
template <typename To, typename Ti> __device__ inline To cuda_max(Ti val){ return cuda_cast<To>(val); }
template <> __device__ inline float         cuda_max(float2 v) { return fmaxf(v.x, v.y); }
template <> __device__ inline half          cuda_max(half2 v)  { return __hmax(v.x, v.y); }
template <> __device__ inline __nv_bfloat16 cuda_max(__nv_bfloat162 v) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
  return __hmax(v.x, v.y);
#else
  assert(0); asm volatile("brkpt;\n" ::); return __nv_bfloat16(0);
#endif
}
template <typename T> __device__ inline T   cuda_max(T a, T b) { return (a > b) ? a : b; }
template <> __device__ inline float2        cuda_max(float2 a, float2 b){ return make_float2(fmaxf(a.x,b.x), fmaxf(a.y,b.y)); }
template <> __device__ inline half2         cuda_max(half2 a, half2 b)  { return __hmax2(a,b); }
template <> __device__ inline __nv_bfloat162 cuda_max(__nv_bfloat162 a, __nv_bfloat162 b){ return __hmax2(a,b); }

// Reciprocal (fast)
inline __device__ float reciprocal_approximate_ftz(float a) {
  float b; asm volatile("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(b) : "f"(a)); return b;
}
}  // namespace maths

// ============================================================================
// utils (FP4 converters, reductions; mostly unchanged)
// ============================================================================
namespace utils {
#define FINAL_MASK 0xffffffff

template <typename T, int NUM>
__inline__ __device__ T warpReduceSumV2(T* val) {
#pragma unroll
  for (int i = 0; i < NUM; i++) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) val[i] += __shfl_xor_sync(FINAL_MASK, val[i], mask, 32);
  }
  return (T)(0.0f);
}

template <typename T, int NUM>
__inline__ __device__ T blockReduceSumV2(T* val) {
  static __shared__ T shared[NUM][33];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  warpReduceSumV2<T, NUM>(val);
  if (lane == 0) {
#pragma unroll
    for (int i = 0; i < NUM; i++) shared[i][wid] = val[i];
  }
  __syncthreads();

  bool is_mask = threadIdx.x < (blockDim.x / 32.f);
#pragma unroll
  for (int i = 0; i < NUM; i++) val[i] = is_mask ? shared[i][lane] : (T)(0.0f);
  warpReduceSumV2<T, NUM>(val);
  return (T)0.0f;
}

inline int getSMVersion() {
  int device{-1};
  FLASHINFER_CUDA_CALL(cudaGetDevice(&device));
  int sm_major = 0, sm_minor = 0;
  FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(&sm_major, cudaDevAttrComputeCapabilityMajor, device));
  FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(&sm_minor, cudaDevAttrComputeCapabilityMinor, device));
  return sm_major * 10 + sm_minor;
}

inline int getSMRegisters() {
  int device{-1};
  FLASHINFER_CUDA_CALL(cudaGetDevice(&device));
  int regs_per_block;
  FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(&regs_per_block, cudaDevAttrMaxRegistersPerBlock, device));
  return regs_per_block;
}

// ---- FP4 SF offset helpers (unchanged) ----
inline __device__ int64_t get_sf_out_offset_128x4(std::optional<int> batchIdx, int mIdx, int kIdx,
                                                  std::optional<int> numRows, int numCols) {
  int32_t innerKIdx = (kIdx % 4);
  int64_t innerKStride = 1;

  int32_t innerMIdx = (mIdx % (32 * 4)) / 32;
  int64_t innerMStride = 4 * innerKStride;

  int32_t outerMIdx = (mIdx % 32);
  int64_t outerMStride = 4 * innerMStride;

  int32_t kTileIdx = (kIdx / 4);
  int64_t kTileStride = 32 * outerMStride;

  int factor = details::CVT_FP4_SF_VEC_SIZE * 4;
  int32_t numKTiles = (numCols + factor - 1) / factor;
  int32_t mTileIdx = mIdx / (32 * 4);
  int64_t mTileStride = numKTiles * kTileStride;

  int32_t numMTiles = (numRows.value_or(0) + 128 - 1) / 128;
  int64_t bTileStride = numMTiles * mTileStride;

  int64_t SFOffset = batchIdx.value_or(0) * bTileStride + mTileIdx * mTileStride +
                     kTileIdx * kTileStride + outerMIdx * outerMStride + innerMIdx * innerMStride +
                     innerKIdx * innerKStride;

  return SFOffset;
}

template <class SFType, int CVT_FP4_NUM_THREADS_PER_SF>
__device__ uint8_t* cvt_quant_to_fp4_get_sf_out_offset(std::optional<int> batchIdx, int rowIdx,
                                                       int colIdx, std::optional<int> numRows,
                                                       int numCols, SFType* SFout,
                                                       QuantizationSFLayout layout) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  static_assert(CVT_FP4_NUM_THREADS_PER_SF == 1 || CVT_FP4_NUM_THREADS_PER_SF == 2);
  if (threadIdx.x % CVT_FP4_NUM_THREADS_PER_SF == 0) {
    if (layout == QuantizationSFLayout::SWIZZLED_128x4) {
      int32_t kIdx = colIdx / CVT_FP4_NUM_THREADS_PER_SF;
      int32_t mIdx = rowIdx;
      auto SFOffset = get_sf_out_offset_128x4(batchIdx, mIdx, kIdx, numRows, numCols);
      return reinterpret_cast<uint8_t*>(SFout) + SFOffset;
    } else if (layout == QuantizationSFLayout::LINEAR) {
      int32_t KTileIdx = colIdx / CVT_FP4_NUM_THREADS_PER_SF;
      int32_t numKTiles = numCols / details::CVT_FP4_SF_VEC_SIZE;
      int64_t mTileStride = numKTiles;
      int64_t BTileStride = numRows.value_or(0) * mTileStride;
      int64_t SFOffset = batchIdx.value_or(0) * BTileStride + rowIdx * mTileStride + KTileIdx;
      return reinterpret_cast<uint8_t*>(SFout) + SFOffset;
    } else {
      return nullptr;
    }
  }
#endif
  return nullptr;
}

__forceinline__ __device__ uint32_t pack_bytes(uint8_t c0, uint8_t c1, uint8_t c2, uint8_t c3) {
  uint32_t val0 = c0, val1 = c1, val2 = c2, val3 = c3;
  return (val3 << 24) | (val2 << 16) | (val1 << 8) | val0;
}

#if CUDA_VERSION >= 12080
inline __device__ uint32_t fp32_vec_to_e2m1(float (&array)[8]) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  uint32_t val;
  asm volatile(
      "{\n"
      ".reg .b8 byte0;\n"
      ".reg .b8 byte1;\n"
      ".reg .b8 byte2;\n"
      ".reg .b8 byte3;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte0, %2, %1;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte1, %4, %3;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte2, %6, %5;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte3, %8, %7;\n"
      "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
      "}"
      : "=r"(val)
      : "f"(array[0]), "f"(array[1]), "f"(array[2]), "f"(array[3]),
        "f"(array[4]), "f"(array[5]), "f"(array[6]), "f"(array[7]));
  return val;
#else
  uint32_t val;
  __nv_fp4x2_storage_t vals[4];
#pragma unroll
  for (int i = 0; i < 4; i++) {
    vals[i] = __nv_cvt_float2_to_fp4x2(*(((float2*)array) + i), __NV_E2M1, cudaRoundNearest);
  }
  val = pack_bytes(vals[0], vals[1], vals[2], vals[3]);
  return val;
#endif
}

inline __device__ uint32_t fp32_vec_to_e2m1(float2 (&array)[4]) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  uint32_t val;
  asm volatile(
      "{\n"
      ".reg .b8 byte0;\n"
      ".reg .b8 byte1;\n"
      ".reg .b8 byte2;\n"
      ".reg .b8 byte3;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte0, %2, %1;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte1, %4, %3;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte2, %6, %5;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte3, %8, %7;\n"
      "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
      "}"
      : "=r"(val)
      : "f"(array[0].x), "f"(array[0].y), "f"(array[1].x), "f"(array[1].y),
        "f"(array[2].x), "f"(array[2].y), "f"(array[3].x), "f"(array[3].y));
  return val;
#else
  uint32_t val;
  __nv_fp4x2_storage_t vals[4];
#pragma unroll
  for (int i = 0; i < 4; i++) {
    vals[i] = __nv_cvt_float2_to_fp4x2(array[i], __NV_E2M1, cudaRoundNearest);
  }
  val = pack_bytes(vals[0], vals[1], vals[2], vals[3]);
  return val;
#endif
}

// Quantizes vec_t<T, VEC_SIZE> into one uint32 via FP4 e2m1
template <typename T, uint32_t VEC_SIZE, bool UE8M0_SF = false>
__device__ uint32_t cvt_warp_fp16_to_fp4(vec_t<T, VEC_SIZE>& vec, float SFScaleVal, uint8_t* SFout) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  auto localMax = maths::cuda_abs(get_vec2_element(vec, 0));
#pragma unroll
  for (int i = 1; i < details::CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    localMax = maths::cuda_max(localMax, maths::cuda_abs(get_vec2_element(vec, i)));
  }
  localMax = maths::cuda_max(__shfl_xor_sync(uint32_t(-1), localMax, 1), localMax);
  float vecMax = float(maths::cuda_max(localMax.x, localMax.y));
  float SFValue = SFScaleVal * (vecMax * maths::reciprocal_approximate_ftz(6.0f));
  uint8_t fp8SFVal;
  if constexpr (UE8M0_SF) {
#if (__CUDACC_VER_MAJOR__ * 1000 + __CUDACC_VER_MINOR__ * 10 >= 12080)
    __nv_fp8_e8m0 tmp;
    tmp.__x = __nv_cvt_float_to_e8m0(SFValue, __NV_SATFINITE, cudaRoundPosInf);
    SFValue = static_cast<float>(tmp);
    fp8SFVal = tmp.__x;
#else
#error "FP8 E8M0 support requires CUDA 12.8 or newer."
#endif
  } else {
    __nv_fp8_e4m3 tmp = __nv_fp8_e4m3(SFValue);
    fp8SFVal = tmp.__x;
    SFValue = static_cast<float>(tmp);
  }
  float outputScale = SFValue != 0 ? maths::reciprocal_approximate_ftz(
                                         SFValue * maths::reciprocal_approximate_ftz(SFScaleVal))
                                   : 0.0f;

  if (SFout) *SFout = fp8SFVal;

  float2 fp2Vals[details::CVT_FP4_ELTS_PER_THREAD / 2];
#pragma unroll
  for (int i = 0; i < details::CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    if constexpr (std::is_same_v<T, half>) {
      fp2Vals[i] = __half22float2(get_vec2_element(vec, i));
    } else {
      fp2Vals[i] = __bfloat1622float2(get_vec2_element(vec, i));
    }
    fp2Vals[i].x *= outputScale;
    fp2Vals[i].y *= outputScale;
  }
  uint32_t e2m1Vec = fp32_vec_to_e2m1(fp2Vals);
  return e2m1Vec;
#else
  return 0;
#endif
}
#endif // CUDA_VERSION >= 12080

}  // namespace utils

// ============================================================================
// generic vec add
// ============================================================================
template <typename T, uint32_t VEC_SIZE>
__device__ __forceinline__ vec_t<T, VEC_SIZE> vec_add(const vec_t<T, VEC_SIZE>& a,
                                                      const vec_t<T, VEC_SIZE>& b) {
  vec_t<T, VEC_SIZE> ret;
#pragma unroll
  for (int i = 0; i < VEC_SIZE; ++i) ret[i] = static_cast<float>(a[i]) + static_cast<float>(b[i]);
  return ret;
}

enum class AllReduceFusionPattern : int {
  kAllReduce = 0,
  kARResidualRMSNorm = 1,
  kARResidualRMSNormFP8Quant = 2,
  kARResidualRMSNormFP4Quant = 3,
  kARResidualRMSNormOutFP8Quant = 4,
  kARResidualRMSNormOutFP4Quant = 5
};

enum class QuantType : int { kNone = 0, kFP8 = 1, kFP4 = 2, };

template <AllReduceFusionPattern Pattern>
struct FusionPatternTraits;

#define DEFINE_FUSION_PATTERN_TRAITS(pattern, hasAllReduceOut, hasResidual, hasResidualOut, hasRMSNorm, hasNormOut, quantType) \
  template <> struct FusionPatternTraits<pattern> { \
    static constexpr bool kHasAllReduceOut = hasAllReduceOut; \
    static constexpr bool kHasResidual     = hasResidual; \
    static constexpr bool kHasResidualOut  = hasResidualOut; \
    static constexpr bool kHasRMSNorm      = hasRMSNorm; \
    static constexpr bool kHasNormOut      = hasNormOut; \
    static constexpr QuantType kQuantType  = quantType; \
  };

DEFINE_FUSION_PATTERN_TRAITS(AllReduceFusionPattern::kAllReduce, true,  false, false, false, false, QuantType::kNone)
DEFINE_FUSION_PATTERN_TRAITS(AllReduceFusionPattern::kARResidualRMSNorm, false, true,  true,  true,  true,  QuantType::kNone)
DEFINE_FUSION_PATTERN_TRAITS(AllReduceFusionPattern::kARResidualRMSNormFP8Quant, false, true, true,  true,  false, QuantType::kFP8)
DEFINE_FUSION_PATTERN_TRAITS(AllReduceFusionPattern::kARResidualRMSNormFP4Quant, false, true, true,  true,  false, QuantType::kFP4)
DEFINE_FUSION_PATTERN_TRAITS(AllReduceFusionPattern::kARResidualRMSNormOutFP8Quant, false, true, true, true, true, QuantType::kFP8)
DEFINE_FUSION_PATTERN_TRAITS(AllReduceFusionPattern::kARResidualRMSNormOutFP4Quant, false, true, true, true, true, QuantType::kFP4)
#undef DEFINE_FUSION_PATTERN_TRAITS

template <AllReduceFusionPattern Pattern> constexpr bool HasResidual   = FusionPatternTraits<Pattern>::kHasResidual;
template <AllReduceFusionPattern Pattern> constexpr bool HasRMSNorm    = FusionPatternTraits<Pattern>::kHasRMSNorm;
template <AllReduceFusionPattern Pattern> constexpr bool HasAllReduceOut = FusionPatternTraits<Pattern>::kHasAllReduceOut;
template <AllReduceFusionPattern Pattern> constexpr bool HasResidualOut  = FusionPatternTraits<Pattern>::kHasResidualOut;
template <AllReduceFusionPattern Pattern> constexpr bool HasNormOut      = FusionPatternTraits<Pattern>::kHasNormOut;
template <AllReduceFusionPattern Pattern> constexpr QuantType GetQuantType = FusionPatternTraits<Pattern>::kQuantType;

template <typename T>
struct AllReduceFusionParams {
  int nranks;
  int rank;
  int size;
  int hidden_dim;
  void** workspace;
  void* allreduce_in;
  void* allreduce_out;
  void* residual_in;
  void* residual_out;
  void* norm_out;
  void* quant_out;
  void* scale_out;
  void* rms_gamma;
  float rms_eps;
  float* scale_factor;
  bool use_oneshot;
  QuantizationSFLayout layout = QuantizationSFLayout::SWIZZLED_128x4;
  cudaStream_t stream;
  AllReduceFusionPattern pattern;
  bool trigger_completion_at_end = false;
};

// ============================================================================
// Comm structs + Barrier (unchanged)
// ============================================================================
template <int NRanks>
struct SyncComm {
  __device__ __forceinline__ SyncComm(void** workspace) {
    counter_ptr = &reinterpret_cast<int*>(workspace[NRanks * 3])[0];
    flag_ptr = &reinterpret_cast<int*>(workspace[NRanks * 3])[1];
    flag_value = *flag_ptr;
    for (int r = 0; r < NRanks; ++r) {
      comm_bufs[r] = workspace[r];
      barrier_flags[r] = workspace[NRanks + r];
    }
    __syncthreads();
    if (threadIdx.x == 0) { atomicAdd(counter_ptr, 1); }
  }

  __device__ __forceinline__ void update(int new_flag_value) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
      while (*reinterpret_cast<int volatile*>(counter_ptr) != gridDim.x) {}
      *flag_ptr = new_flag_value;
      *counter_ptr = 0;
    }
  }

  int* counter_ptr;
  int* flag_ptr;
  void* comm_bufs[NRanks];
  void* barrier_flags[NRanks];
  int flag_value;
};

template <int NRanks>
struct LamportComm {
  __device__ __forceinline__ LamportComm(void** workspace, int rank) {
    counter_ptr = &reinterpret_cast<int*>(workspace[NRanks * 3])[0];
    flag_ptr = &reinterpret_cast<int*>(workspace[NRanks * 3])[2];
    clear_ptr = &reinterpret_cast<int*>(workspace[NRanks * 3])[4];
    flag_value = *flag_ptr;
    int comm_size = reinterpret_cast<int*>(workspace[NRanks * 3])[3];
    clear_size = *clear_ptr;
    int data_offset = flag_value % 3;
    int clear_offset = (flag_value + 2) % 3;
    for (int r = 0; r < NRanks; ++r) {
      data_bufs[r] = reinterpret_cast<uint8_t*>(workspace[2 * NRanks + r]) +
                     static_cast<int64_t>(data_offset) * comm_size;
    }
    clear_buf = reinterpret_cast<uint8_t*>(workspace[2 * NRanks + rank]) + clear_offset * comm_size;
    __syncthreads();
    if (threadIdx.x == 0) { atomicAdd(counter_ptr, 1); }
  }

  __device__ __forceinline__ void update(int new_clear_size) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
      while (*reinterpret_cast<int volatile*>(counter_ptr) != gridDim.x) {}
      *flag_ptr = (flag_value + 1) % 3;
      *clear_ptr = new_clear_size;
      *counter_ptr = 0;
    }
  }

  int* counter_ptr;
  int* flag_ptr;
  int* clear_ptr;
  uint8_t* data_bufs[NRanks];
  uint8_t* clear_buf;
  int clear_size;
  int flag_value;
};

template <int NRanks>
class Barrier {
 public:
  __device__ __forceinline__ Barrier(int rank, SyncComm<NRanks> const& comm) {
    if (threadIdx.x < NRanks) {
      m_flag_value = comm.flag_value;
      int current_rank = rank;
      int target_rank = threadIdx.x;
      m_target_flag = reinterpret_cast<int*>(comm.barrier_flags[target_rank]) + current_rank;
      m_current_flag = reinterpret_cast<int*>(comm.barrier_flags[current_rank]) +
                       blockIdx.x * NRanks + target_rank;
    }
  }

  __device__ __forceinline__ void sync() {
    __syncthreads();
    if (threadIdx.x < NRanks) {
      m_flag_value = next_flag(m_flag_value);
      for (int flag_idx = blockIdx.x; flag_idx < details::kBarrierFlagCount; flag_idx += gridDim.x) {
        st_flag(m_target_flag + flag_idx * NRanks, m_flag_value);
      }
      while (ld_flag(m_current_flag) == prev_flag(m_flag_value)) {}
    }
    __syncthreads();
  }

 protected:
  __device__ __forceinline__ void st_flag(int* addr, int flag) {
    asm volatile("st.global.release.sys.b32 [%1], %0;" ::"r"(flag), "l"(addr));
  }
  __device__ __forceinline__ int ld_flag(int* addr) {
    int flag; asm volatile("ld.global.acquire.sys.b32 %0, [%1];" : "=r"(flag) : "l"(addr)); return flag;
  }
  __device__ __forceinline__ int next_flag(int flag) { return flag == 2 ? 0 : flag + 1; }
  __device__ __forceinline__ int prev_flag(int flag) { return flag == 0 ? 2 : flag - 1; }

 public:
  int m_flag_value;

 private:
  int* m_target_flag;
  int* m_current_flag;
};

// ============================================================================
// ONE-SHOT float4 helpers + FusedOp128 (fast path)
// ============================================================================
namespace ar128 {
__device__ __forceinline__ bool is_neg_zero32(float v) { return __float_as_uint(v) == 0x80000000u; }
__device__ __forceinline__ bool is_neg_zero128(float4 v) {
  return is_neg_zero32(v.x) | is_neg_zero32(v.y) | is_neg_zero32(v.z) | is_neg_zero32(v.w);
}
__device__ __forceinline__ float4 get_neg_zero128() {
  float4 vec;
#pragma unroll
  for (int i = 0; i < 4; ++i) reinterpret_cast<uint32_t*>(&vec)[i] = 0x80000000u;
  return vec;
}
__device__ __forceinline__ float4 ld_global_volatile_f4(const float4* p) {
  float4 out;
  asm volatile("ld.volatile.global.v4.f32 {%0,%1,%2,%3}, [%4];"
               : "=f"(out.x), "=f"(out.y), "=f"(out.z), "=f"(out.w) : "l"(p));
  return out;
}
template <typename DType>
__device__ __forceinline__ float4 add128(float4 const& a, float4 const& b) {
  float4 c;
#pragma unroll
  for (int i = 0; i < int(details::kBytesPerAccess / sizeof(DType)); ++i) {
    reinterpret_cast<DType*>(&c)[i] =
        reinterpret_cast<DType const*>(&a)[i] + reinterpret_cast<DType const*>(&b)[i];
  }
  return c;
}
template <typename DType, int NRanks, bool Fp32Acc>
__device__ __forceinline__ float4 allreduce_sum128(float4* vals) {
  if constexpr (Fp32Acc) {
    static_assert(!std::is_same_v<DType, float>);
    float acc[details::kBytesPerAccess / sizeof(DType)];
#pragma unroll
    for (int i = 0; i < int(details::kBytesPerAccess / sizeof(DType)); ++i)
      acc[i] = static_cast<float>(reinterpret_cast<DType*>(&vals[0])[i]);
#pragma unroll
    for (int r = 1; r < NRanks; ++r) {
#pragma unroll
      for (int i = 0; i < int(details::kBytesPerAccess / sizeof(DType)); ++i)
        acc[i] += static_cast<float>(reinterpret_cast<DType*>(&vals[r])[i]);
    }
    float4 out;
#pragma unroll
    for (int i = 0; i < int(details::kBytesPerAccess / sizeof(DType)); ++i)
      reinterpret_cast<DType*>(&out)[i] = static_cast<DType>(acc[i]);
    return out;
  } else {
    float4 acc = vals[0];
#pragma unroll
    for (int r = 1; r < NRanks; ++r) acc = add128<DType>(acc, vals[r]);
    return acc;
  }
}
} // namespace ar128

template <AllReduceFusionPattern Pattern, typename DType>
class FusedOp128 {
  static constexpr int kMathCount = details::kBytesPerAccess / sizeof(DType);

 public:
  __device__ __forceinline__ FusedOp128(AllReduceFusionParams<DType> const& params,
                                        int access_id, int access_id_in_token)
      : m_params(params), m_access_id(access_id), m_access_id_in_token(access_id_in_token) {
    if constexpr (HasRMSNorm<Pattern>) { m_gamma_val = reinterpret_cast<float4*>(params.rms_gamma)[m_access_id_in_token]; }
    if constexpr (HasResidual<Pattern>) { m_residual_val = reinterpret_cast<float4*>(params.residual_in)[m_access_id]; }
    if constexpr (GetQuantType<Pattern> == QuantType::kFP8)      m_scale_factor = 1.f / *params.scale_factor;
    else if constexpr (GetQuantType<Pattern> == QuantType::kFP4) m_scale_factor = *params.scale_factor;
  }

  __device__ __forceinline__ void update(int access_id) {
    if (m_access_id != access_id) {
      m_access_id = access_id;
      if constexpr (HasResidual<Pattern>) m_residual_val = reinterpret_cast<float4*>(m_params.residual_in)[m_access_id];
    }
  }

  __device__ __forceinline__ void operator()(float4 val, int token_id) {
    if constexpr (HasAllReduceOut<Pattern>) reinterpret_cast<float4*>(m_params.allreduce_out)[m_access_id] = val;
    if constexpr (HasResidual<Pattern>) {
      val = ar128::add128<DType>(val, m_residual_val);
      if constexpr (HasResidualOut<Pattern>) reinterpret_cast<float4*>(m_params.residual_out)[m_access_id] = val;
    }
    if constexpr (HasRMSNorm<Pattern>) {
      val = rms_norm(val, m_gamma_val);
      if constexpr (HasNormOut<Pattern>) reinterpret_cast<float4*>(m_params.norm_out)[m_access_id] = val;
    }

#if CUDA_VERSION >= 12080
    if constexpr (GetQuantType<Pattern> == QuantType::kFP4) {
      using PackedVec = vec_t<DType, details::kBytesPerAccess / sizeof(DType)>;
      PackedVec pack_val = *reinterpret_cast<PackedVec const*>(&val);
      auto sf_out = utils::cvt_quant_to_fp4_get_sf_out_offset<uint32_t, 2>(
          std::nullopt, token_id, m_access_id_in_token, std::nullopt,
          m_params.hidden_dim, reinterpret_cast<uint32_t*>(m_params.scale_out), m_params.layout);
      reinterpret_cast<uint32_t*>(m_params.quant_out)[m_access_id] =
          utils::cvt_warp_fp16_to_fp4<DType, details::kBytesPerAccess / sizeof(DType)>(
              pack_val, m_scale_factor, sf_out);
    } else
#endif
    if constexpr (GetQuantType<Pattern> == QuantType::kFP8) {
      using PackedQ = std::conditional_t<std::is_same_v<DType, float>, float, float2>;
      PackedQ ret;
#pragma unroll
      for (int i = 0; i < kMathCount; ++i) {
        reinterpret_cast<__nv_fp8_e4m3*>(&ret)[i] =
            static_cast<__nv_fp8_e4m3>( static_cast<float>(reinterpret_cast<DType*>(&val)[i]) * m_scale_factor );
      }
      reinterpret_cast<PackedQ*>(m_params.quant_out)[m_access_id] = ret;
    } else {
      static_assert(GetQuantType<Pattern> == QuantType::kNone, "Invalid quant type");
    }
  }

 private:
  __device__ __forceinline__ float4 rms_norm(float4 const& residual, float4 const& gamma) {
    __shared__ float s_val;
    float acc = 0.f;
#pragma unroll
    for (int i = 0; i < kMathCount; ++i) {
      float v = static_cast<float>(reinterpret_cast<DType const*>(&residual)[i]);
      acc += v * v;
    }
    utils::blockReduceSumV2<float, 1>(&acc);
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    namespace cg = cooperative_groups;
    cg::cluster_group cluster = cg::this_cluster();
    if (cluster.num_blocks() > 1) {
      if (threadIdx.x == 0) { s_val = acc; acc = 0.f; }
      cluster.sync();
      if (threadIdx.x == 0) for (int i = 0; i < cluster.num_blocks(); ++i) acc += *cluster.map_shared_rank(&s_val, i);
      cluster.sync();
    }
#endif
    if (threadIdx.x == 0) s_val = rsqrtf(acc / m_params.hidden_dim + m_params.rms_eps);
    __syncthreads();

    float4 out;
#pragma unroll
    for (int i = 0; i < kMathCount; ++i) {
      reinterpret_cast<DType*>(&out)[i] =
          static_cast<DType>( static_cast<float>(reinterpret_cast<DType const*>(&residual)[i]) * s_val *
                              static_cast<float>(reinterpret_cast<DType const*>(&gamma)[i]) );
    }
    return out;
  }

 private:
  AllReduceFusionParams<DType> const& m_params;
  int m_access_id;
  int m_access_id_in_token;
  float m_scale_factor;
  float4 m_residual_val;
  float4 m_gamma_val;
};

// ============================================================================
// Generic FusedOp (vec_t path) for TWO-SHOT and any generic uses
// ============================================================================
template <AllReduceFusionPattern Pattern, typename T>
class FusedOp {
  static constexpr int VEC_SIZE = details::kBytesPerAccess / sizeof(T);
 public:
  __device__ __forceinline__ FusedOp(AllReduceFusionParams<T> const& params, int access_id, int access_id_in_token)
      : m_params(params), m_access_id(access_id), m_access_id_in_token(access_id_in_token) {
    if constexpr (HasRMSNorm<Pattern>) m_gamma_val.load(reinterpret_cast<T*>(params.rms_gamma) + m_access_id_in_token * VEC_SIZE);
    if constexpr (HasResidual<Pattern>) m_residual_val.load(reinterpret_cast<T*>(params.residual_in) + m_access_id * VEC_SIZE);
    if constexpr (GetQuantType<Pattern> == QuantType::kFP8)      m_scale_factor = 1.f / *params.scale_factor;
    else if constexpr (GetQuantType<Pattern> == QuantType::kFP4) m_scale_factor = *params.scale_factor;
  }

  __device__ __forceinline__ void update(int access_id) {
    if (m_access_id != access_id) {
      m_access_id = access_id;
      if constexpr (HasResidual<Pattern>) {
        m_residual_val.load(reinterpret_cast<T*>(m_params.residual_in) + m_access_id * VEC_SIZE);
      }
    }
  }

  __device__ __forceinline__ void operator()(vec_t<T, VEC_SIZE> val, int token_id) {
    if constexpr (HasAllReduceOut<Pattern>) {
      val.store(reinterpret_cast<T*>(m_params.allreduce_out) + m_access_id * VEC_SIZE);
    }
    if constexpr (HasResidual<Pattern>) {
      val = vec_add<T, VEC_SIZE>(val, m_residual_val);
      if constexpr (HasResidualOut<Pattern>) {
        val.store(reinterpret_cast<T*>(m_params.residual_out) + m_access_id * VEC_SIZE);
      }
    }
    if constexpr (HasRMSNorm<Pattern>) {
      val = rms_norm(val, m_gamma_val);
      if constexpr (HasNormOut<Pattern>) {
        val.store(reinterpret_cast<T*>(m_params.norm_out) + m_access_id * VEC_SIZE);
      }
    }

#if CUDA_VERSION >= 12080
    if constexpr (GetQuantType<Pattern> == QuantType::kFP4) {
      auto sf_out = utils::cvt_quant_to_fp4_get_sf_out_offset<uint32_t, 2>(
          std::nullopt, token_id, m_access_id_in_token, std::nullopt,
          m_params.hidden_dim, reinterpret_cast<uint32_t*>(m_params.scale_out), m_params.layout);
      reinterpret_cast<uint32_t*>(m_params.quant_out)[m_access_id] =
          utils::cvt_warp_fp16_to_fp4<T, VEC_SIZE>(val, m_scale_factor, sf_out);
    } else
#endif
    if constexpr (GetQuantType<Pattern> == QuantType::kFP8) {
      using PackedQuantizedType = std::conditional_t<std::is_same_v<T, float>, float, float2>;
      PackedQuantizedType ret;
#pragma unroll
      for (int i = 0; i < VEC_SIZE; ++i) {
        reinterpret_cast<__nv_fp8_e4m3*>(&ret)[i] =
            static_cast<__nv_fp8_e4m3>( static_cast<float>(reinterpret_cast<T*>(&val)[i]) * m_scale_factor );
      }
      reinterpret_cast<PackedQuantizedType*>(m_params.quant_out)[m_access_id] = ret;
    } else {
      static_assert(GetQuantType<Pattern> == QuantType::kNone, "Invalid quant type");
    }
  }

 protected:
  __device__ __forceinline__ vec_t<T, VEC_SIZE> rms_norm(vec_t<T, VEC_SIZE> const& residual,
                                                         vec_t<T, VEC_SIZE> const& gamma) {
    __shared__ float s_val;
    vec_t<T, VEC_SIZE> norm_out;
    float acc = 0.f;
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
      float v = static_cast<float>(reinterpret_cast<T const*>(&residual)[i]);
      acc += v * v;
    }
    utils::blockReduceSumV2<float, 1>(&acc);
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    namespace cg = cooperative_groups;
    cg::cluster_group cluster = cg::this_cluster();
    if (cluster.num_blocks() > 1) {
      if (threadIdx.x == 0) { s_val = acc; acc = 0.f; }
      cluster.sync();
      if (threadIdx.x == 0) {
        for (int i = 0; i < cluster.num_blocks(); ++i) acc += *cluster.map_shared_rank(&s_val, i);
      }
      cluster.sync();
    }
#endif
    if (threadIdx.x == 0) s_val = rsqrtf(acc / m_params.hidden_dim + m_params.rms_eps);
    __syncthreads();
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
      reinterpret_cast<T*>(&norm_out)[i] =
          static_cast<T>( static_cast<float>(reinterpret_cast<T const*>(&residual)[i]) * s_val *
                          static_cast<float>(reinterpret_cast<T const*>(&gamma)[i]) );
    }
    return norm_out;
  }

 private:
  AllReduceFusionParams<T> const& m_params;
  int m_access_id;
  int m_access_id_in_token;
  float m_scale_factor;
  vec_t<T, VEC_SIZE> m_residual_val;
  vec_t<T, VEC_SIZE> m_gamma_val;
};

// ============================================================================
// Original generic negative-zero helpers (kept for two-shot path)
// ============================================================================
template <typename T> struct neg_zero { static constexpr T value = -T(0); };
template <> struct neg_zero<half> {
  static constexpr unsigned short neg_zero_bits = 0x8000U;
  static constexpr __half value = __half_raw{neg_zero_bits};
};
template <> struct neg_zero<__nv_bfloat16> {
  static constexpr unsigned short neg_zero_bits = 0x8000U;
  static constexpr __nv_bfloat16 value = __nv_bfloat16_raw{neg_zero_bits};
};
template <> struct neg_zero<float> {
  static constexpr unsigned int neg_zero_bits = 0x80000000U;
  static constexpr float value = -0.0f;
};
template <typename T> __device__ static constexpr T neg_zero_v = neg_zero<T>::value;

template <typename T> __device__ bool is_negative_zero(T) { return false; }
template <> __device__ bool is_negative_zero<float>(float x) { return (__float_as_int(x) == 0x80000000); }
template <> __device__ bool is_negative_zero<double>(double x) {
  return (__double_as_longlong(x) == 0x8000000000000000ULL);
}
template <> __device__ bool is_negative_zero<__half>(__half x) { return (__half_as_ushort(x) == 0x8000); }
template <> __device__ bool is_negative_zero<__nv_bfloat16>(__nv_bfloat16 x) { return (__bfloat16_as_ushort(x) == 0x8000); }

template <typename T, uint32_t VEC_SIZE>
__device__ __forceinline__ bool has_neg_zero(const vec_t<T, VEC_SIZE>& vec) {
#pragma unroll
  for (int i = 0; i < VEC_SIZE; ++i) if (is_negative_zero(vec[i])) return true;
  return false;
}
template <typename T, uint32_t VEC_SIZE>
__device__ __forceinline__ void remove_neg_zero(vec_t<T, VEC_SIZE>& vec) {
#pragma unroll
  for (int i = 0; i < VEC_SIZE; ++i) vec[i] = (is_negative_zero(vec[i])) ? static_cast<T>(0.f) : vec[i];
}
template <typename T>
__device__ __forceinline__ void set_neg_zero(T* addr) {
  vec_t<T, details::kBytesPerAccess / sizeof(T)> val;
  val.fill(neg_zero_v<T>);
  val.store_global_volatile(addr);
}

// ============================================================================
// vec_t based sum (used by two-shot path)
// ============================================================================
template <typename T, uint32_t VEC_SIZE, int NRanks, bool Fp32Acc>
__device__ __forceinline__ vec_t<T, VEC_SIZE> allreduce_sum(vec_t<T, VEC_SIZE>* vals) {
  if constexpr (Fp32Acc) {
    static_assert(!std::is_same_v<T, float>);
    float acc_f32[VEC_SIZE];
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) acc_f32[i] = static_cast<float>(reinterpret_cast<T*>(&vals[0])[i]);
#pragma unroll
    for (int r = 1; r < NRanks; ++r) {
#pragma unroll
      for (int i = 0; i < VEC_SIZE; ++i) acc_f32[i] += static_cast<float>(reinterpret_cast<T*>(&vals[r])[i]);
    }
    vec_t<T, VEC_SIZE> acc;
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) acc[i] = static_cast<T>(acc_f32[i]);
    return acc;
  } else {
    vec_t<T, VEC_SIZE> acc = vals[0];
#pragma unroll
    for (int r = 1; r < NRanks; ++r) acc = vec_add<T, VEC_SIZE>(acc, vals[r]);
    return acc;
  }
}

// ============================================================================
// Index helper
// ============================================================================
template <typename T>
class IndexHelper {
 public:
  __device__ __forceinline__ IndexHelper(AllReduceFusionParams<T> const& params) {
    static constexpr int VEC_SIZE = details::kBytesPerAccess / sizeof(T);
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    namespace cg = cooperative_groups;
    cg::cluster_group cluster = cg::this_cluster();
    cg::grid_group grid = cg::this_grid();
    token_id = grid.cluster_rank();
    access_id_in_token = cluster.thread_rank();
    token_stride = grid.num_clusters();
#else
    token_id = blockIdx.x;
    access_id_in_token = threadIdx.x;
    token_stride = gridDim.x;
#endif
    access_id = token_id * params.hidden_dim / VEC_SIZE + access_id_in_token;
    access_stride = token_stride * params.hidden_dim / VEC_SIZE;
    tot_access = params.size / VEC_SIZE;
  }

  int token_id;
  int access_id_in_token;
  int token_stride;
  int access_id;
  int access_stride;
  int tot_access;
};

// ============================================================================
// ONE-SHOT Lamport kernel (float4 mailbox)
// ============================================================================
template <AllReduceFusionPattern Pattern, typename T, int NRanks, bool Fp32Acc,
          bool TriggerCompletionAtEnd = true>
__global__ void __launch_bounds__(1024)
allreduce_fusion_kernel_oneshot_lamport(AllReduceFusionParams<T> params) {
  static constexpr int VEC_SIZE = details::kBytesPerAccess / sizeof(T);

  IndexHelper<T> index_helper(params);
  int token_id            = index_helper.token_id;
  int access_id_in_token  = index_helper.access_id_in_token;
  int token_stride        = index_helper.token_stride;
  int access_id           = index_helper.access_id;
  int access_stride       = index_helper.access_stride;
  int tot_access          = index_helper.tot_access;

  float4 clear_vec = ar128::get_neg_zero128();
  FusedOp128<Pattern, T> fused_op(params, access_id, access_id_in_token);

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
  if constexpr (!TriggerCompletionAtEnd) { cudaTriggerProgrammaticLaunchCompletion(); }
#endif

  LamportComm<NRanks> comm(params.workspace, params.rank);
  int clear_access = comm.clear_size / VEC_SIZE;

  // Producer: push 16B lanes to all ranks, scrub -0 sentinel per 32b word
  for (int idx = access_id; idx < tot_access; idx += access_stride) {
    float4 v = reinterpret_cast<float4 const*>(params.allreduce_in)[idx];
    if (ar128::is_neg_zero128(v)) {
      alignas(16) float tmp[4]; *reinterpret_cast<float4*>(tmp) = v;
#pragma unroll
      for (int i = 0; i < 4; ++i) if (ar128::is_neg_zero32(tmp[i])) tmp[i] = 0.f;
      v = *reinterpret_cast<float4*>(tmp);
    }
#pragma unroll
    for (int r = 0; r < NRanks; ++r) {
      reinterpret_cast<float4*>(comm.data_bufs[r])[params.rank * tot_access + idx] = v;
    }
  }

  // Clear previous comm buffer region with -0 sentinel (128-bit store)
  for (int idx = access_id; idx < clear_access; idx += access_stride) {
    reinterpret_cast<float4*>(comm.clear_buf)[idx] = clear_vec;
  }

  // Consumer: volatile 128-bit polling
  for (int idx = access_id, tidx = token_id; idx < tot_access; idx += access_stride, tidx += token_stride) {
    fused_op.update(idx);
    float4 vals[NRanks];
    bool done = false;
    while (!done) {
      done = true;
#pragma unroll
      for (int r = 0; r < NRanks; ++r) {
        auto* base = &reinterpret_cast<float4*>(comm.data_bufs[params.rank])[r * tot_access + idx];
        vals[r] = ar128::ld_global_volatile_f4(base);
        done &= !ar128::is_neg_zero128(vals[r]);
      }
    }
    float4 sum_val = ar128::allreduce_sum128<T, NRanks, Fp32Acc>(vals);
    fused_op(sum_val, tidx);
  }

  comm.update(params.size * NRanks);

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  if constexpr (TriggerCompletionAtEnd) { cudaTriggerProgrammaticLaunchCompletion(); }
#endif
}

// ============================================================================
// TWO-SHOT Sync kernel (generic path; vec_t + FusedOp)
// ============================================================================
template <AllReduceFusionPattern Pattern, typename T, int NRanks, bool Fp32Acc>
__global__ void __launch_bounds__(1024)
allreduce_fusion_kernel_twoshot_sync(AllReduceFusionParams<T> params,
                                     std::array<int, NRanks> begin_tokens,
                                     std::array<int, NRanks> token_num_per_ranks) {
  static constexpr int VEC_SIZE = details::kBytesPerAccess / sizeof(T);
  IndexHelper<T> index_helper(params);
  int token_id = index_helper.token_id;
  int access_id_in_token = index_helper.access_id_in_token;
  int token_stride = index_helper.token_stride;
  int access_id = index_helper.access_id;
  int access_stride = index_helper.access_stride;
  int tot_access = index_helper.tot_access;

  FusedOp<Pattern, T> fused_op(params, access_id, access_id_in_token);

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif

  SyncComm<NRanks> comm(params.workspace);
#pragma unroll
  for (int r = 0; r < NRanks; ++r) {
    int comm_access_id = access_id + begin_tokens[r] * params.hidden_dim / VEC_SIZE;
    int comm_tot_access = (begin_tokens[r] + token_num_per_ranks[r]) * params.hidden_dim / VEC_SIZE;
    for (int idx = comm_access_id; idx < comm_tot_access; idx += access_stride) {
      reinterpret_cast<float4*>(comm.comm_bufs[params.rank])[idx] =
          reinterpret_cast<float4*>(params.allreduce_in)[idx];
    }
  }

  Barrier<NRanks> barrier(params.rank, comm);
  barrier.sync();

  int comm_access_id = access_id + begin_tokens[params.rank] * params.hidden_dim / VEC_SIZE;
  int comm_tot_access =
      (begin_tokens[params.rank] + token_num_per_ranks[params.rank]) * params.hidden_dim / VEC_SIZE;

  for (int idx = comm_access_id; idx < comm_tot_access; idx += access_stride) {
    vec_t<T, VEC_SIZE> vals[NRanks];
#pragma unroll
    for (int r = 0; r < NRanks; ++r) {
      vals[r].load(reinterpret_cast<T*>(comm.comm_bufs[r]) + idx * VEC_SIZE);
    }
    vec_t<T, VEC_SIZE> sum_val = allreduce_sum<T, VEC_SIZE, NRanks, Fp32Acc>(vals);
#pragma unroll
    for (int r = 0; r < NRanks; ++r) {
      sum_val.store(reinterpret_cast<T*>(comm.comm_bufs[r]) + (tot_access + idx) * VEC_SIZE);
    }
  }

  barrier.sync();

#pragma unroll
  for (int r = 0; r < NRanks; ++r) {
    int comm_access_id = access_id + begin_tokens[r] * params.hidden_dim / VEC_SIZE;
    int comm_token_id = token_id + begin_tokens[r];
    int comm_tot_access = (begin_tokens[r] + token_num_per_ranks[r]) * params.hidden_dim / VEC_SIZE;
    for (int idx = comm_access_id, tidx = comm_token_id; idx < comm_tot_access;
         idx += access_stride, tidx += token_stride) {
      fused_op.update(idx);
      vec_t<T, VEC_SIZE> sum_val;
      sum_val.load(reinterpret_cast<T*>(comm.comm_bufs[params.rank]) +
                   (tot_access + idx) * VEC_SIZE);
      fused_op(sum_val, tidx);
    }
  }

  comm.update(barrier.m_flag_value);

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}

// ============================================================================
// Device/SM helpers and launchers
// ============================================================================
int get_sm_count() {
  static int sm_count = 0;
  if (sm_count == 0) {
    int device_id;
    FLASHINFER_CUDA_CALL(cudaGetDevice(&device_id));
    FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_id));
  }
  return sm_count;
}

template <AllReduceFusionPattern Pattern, typename T, int NRanks, bool Fp32Acc,
          bool TriggerCompletionAtEnd = true>
cudaError_t launch_oneshot_lamport(AllReduceFusionParams<T> const& params, cudaLaunchConfig_t& cfg) {
  FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(
      &cfg, allreduce_fusion_kernel_oneshot_lamport<Pattern, T, NRanks, Fp32Acc, TriggerCompletionAtEnd>, params));
  return cudaSuccess;
}

template <AllReduceFusionPattern Pattern, typename T, int NRanks, bool Fp32Acc>
cudaError_t launch_twoshot_sync(AllReduceFusionParams<T> const& params, cudaLaunchConfig_t& cfg,
                                std::array<int, NRanks> begin_tokens,
                                std::array<int, NRanks> token_num_per_ranks) {
  FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(
      &cfg, allreduce_fusion_kernel_twoshot_sync<Pattern, T, NRanks, Fp32Acc>,
      params, begin_tokens, token_num_per_ranks));
  return cudaSuccess;
}

bool use_oneshot(int token_num) { return token_num <= details::kOneShotMaxToken; }

template <AllReduceFusionPattern Pattern, typename T, int NRanks, bool Fp32Acc>
cudaError_t allreduce_fusion_kernel_launcher(AllReduceFusionParams<T> const& params,
                                             bool launch_with_pdl) {
  static constexpr int VEC_SIZE = details::kBytesPerAccess / sizeof(T);
  FLASHINFER_CHECK(params.size % params.hidden_dim == 0, "params.size % params.hidden_dim != 0");
  FLASHINFER_CHECK(params.hidden_dim % VEC_SIZE == 0, "params.hidden_dim % VEC_SIZE != 0");
  static int SM = utils::getSMVersion();

  int token_num = params.size / params.hidden_dim;
  bool oneshot = params.use_oneshot;

  int cluster_num = token_num;
  std::array<int, NRanks> begin_tokens{}, token_num_per_ranks{};

  if (!oneshot) {
    int remaining_token = token_num % NRanks;
    int token_num_per_rank = token_num / NRanks;
    cluster_num = token_num_per_rank + (remaining_token ? 1 : 0);
    for (int r = 0; r < NRanks; ++r) {
      begin_tokens[r] = r * token_num_per_rank + (remaining_token > r ? r : remaining_token);
      token_num_per_ranks[r] = token_num_per_rank + (remaining_token > r ? 1 : 0);
    }
  }

  int threads_per_token = params.hidden_dim / VEC_SIZE;
  int cluster_size = (SM >= 90) ? 8 : 1;

  while (threads_per_token % cluster_size != 0 && cluster_size > 1) cluster_size /= 2;
  int threads_per_block = threads_per_token / cluster_size;
  while (threads_per_block < 128 && cluster_size >= 2) {
    threads_per_block *= 2; cluster_size /= 2;
  }

  int sm_count = get_sm_count();
  while (cluster_num * cluster_size > sm_count && cluster_size > 1 && threads_per_block <= 512) {
    threads_per_block *= 2; cluster_size /= 2;
  }

  FLASHINFER_CHECK(oneshot || threads_per_block >= params.nranks,
                   "not oneshot, or threads_per_block < nranks");
  int block_size = threads_per_block;
  FLASHINFER_CHECK(block_size <= 1024 && cluster_size > 0,
                   "block_size > 1024 or cluster_size <= 0");

  int grid_size = (std::min(sm_count, cluster_num * cluster_size) / cluster_size) * cluster_size;

  cudaLaunchConfig_t cfg;
  cudaLaunchAttribute attribute[2];
  cfg.gridDim = grid_size;
  cfg.blockDim = block_size;
  cfg.dynamicSmemBytes = 0;
  cfg.stream = params.stream;
  attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attribute[0].val.programmaticStreamSerializationAllowed = launch_with_pdl ? 1 : 0;
  attribute[1].id = cudaLaunchAttributeClusterDimension;
  attribute[1].val.clusterDim.x = cluster_size;
  attribute[1].val.clusterDim.y = 1;
  attribute[1].val.clusterDim.z = 1;
  cfg.attrs = attribute;
  cfg.numAttrs = SM >= 90 ? 2 : 0;

  if (oneshot) {
    // bool trigger_completion_at_end = params.trigger_completion_at_end;
    bool trigger_completion_at_end = false; // for better PDL overlap
    if (trigger_completion_at_end) {
      FLASHINFER_CUDA_CALL((launch_oneshot_lamport<Pattern, T, NRanks, Fp32Acc, true>(params, cfg)));
    } else {
      FLASHINFER_CUDA_CALL((launch_oneshot_lamport<Pattern, T, NRanks, Fp32Acc, false>(params, cfg)));
    }
  } else {
    FLASHINFER_CUDA_CALL((launch_twoshot_sync<Pattern, T, NRanks, Fp32Acc>(
        params, cfg, begin_tokens, token_num_per_ranks)));
  }
  return cudaSuccess;
}

// ============================================================================
// Public entry (pattern/rank dispatch). We keep fp32_acc forced off for parity.
// ============================================================================
template <typename T>
cudaError_t allreduce_fusion_op(AllReduceFusionParams<T> const& params, bool launch_with_pdl,
                                bool fp32_acc) {
  // keep parity with original: force disable fp32 accumulation for now
  fp32_acc = false;
#define DISPATCH_ACC_TYPE(T_, Pattern, NRanks)                                                     \
  if constexpr (std::is_same_v<T_, float>) {                                                       \
    return allreduce_fusion_kernel_launcher<Pattern, T_, NRanks, false>(params, launch_with_pdl);  \
  } else {                                                                                         \
    if (fp32_acc) {                                                                                \
      return allreduce_fusion_kernel_launcher<Pattern, T_, NRanks, true>(params, launch_with_pdl); \
    } else {                                                                                       \
      return allreduce_fusion_kernel_launcher<Pattern, T_, NRanks, false>(params, launch_with_pdl);\
    }                                                                                              \
  }

#define DISPATCH_PATTERN(T_, NRanks)                                                        \
  switch (params.pattern) {                                                                 \
    case AllReduceFusionPattern::kAllReduce:                                                \
      DISPATCH_ACC_TYPE(T_, AllReduceFusionPattern::kAllReduce, NRanks);                    \
      break;                                                                                \
    case AllReduceFusionPattern::kARResidualRMSNorm:                                        \
      DISPATCH_ACC_TYPE(T_, AllReduceFusionPattern::kARResidualRMSNorm, NRanks);            \
      break;                                                                                \
    case AllReduceFusionPattern::kARResidualRMSNormFP8Quant:                                \
      DISPATCH_ACC_TYPE(T_, AllReduceFusionPattern::kARResidualRMSNormFP8Quant, NRanks);    \
      break;                                                                                \
    case AllReduceFusionPattern::kARResidualRMSNormFP4Quant:                                \
      if constexpr (!std::is_same_v<T_, float> && CUDA_VERSION >= 12080) {                  \
        DISPATCH_ACC_TYPE(T_, AllReduceFusionPattern::kARResidualRMSNormFP4Quant, NRanks);  \
      } else {                                                                              \
        FLASHINFER_CHECK(CUDA_VERSION >= 12080, "FP4Quant requires CUDA 12.8 or higher");   \
        FLASHINFER_CHECK(false, "FP4Quant pattern cannot work with DType=float");           \
      }                                                                                     \
      break;                                                                                \
    case AllReduceFusionPattern::kARResidualRMSNormOutFP8Quant:                             \
      DISPATCH_ACC_TYPE(T_, AllReduceFusionPattern::kARResidualRMSNormOutFP8Quant, NRanks); \
      break;                                                                                \
    case AllReduceFusionPattern::kARResidualRMSNormOutFP4Quant:                             \
      if constexpr (!std::is_same_v<T_, float> && CUDA_VERSION >= 12080) {                  \
        DISPATCH_ACC_TYPE(T_, AllReduceFusionPattern::kARResidualRMSNormOutFP4Quant, NRanks);\
      } else {                                                                              \
        FLASHINFER_CHECK(CUDA_VERSION >= 12080, "OutFP4Quant requires CUDA 12.8 or higher");\
        FLASHINFER_CHECK(false, "OutFP4Quant pattern cannot work with DType=float");        \
      }                                                                                     \
      break;                                                                                \
    default:                                                                                \
      FLASHINFER_CHECK(false, "Unsupported allreduce fusion pattern");                      \
  }

  switch (params.nranks) {
    case 2:  DISPATCH_PATTERN(T, 2);  break;
    case 4:  DISPATCH_PATTERN(T, 4);  break;
    case 8:  DISPATCH_PATTERN(T, 8);  break;
    case 16: DISPATCH_PATTERN(T, 16); break;
    default:
      FLASHINFER_ERROR("allreduce_fusion_kernel: unsupported ranks number! Supported ranks: 2, 4, 8, 16.");
  }
}

}  // namespace trtllm_allreduce_fusion

}  // namespace flashinfer
