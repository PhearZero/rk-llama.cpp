#include "ggml-rknpu2.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "ggml-quants.h"
#include "ggml-cpu.h"

#include "rknpu2-allocation.h"
#include "rknpu2-quantization.h"
#include "rknpu2-calibration.h"
#include "rknpu2-configuration.h"

#include <rknn_api.h>
#include <rknn_matmul_api.h>

#include <arm_neon.h>
#include <omp.h>

#include <chrono>
#include <cassert>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>
#include <tuple>
#include <algorithm>
#include <memory>
#include <unordered_map>
#include <cstdio>
#include <cmath>

#define UNUSED(x) (void)(x)

// Macro for RKNN API calls
#define RKNN_CHECK(stmt, msg)                                           \
    do {                                                                \
        int ret = (stmt);                                               \
        if (ret < 0) {                                                  \
            fprintf(stderr,"RKNN error %d at %s:%d: %s\n", ret,         \
                __FILE__, __LINE__, msg);                               \
            assert(false);                                              \
        }                                                               \
    } while (0)

// --- Hashers ---

template <class T>
inline void hash_combine(std::size_t& seed, const T& v) {
    std::hash<T> hasher;
    seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

struct PairHasher {
    template <class T1, class T2>
    std::size_t operator()(const std::pair<T1, T2>& p) const {
        std::size_t seed = 0;
        hash_combine(seed, p.first);
        hash_combine(seed, p.second);
        return seed;
    }
};

struct TupleHasher {
    template <typename... Ts>
    std::size_t operator()(const std::tuple<Ts...>& t) const {
        std::size_t seed = 0;
        std::apply([&](const auto&... args) {
            (hash_combine(seed, args), ...);
        }, t);
        return seed;
    }
};

// --- Segmenters ---

struct MatrixSegment {
    int offset_n;
    int size_n;
    int core_id;
};

static std::vector<MatrixSegment> compute_matrix_segments(int N, int num_cores, int alignment) {
    std::vector<MatrixSegment> segments;
    int base_segment_size = (N / num_cores / alignment) * alignment;
    int remaining = N - (base_segment_size * num_cores);
    int offset = 0;
    for (int i = 0; i < num_cores; i++) {
        MatrixSegment seg;
        seg.offset_n = offset;
        seg.size_n = base_segment_size;
        if (i == num_cores - 1) seg.size_n += remaining;
        seg.core_id = i;
        segments.push_back(seg);
        offset += seg.size_n;
    }
    return segments;
}

// --- Memory Context ---

struct rknpu_memory_context {
    rknn_context ctx = 0;
    rknpu_memory_context() {
        rknn_matmul_info info = {};
        info.M = 128;
        info.K = 128;
        info.N = 128;
        info.type = RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32;
        info.B_layout = RKNN_MM_LAYOUT_NATIVE;
        info.AC_layout = RKNN_MM_LAYOUT_NORM;
        rknn_matmul_io_attr io_attr;
        int ret = rknn_matmul_create(&ctx, &info, &io_attr);
        if (ret < 0) {
            fprintf(stderr, "RKNPU2: Failed to create initial rknn context (%d)\n", ret);
            ctx = 0;
        }
    }
    ~rknpu_memory_context() {
        if (ctx != 0) rknn_matmul_destroy(ctx);
    }
};

static rknpu_memory_context & get_rknpu_memory_context() {
    static rknpu_memory_context instance;
    return instance;
}

// --- Backend structures ---

struct ggml_backend_rknpu_buffer_context {
    rknpu2_allocation::DmaBuffer dma_buf;
    std::string name;
    std::unordered_map<size_t, float> quantized_tensor_scales;
    std::unordered_map<size_t, std::vector<float>> hadamard_s_vectors;
    std::mutex mutex;
};

struct rknpu_matmul_context {
    rknn_matmul_info info;
    rknn_matmul_io_attr io_attr;
    rknn_matmul_ctx ctx = 0;
    rknpu_matmul_context(int M, int K, int N, rknn_matmul_type type) {
        memset(&info, 0, sizeof(info));
        info.M = M;
        info.K = K;
        info.N = N;
        info.type = type;
        info.B_layout = RKNN_MM_LAYOUT_NATIVE;
        info.AC_layout = RKNN_MM_LAYOUT_NORM;
        int ret = rknn_matmul_create(&ctx, &info, &io_attr);
        if (ret < 0) {
            fprintf(stderr, "RKNPU2: matmul context creation failed (%d)\n", ret);
            ctx = 0;
        }
    }
    ~rknpu_matmul_context() {
        if (ctx != 0) rknn_matmul_destroy(ctx);
    }
};

struct ggml_backend_rknpu_context {
    std::string name;
    std::mutex mutex;
    std::unordered_map<std::tuple<int, int, int, int, int>, std::shared_ptr<rknpu_matmul_context>, TupleHasher> matmul_ctx_cache;
    std::unordered_map<std::pair<ggml_backend_buffer_t, size_t>, std::shared_ptr<rknn_tensor_mem>, PairHasher> b_mem_handle_cache;
    std::unordered_map<std::tuple<int, int, int>, std::shared_ptr<rknn_tensor_mem>, TupleHasher> a_buffer_cache;
    std::unordered_map<std::tuple<int, int, int>, std::shared_ptr<rknn_tensor_mem>, TupleHasher> c_buffer_cache;

    std::shared_ptr<rknpu_matmul_context> get_matmul_ctx(int M, int K, int N, int core_id, rknn_matmul_type type) {
        std::lock_guard<std::mutex> lock(mutex);
        auto key = std::make_tuple(M, K, N, core_id, (int)type);
        auto it = matmul_ctx_cache.find(key);
        if (it != matmul_ctx_cache.end()) return it->second;
        
        auto ctx = std::make_shared<rknpu_matmul_context>(M, K, N, type);
        if (!ctx || ctx->ctx == 0) return nullptr;
        
        rknn_core_mask core_mask;
        switch(core_id) {
            case 0: core_mask = RKNN_NPU_CORE_0; break;
            case 1: core_mask = RKNN_NPU_CORE_1; break;
            case 2: core_mask = RKNN_NPU_CORE_2; break;
            default: core_mask = RKNN_NPU_CORE_AUTO; break;
        }
        rknn_matmul_set_core_mask(ctx->ctx, core_mask);
        
        matmul_ctx_cache[key] = ctx;
        return ctx;
    }
};

// --- Backend interface helpers ---

static void * ggml_backend_rknpu_buffer_get_base(ggml_backend_buffer_t buffer);

static size_t ggml_backend_rknpu_get_tensor_offset(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor) {
    if (!buffer || !tensor) return 0;
    void * base = ggml_backend_rknpu_buffer_get_base(buffer);
    return (size_t)((uint8_t*)tensor->data - (uint8_t*)base);
}

static void* ggml_rknpu_get_system_ptr(const struct ggml_tensor* tensor) {
    if (tensor && tensor->buffer && tensor->buffer->iface.get_base == ggml_backend_rknpu_buffer_get_base) {
        auto* buf_ctx = (ggml_backend_rknpu_buffer_context*)tensor->buffer->context;
        size_t offset = ggml_backend_rknpu_get_tensor_offset(tensor->buffer, tensor);
        return (uint8_t*)buf_ctx->dma_buf.virt_addr + offset;
    }
    return tensor ? (void*)tensor->data : nullptr;
}

static void rknpu_sync_tensor(const struct ggml_tensor * tensor, rknn_mem_sync_mode type, size_t offset, size_t size) {
    if (!tensor || !tensor->buffer) return;
    auto* buf_ctx = (ggml_backend_rknpu_buffer_context*)tensor->buffer->context;
    if (!buf_ctx || !buf_ctx->dma_buf.virt_addr) return;

    size_t total_offset = ggml_backend_rknpu_get_tensor_offset(tensor->buffer, tensor) + offset;
    
    rknn_tensor_mem sync_mem = {};
    sync_mem.virt_addr = buf_ctx->dma_buf.virt_addr;
    sync_mem.fd = buf_ctx->dma_buf.fd;
    sync_mem.offset = (uint64_t)total_offset;
    sync_mem.size = (uint32_t)size;
    
    rknn_context mem_ctx = get_rknpu_memory_context().ctx;
    if (mem_ctx != 0) {
        rknn_mem_sync(mem_ctx, &sync_mem, type);
    }
}

// --- Compute fallbacks ---

static void rknpu_compute_forward_set_rows(ggml_tensor* dst) {
    const struct ggml_tensor* src1 = dst->src[1]; // values
    const struct ggml_tensor* src2 = dst->src[2]; // indices

    rknpu_sync_tensor(src1, RKNN_MEMORY_SYNC_FROM_DEVICE, 0, ggml_nbytes(src1));
    rknpu_sync_tensor(src2, RKNN_MEMORY_SYNC_FROM_DEVICE, 0, ggml_nbytes(src2));
    rknpu_sync_tensor(dst, RKNN_MEMORY_SYNC_FROM_DEVICE, 0, ggml_nbytes(dst));

    const void* src1_data = ggml_rknpu_get_system_ptr(src1);
    const void* src2_data = ggml_rknpu_get_system_ptr(src2);
    void* dst_data = ggml_rknpu_get_system_ptr(dst);

    for (int64_t i = 0; i < ggml_nelements(src2); ++i) {
        int64_t i01 = (src2->type == GGML_TYPE_I32) ? ((const int32_t*)src2_data)[i] : ((const int64_t*)src2_data)[i];
        if (i01 >= 0 && i01 < dst->src[0]->ne[1]) {
            memcpy((uint8_t*)dst_data + i01 * dst->nb[1], (const uint8_t*)src1_data + i * src1->nb[1], src1->nb[1]);
        }
    }

    rknpu_sync_tensor(dst, RKNN_MEMORY_SYNC_TO_DEVICE, 0, ggml_nbytes(dst));
}

static void rknpu_compute_forward_get_rows(ggml_tensor* dst) {
    const struct ggml_tensor* src0 = dst->src[0];
    const struct ggml_tensor* src1 = dst->src[1];

    rknpu_sync_tensor(src0, RKNN_MEMORY_SYNC_FROM_DEVICE, 0, ggml_nbytes(src0));
    rknpu_sync_tensor(src1, RKNN_MEMORY_SYNC_FROM_DEVICE, 0, ggml_nbytes(src1));

    const void* src0_data = ggml_rknpu_get_system_ptr(src0);
    const void* src1_data = ggml_rknpu_get_system_ptr(src1);
    void* dst_data = ggml_rknpu_get_system_ptr(dst);

    for (int64_t i = 0; i < ggml_nelements(src1); ++i) {
        int64_t i01 = (src1->type == GGML_TYPE_I32) ? ((const int32_t*)src1_data)[i] : ((const int64_t*)src1_data)[i];
        if (i01 >= 0 && i01 < src0->ne[1]) {
            memcpy((uint8_t*)dst_data + i * dst->nb[1], (const uint8_t*)src0_data + i01 * src0->nb[1], dst->nb[1]);
        }
    }

    rknpu_sync_tensor(dst, RKNN_MEMORY_SYNC_TO_DEVICE, 0, ggml_nbytes(dst));
}

static void rknpu_compute_forward_cpy(ggml_tensor* dst) {
    const struct ggml_tensor* src = dst->src[0];
    rknpu_sync_tensor(src, RKNN_MEMORY_SYNC_FROM_DEVICE, 0, ggml_nbytes(src));
    rknpu_sync_tensor(dst, RKNN_MEMORY_SYNC_FROM_DEVICE, 0, ggml_nbytes(dst));

    ggml_to_float_t to_f = ggml_get_type_traits(src->type)->to_float;
    ggml_from_float_t from_f = ggml_get_type_traits_cpu(dst->type)->from_float;

    if (to_f && from_f) {
        std::vector<float> row(src->ne[0]);
        for (int64_t i = 0; i < ggml_nelements(src)/src->ne[0]; i++) {
            to_f((const char*)src->data + i*src->nb[1], row.data(), src->ne[0]);
            from_f(row.data(), (char*)dst->data + i*dst->nb[1], src->ne[0]);
        }
    } else if (src->type == dst->type) {
        memcpy(dst->data, src->data, ggml_nbytes(dst));
    }

    rknpu_sync_tensor(dst, RKNN_MEMORY_SYNC_TO_DEVICE, 0, ggml_nbytes(dst));
}

// --- Main graph compute ---

static std::shared_ptr<rknn_tensor_mem> get_or_create_npu_buffer(
    ggml_backend_rknpu_context* backend_ctx,
    rknn_matmul_ctx matmul_ctx,
    size_t size,
    const std::tuple<int, int, int>& key,
    std::unordered_map<std::tuple<int, int, int>, std::shared_ptr<rknn_tensor_mem>, TupleHasher>& cache
) {
    std::lock_guard<std::mutex> lock(backend_ctx->mutex);
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;
    
    if (size == 0) return nullptr;
    rknn_tensor_mem* mem = rknn_create_mem(matmul_ctx, size);
    if (!mem) return nullptr;
    
    auto deleter = [matmul_ctx](rknn_tensor_mem* m) {
        if (m) rknn_destroy_mem(matmul_ctx, m);
    };
    
    auto mem_shared = std::shared_ptr<rknn_tensor_mem>(mem, deleter);
    cache[key] = mem_shared;
    return mem_shared;
}

static enum ggml_status ggml_backend_rknpu_graph_compute(ggml_backend_t backend, struct ggml_cgraph* cgraph) {
    auto* bctx = (ggml_backend_rknpu_context*)backend->context;
    const auto& cfg = rknpu2_configuration::Rknpu2ConfigManager::get_instance().get_current_config();

    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor* node = cgraph->nodes[i];
        if (!node) continue;
        
        if (node->op == GGML_OP_SET_ROWS) {
            rknpu_compute_forward_set_rows(node);
            continue;
        } else if (node->op == GGML_OP_GET_ROWS) {
            rknpu_compute_forward_get_rows(node);
            continue;
        } else if (node->op == GGML_OP_CPY || node->op == GGML_OP_DUP || node->op == GGML_OP_CONT) {
            rknpu_compute_forward_cpy(node);
            continue;
        } else if (node->op != GGML_OP_MUL_MAT) {
            continue;
        }

        const struct ggml_tensor* src0 = node->src[0]; 
        const struct ggml_tensor* src1 = node->src[1]; 
        const int M = (int)src1->ne[1];
        const int K = (int)src0->ne[0];
        const int N = (int)src0->ne[1];

        const int Mop = (M == 1) ? 1 : (M <= 32 ? (M <= 8 ? 8 : 32) : (M <= 512 ? 512 : ((M + 7) / 8 * 8)));
        const int Kop = (src0->type == GGML_TYPE_Q4_0) ? rknpu2_calibration::next_power_of_two(K) : K;

        const auto* op_support = cfg.find_op_support(src0->type);
        if (!op_support) return GGML_STATUS_FAILED;

        auto segs = compute_matrix_segments(N, cfg.core_count, op_support->n_align);
        const size_t nseg = segs.size();

        std::vector<std::shared_ptr<rknpu_matmul_context>> ctxs(nseg);
        std::vector<std::shared_ptr<rknn_tensor_mem>> mB(nseg), mC(nseg);
        
        size_t boff = 0;
        for (size_t j = 0; j < nseg; j++) {
            ctxs[j] = bctx->get_matmul_ctx(Mop, Kop, segs[j].size_n, segs[j].core_id, op_support->mm_type);
            if (!ctxs[j]) return GGML_STATUS_FAILED;

            size_t bsz = ctxs[j]->io_attr.B.size;
            size_t aoff = ggml_backend_rknpu_get_tensor_offset(src0->buffer, src0) + boff;
            auto keyb = std::make_pair(src0->buffer, aoff);
            
            {
                std::lock_guard<std::mutex> lock(bctx->mutex);
                if (bctx->b_mem_handle_cache.count(keyb)) mB[j] = bctx->b_mem_handle_cache[keyb];
            }

            if (!mB[j]) {
                rknn_tensor_mem* m = rknn_create_mem_from_fd(ctxs[j]->ctx, ((ggml_backend_rknpu_buffer_context*)src0->buffer->context)->dma_buf.fd, ((ggml_backend_rknpu_buffer_context*)src0->buffer->context)->dma_buf.virt_addr, bsz, (int32_t)aoff);
                if (!m) return GGML_STATUS_FAILED;
                
                rknn_context cp = ctxs[j]->ctx;
                mB[j] = std::shared_ptr<rknn_tensor_mem>(m, [cp](rknn_tensor_mem* rm) {
                    if (rm) rknn_destroy_mem(cp, rm);
                });
                
                std::lock_guard<std::mutex> lock(bctx->mutex);
                bctx->b_mem_handle_cache[keyb] = mB[j];
            }
            
            RKNN_CHECK(rknn_matmul_set_io_mem(ctxs[j]->ctx, mB[j].get(), &ctxs[j]->io_attr.B), "set_io_mem B");
            boff += bsz;
        }

        auto mA = get_or_create_npu_buffer(bctx, ctxs[0]->ctx, ctxs[0]->io_attr.A.size, std::make_tuple(Mop, Kop, (int)op_support->mm_type), bctx->a_buffer_cache);
        if (!mA) return GGML_STATUS_FAILED;

        rknpu_sync_tensor(src1, RKNN_MEMORY_SYNC_FROM_DEVICE, 0, ggml_nbytes(src1));
        const void* s1p = ggml_rknpu_get_system_ptr(src1);
        float scA[2048] = {0};
        
        if (op_support->npu_type_a == rknpu2_configuration::NPU_TYPE_FP16) {
#pragma omp parallel for
            for (int m = 0; m < Mop; m++) {
                uint16_t* d = (uint16_t*)mA->virt_addr + (size_t)m * Kop;
                if (m < M) {
                    const void* r = (const char*)s1p + m*src1->nb[1];
                    if (src1->type == GGML_TYPE_F32) rknpu2_quantization::convert_fp32_to_fp16((const float*)r, d, K);
                    else memcpy(d, r, K * 2);
                    if (Kop > K) memset(d + K, 0, (Kop - K) * 2);
                } else memset(d, 0, Kop * 2);
            }
        } else if (op_support->npu_type_a == rknpu2_configuration::NPU_TYPE_INT8) {
#pragma omp parallel for
            for (int m = 0; m < Mop; m++) {
                int8_t* d = (int8_t*)mA->virt_addr + (size_t)m * Kop;
                if (m < M) {
                    const void* r = (const char*)s1p + m*src1->nb[1];
                    float am = (src1->type == GGML_TYPE_F32) ? rknpu2_quantization::calculate_amax_fp32((const float*)r, K) : rknpu2_quantization::calculate_amax_fp16((const uint16_t*)r, K);
                    scA[m] = am / 127.0f;
                    if (src1->type == GGML_TYPE_F32) rknpu2_quantization::quantize_fp32_to_int8((const float*)r, d, K, scA[m]);
                    else rknpu2_quantization::quantize_fp16_to_int8((const uint16_t*)r, d, K, scA[m]);
                    if (Kop > K) memset(d + K, 0, Kop - K);
                } else { memset(d, 0, Kop); scA[m] = 0.0f; }
            }
        } else if (op_support->npu_type_a == rknpu2_configuration::NPU_TYPE_INT4) {
             std::vector<float> sv;
             { 
                 std::lock_guard<std::mutex> l(((ggml_backend_rknpu_buffer_context*)src0->buffer->context)->mutex); 
                 sv = ((ggml_backend_rknpu_buffer_context*)src0->buffer->context)->hadamard_s_vectors[ggml_backend_rknpu_get_tensor_offset(src0->buffer, src0)]; 
             }
#pragma omp parallel for
             for (int m = 0; m < Mop; m++) {
                 uint8_t* d = (uint8_t*)mA->virt_addr + (size_t)m * Kop / 2;
                 if (m < M) {
                     std::vector<float> t(Kop, 0.0f); const void* r = (const char*)s1p + m*src1->nb[1];
                     if (src1->type == GGML_TYPE_F32) memcpy(t.data(), r, K * 4);
                     else { for (int k=0; k<K; k++) t[k] = ggml_fp16_to_fp32(((const uint16_t*)r)[k]); }
                     for (int k=0; k<Kop; k++) t[k] *= sv[k];
                     std::vector<float> rot(Kop); rknpu2_calibration::hadamard_transform(rot.data(), t.data(), K, Kop);
                     scA[m] = rknpu2_quantization::calculate_amax_fp32(rot.data(), Kop) / 7.0f;
                     rknpu2_quantization::quantize_fp32_to_int4_packed(rot.data(), d, Kop, scA[m]);
                 } else { memset(d, 0, Kop/2); scA[m] = 0.0f; }
             }
        }

        for (size_t j = 0; j < nseg; j++) RKNN_CHECK(rknn_matmul_set_io_mem(ctxs[j]->ctx, mA.get(), &ctxs[j]->io_attr.A), "set_io_mem A shared");
        
        {
            rknn_tensor_mem sync_mA = *mA;
            RKNN_CHECK(rknn_mem_sync(ctxs[0]->ctx, &sync_mA, RKNN_MEMORY_SYNC_TO_DEVICE), "sync mA TO_DEVICE");
        }

#pragma omp parallel num_threads(nseg)
        {
            int j = omp_get_thread_num();
            mC[j] = get_or_create_npu_buffer(bctx, ctxs[j]->ctx, ctxs[j]->io_attr.C.size, std::make_tuple(Mop, segs[j].size_n, segs[j].core_id), bctx->c_buffer_cache);
            RKNN_CHECK(rknn_matmul_set_io_mem(ctxs[j]->ctx, mC[j].get(), &ctxs[j]->io_attr.C), "set_io_mem C");
            RKNN_CHECK(rknn_matmul_run(ctxs[j]->ctx), "run");
            
            rknn_tensor_mem sync_mC = *mC[j];
            RKNN_CHECK(rknn_mem_sync(ctxs[j]->ctx, &sync_mC, RKNN_MEMORY_SYNC_FROM_DEVICE), "sync mC FROM_DEVICE");
        }

        float scB = 1.0f;
        if (op_support->mm_type != RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32) {
            std::lock_guard<std::mutex> l(((ggml_backend_rknpu_buffer_context*)src0->buffer->context)->mutex);
            scB = ((ggml_backend_rknpu_buffer_context*)src0->buffer->context)->quantized_tensor_scales[ggml_backend_rknpu_get_tensor_offset(src0->buffer, src0)];
        }

        rknpu_sync_tensor(node, RKNN_MEMORY_SYNC_FROM_DEVICE, 0, ggml_nbytes(node));
        void* dp = ggml_rknpu_get_system_ptr(node);

        for (size_t j = 0; j < nseg; j++) {
            const auto& seg = segs[j];
            void* sC = mC[j]->virt_addr;
#pragma omp parallel for
            for (int m = 0; m < M; m++) {
                float* dr = (float*)((uint8_t*)dp + (size_t)m * node->nb[1]) + seg.offset_n;
                if (op_support->mm_type == RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32) {
                    memcpy(dr, (float*)sC + (size_t)m * seg.size_n, seg.size_n * 4);
                } else {
                    float fsc = scB * scA[m] * (op_support->mm_type == RKNN_INT4_MM_INT4_TO_INT16 ? 1.0f/sqrt(Kop) : 1.0f);
                    if (op_support->mm_type == RKNN_INT8_MM_INT8_TO_INT32) {
                        const int32_t* sc_row = (const int32_t*)sC + (size_t)m * seg.size_n;
                        for (int n=0; n<seg.size_n; n++) dr[n] = sc_row[n] * fsc;
                    } else {
                        const int16_t* sc_row = (const int16_t*)sC + (size_t)m * seg.size_n;
                        for (int n=0; n<seg.size_n; n++) dr[n] = sc_row[n] * fsc;
                    }
                }
            }
        }
        rknpu_sync_tensor(node, RKNN_MEMORY_SYNC_TO_DEVICE, 0, ggml_nbytes(node));
    }
    return GGML_STATUS_SUCCESS;
}

// --- Backend interface implementation ---

static void ggml_backend_rknpu_free(ggml_backend_t backend) {
    if (backend->context) delete (ggml_backend_rknpu_context *)backend->context;
    delete backend;
}

static void ggml_backend_rknpu_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    if (buffer && buffer->context) {
        auto* ctx = (ggml_backend_rknpu_buffer_context *)buffer->context;
        rknpu2_allocation::free(ctx->dma_buf);
        delete ctx;
    }
}

static void * ggml_backend_rknpu_buffer_get_base(ggml_backend_buffer_t buffer) {
    return ((ggml_backend_rknpu_buffer_context *)buffer->context)->dma_buf.virt_addr;
}

static enum ggml_status ggml_backend_rknpu_buffer_init_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor) {
    UNUSED(buffer);
    if (tensor->view_src && !tensor->data) {
        tensor->data = (uint8_t *)tensor->view_src->data + tensor->view_offs;
    }
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_rknpu_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    auto* bctx = (ggml_backend_rknpu_buffer_context*)buffer->context;
    uint8_t* ptr = (uint8_t*)bctx->dma_buf.virt_addr + ggml_backend_rknpu_get_tensor_offset(buffer, tensor);
    if (!data) return;

    const auto& cfg = rknpu2_configuration::Rknpu2ConfigManager::get_instance().get_current_config();
    const auto* op = cfg.find_op_support(tensor->type);

    if (op && op->pack_func && offset == 0 && size == ggml_nbytes(tensor)) {
        int K = (int)tensor->ne[0], N = (int)tensor->ne[1];
        size_t off = ggml_backend_rknpu_get_tensor_offset(buffer, tensor);
        std::vector<uint8_t> pdata;

        if (tensor->type == GGML_TYPE_Q8_0) {
            float am = 0.0f;
#pragma omp parallel
            {
                std::vector<float> r(K);
#pragma omp for reduction(max:am)
                for (int n=0; n<N; n++) {
                    dequantize_row_q8_0((const block_q8_0*)data + n*(K/QK8_0), r.data(), K);
                    for (int k=0; k<K; k++) am = std::max(am, std::abs(r[k]));
                }
            }
            float sc = am / 127.0f;
            { std::lock_guard<std::mutex> l(bctx->mutex); bctx->quantized_tensor_scales[off] = sc; }
            pdata.resize((size_t)K * N);
#pragma omp parallel
            {
                std::vector<float> r(K);
#pragma omp for
                for (int n=0; n<N; n++) {
                    dequantize_row_q8_0((const block_q8_0*)data + n*(K/QK8_0), r.data(), K);
                    rknpu2_quantization::quantize_fp32_to_int8(r.data(), (int8_t*)pdata.data() + n*K, K, sc);
                }
            }
        } else if (tensor->type == GGML_TYPE_Q4_0 && op->npu_type_a == rknpu2_configuration::NPU_TYPE_INT4) {
            int Kop = rknpu2_calibration::next_power_of_two(K);
            std::vector<float> sv = rknpu2_calibration::generate_random_sign_vector(Kop);
            { std::lock_guard<std::mutex> l(bctx->mutex); bctx->hadamard_s_vectors[off] = sv; }
            std::vector<float> rotd((size_t)Kop * N);
#pragma omp parallel
            {
                std::vector<float> r(Kop, 0.0f);
#pragma omp for
                for (int n=0; n<N; n++) {
                    dequantize_row_q4_0((const block_q4_0*)data + n*(K/QK4_0), r.data(), K);
                    for (int k=0; k<Kop; k++) r[k] *= sv[k];
                    rknpu2_calibration::hadamard_transform(rotd.data() + n*Kop, r.data(), K, Kop);
                }
            }
            float am = 0.0f; for (float v : rotd) am = std::max(am, std::abs(v));
            float sc = am / 7.0f;
            { std::lock_guard<std::mutex> l(bctx->mutex); bctx->quantized_tensor_scales[off] = sc; }
            pdata.resize((rotd.size() + 1)/2);
#pragma omp parallel for
            for (int n=0; n<N; n++) rknpu2_quantization::quantize_fp32_to_int4_packed(rotd.data() + n*Kop, pdata.data() + n*Kop/2, Kop, sc);
        } else if (tensor->type != GGML_TYPE_F16 && tensor->type != GGML_TYPE_Q4_0) {
            pdata.resize((size_t)K * N * 2);
            uint16_t* d16 = (uint16_t*)pdata.data();
            ggml_to_float_t to_f = ggml_get_type_traits(tensor->type)->to_float;
#pragma omp parallel
            {
                std::vector<float> r(K);
#pragma omp for
                for (int n=0; n<N; n++) {
                    to_f((const uint8_t*)data + n*(ggml_nbytes(tensor)/N), r.data(), K);
                    for (int k=0; k<K; k++) d16[n*K + k] = ggml_fp32_to_fp16(r[k]);
                }
            }
        }

        const uint8_t* sptr = pdata.empty() ? (const uint8_t*)data : pdata.data();
        int Kpk = (tensor->type == GGML_TYPE_Q4_0 && op->npu_type_a == rknpu2_configuration::NPU_TYPE_INT4) ? rknpu2_calibration::next_power_of_two(K) : K;
        auto segs = compute_matrix_segments(N, cfg.core_count, op->n_align);
        uint8_t* wp = ptr;
        for (const auto& s : segs) {
            if (s.size_n > 0) {
                op->pack_func(wp, sptr, Kpk, N, s.offset_n, s.size_n);
                size_t psz = (op->mm_type == RKNN_INT8_MM_INT8_TO_INT32) ? (size_t)s.size_n * Kpk : ((op->mm_type == RKNN_INT4_MM_INT4_TO_INT16) ? (size_t)s.size_n * Kpk / 2 : (size_t)s.size_n * Kpk * 2);
                wp += psz;
            }
        }
        rknpu_sync_tensor(tensor, RKNN_MEMORY_SYNC_TO_DEVICE, 0, (size_t)(wp - ptr));
        return;
    }
    if (size > 0) {
        memcpy(ptr + offset, data, size);
        rknpu_sync_tensor(tensor, RKNN_MEMORY_SYNC_TO_DEVICE, offset, size);
    }
}

static void ggml_backend_rknpu_buffer_get_tensor(ggml_backend_buffer_t b, const struct ggml_tensor * t, void * d, size_t off, size_t sz) {
    if (sz > 0) {
        rknpu_sync_tensor(t, RKNN_MEMORY_SYNC_FROM_DEVICE, off, sz);
        memcpy(d, (uint8_t*)ggml_backend_rknpu_buffer_get_base(b) + ggml_backend_rknpu_get_tensor_offset(b, t) + off, sz);
    }
}

static void ggml_backend_rknpu_buffer_clear(ggml_backend_buffer_t b, uint8_t v) {
    auto* c = (ggml_backend_rknpu_buffer_context *)b->context;
    memset(c->dma_buf.virt_addr, v, c->dma_buf.size);
    rknn_tensor_mem sm = {};
    sm.virt_addr = c->dma_buf.virt_addr;
    sm.fd = c->dma_buf.fd;
    sm.size = (uint32_t)c->dma_buf.size;
    rknn_mem_sync(get_rknpu_memory_context().ctx, &sm, RKNN_MEMORY_SYNC_TO_DEVICE);
}

static void ggml_backend_rknpu_buffer_reset(ggml_backend_buffer_t buffer) { UNUSED(buffer); }

static ggml_backend_buffer_t ggml_backend_rknpu_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t sz) {
    rknpu2_allocation::DmaBuffer db = rknpu2_allocation::alloc(sz);
    if (db.fd < 0 || !db.virt_addr) return NULL;
    auto* ctx = new ggml_backend_rknpu_buffer_context();
    ctx->dma_buf = db;
    ctx->name = "rknpu_dma_buffer";
    static struct ggml_backend_buffer_i iface = {
        ggml_backend_rknpu_buffer_free_buffer,
        ggml_backend_rknpu_buffer_get_base,
        ggml_backend_rknpu_buffer_init_tensor,
        NULL,
        ggml_backend_rknpu_buffer_set_tensor,
        ggml_backend_rknpu_buffer_get_tensor,
        NULL,
        ggml_backend_rknpu_buffer_clear,
        ggml_backend_rknpu_buffer_reset,
    };
    return ggml_backend_buffer_init(buft, iface, ctx, sz);
}

static size_t ggml_backend_rknpu_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const struct ggml_tensor * t) {
    UNUSED(buft);
    if (t->type == GGML_TYPE_Q4_0) {
        const auto& cfg = rknpu2_configuration::Rknpu2ConfigManager::get_instance().get_current_config();
        const auto* op = cfg.find_op_support(t->type);
        if (op && op->npu_type_a == rknpu2_configuration::NPU_TYPE_INT4) {
            int Kop = rknpu2_calibration::next_power_of_two((int)t->ne[0]);
            auto segs = compute_matrix_segments((int)t->ne[1], cfg.core_count, op->n_align);
            size_t tot = 0;
            for (const auto& s : segs) tot += (size_t)s.size_n * Kop / 2;
            return tot;
        }
    }
    return ggml_nbytes(t);
}

static ggml_backend_t ggml_backend_rknpu_device_init(ggml_backend_dev_t dev, const char * params) {
    UNUSED(params);
    auto* c = new ggml_backend_rknpu_context();
    static struct ggml_backend_i iface = {
        [](ggml_backend_t b) { UNUSED(b); return "RKNPU"; },
        ggml_backend_rknpu_free,
        NULL, NULL, NULL,
        [](ggml_backend_t b) { UNUSED(b); },
        NULL, NULL, NULL, NULL,
        ggml_backend_rknpu_graph_compute,
        NULL, NULL, NULL
    };
    ggml_backend_t backend = new ggml_backend { {0}, iface, dev, c };
    return backend;
}

static ggml_backend_buffer_type_t ggml_backend_rknpu_device_get_buffer_type(ggml_backend_dev_t dev) {
    static struct ggml_backend_buffer_type_i iface = {
        [](ggml_backend_buffer_type_t buft) { UNUSED(buft); return "RKNPU"; },
        ggml_backend_rknpu_buffer_type_alloc_buffer,
        [](ggml_backend_buffer_type_t buft) { UNUSED(buft); return (size_t)64; },
        NULL,
        ggml_backend_rknpu_buffer_type_get_alloc_size,
        NULL
    };
    static struct ggml_backend_buffer_type buft = { iface, dev, NULL };
    return &buft;
}

static bool ggml_backend_rknpu_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    UNUSED(dev);
    const auto& cfg = rknpu2_configuration::Rknpu2ConfigManager::get_instance().get_current_config();
    switch (op->op) {
        case GGML_OP_NONE: case GGML_OP_RESHAPE: case GGML_OP_VIEW: case GGML_OP_PERMUTE: case GGML_OP_TRANSPOSE:
        case GGML_OP_SET_ROWS: case GGML_OP_GET_ROWS: case GGML_OP_CPY: case GGML_OP_DUP: case GGML_OP_CONT:
            return true;
        case GGML_OP_MUL_MAT:
            return cfg.find_op_support(op->src[0]->type) != nullptr;
        default:
            return false;
    }
}

static const struct ggml_backend_device_i rknpu_device_i = {
    [](ggml_backend_dev_t d) { UNUSED(d); return "RKNPU"; },
    [](ggml_backend_dev_t d) { UNUSED(d); return "Rockchip NPU"; },
    [](ggml_backend_dev_t d, size_t * f, size_t * t) { UNUSED(d); *f = 1024*1024*1024; *t = 1024*1024*1024; },
    [](ggml_backend_dev_t d) { UNUSED(d); return GGML_BACKEND_DEVICE_TYPE_ACCEL; },
    [](ggml_backend_dev_t d, struct ggml_backend_dev_props * p) { UNUSED(d); p->name = "RKNPU"; p->description = "Rockchip NPU"; p->type = GGML_BACKEND_DEVICE_TYPE_ACCEL; },
    ggml_backend_rknpu_device_init,
    ggml_backend_rknpu_device_get_buffer_type,
    NULL, NULL,
    ggml_backend_rknpu_device_supports_op,
    [](ggml_backend_dev_t d, ggml_backend_buffer_type_t b) { UNUSED(d); return b->iface.alloc_buffer == ggml_backend_rknpu_buffer_type_alloc_buffer; },
    NULL, NULL, NULL, NULL
};

static struct ggml_backend_device rknpu_device = { rknpu_device_i, NULL, NULL };

extern "C" {
    GGML_BACKEND_API ggml_backend_reg_t ggml_backend_rknpu2_reg(void) {
        static struct ggml_backend_reg_i iface = {
            [](ggml_backend_reg_t r) { UNUSED(r); return "RKNPU"; },
            [](ggml_backend_reg_t r) { UNUSED(r); return (size_t)1; },
            [](ggml_backend_reg_t r, size_t i) {
                GGML_ASSERT(i == 0);
                rknpu_device.reg = r;
                return &rknpu_device;
            },
            NULL
        };
        static struct ggml_backend_reg reg = { GGML_BACKEND_API_VERSION, iface, NULL };
        return &reg;
    }
}
