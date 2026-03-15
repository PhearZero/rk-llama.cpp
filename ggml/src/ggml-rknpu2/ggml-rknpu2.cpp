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
#include <map>
#include <unordered_set>
#include <tuple>
#include <algorithm>
#include <memory>
#include <unordered_map>
#include <random>
#include <limits>
#include <fstream>
#include <sys/sysinfo.h>
#include <cstdio>

#define UNUSED(x) (void)(x)

// Macro for RKNN API calls
#define RKNN_CHECK(stmt, msg)                                           \
    do {                                                                \
        int ret = (stmt);                                               \
        if (ret < 0) {                                                  \
            fprintf(stderr,"RKNN error %d at %s:%d: %s\n", ret,                \
                __FILE__, __LINE__, msg);                               \
            assert(false);                                              \
        }                                                               \
    } while (0)

// --- Hashers ---

// Function for hash combinations
template <class T>
inline void hash_combine(std::size_t& seed, const T& v) {
    std::hash<T> hasher;
    seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

// Hasher for std::pair
struct PairHasher {
    template <class T1, class T2>
    std::size_t operator()(const std::pair<T1, T2>& p) const {
        std::size_t seed = 0;
        hash_combine(seed, p.first);
        hash_combine(seed, p.second);
        return seed;
    }
};

// Hasher for std::tuple
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

// Matrix segment information
struct MatrixSegment {
    int offset_n;  // Segment offset
    int size_n;    // Segment size
    int core_id;   // Segment core ID
};

// Split B-matrix into segments
static std::vector<MatrixSegment> compute_matrix_segments(int N, int num_cores, int alignment) {
    std::vector<MatrixSegment> segments;

    int base_segment_size = (N / num_cores / alignment) * alignment;
    int remaining = N - (base_segment_size * num_cores);
    
    int offset = 0;
    for (int i = 0; i < num_cores; i++) {
        MatrixSegment seg;
        seg.offset_n = offset;
        seg.size_n = base_segment_size;
        seg.core_id = i;
        
        if (i < remaining / alignment) {
            seg.size_n += alignment;
        }
        
        offset += seg.size_n;
        segments.push_back(seg);
    }
    
    return segments;
}

// --- Structs ---

// --- Structs ---

// RKNN memory global context
struct rknpu_memory_context {
    rknn_matmul_ctx mem_ctx = 0;
    std::mutex mutex;

    rknpu_memory_context() {
        rknn_matmul_info dummy_info;
        memset(&dummy_info, 0, sizeof(dummy_info));
        dummy_info.M = 32;
        dummy_info.K = 32;
        dummy_info.N = 32;
        dummy_info.type = RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32;

        rknn_matmul_io_attr dummy_io_attr;
        // fprintf(stderr, "RKNPU_INIT: Calling rknn_matmul_create...\n");
        int ret = rknn_matmul_create(&mem_ctx, &dummy_info, &dummy_io_attr);
        if (ret < 0) {
            fprintf(stderr, "RKNPU_INIT: Failed to create dummy matmul context for memory management! error=%d\n", ret);
            mem_ctx = 0;
        }
    }

    ~rknpu_memory_context() {
        if (mem_ctx != 0) {
            rknn_matmul_destroy(mem_ctx);
        }
    }

    rknn_matmul_ctx get_ctx() {
        return mem_ctx;
    }
};

static rknpu_memory_context & get_rknpu_memory_context() {
    static rknpu_memory_context instance;
    return instance;
}

// RKNN buffer context
struct ggml_backend_rknpu_buffer_context {
    rknpu2_allocation::DmaBuffer dma_buf;
    std::string name;

    // Use offset in the DMA buffer as the key (instead of pointer) to support RPC.
    // Maps: offset -> data.
    std::unordered_map<size_t, float> quantized_tensor_scales;
    std::unordered_map<size_t, std::vector<float>> hadamard_s_vectors;
    
    // Cache for packed weight buffers (offset -> rknn_tensor_mem).
    // These buffers are allocated via rknn_create_mem and managed by shared_ptr.
    struct rknn_tensor_mem_deleter {
        void operator()(rknn_tensor_mem* mem) {
            if (mem) {
                // RKNN view handles created via rknn_create_mem_from_fd 
                // must also be destroyed, but we use a single context for all memory operations.
                auto mem_ctx = get_rknpu_memory_context().get_ctx();
                if (mem_ctx != 0) {
                    rknn_destroy_mem(mem_ctx, mem);
                }
            }
        }
    };
    std::unordered_map<size_t, std::shared_ptr<rknn_tensor_mem>> packed_weight_cache;

    std::mutex mutex;
};

// RKNN matmul operation context
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
            fprintf(stderr, "RKNPU2: Failed to create matmul context! M=%d, K=%d, N=%d, type=%d, error=%d\n", M, K, N, (int)type, ret);
            ctx = 0;
        }
    }

    ~rknpu_matmul_context() {
        if (ctx != 0) {
            rknn_matmul_destroy(ctx);
        }
    }
};


// Backend main context
struct ggml_backend_rknpu_context {
    std::string name;
    std::mutex mutex;

    // RKNN matmul contexts cache (M_supported, K, N, core_id, type)
    std::unordered_map<std::tuple<int, int, int, int, int>, std::shared_ptr<rknpu_matmul_context>, TupleHasher> matmul_ctx_cache;

    // B-matrices handle cache (from fd)
    std::unordered_map<std::pair<ggml_backend_buffer_t, size_t>, std::shared_ptr<rknn_tensor_mem>, PairHasher> b_mem_handle_cache;

    // A- and C-matrices cache (M, K, type) or (M, size_n, core_id)
    std::unordered_map<std::tuple<int, int, int>, std::shared_ptr<rknn_tensor_mem>, TupleHasher> a_buffer_cache;
    std::unordered_map<std::tuple<int, int, int>, std::shared_ptr<rknn_tensor_mem>, TupleHasher> c_buffer_cache;

    std::shared_ptr<rknpu_matmul_context> get_matmul_ctx(int M, int K, int N, int core_id, rknn_matmul_type type) {
        std::lock_guard<std::mutex> lock(mutex);
        auto key = std::make_tuple(M, K, N, core_id, (int)type);
        auto it = matmul_ctx_cache.find(key);
        if (it != matmul_ctx_cache.end()) {
            return it->second;
        }
        // fprintf(stderr, "RKNPU2: Creating new matmul context M=%d, K=%d, N=%d, type=%d, core=%d, cache_size=%zu\n", M, K, N, (int)type, core_id, matmul_ctx_cache.size());
        auto ctx = std::make_shared<rknpu_matmul_context>(M, K, N, type);
        if (ctx->ctx == 0) {
            return nullptr;
        }

        rknn_core_mask core_mask;
        switch(core_id) {
            case 0: core_mask = RKNN_NPU_CORE_0; break;
            case 1: core_mask = RKNN_NPU_CORE_1; break;
            case 2: core_mask = RKNN_NPU_CORE_2; break;
            default: core_mask = RKNN_NPU_CORE_AUTO; break;
        }

        int ret = rknn_matmul_set_core_mask(ctx->ctx, core_mask);
        if (ret < 0) {
            fprintf(stderr, "RKNPU2: Failed to set core mask! error=%d\n", ret);
        }
        
        matmul_ctx_cache[key] = ctx;
        return ctx;
    }
};

//
// Backend
//

static const char * ggml_backend_rknpu_name(ggml_backend_t backend) {
    UNUSED(backend);
    return "RKNPU";
}

static void ggml_backend_rknpu_free(ggml_backend_t backend) {
    ggml_backend_rknpu_context * ctx = (ggml_backend_rknpu_context *)backend->context;
    delete ctx;
    delete backend;
}

// Function for getting buffer from cache or creating new one
static std::shared_ptr<rknn_tensor_mem> get_or_create_npu_buffer(
    ggml_backend_rknpu_context* backend_ctx,
    rknn_matmul_ctx matmul_ctx,
    size_t size,
    const std::tuple<int, int, int>& key,
    std::unordered_map<std::tuple<int, int, int>, std::shared_ptr<rknn_tensor_mem>, TupleHasher>& cache
) {
    (void)matmul_ctx;
    std::lock_guard<std::mutex> lock(backend_ctx->mutex);
    auto it = cache.find(key);
    if (it != cache.end()) {
        return it->second;
    }

    if (size == 0) return nullptr;
    rknn_tensor_mem* mem = rknn_create_mem(get_rknpu_memory_context().get_ctx(), size);
    if (!mem) { return nullptr; }

    auto mem_ctx_for_deleter = get_rknpu_memory_context().get_ctx();
    auto deleter = [mem_ctx_for_deleter](rknn_tensor_mem* m) {
        if (m && mem_ctx_for_deleter != 0) {
            rknn_destroy_mem(mem_ctx_for_deleter, m);
        }
    };

    std::shared_ptr<rknn_tensor_mem> mem_shared(mem, deleter);
    cache[key] = mem_shared;
    return mem_shared;
}

static void * ggml_backend_rknpu_buffer_get_base(ggml_backend_buffer_t buffer);

static void* ggml_rknpu_get_system_ptr(const struct ggml_tensor* tensor) {
    if (tensor && tensor->buffer && tensor->buffer->iface.get_base == ggml_backend_rknpu_buffer_get_base) {
        void* base = ggml_backend_rknpu_buffer_get_base(tensor->buffer);
        if (base && tensor->data) {
            return (void*)((uintptr_t)base + (uintptr_t)tensor->data - (uintptr_t)base);
        }
    }
    return tensor ? tensor->data : nullptr;
}

static size_t ggml_backend_rknpu_get_tensor_offset(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor) {
    if (!buffer || !tensor || !tensor->data) return 0;
    void* base_addr = ggml_backend_rknpu_buffer_get_base(buffer);
    if (!base_addr) return 0;
    return (uint8_t*)tensor->data - (uint8_t*)base_addr;
}


static void rknpu_sync_tensor(const struct ggml_tensor * tensor, rknn_mem_sync_mode type, size_t offset, size_t size) {
    if (!tensor || !tensor->data || !tensor->buffer || !tensor->buffer->context) return;
    if (size == 0) return;
    
    // Check if buffer is RKNPU
    const char * name = ggml_backend_buffer_name(tensor->buffer);
    if (strcmp(name, "RKNPU") != 0) return;

    ggml_backend_rknpu_buffer_context * ctx = (ggml_backend_rknpu_buffer_context *)tensor->buffer->context;
    if (!ctx->dma_buf.virt_addr) return;
    
    size_t tensor_offset = ggml_backend_rknpu_get_tensor_offset(tensor->buffer, tensor);
    size_t total_offset = tensor_offset + offset;

    // Use virt_addr = base + offset and offset = 0 which is often safer for driver consistency
    rknn_tensor_mem sync_mem = {};
    sync_mem.virt_addr = (uint8_t*)ctx->dma_buf.virt_addr + total_offset;
    sync_mem.fd = ctx->dma_buf.fd;
    sync_mem.offset = 0; 
    sync_mem.size = (uint32_t)size;
    
    auto mem_ctx = get_rknpu_memory_context().get_ctx();
    if (mem_ctx != 0) {
        // Memory barrier before sync
        std::atomic_thread_fence(std::memory_order_seq_cst);
        
        int ret = rknn_mem_sync(mem_ctx, &sync_mem, type);
        
        // Memory barrier after sync
        std::atomic_thread_fence(std::memory_order_seq_cst);

        if (ret != 0) {
            fprintf(stderr, "RKNPU2: rknpu_sync_tensor FAILED ret=%d, node=%s, type=%d, fd=%d, addr=%p, size=%u\n",
                ret, tensor->name, (int)type, sync_mem.fd, sync_mem.virt_addr, sync_mem.size);
            fflush(stderr);
        }
    }
}

static void rknpu_compute_forward_set_rows(struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0]; // new rows
    const struct ggml_tensor * src1 = dst->src[1]; // indices
    const struct ggml_tensor * src2 = dst->src[2]; // original cache (dst is a view of this)

    if (src0->data == NULL || src1->data == NULL || src2->data == NULL || dst->data == NULL) {
        fprintf(stderr, "RKNPU2: set_rows NULL DATA: src0=%p, src1=%p, src2=%p, dst=%p\n", src0->data, src1->data, src2->data, dst->data);
        return;
    }

    rknpu_sync_tensor(src0, RKNN_MEMORY_SYNC_FROM_DEVICE, 0, ggml_nbytes(src0));
    rknpu_sync_tensor(src1, RKNN_MEMORY_SYNC_FROM_DEVICE, 0, ggml_nbytes(src1));
    rknpu_sync_tensor(src2, RKNN_MEMORY_SYNC_FROM_DEVICE, 0, ggml_nbytes(src2));

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];

    const size_t nb01 = src0->nb[1];
    const size_t nb02 = src0->nb[2];
    const size_t nb03 = src0->nb[3];

    const size_t nb10 = src1->nb[0];
    const size_t nb11 = src1->nb[1];
    const size_t nb12 = src1->nb[2];

    const int64_t ne1 = dst->ne[1];
    const size_t nb1 = dst->nb[1];
    const size_t nb2 = dst->nb[2];
    const size_t nb3 = dst->nb[3];

    ggml_from_float_t const from_float = ggml_get_type_traits_cpu(dst->type)->from_float;
    GGML_ASSERT(from_float != NULL);

    for (int64_t i03 = 0; i03 < ne03; ++i03) {
        for (int64_t i02 = 0; i02 < ne02; ++i02) {
            for (int64_t i = 0; i < ne01; ++i) {
                const int64_t i12 = i03 % src1->ne[2];
                const int64_t i11 = i02 % src1->ne[1];
                const int64_t i10 = i;

                int64_t i1;
                if (src1->type == GGML_TYPE_I32) {
                    i1 = *(int32_t *) ((char *) src1->data + i10*nb10 + i11*nb11 + i12*nb12);
                } else if (src1->type == GGML_TYPE_I64) {
                    i1 = *(int64_t *) ((char *) src1->data + i10*nb10 + i11*nb11 + i12*nb12);
                } else {
                    fprintf(stderr, "RKNPU2: set_rows INVALID src1 type=%d, name=%s\n", src1->type, src1->name);
                    GGML_ABORT("unsupported src1 type");
                }

                if (i1 < 0 || i1 >= ne1) {
                    fprintf(stderr, "RKNPU2: set_rows index OUT OF BOUNDS: i=%ld, i1=%ld, ne1=%ld, src1_type=%d, src1_name=%s\n",
                           i, i1, ne1, src1->type, src1->name);
                    if (i == 0 && src1->data) {
                        const int32_t * p = (const int32_t *)src1->data;
                        int n_dump = std::min((int64_t)4, ggml_nelements(src1));
                        fprintf(stderr, "RKNPU2: set_rows src1[0..%d] = [", n_dump - 1);
                        for (int j = 0; j < n_dump; ++j) {
                            fprintf(stderr, " %d%s", p[j], (j < n_dump - 1 ? "," : ""));
                        }
                        fprintf(stderr, " ]\n");
                    }
                    fflush(stderr);
                }
                GGML_ASSERT(i1 >= 0 && i1 < ne1);

                from_float(
                        (const float *) ((char *) src0->data + i*nb01 + i02*nb02 + i03*nb03),
                                        ((char *)  dst->data + i1*nb1  + i02*nb2  + i03*nb3), ne00);
            }
        }
    }

    rknpu_sync_tensor(dst, RKNN_MEMORY_SYNC_TO_DEVICE, 0, ggml_nbytes(dst));
}

static void rknpu_compute_forward_get_rows(struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    if (src0->data == NULL || src1->data == NULL || dst->data == NULL) {
        fprintf(stderr, "RKNPU2: get_rows NULL DATA: src0=%p, src1=%p, dst=%p\n", src0->data, src1->data, dst->data);
        return;
    }

    rknpu_sync_tensor(src0, RKNN_MEMORY_SYNC_FROM_DEVICE, 0, ggml_nbytes(src0));
    rknpu_sync_tensor(src1, RKNN_MEMORY_SYNC_FROM_DEVICE, 0, ggml_nbytes(src1));

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const size_t nb01 = src0->nb[1];
    const size_t nb02 = src0->nb[2];
    const size_t nb03 = src0->nb[3];

    const size_t nb10 = src1->nb[0];
    const size_t nb11 = src1->nb[1];
    const size_t nb12 = src1->nb[2];

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];

    const size_t nb1 = dst->nb[1];
    const size_t nb2 = dst->nb[2];
    const size_t nb3 = dst->nb[3];

    ggml_to_float_t   const to_float   = (src0->type == GGML_TYPE_F32) ? (ggml_to_float_t)ggml_cpu_fp32_to_fp32 : ggml_get_type_traits(src0->type)->to_float;
    ggml_from_float_t const from_float = ggml_get_type_traits_cpu(dst->type)->from_float;
    GGML_ASSERT(to_float != NULL);
    GGML_ASSERT(from_float != NULL);

    std::vector<float> row_f32(ne00);
    for (int64_t i = 0; i < ggml_nelements(src1); ++i) {
        const int64_t i12 = i / (ne11*ne10);
        const int64_t i11 = (i - i12*ne11*ne10) / ne10;
        const int64_t i10 = (i - i12*ne11*ne10 - i11*ne10);

        int64_t i01;
        if (src1->type == GGML_TYPE_I32) {
            i01 = *(int32_t *) ((char *) src1->data + i10*nb10 + i11*nb11 + i12*nb12);
        } else if (src1->type == GGML_TYPE_I64) {
            i01 = *(int64_t *) ((char *) src1->data + i10*nb10 + i11*nb11 + i12*nb12);
        } else {
            fprintf(stderr, "RKNPU2: get_rows INVALID src1 type=%d, name=%s\n", src1->type, src1->name);
            GGML_ABORT("unsupported src1 type");
        }

        if (i01 < 0 || i01 >= ne01) {
            fprintf(stderr, "RKNPU2: get_rows index OUT OF BOUNDS: i=%ld, i01=%ld, ne01=%ld, src1_type=%d, src1_name=%s, src0_name=%s\n",
                   i, i01, ne01, src1->type, src1->name, src0->name);
            if (i == 0 && src1->data) {
                const uint8_t * p = (const uint8_t *)src1->data;
                int n_dump = std::min((size_t)16, ggml_nbytes(src1));
                fprintf(stderr, "RKNPU2: get_rows src1[0..%d] (raw hex) = [", n_dump - 1);
                for (int j = 0; j < n_dump; ++j) {
                    fprintf(stderr, " %02x%s", p[j], (j < n_dump - 1 ? "," : ""));
                }
                fprintf(stderr, " ]\n");
                
                if (src1->type == GGML_TYPE_I32) {
                    const int32_t * pi = (const int32_t *)src1->data;
                    int ni_dump = std::min((int64_t)4, ggml_nelements(src1));
                    fprintf(stderr, "RKNPU2: get_rows src1[0..%d] (int32) = [", ni_dump - 1);
                    for (int j = 0; j < ni_dump; ++j) {
                        fprintf(stderr, " %d%s", pi[j], (j < ni_dump - 1 ? "," : ""));
                    }
                    fprintf(stderr, " ]\n");
                }
            }
            fflush(stderr);
        }
        GGML_ASSERT(i01 >= 0 && i01 < ne01);

        to_float((const void *) ((char *) src0->data + i01*nb01 + i11*nb02 + i12*nb03), row_f32.data(), ne00);
        from_float(row_f32.data(), (void *) ((char *)  dst->data + i10*nb1  + i11*nb2  + i12*nb3), ne00);
    }

    rknpu_sync_tensor(dst, RKNN_MEMORY_SYNC_TO_DEVICE, 0, ggml_nbytes(dst));
}

static void rknpu_compute_forward_cpy(struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];

    if (src0->data == NULL || dst->data == NULL) {
        fprintf(stderr, "RKNPU2: cpy NULL DATA: src0=%p, dst=%p\n", src0->data, dst->data);
        return;
    }

    rknpu_sync_tensor(src0, RKNN_MEMORY_SYNC_FROM_DEVICE, 0, ggml_nbytes(src0));
    rknpu_sync_tensor(dst, RKNN_MEMORY_SYNC_FROM_DEVICE, 0, ggml_nbytes(dst));

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];

    const size_t nb01 = src0->nb[1];
    const size_t nb02 = src0->nb[2];
    const size_t nb03 = src0->nb[3];

    const size_t nb1 = dst->nb[1];
    const size_t nb2 = dst->nb[2];
    const size_t nb3 = dst->nb[3];

    ggml_to_float_t   const to_float   = (src0->type == GGML_TYPE_F32) ? (ggml_to_float_t)ggml_cpu_fp32_to_fp32 : ggml_get_type_traits(src0->type)->to_float;
    ggml_from_float_t const from_float = ggml_get_type_traits_cpu(dst->type)->from_float;
    GGML_ASSERT(to_float != NULL);
    GGML_ASSERT(from_float != NULL);

    std::vector<float> row_f32(ne00);
    for (int64_t i03 = 0; i03 < ne03; i03++) {
        for (int64_t i02 = 0; i02 < ne02; i02++) {
            for (int64_t i01 = 0; i01 < ne01; i01++) {
                to_float((char *)src0->data + i01*nb01 + i02*nb02 + i03*nb03, row_f32.data(), ne00);
                from_float(row_f32.data(), (char *)dst->data + i01*nb1 + i02*nb2 + i03*nb3, ne00);
            }
        }
    }

    rknpu_sync_tensor(dst, RKNN_MEMORY_SYNC_TO_DEVICE, 0, ggml_nbytes(dst));
}

static enum ggml_status ggml_backend_rknpu_graph_compute(ggml_backend_t backend, struct ggml_cgraph* cgraph) {
    auto* backend_ctx = (ggml_backend_rknpu_context*)backend->context;

    // Getting the current device configuration once
    const auto& config = rknpu2_configuration::Rknpu2ConfigManager::get_instance().get_current_config();

    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor* node = cgraph->nodes[i];
        if (!node) continue;
        
        if (node->op == GGML_OP_MUL_MAT) {
            // ... MUL_MAT implementation follows ...
        } else if (node->op == GGML_OP_SET_ROWS) {
            rknpu_compute_forward_set_rows(node);
            continue;
        } else if (node->op == GGML_OP_GET_ROWS) {
            rknpu_compute_forward_get_rows(node);
            continue;
        } else if (node->op == GGML_OP_CPY || node->op == GGML_OP_DUP || node->op == GGML_OP_CONT) {
            rknpu_compute_forward_cpy(node);
            continue;
        } else if (ggml_op_is_empty(node->op)) {
            continue;
        } else {
            // Check if we should ignore or fail
            if (node->op == GGML_OP_NONE) continue;
            continue; // Skip unknown nodes for now, hopefully supports_op caught them
        }

        const struct ggml_tensor* src0 = node->src[0]; // Weights      :  (K x N)
        const struct ggml_tensor* src1 = node->src[1]; // Activations  :  (M x K)
        struct ggml_tensor* dst = node;

        if (!src0 || !src1) continue;

        const ggml_type w_type = src0->type;

        const int M = (int)src1->ne[1];
        const int K = (int)src0->ne[0];
        const int N = (int)src0->ne[1];

        // M rounding for driver stability and context reuse
        // Bucket 1: Generation (M=1)
        // Bucket 2: Small Prefill (M <= 8, use 8 for stability)
        // Bucket 3: Medium Prefill (M <= 512, use 512 for stability)
        const int M_op = (M == 1) ? 1 : (M <= 32 ? (M <= 8 ? 8 : 32) : (M <= 512 ? 512 : ((M + 7) / 8 * 8)));

        const bool is_q4_hadamard = (src0->type == GGML_TYPE_Q4_0);
        const int K_op = is_q4_hadamard ? rknpu2_calibration::next_power_of_two(K) : K;

        const auto* op_support = config.find_op_support(src0->type);
        if (!op_support) {
            fprintf(stderr, "RKNPU2: node %s has unsupported weight type %s\n", node->name, ggml_type_name(src0->type));
            fflush(stderr);
            return GGML_STATUS_FAILED;
        }

        if (src1->type != op_support->type_a) {
            fprintf(stderr, "RKNPU2: node %s has unsupported activation type %s (expected %s)\n", node->name, ggml_type_name(src1->type), ggml_type_name(op_support->type_a));
            fflush(stderr);
            return GGML_STATUS_FAILED;
        }

        if (src0->ne[0] % op_support->k_align != 0) {
            fprintf(stderr, "RKNPU2: node %s has unaligned K=%d (alignment required: %d)\n", node->name, (int)src0->ne[0], op_support->k_align);
            fflush(stderr);
            return GGML_STATUS_FAILED;
        }

        if (src0->ne[1] % op_support->n_align != 0) {
            fprintf(stderr, "RKNPU2: node %s has unaligned N=%d (alignment required: %d)\n", node->name, (int)src0->ne[1], op_support->n_align);
            fflush(stderr);
            return GGML_STATUS_FAILED;
        }

        if (src1->ne[0] != src0->ne[0]) {
            fprintf(stderr, "RKNPU2: node %s has mismatched K: src0 K=%d, src1 K=%d\n", node->name, (int)src0->ne[0], (int)src1->ne[0]);
            fflush(stderr);
            return GGML_STATUS_FAILED;
        }

        const rknn_matmul_type matmul_type = op_support->mm_type;
        const int alignment = op_support->n_align;

        auto all_segments = compute_matrix_segments(N, config.core_count, alignment);

        std::vector<MatrixSegment> active_segments;
        for (const auto& seg : all_segments) {
            if (seg.size_n > 0) {
                active_segments.push_back(seg);
            }
        }

        if (active_segments.empty()) continue;

        const size_t num_active_segments = active_segments.size();
        std::vector<std::shared_ptr<rknpu_matmul_context>> matmul_ctxs(num_active_segments);
        std::vector<rknn_matmul_io_attr> segments_io_attrs(num_active_segments);
        std::vector<std::shared_ptr<rknn_tensor_mem>> mem_B_segments(num_active_segments);
        std::shared_ptr<rknn_tensor_mem> mem_A_shared;
        std::vector<std::shared_ptr<rknn_tensor_mem>> mem_C_segments(num_active_segments);

        // ===========================================
        // ========== 1. Preparing contexts ==========
        // ===========================================
        {
            for (size_t i = 0; i < num_active_segments; ++i) {
                const auto& seg = active_segments[i];
                matmul_ctxs[i] = backend_ctx->get_matmul_ctx(M_op, K_op, seg.size_n, seg.core_id, matmul_type);
                if (!matmul_ctxs[i] || matmul_ctxs[i]->ctx == 0) return GGML_STATUS_FAILED;
                segments_io_attrs[i] = matmul_ctxs[i]->io_attr;
            }
        }

        // ===========================================
        // ========== 2. Preparing B-matrix ==========
        // ===========================================
        {
            ggml_backend_buffer_t src0_buffer = src0->buffer;
            if (!src0_buffer || !src0_buffer->context) {
                return GGML_STATUS_FAILED;
            }
            auto* src0_buf_ctx = (ggml_backend_rknpu_buffer_context*)src0_buffer->context;
            
            // Resolve the base tensor and its offset to use as a cache key
            size_t tensor_offset = ggml_backend_rknpu_get_tensor_offset(src0_buffer, src0);
            
            std::shared_ptr<rknn_tensor_mem> packed_mem;
            {
                std::lock_guard<std::mutex> lock(src0_buf_ctx->mutex);
                auto it = src0_buf_ctx->packed_weight_cache.find(tensor_offset);
                if (it != src0_buf_ctx->packed_weight_cache.end()) {
                    packed_mem = it->second;
                }
            }

            if (!packed_mem) {
                // Not in cache, need to pack it now
                size_t total_packed_size = 0;
                for (const auto& seg : active_segments) {
                    // Find the IO attr for this segment to get the required packed size
                    // We can use any matmul_ctx that has the same K_op and segment size
                    auto temp_matmul_ctx = backend_ctx->get_matmul_ctx(M_op, K_op, seg.size_n, seg.core_id, matmul_type);
                    if (!temp_matmul_ctx) {
                        fprintf(stderr, "RKNPU2: Failed to get temporary matmul context for weight packing!\n");
                        fflush(stderr);
                        return GGML_STATUS_FAILED;
                    }
                    total_packed_size += temp_matmul_ctx->io_attr.B.size;
                }

                auto mem_ctx = get_rknpu_memory_context().get_ctx();
                if (mem_ctx == 0) {
                    fprintf(stderr, "RKNPU2: Failed to get global memory context for weight packing!\n");
                    fflush(stderr);
                    return GGML_STATUS_FAILED;
                }
                rknn_tensor_mem* mem_ptr = rknn_create_mem(mem_ctx, total_packed_size);
                if (!mem_ptr) {
                    fprintf(stderr, "RKNPU2: Failed to allocate packed weight buffer of size %zu\n", total_packed_size);
                    return GGML_STATUS_FAILED;
                }
                packed_mem = std::shared_ptr<rknn_tensor_mem>(mem_ptr, ggml_backend_rknpu_buffer_context::rknn_tensor_mem_deleter{});

                // Pack the data from the original DMA buffer into the new NPU buffer
                rknpu_sync_tensor(src0, RKNN_MEMORY_SYNC_FROM_DEVICE, 0, ggml_nbytes(src0));
                uint8_t* src_ptr = (uint8_t*)src0_buf_ctx->dma_buf.virt_addr + tensor_offset;
                uint8_t* dst_ptr = (uint8_t*)packed_mem->virt_addr;

                const int K_orig = (int)src0->ne[0];
                const int N_orig = (int)src0->ne[1];

                if (src0->type == GGML_TYPE_F16) {
                    for (const auto& seg : active_segments) {
                        size_t segment_packed_size = (size_t)seg.size_n * K_orig * sizeof(uint16_t);
                        op_support->pack_func(dst_ptr, src_ptr, K_orig, N_orig, seg.offset_n, seg.size_n);
                        dst_ptr += segment_packed_size;
                    }
                } else if (src0->type == GGML_TYPE_Q8_0) {
                    float scale_B_val;
                    {
                        std::lock_guard<std::mutex> lock(src0_buf_ctx->mutex);
                        scale_B_val = src0_buf_ctx->quantized_tensor_scales[tensor_offset];
                    }
                    
                    std::vector<int8_t> requantized_data((size_t)K_orig * N_orig);
                    #pragma omp parallel
                    {
                        std::vector<float> tmp_row(K_orig);
                        #pragma omp for
                        for (int n = 0; n < N_orig; ++n) {
                            dequantize_row_q8_0((const block_q8_0*)src_ptr + (size_t)n * (K_orig / QK8_0), tmp_row.data(), K_orig);
                            rknpu2_quantization::quantize_fp32_to_int8(tmp_row.data(), requantized_data.data() + (size_t)n * K_orig, K_orig, scale_B_val);
                        }
                    }

                    for (const auto& seg : active_segments) {
                        size_t segment_packed_size = (size_t)seg.size_n * K_orig;
                        op_support->pack_func(dst_ptr, (const uint8_t*)requantized_data.data(), K_orig, N_orig, seg.offset_n, seg.size_n);
                        dst_ptr += segment_packed_size;
                    }
                } else if (src0->type == GGML_TYPE_Q4_0) {
                    if (op_support->npu_type_a == rknpu2_configuration::NPU_TYPE_INT4) {
                        float scale_B_val;
                        std::vector<float> s_vec;
                        {
                            std::lock_guard<std::mutex> lock(src0_buf_ctx->mutex);
                            scale_B_val = src0_buf_ctx->quantized_tensor_scales[tensor_offset];
                            s_vec = src0_buf_ctx->hadamard_s_vectors[tensor_offset];
                        }

                        std::vector<float> rotated_data((size_t)K_op * N_orig);
                        #pragma omp parallel
                        {
                            std::vector<float> tmp_row(K_op, 0.0f);
                            #pragma omp for
                            for (int n = 0; n < N_orig; ++n) {
                                dequantize_row_q4_0((const block_q4_0*)src_ptr + (size_t)n * (K_orig / QK4_0), tmp_row.data(), K_orig);
                                for (int k = 0; k < K_op; ++k) tmp_row[k] *= s_vec[k];
                                rknpu2_calibration::hadamard_transform(rotated_data.data() + (size_t)n * K_op, tmp_row.data(), K_orig, K_op);
                            }
                        }

                        std::vector<uint8_t> requantized_data((size_t)K_op * N_orig / 2);
                        #pragma omp parallel for
                        for (int n = 0; n < N_orig; ++n) {
                            rknpu2_quantization::quantize_fp32_to_int4_packed(rotated_data.data() + (size_t)n * K_op, requantized_data.data() + (size_t)n * (K_op / 2), K_op, scale_B_val);
                        }

                        for (const auto& seg : active_segments) {
                            size_t segment_packed_size = (size_t)seg.size_n * K_op / 2;
                            op_support->pack_func(dst_ptr, (const uint8_t*)requantized_data.data(), K_op, N_orig, seg.offset_n, seg.size_n);
                            dst_ptr += segment_packed_size;
                        }
                    } else {
                        // Q4_0 fallback
                        float scale_B_val;
                        {
                            std::lock_guard<std::mutex> lock(src0_buf_ctx->mutex);
                            scale_B_val = src0_buf_ctx->quantized_tensor_scales[tensor_offset];
                        }

                        std::vector<uint8_t> requantized_data((size_t)K_orig * N_orig / 2);
                        #pragma omp parallel
                        {
                            std::vector<float> tmp_row(K_orig);
                            #pragma omp for
                            for (int n = 0; n < N_orig; ++n) {
                                dequantize_row_q4_0((const block_q4_0*)src_ptr + (size_t)n * (K_orig / QK4_0), tmp_row.data(), K_orig);
                                rknpu2_quantization::quantize_fp32_to_int4_packed(tmp_row.data(), requantized_data.data() + (size_t)n * (K_orig / 2), K_orig, scale_B_val);
                            }
                        }

                        for (const auto& seg : active_segments) {
                            size_t segment_packed_size = (size_t)seg.size_n * K_orig / 2;
                            op_support->pack_func(dst_ptr, (const uint8_t*)requantized_data.data(), K_orig, N_orig, seg.offset_n, seg.size_n);
                            dst_ptr += segment_packed_size;
                        }
                    }
                } else {
                    // Generic fallback for other supported types (e.g. Q5_0, Q4_K, etc.)
                    // These types have a pack_func that handles the original source data directly.
                    for (size_t i = 0; i < num_active_segments; ++i) {
                        const auto& seg = active_segments[i];
                        size_t segment_packed_size = segments_io_attrs[i].B.size;
                        op_support->pack_func(dst_ptr, src_ptr, K_orig, N_orig, seg.offset_n, seg.size_n);
                        dst_ptr += segment_packed_size;
                    }
                }

                // Sync the packed weights to the NPU device
                rknn_mem_sync(get_rknpu_memory_context().get_ctx(), packed_mem.get(), RKNN_MEMORY_SYNC_TO_DEVICE);

                {
                    std::lock_guard<std::mutex> lock(src0_buf_ctx->mutex);
                    src0_buf_ctx->packed_weight_cache[tensor_offset] = packed_mem;
                }
            }

            // Map the packed segments to the matmul contexts
            size_t current_offset_in_packed_buffer = 0;
            for (size_t i = 0; i < num_active_segments; ++i) {
                auto& matmul_ctx = matmul_ctxs[i];
                size_t segment_size_bytes = segments_io_attrs[i].B.size;

                // Create a sub-handle for this segment
                rknn_tensor_mem* seg_mem_ptr = rknn_create_mem_from_fd(get_rknpu_memory_context().get_ctx(), packed_mem->fd, packed_mem->virt_addr, segment_size_bytes, current_offset_in_packed_buffer);
                if (!seg_mem_ptr) return GGML_STATUS_FAILED;
                
                mem_B_segments[i] = std::shared_ptr<rknn_tensor_mem>(seg_mem_ptr, ggml_backend_rknpu_buffer_context::rknn_tensor_mem_deleter{});

                RKNN_CHECK(rknn_matmul_set_io_mem(matmul_ctx->ctx, mem_B_segments[i].get(), &segments_io_attrs[i].B), "set_io_mem B segment");
                current_offset_in_packed_buffer += segment_size_bytes;
            }
        }

        // ===========================================
        // ========== 3. Preparing A-matrix ==========
        // ===========================================
        // Use a fixed-size stack array for scales_A to avoid dynamic allocation
        float scales_A[2048];
        if (M_op > 2048) {
            // Fallback to CPU if batch size is too large for stack buffer
            return GGML_STATUS_FAILED;
        }
        memset(scales_A, 0, M_op * sizeof(float));

        float scale_B = 1.0f;
        {
            auto cache_key = std::make_tuple(M_op, K, (int)w_type);
            auto& matmul_ctx_0 = matmul_ctxs[0];

            mem_A_shared = get_or_create_npu_buffer(backend_ctx, matmul_ctx_0->ctx, segments_io_attrs[0].A.size, cache_key, backend_ctx->a_buffer_cache);
            if (!mem_A_shared) return GGML_STATUS_FAILED;

            // Sync src1 to CPU if it's coming from RKNPU
            rknpu_sync_tensor(src1, RKNN_MEMORY_SYNC_FROM_DEVICE, 0, ggml_nbytes(src1));
            const void* src1_data = ggml_rknpu_get_system_ptr(src1);
            // No need to sync FROM device if src1 was produced on CPU (which is usually the case in llama.cpp)

            void* dst_base = mem_A_shared->virt_addr;

            switch (op_support->npu_type_a) {
                case rknpu2_configuration::NPU_TYPE_FP16: {
                    uint16_t* dst_ptr = (uint16_t*)dst_base;
                    #pragma omp parallel for
                    for (int m = 0; m < M_op; ++m) {
                        uint16_t* dst_row = dst_ptr + (size_t)m * K;
                        if (m < M) {
                            const void* src_row = (const uint8_t*)src1_data + (size_t)m * src1->nb[1];
                            if (src1->type == GGML_TYPE_F32) {
                                rknpu2_quantization::convert_fp32_to_fp16((const float*)src_row, dst_row, K);
                            } else if (src1->type == GGML_TYPE_F16) {
                                memcpy(dst_row, src_row, K * sizeof(uint16_t));
                            }
                        } else {
                            memset(dst_row, 0, K * sizeof(uint16_t));
                        }
                    }
                    break;
                }

                case rknpu2_configuration::NPU_TYPE_INT8: {
                    int8_t* dst_ptr = (int8_t*)dst_base;
                    #pragma omp parallel for
                    for (int m = 0; m < M_op; ++m) {
                        int8_t* dst_row = dst_ptr + (size_t)m * K;
                        if (m < M) {
                            const void* src_row = (const uint8_t*)src1_data + (size_t)m * src1->nb[1];
                            if (src1->type == GGML_TYPE_F32) {
                                const float* f32_row = (const float*)src_row;
                                float amax_m = rknpu2_quantization::calculate_amax_fp32(f32_row, K);
                                scales_A[m] = amax_m / 127.0f;
                                rknpu2_quantization::quantize_fp32_to_int8(f32_row, dst_row, K, scales_A[m]);
                            } else if (src1->type == GGML_TYPE_F16) {
                                const uint16_t* f16_row = (const uint16_t*)src_row;
                                float amax_m = rknpu2_quantization::calculate_amax_fp16(f16_row, K);
                                scales_A[m] = amax_m / 127.0f;
                                rknpu2_quantization::quantize_fp16_to_int8(f16_row, dst_row, K, scales_A[m]);
                            }
                        } else {
                            memset(dst_row, 0, K);
                            scales_A[m] = 0.0f;
                        }
                    }
                    break;
                }

                case rknpu2_configuration::NPU_TYPE_INT4: {
                    auto* src0_buf_ctx = (ggml_backend_rknpu_buffer_context*)src0->buffer->context;
                    size_t offset = ggml_backend_rknpu_get_tensor_offset(src0->buffer, src0);
                    
                    std::vector<float> s_vec;
                    {
                        std::lock_guard<std::mutex> lock(src0_buf_ctx->mutex);
                        auto it = src0_buf_ctx->hadamard_s_vectors.find(offset);
                        GGML_ASSERT(it != src0_buf_ctx->hadamard_s_vectors.end() && "Hadamard 's' vector not found");
                        s_vec = it->second;
                    }

                    uint8_t* dst_ptr = (uint8_t*)dst_base;
                    #pragma omp parallel for
                    for (int m = 0; m < M_op; ++m) {
                        uint8_t* dst_row = dst_ptr + (size_t)m * (K_op / 2);
                        if (m < M) {
                            const void* src_row = (const uint8_t*)src1_data + (size_t)m * src1->nb[1];
                            std::vector<float> f32_row(K);
                            if (src1->type == GGML_TYPE_F32) {
                                memcpy(f32_row.data(), src_row, K * sizeof(float));
                            } else if (src1->type == GGML_TYPE_F16) {
                                rknpu2_quantization::convert_fp16_to_fp32((const uint16_t*)src_row, f32_row.data(), K);
                            }
                            
                            std::vector<float> signed_row(K);
                            for(int k=0; k<K; ++k) signed_row[k] = f32_row[k] * s_vec[k];

                            std::vector<float> rotated_row(K_op);
                            rknpu2_calibration::hadamard_transform(rotated_row.data(), signed_row.data(), K, K_op);

                            float amax_m = 0.0f;
                            for (int k = 0; k < K_op; ++k) amax_m = std::max(amax_m, std::abs(rotated_row[k]));
                            scales_A[m] = amax_m / 7.0f;
                            rknpu2_quantization::quantize_fp32_to_int4_packed(rotated_row.data(), dst_row, K_op, scales_A[m]);
                        } else {
                            memset(dst_row, 0, K_op / 2);
                            scales_A[m] = 0.0f;
                        }
                    }
                    break;
                }

                default:
                    // This should not be reached if config is correct
                    return GGML_STATUS_FAILED;
            }

            if (op_support->npu_type_a == rknpu2_configuration::NPU_TYPE_INT8 || op_support->npu_type_a == rknpu2_configuration::NPU_TYPE_INT4) {
                ggml_backend_buffer_t src0_buffer = src0->buffer;
                auto* src0_buf_ctx = (ggml_backend_rknpu_buffer_context*)src0_buffer->context;
                size_t offset = ggml_backend_rknpu_get_tensor_offset(src0_buffer, src0);

                {
                    std::lock_guard<std::mutex> lock(src0_buf_ctx->mutex);
                    auto it = src0_buf_ctx->quantized_tensor_scales.find(offset);
                    GGML_ASSERT(it != src0_buf_ctx->quantized_tensor_scales.end() && "Quantized scale not found");
                    scale_B = it->second;
                }
            }
        }

        // ===========================================
        // ========== 4. Preparing C-matrix ==========
        // ===========================================
        {            
            for (size_t i = 0; i < num_active_segments; i++) {
                auto& matmul_ctx = matmul_ctxs[i];
                auto cache_key = std::make_tuple(M_op, active_segments[i].size_n, active_segments[i].core_id);
                mem_C_segments[i] = get_or_create_npu_buffer(backend_ctx, matmul_ctx->ctx, segments_io_attrs[i].C.size, cache_key, backend_ctx->c_buffer_cache);
                if (!mem_C_segments[i]) return GGML_STATUS_FAILED;
                RKNN_CHECK(rknn_matmul_set_io_mem(matmul_ctx->ctx, mem_C_segments[i].get(), &segments_io_attrs[i].C), "set_io_mem C");
            }
        }

        // ==========================================
        // ========== 5. Running operation ==========
        // ==========================================
        {            
            int run_ret = RKNN_SUCC;

            auto mem_ctx = get_rknpu_memory_context().get_ctx();
            RKNN_CHECK(rknn_mem_sync(mem_ctx, mem_A_shared.get(), RKNN_MEMORY_SYNC_TO_DEVICE), "sync A TO_DEVICE");

            for (size_t i = 0; i < num_active_segments; i++) {
                RKNN_CHECK(rknn_matmul_set_io_mem(matmul_ctxs[i]->ctx, mem_A_shared.get(), &segments_io_attrs[i].A), "set_io_mem A for core");
            }

            // Running segments sequentially for better stability on some driver versions
            for (size_t i = 0; i < num_active_segments; i++) {
                int ret = rknn_matmul_run(matmul_ctxs[i]->ctx);
                if (ret != RKNN_SUCC) {
                    run_ret = ret;
                    fprintf(stderr, "RKNPU2: Failed to run matmul for node %s segment %zu core %d error %d! M=%d K=%d N=%d\n", 
                           node->name, i, active_segments[i].core_id, ret, M_op, K_op, active_segments[i].size_n);
                    fflush(stderr);
                }
            }
            if (run_ret != RKNN_SUCC) return GGML_STATUS_FAILED;
        }

        // ===========================================
        // ========== 6. Collecting results ==========
        // ===========================================
        {
            float* dst_data = (float*)dst->data;
            
            for (size_t i = 0; i < num_active_segments; i++) {
                RKNN_CHECK(rknn_mem_sync(matmul_ctxs[i]->ctx, mem_C_segments[i].get(), RKNN_MEMORY_SYNC_FROM_DEVICE), "sync C FROM_DEVICE");
            }

            #pragma omp parallel for collapse(2)
            for (int m = 0; m < M; m++) {
                for (size_t i = 0; i < num_active_segments; i++) {
                    switch (op_support->npu_type_c) {
                        case rknpu2_configuration::NPU_TYPE_FP32: {
                            int N_offset = active_segments[i].offset_n;
                            int N_segment = active_segments[i].size_n;
                            float* src_segment_base = (float*)mem_C_segments[i]->virt_addr;
                            memcpy(dst_data + (size_t)m * N + N_offset,
                                src_segment_base + (size_t)m * N_segment,
                                N_segment * sizeof(float));
                            break;
                        }

                        case rknpu2_configuration::NPU_TYPE_INT32: {
                            float dequant_scale = scales_A[m] * scale_B;
                            int N_offset = active_segments[i].offset_n;
                            int N_segment = active_segments[i].size_n;
                            float* dst_ptr = dst_data + (size_t)m * N + N_offset;
                            int32_t* src_segment_base = (int32_t*)mem_C_segments[i]->virt_addr;
                            int32_t* src_ptr = src_segment_base + (size_t)m * N_segment;
                            rknpu2_quantization::dequantize_int32_to_fp32(src_ptr, dst_ptr, N_segment, dequant_scale);
                            break;
                        }

                        case rknpu2_configuration::NPU_TYPE_INT16: {
                            float dequant_scale = scales_A[m] * scale_B;
                            if (op_support->npu_type_a == rknpu2_configuration::NPU_TYPE_INT4) {
                                dequant_scale /= (float)K_op;
                            }

                            int N_offset = active_segments[i].offset_n;
                            int N_segment = active_segments[i].size_n;
                            float* dst_ptr = dst_data + (size_t)m * N + N_offset;
                            int16_t* src_segment_base = (int16_t*)mem_C_segments[i]->virt_addr;
                            int16_t* src_ptr = src_segment_base + (size_t)m * N_segment;
                            rknpu2_quantization::dequantize_int16_to_fp32(src_ptr, dst_ptr, N_segment, dequant_scale);
                            break;
                        }
                        
                        default:
                            break;
                    }
                }
            }
        }
    }

    return GGML_STATUS_SUCCESS;
}


//
// Buffer
//

static void ggml_backend_rknpu_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_rknpu_buffer_context * ctx = (ggml_backend_rknpu_buffer_context *)buffer->context;
    rknpu2_allocation::free(ctx->dma_buf);
    delete ctx;
}

static void * ggml_backend_rknpu_buffer_get_base(ggml_backend_buffer_t buffer) {
    if (!buffer || !buffer->context) return NULL;
    ggml_backend_rknpu_buffer_context * ctx = (ggml_backend_rknpu_buffer_context *)buffer->context;
    return ctx->dma_buf.virt_addr;
}

static enum ggml_status ggml_backend_rknpu_buffer_init_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor) {
    if (!buffer || !buffer->context || !tensor) return GGML_STATUS_FAILED;
    
    ggml_backend_rknpu_buffer_context * ctx = (ggml_backend_rknpu_buffer_context *)buffer->context;
    if (ctx->dma_buf.virt_addr) {
        uintptr_t data_val = (uintptr_t)tensor->data;
        uintptr_t base_val = (uintptr_t)ctx->dma_buf.virt_addr;
        uintptr_t end_val = base_val + ctx->dma_buf.size;
        
        // If data is NOT within the buffer range, treat it as an offset and add base
        if (data_val < base_val || data_val >= end_val) {
            tensor->data = (uint8_t*)ctx->dma_buf.virt_addr + data_val;
        }
    }
    
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_rknpu_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    if (!buffer || !buffer->context || !tensor || !tensor->data) {
        if (!tensor || !tensor->data) {
            fprintf(stderr, "RKNPU2: set_tensor INVALID ARGS: buffer=%p, tensor=%p, data=%p\n", (void*)buffer, (void*)tensor, (void*)(tensor ? tensor->data : NULL));
            fflush(stderr);
        }
        return;
    }

    auto * ctx = (ggml_backend_rknpu_buffer_context *) buffer->context;
    uint8_t* dma_base = (uint8_t*)ctx->dma_buf.virt_addr;
    if (!dma_base) return;

    void* base_addr = ggml_backend_rknpu_buffer_get_base(buffer);
    if (!base_addr) return;

    const size_t tensor_offset_in_buffer = (uint8_t*)tensor->data - (uint8_t*)base_addr;
    if (tensor_offset_in_buffer + offset + size > ctx->dma_buf.size) {
        fprintf(stderr, "RKNPU2: set_tensor OUT OF BOUNDS! buffer_size=%zu, offset_in_buffer=%zu, offset=%zu, size=%zu\n", 
                ctx->dma_buf.size, tensor_offset_in_buffer, offset, size);
        fflush(stderr);
        return;
    }

    uint8_t* tensor_dma_ptr = dma_base + tensor_offset_in_buffer;
    
    if (!data) {
        return;
    }

    // Getting the current device configuration to drive the packing logic
    const auto& config = rknpu2_configuration::Rknpu2ConfigManager::get_instance().get_current_config();
    const auto* op_support = config.find_op_support(tensor->type);

    // If there is a specific packing function defined for this tensor type, it's a weight matrix
    // We only calculate and store metadata here (scales, etc). 
    // The actual packing will be performed lazily in graph_compute to allow 
    // the same tensor to be used by CPU nodes (like token_embd lookup).
    if (op_support && op_support->pack_func) {
        const int K = (int)tensor->ne[0];
        const int N = (int)tensor->ne[1];

        // Metadata key (use base tensor offset for the status)
        size_t offset_key = ggml_backend_rknpu_get_tensor_offset(buffer, tensor);

        // GGML_TYPE_F16
        if (tensor->type == GGML_TYPE_F16) {
            // No metadata needed for F16
        // GGML_TYPE_Q8_0
        } else if (tensor->type == GGML_TYPE_Q8_0) {
            const block_q8_0* src_blocks = (const block_q8_0*)data;

            float amax = 0.0f;
            #pragma omp parallel
            {
                std::vector<float> tmp_row(K);
                #pragma omp for reduction(max:amax)
                for (int n = 0; n < N; ++n) {
                    dequantize_row_q8_0(src_blocks + (size_t)n * (K / QK8_0), tmp_row.data(), K);
                    for (int k = 0; k < K; ++k) amax = std::max(amax, std::abs(tmp_row[k]));
                }
            }

            const float global_scale_b = amax / 127.0f;
            {
                std::lock_guard<std::mutex> lock(ctx->mutex);
                ctx->quantized_tensor_scales[offset_key] = global_scale_b;
            }
        // GGML_TYPE_Q4_0
        } else if (tensor->type == GGML_TYPE_Q4_0) {
            if (op_support->npu_type_a == rknpu2_configuration::NPU_TYPE_INT4) {
                const int K_op = rknpu2_calibration::next_power_of_two(K);
                std::vector<float> s_vec = rknpu2_calibration::generate_random_sign_vector(K_op);
                
                {
                    std::lock_guard<std::mutex> lock(ctx->mutex);
                    ctx->hadamard_s_vectors[offset_key] = s_vec;
                }

                std::vector<float> rotated_data((size_t)K_op * N);
                #pragma omp parallel
                {
                    std::vector<float> tmp_row(K_op, 0.0f);
                    #pragma omp for
                    for (int n = 0; n < N; ++n) {
                        dequantize_row_q4_0((const block_q4_0*)data + (size_t)n * (K / QK4_0), tmp_row.data(), K);
                        // Hadamard
                        for (int k = 0; k < K_op; ++k) tmp_row[k] *= s_vec[k];
                        rknpu2_calibration::hadamard_transform(rotated_data.data() + (size_t)n * K_op, tmp_row.data(), K, K_op);
                    }
                }

                float amax = 0.0f;
                for (const auto& val : rotated_data) amax = std::max(amax, std::abs(val));
                const float global_scale_b = amax / 7.0f;

                {
                    std::lock_guard<std::mutex> lock(ctx->mutex);
                    ctx->quantized_tensor_scales[offset_key] = global_scale_b;
                }
            } else {
                // Basic Q4_0 (non-Hadamard) fallback
                const block_q4_0* src_blocks = (const block_q4_0*)data;

                float amax = 0.0f;
                #pragma omp parallel
                {
                    std::vector<float> tmp_row(K);
                    #pragma omp for reduction(max:amax)
                    for (int n = 0; n < N; ++n) {
                        dequantize_row_q4_0(src_blocks + (size_t)n * (K / QK4_0), tmp_row.data(), K);
                        for (int k = 0; k < K; ++k) amax = std::max(amax, std::abs(tmp_row[k]));
                    }
                }

                const float global_scale_b = amax / 7.0f;

                {
                    std::lock_guard<std::mutex> lock(ctx->mutex);
                    ctx->quantized_tensor_scales[offset_key] = global_scale_b;
                }
            }
        }
    }

    // ALWAYS copy the original raw data to the DMA buffer.
    // This allows CPU nodes to work with the data in its original format.
    if (size > 0) {
        memcpy(tensor_dma_ptr + offset, data, size);
    }

    // Syncing the raw data to the NPU
    if (size > 0) {
        rknpu_sync_tensor(tensor, RKNN_MEMORY_SYNC_TO_DEVICE, offset, size);
    }
}

static void ggml_backend_rknpu_buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    if (!buffer || !buffer->context || !tensor || !tensor->data) {
        if (!tensor || !tensor->data) {
            fprintf(stderr, "RKNPU2: get_tensor INVALID ARGS: buffer=%p, tensor=%p, data=%p\n", (void*)buffer, (const void*)tensor, (void*)(tensor ? tensor->data : NULL));
            fflush(stderr);
        }
        return;
    }

    ggml_backend_rknpu_buffer_context * ctx = (ggml_backend_rknpu_buffer_context *)buffer->context;
    uint8_t* dma_base = (uint8_t*)ctx->dma_buf.virt_addr;
    if (!dma_base) return;

    void* base_addr = ggml_backend_rknpu_buffer_get_base(buffer);
    if (!base_addr) return;

    const size_t tensor_offset_in_buffer = (uint8_t*)tensor->data - (uint8_t*)base_addr;
    if (tensor_offset_in_buffer + offset + size > ctx->dma_buf.size) {
        fprintf(stderr, "RKNPU2: get_tensor OUT OF BOUNDS! buffer_size=%zu, offset_in_buffer=%zu, offset=%zu, size=%zu\n", 
                ctx->dma_buf.size, tensor_offset_in_buffer, offset, size);
        fflush(stderr);
        return;
    }

    uint8_t* tensor_dma_ptr = dma_base + tensor_offset_in_buffer;
    
    // Sync from NPU before reading with CPU
    if (size > 0) {
        rknpu_sync_tensor(tensor, RKNN_MEMORY_SYNC_FROM_DEVICE, offset, size);
    }

    memcpy(data, tensor_dma_ptr + offset, size);
}

static void ggml_backend_rknpu_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    if (!buffer || !buffer->context) return;
    ggml_backend_rknpu_buffer_context * ctx = (ggml_backend_rknpu_buffer_context *)buffer->context;
    if (!ctx->dma_buf.virt_addr) return;
    memset(ctx->dma_buf.virt_addr, value, ctx->dma_buf.size);

    // Sync to device after clearing
    rknn_tensor_mem sync_mem = {};
    sync_mem.virt_addr = ctx->dma_buf.virt_addr;
    sync_mem.fd = ctx->dma_buf.fd;
    sync_mem.offset = 0;
    sync_mem.size = (uint32_t)ctx->dma_buf.size;
    auto mem_ctx = get_rknpu_memory_context().get_ctx();
    if (mem_ctx != 0) {
        rknn_mem_sync(mem_ctx, &sync_mem, RKNN_MEMORY_SYNC_TO_DEVICE);
    }
}


//
// Buffer Type
//

static const char * ggml_backend_rknpu_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    UNUSED(buft);
    return "RKNPU";
}

static ggml_backend_buffer_t ggml_backend_rknpu_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    UNUSED(buft);

    rknpu2_allocation::DmaBuffer dma_buf = rknpu2_allocation::alloc(size);
    if (dma_buf.fd < 0 || !dma_buf.virt_addr) {
        return NULL;
    }

    ggml_backend_rknpu_buffer_context * ctx = new ggml_backend_rknpu_buffer_context();
    ctx->dma_buf = dma_buf;
    ctx->name = "rknpu_dma_buffer";

    static const ggml_backend_buffer_i rknpu_buffer_interface = {
        /* .free_buffer   = */ ggml_backend_rknpu_buffer_free_buffer,
        /* .get_base      = */ ggml_backend_rknpu_buffer_get_base,
        /* .init_tensor   = */ ggml_backend_rknpu_buffer_init_tensor,
        /* .memset_tensor = */ NULL,
        /* .set_tensor    = */ ggml_backend_rknpu_buffer_set_tensor,
        /* .get_tensor    = */ ggml_backend_rknpu_buffer_get_tensor,
        /* .cpy_tensor    = */ NULL,
        /* .clear         = */ ggml_backend_rknpu_buffer_clear,
        /* .reset         = */ NULL,
    };

    return ggml_backend_buffer_init(buft, rknpu_buffer_interface, ctx, size);
}

static size_t ggml_backend_rknpu_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    UNUSED(buft);
    return 64;
}

static size_t ggml_backend_rknpu_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor) {
    UNUSED(buft);
    if (!tensor) return 0;

    // Getting the current device configuration
    const auto& config = rknpu2_configuration::Rknpu2ConfigManager::get_instance().get_current_config();
    const auto* op_support = config.find_op_support(tensor->type);

    // Padding Q4_0 weights to the next power of two for Hadamard Transform.
    if (tensor->type == GGML_TYPE_Q4_0) {
        if (op_support && op_support->npu_type_a == rknpu2_configuration::NPU_TYPE_INT4) {
            const int K = (int)tensor->ne[0];
            const int N = (int)tensor->ne[1];

            const int padded_K = rknpu2_calibration::next_power_of_two(K);

            auto segments = compute_matrix_segments(N, config.core_count, op_support->n_align);
            size_t total_size = 0;
            for (const auto& seg : segments) {
                if (seg.size_n > 0) {
                    total_size += (size_t)seg.size_n * padded_K / 2;
                }
            }
            return total_size;
        }
    }

    // Fallback to default size calculation for other types.
    return ggml_nbytes(tensor);
}


//
// Device
//

static const char * ggml_backend_rknpu_device_get_name(ggml_backend_dev_t dev) {
    UNUSED(dev);
    return "RKNPU";
}

static const char * ggml_backend_rknpu_device_get_description(ggml_backend_dev_t dev) {
    UNUSED(dev);
    return "Rockchip NPU";
}

static void ggml_backend_rknpu_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    UNUSED(dev);
    *free = 0;
    *total = 0;

    std::ifstream meminfo("/proc/meminfo");
    if (meminfo.is_open()) {
        std::string line;
        while (std::getline(meminfo, line)) {
            if (line.compare(0, 9, "MemTotal:") == 0) {
                size_t val;
                if (sscanf(line.c_str(), "MemTotal: %zu kB", &val) == 1) {
                    *total = val * 1024;
                }
            } else if (line.compare(0, 13, "MemAvailable:") == 0) {
                size_t val;
                if (sscanf(line.c_str(), "MemAvailable: %zu kB", &val) == 1) {
                    *free = val * 1024;
                }
            }
        }
    }
}

static enum ggml_backend_dev_type ggml_backend_rknpu_device_get_type(ggml_backend_dev_t dev) {
    UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;
}

static void ggml_backend_rknpu_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name = ggml_backend_rknpu_device_get_name(dev);
    props->description = ggml_backend_rknpu_device_get_description(dev);
    props->type = ggml_backend_rknpu_device_get_type(dev);
    ggml_backend_rknpu_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->device_id = NULL;

    props->caps.async = false;
    props->caps.host_buffer = false;
    props->caps.buffer_from_host_ptr = false;
    props->caps.events = false;
}

static bool ggml_backend_rknpu_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    UNUSED(dev);

    // Getting the current device configuration
    const auto& config = rknpu2_configuration::Rknpu2ConfigManager::get_instance().get_current_config();

    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;

        case GGML_OP_SET_ROWS:
        case GGML_OP_GET_ROWS:
        case GGML_OP_CPY:
        case GGML_OP_DUP:
        case GGML_OP_CONT:
            return true;

        case GGML_OP_MUL_MAT: {
            const struct ggml_tensor * src0 = op->src[0]; // Weights
            const struct ggml_tensor * src1 = op->src[1]; // Activations

            if (!src0 || !src1) {
                return false;
            }

            // Finding if there is a supported operation for the given weight type
            const auto* op_support = config.find_op_support(src0->type);
            if (!op_support) {
                return false;
            }

            // Checking if activation type matches the supported operation
            if (src1->type != op_support->type_a) {
                return false;
            }

            // Checking for K alignment
            if (src0->ne[0] % op_support->k_align != 0) {
                return false;
            }

            // Checking for N alignment
            if (src0->ne[1] % op_support->n_align != 0) {
                return false;
            }

            // Checking for exact dimensions
            if (src1->ne[0] != src0->ne[0]) {
                 return false;
            }

            // Removed CPU fallback for batched matmuls (M > 1) to enable RKNPU offloading
            // if (src1->ne[1] > 1) {
            //     return false;
            // }

            // Checking contiguous memory
            if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1)) {
                return false;
            }

            return true;
        }
        default:
            return false;
    }
}

static ggml_backend_t ggml_backend_rknpu_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    UNUSED(dev);
    UNUSED(params);

    // TODO: Make device selection dynamic (e.g., from params or env var)
    if (!rknpu2_configuration::Rknpu2ConfigManager::get_instance().select_device("RK3588")) return NULL;

    ggml_backend_rknpu_context * ctx = new ggml_backend_rknpu_context();
    
    static const struct ggml_backend_i rknpu_backend_interface = {
        /* .get_name           = */ ggml_backend_rknpu_name,
        /* .free               = */ ggml_backend_rknpu_free,
        /* .set_tensor_async   = */ NULL,
        /* .get_tensor_async   = */ NULL,
        /* .cpy_tensor_async   = */ NULL,
        /* .synchronize        = */ NULL,
        /* .graph_plan_create  = */ NULL,
        /* .graph_plan_free    = */ NULL,
        /* .graph_plan_update  = */ NULL,
        /* .graph_plan_compute = */ NULL,
        /* .graph_compute      = */ ggml_backend_rknpu_graph_compute,
        /* .event_record       = */ NULL,
        /* .event_wait         = */ NULL,
        /* .graph_optimize     = */ NULL,
    };

    return new ggml_backend{
        /* .guid    = */ {0},
        /* .iface   = */ rknpu_backend_interface,
        /* .device  = */ dev,
        /* .context = */ ctx,
    };
}


//
// Registry
//

static const char * ggml_backend_rknpu_reg_get_name(ggml_backend_reg_t reg) {
    UNUSED(reg);
    return "RKNPU";
}

static size_t ggml_backend_rknpu_reg_get_device_count(ggml_backend_reg_t reg) {
    UNUSED(reg);
    if (get_rknpu_memory_context().get_ctx() != 0) {
        return 1;
    }
    return 0;
}

static ggml_backend_dev_t ggml_backend_rknpu_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    if (index != 0) {
        return NULL;
    }
    
    static const struct ggml_backend_buffer_type_i rknpu_buffer_type_interface = {
        /* .get_name       = */ ggml_backend_rknpu_buffer_type_get_name,
        /* .alloc_buffer   = */ ggml_backend_rknpu_buffer_type_alloc_buffer,
        /* .get_alignment  = */ ggml_backend_rknpu_buffer_type_get_alignment,
        /* .get_max_size   = */ NULL,
        /* .get_alloc_size = */ ggml_backend_rknpu_buffer_type_get_alloc_size,
        /* .is_host        = */ NULL,
    };

    static struct ggml_backend_buffer_type rknpu_buffer_type = {
        /* .iface   = */ rknpu_buffer_type_interface,
        /* .device  = */ NULL,
        /* .context = */ NULL,
    };

    static const struct ggml_backend_device_i rknpu_device_interface = {
        /* .get_name             = */ ggml_backend_rknpu_device_get_name,
        /* .get_description      = */ ggml_backend_rknpu_device_get_description,
        /* .get_memory           = */ ggml_backend_rknpu_device_get_memory,
        /* .get_type             = */ ggml_backend_rknpu_device_get_type,
        /* .get_props            = */ ggml_backend_rknpu_device_get_props,
        /* .init_backend         = */ ggml_backend_rknpu_device_init_backend,
        /* .get_buffer_type      = */ [](ggml_backend_dev_t dev) { UNUSED(dev); return &rknpu_buffer_type; },
        /* .get_host_buffer_type = */ NULL,
        /* .buffer_from_host_ptr = */ NULL,
        /* .supports_op          = */ ggml_backend_rknpu_device_supports_op,
        /* .supports_buft        = */ [](ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) { UNUSED(dev); return buft == &rknpu_buffer_type; },
        /* .offload_op           = */ NULL,
        /* .event_new            = */ NULL,
        /* .event_free           = */ NULL,
        /* .event_synchronize    = */ NULL,
    };

    static struct ggml_backend_device rknpu_device = {
        /* .iface   = */ rknpu_device_interface,
        /* .reg     = */ reg,
        /* .context = */ NULL,
    };
    
    if (rknpu_buffer_type.device == NULL) {
        rknpu_buffer_type.device = &rknpu_device;
    }

    return &rknpu_device;
}


//
// Public API
//

GGML_API ggml_backend_reg_t ggml_backend_rknpu2_reg(void) {
    static const struct ggml_backend_reg_i rknpu_reg_interface = {
        /* .get_name         = */ ggml_backend_rknpu_reg_get_name,
        /* .get_device_count = */ ggml_backend_rknpu_reg_get_device_count,
        /* .get_device       = */ ggml_backend_rknpu_reg_get_device,
        /* .get_proc_address = */ NULL,
    };

    static struct ggml_backend_reg rknpu_backend_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ rknpu_reg_interface,
        /* .context     = */ NULL,
    };

    return &rknpu_backend_reg;
}

#ifdef GGML_BACKEND_DL
GGML_BACKEND_DL_IMPL(ggml_backend_rknpu2_reg)
#endif