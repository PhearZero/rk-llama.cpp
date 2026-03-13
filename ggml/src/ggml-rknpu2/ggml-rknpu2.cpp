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

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif
#include <omp.h>

#include <chrono>
#include <cassert>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>
#include <map>
#include <tuple>
#include <algorithm>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <random>
#include <limits>
#include <thread>
#ifdef __linux__
#include <sys/sysinfo.h>
#endif

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

// RKNN buffer context
struct ggml_backend_rknpu_buffer_context {
    rknpu2_allocation::DmaBuffer dma_buf;
    std::string name;

    // Per-tensor scale for weights
    std::unordered_map<const struct ggml_tensor *, float> quantized_tensor_scales;

    // Per-tensor random sign vector for Hadamard Transform
    std::unordered_map<const struct ggml_tensor *, std::vector<float>> hadamard_s_vectors;

    // Tensors that have been explicitly packed via set_tensor
    std::unordered_set<const struct ggml_tensor *> packed_tensors;

    std::mutex mutex;
    void * base_ptr = nullptr;
    size_t base_ptr_offset = 0;
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
        if (ret < 0) ctx = 0;
    }

    ~rknpu_matmul_context() {
        if (ctx != 0) {
            rknn_matmul_destroy(ctx);
        }
    }
};


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
        int ret = rknn_matmul_create(&mem_ctx, &dummy_info, &dummy_io_attr);
        if (ret < 0) mem_ctx = 0;
    }

    ~rknpu_memory_context() {
        if (mem_ctx != 0) {
            rknn_matmul_destroy(mem_ctx);
        }
    }

    rknn_matmul_ctx get_ctx() {
        std::lock_guard<std::mutex> lock(mutex);
        return mem_ctx;
    }
};

static rknpu_memory_context & get_rknpu_memory_context() {
    static rknpu_memory_context g_mem_ctx;
    return g_mem_ctx;
}

// Backend main context
struct ggml_backend_rknpu_context {
    std::string name;
    std::mutex mutex;

    // RKNN matmul contexts cache
    std::unordered_map<std::tuple<int, int, int, int, int>, std::shared_ptr<rknpu_matmul_context>, TupleHasher> matmul_ctx_cache;

    // B-matrices handle cache (from fd)
    // Key: <fd, offset, size>
    std::unordered_map<std::tuple<int, size_t, size_t>, std::shared_ptr<rknn_tensor_mem>, TupleHasher> b_mem_handle_cache;

    // A- and C-matrices cache (from create_mem)
    // Key: <M, N or K, type>
    std::unordered_map<std::tuple<int, int, int>, std::shared_ptr<rknn_tensor_mem>, TupleHasher> a_buffer_cache;
    std::unordered_map<std::tuple<int, int, int>, std::shared_ptr<rknn_tensor_mem>, TupleHasher> c_buffer_cache;

    ggml_backend_t cpu_fallback = nullptr;
    int n_threads = 1;

    ggml_backend_rknpu_context() {
        cpu_fallback = ggml_backend_cpu_init();
#ifdef __linux__
        n_threads = std::max(1U, std::thread::hardware_concurrency());
#else
        n_threads = 4;
#endif
        if (cpu_fallback) {
            ggml_backend_cpu_set_n_threads(cpu_fallback, n_threads);
        }
    }

    ~ggml_backend_rknpu_context() {
        if (cpu_fallback) {
            ggml_backend_free(cpu_fallback);
        }
    }

    std::shared_ptr<rknpu_matmul_context> get_matmul_ctx(int M, int K, int N, int core_id, rknn_matmul_type type) {
        std::lock_guard<std::mutex> lock(mutex);
        auto key = std::make_tuple(M, K, N, core_id, (int)type);
        auto it = matmul_ctx_cache.find(key);
        if (it != matmul_ctx_cache.end()) {
            return it->second;
        }
        auto ctx = std::make_shared<rknpu_matmul_context>(M, K, N, type);
        if (!ctx || ctx->ctx == 0) {
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
        if (ret != RKNN_SUCC) {
            // Handle error
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
    std::shared_ptr<rknpu_matmul_context> matmul_ctx,
    size_t size,
    const std::tuple<int, int, int>& key,
    std::unordered_map<std::tuple<int, int, int>, std::shared_ptr<rknn_tensor_mem>, TupleHasher>& cache
) {
    std::lock_guard<std::mutex> lock(backend_ctx->mutex);
    auto it = cache.find(key);
    if (it != cache.end()) {
        return it->second;
    }

    rknn_tensor_mem* mem = rknn_create_mem(matmul_ctx->ctx, size);
    if (!mem) { return nullptr; }

    auto deleter = [matmul_ctx](rknn_tensor_mem* m) {
        if (m) {
            rknn_destroy_mem(matmul_ctx->ctx, m);
        }
    };

    std::shared_ptr<rknn_tensor_mem> mem_shared(mem, deleter);
    cache[key] = mem_shared;
    return mem_shared;
}

static float get_quantized_scale(ggml_backend_rknpu_buffer_context* ctx, const struct ggml_tensor* tensor) {
    const struct ggml_tensor* base = tensor;
    while (base->view_src != nullptr) base = base->view_src;

    std::lock_guard<std::mutex> lock(ctx->mutex);
    auto it = ctx->quantized_tensor_scales.find(base);
    if (it != ctx->quantized_tensor_scales.end()) return it->second;
    return 1.0f;
}

static std::vector<float> get_hadamard_s_vector(ggml_backend_rknpu_buffer_context* ctx, const struct ggml_tensor* tensor, int K_op) {
    const struct ggml_tensor* base = tensor;
    while (base->view_src != nullptr) base = base->view_src;

    std::lock_guard<std::mutex> lock(ctx->mutex);
    auto it = ctx->hadamard_s_vectors.find(base);
    if (it != ctx->hadamard_s_vectors.end() && (int)it->second.size() >= K_op) {
        return it->second;
    }

    // Generating the random sign vector 's'
    std::vector<float> s_vec(K_op);
    std::mt19937 gen(reinterpret_cast<uintptr_t>(base));
    std::uniform_int_distribution<int> distrib(0, 1);
    for(int k = 0; k < K_op; ++k) {
        s_vec[k] = (distrib(gen) == 0) ? -1.0f : 1.0f;
    }
    ctx->hadamard_s_vectors[base] = s_vec;
    return s_vec;
}

static void * ggml_backend_rknpu_buffer_get_base(ggml_backend_buffer_t buffer);

static void * ggml_rknpu_get_system_ptr(const struct ggml_tensor * tensor) {
    if (!tensor || !tensor->data) return nullptr;

    ggml_backend_buffer_t buffer = tensor->buffer;
    if (!buffer) return tensor->data;

    if (buffer->iface.get_base != ggml_backend_rknpu_buffer_get_base) {
        return tensor->data;
    }

    auto * ctx = (ggml_backend_rknpu_buffer_context *)buffer->context;
    uintptr_t data_ptr = (uintptr_t)tensor->data;

    // Check if data_ptr is already within the DMA buffer's virtual address range
    uintptr_t dma_start = (uintptr_t)ctx->dma_buf.virt_addr;
    uintptr_t dma_end = dma_start + ctx->dma_buf.size;
    if (data_ptr >= dma_start && data_ptr < dma_end) {
        return (void *)data_ptr;
    }

    // Heuristic: absolute pointers are usually much larger than relative offsets
    // On 64-bit systems, virtual addresses are typically > 0x100000000 (4GB)
    // Small values (< buffer_size) are likely offsets from the buffer base.
    size_t offset = (size_t)data_ptr;
    bool is_offset = (offset < ctx->dma_buf.size);

    if (is_offset) {
        void* ptr = (uint8_t *)ctx->dma_buf.virt_addr + offset;
        return ptr;
    }

    // In Protocol v3, if base_ptr is set, check if data_ptr is in the old CPU range
    if (ctx->base_ptr) {
        uintptr_t cpu_start = (uintptr_t)ctx->base_ptr;
        uintptr_t cpu_end = cpu_start + ctx->dma_buf.size;
        if (data_ptr >= cpu_start && data_ptr < cpu_end) {
            offset = data_ptr - cpu_start;
            return (uint8_t *)ctx->dma_buf.virt_addr + offset;
        }
    }

    return (void *)data_ptr;
}

static void translate_tensor_recursive(struct ggml_tensor * tensor, std::unordered_map<struct ggml_tensor *, void *> & saved_ptrs) {
    if (!tensor || saved_ptrs.count(tensor)) return;

    saved_ptrs[tensor] = tensor->data;
    tensor->data = ggml_rknpu_get_system_ptr(tensor);

    for (int i = 0; i < GGML_MAX_SRC; i++) {
        translate_tensor_recursive(tensor->src[i], saved_ptrs);
    }
    translate_tensor_recursive(tensor->view_src, saved_ptrs);
}

static enum ggml_status ggml_backend_rknpu_graph_compute(ggml_backend_t backend, struct ggml_cgraph* cgraph) {
    auto* backend_ctx = (ggml_backend_rknpu_context*)backend->context;

    // Getting the current device configuration once
    const auto& config = rknpu2_configuration::Rknpu2ConfigManager::get_instance().get_current_config();

    if (config.device_name == "NONE") {
        GGML_LOG_ERROR("[%s] No device configuration found!\n", __func__);
        return GGML_STATUS_FAILED;
    }

    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor* node = cgraph->nodes[i];

        const struct ggml_tensor* src0 = node->src[0]; // Weights      :  (K x N)
        const struct ggml_tensor* src1 = node->src[1]; // Activations  :  (M x K)
        struct ggml_tensor* dst = node;

        const ggml_type w_type = src0 ? src0->type : GGML_TYPE_COUNT;
        const int M = src1 ? (int)src1->ne[1] : 0;
        const int K = src0 ? (int)src0->ne[0] : 0;
        const int N = src0 ? (int)src0->ne[1] : 0;

        if (node->op == GGML_OP_MUL_MAT && (M <= 0 || N <= 0 || K <= 0)) {
            // GGML_LOG_INFO("[%s] Skipping zero-sized MUL_MAT Node %d (M=%d, K=%d, N=%d)\n", __func__, i, M, K, N);
            continue;
        }

        if (node->op == GGML_OP_MUL_MAT) {
            // GGML_LOG_INFO("[%s] Node %d: op=%d (MUL_MAT) M=%d, K=%d, N=%d, type=%d\n", __func__, i, (int)node->op, M, K, N, (int)w_type);
        }

        if (node->op != GGML_OP_MUL_MAT) {
            if (backend_ctx->cpu_fallback) {
                // GGML_LOG_INFO("[%s] Node %d: op=%d (%s) not supported by RKNPU, falling back to CPU\n", __func__, i, (int)node->op, ggml_op_name(node->op));
                struct ggml_tensor * tmp_nodes[1] = { node };
                struct ggml_cgraph temp_graph = {};
                temp_graph.n_nodes = 1;
                temp_graph.nodes = tmp_nodes;

                std::unordered_map<struct ggml_tensor *, void *> saved_ptrs;
                translate_tensor_recursive(node, saved_ptrs);

                ggml_status status = ggml_backend_graph_compute(backend_ctx->cpu_fallback, &temp_graph);

                for (auto & pair : saved_ptrs) {
                    pair.first->data = pair.second;
                }

                if (status != GGML_STATUS_SUCCESS) {
                    GGML_LOG_ERROR("[%s] CPU fallback failed for node %d (op=%d)\n", __func__, i, (int)node->op);
                    return status;
                }
                // GGML_LOG_INFO("[%s] Node %d: op=%d (%s) CPU fallback successful\n", __func__, i, (int)node->op, ggml_op_name(node->op));
            }
            continue;
        }

        const auto* op_support = config.find_op_support(w_type, src1 ? src1->type : GGML_TYPE_F32);
        if (node->op == GGML_OP_MUL_MAT) {
            bool supported_by_npu = (op_support != nullptr);

            // Check if src0 is packed. If it's in a view, we check the base tensor.
            const struct ggml_tensor * src0_base = src0;
            while (src0_base->view_src != nullptr) src0_base = src0_base->view_src;

            bool is_packed = false;
            if (src0_base->buffer && src0_base->buffer->iface.get_base == ggml_backend_rknpu_buffer_get_base) {
                auto * src0_buf_ctx = (ggml_backend_rknpu_buffer_context*)src0_base->buffer->context;
                std::lock_guard<std::mutex> lock(src0_buf_ctx->mutex);
                if (src0_buf_ctx->packed_tensors.count(src0_base)) {
                    is_packed = true;
                }
            }

            if (!supported_by_npu || !is_packed) {
                if (backend_ctx->cpu_fallback) {
                    GGML_LOG_DEBUG("[%s] Node %d: op=%d (MUL_MAT) %s, falling back to CPU\n",
                                  __func__, i, (int)node->op, !supported_by_npu ? "type not supported" : "src0 is not packed");
                    struct ggml_tensor * tmp_nodes[1] = { node };
                    struct ggml_cgraph temp_graph = {};
                    temp_graph.n_nodes = 1;
                    temp_graph.nodes = tmp_nodes;

                    std::unordered_map<struct ggml_tensor *, void *> saved_ptrs;
                    translate_tensor_recursive(node, saved_ptrs);

                    ggml_status status = ggml_backend_graph_compute(backend_ctx->cpu_fallback, &temp_graph);

                    for (auto & pair : saved_ptrs) {
                        pair.first->data = pair.second;
                    }

                    if (status != GGML_STATUS_SUCCESS) {
                        GGML_LOG_ERROR("[%s] CPU fallback failed for node %d (op=%d)\n", __func__, i, (int)node->op);
                        return status;
                    }
                } else {
                    GGML_LOG_ERROR("[%s] Node %d: op=%d (MUL_MAT) %s and no CPU fallback available\n",
                                   __func__, i, (int)node->op, !supported_by_npu ? "type not supported" : "src0 is not packed");
                    return GGML_STATUS_FAILED;
                }
                continue;
            }
        }

        const bool is_q4_hadamard = (w_type == GGML_TYPE_Q4_0);
        const int K_op = is_q4_hadamard ? rknpu2_calibration::next_power_of_two(K) : K;

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
        std::vector<std::shared_ptr<rknn_tensor_mem>> mem_B_segments(num_active_segments);
        std::shared_ptr<rknn_tensor_mem> mem_A_shared;
        std::vector<std::shared_ptr<rknn_tensor_mem>> mem_C_segments(num_active_segments);

        // ===========================================
        // ========== 1. Preparing contexts ==========
        // ===========================================
        // LOG_DBG("[%s] Node %d Step 1: Preparing contexts\n", __func__, i);
        {
            for (size_t i = 0; i < num_active_segments; ++i) {
                const auto& seg = active_segments[i];
                matmul_ctxs[i] = backend_ctx->get_matmul_ctx(M, K_op, seg.size_n, seg.core_id, matmul_type);
                if (!matmul_ctxs[i] || matmul_ctxs[i]->ctx == 0) {
                    GGML_LOG_ERROR("[%s] Failed to get matmul context for Node %d, core %d (M=%d, K=%d, N=%d)\n", __func__, (int)i, seg.core_id, M, K_op, seg.size_n);
                    return GGML_STATUS_FAILED;
                }
            }
        }

        // ===========================================
        // ========== 2. Preparing B-matrix ==========
        // ===========================================
        // LOG_DBG("[%s] Node %d Step 2: Preparing B-matrix\n", __func__, i);
        {
            ggml_backend_buffer_t src0_buffer = src0->buffer;
            auto* src0_buf_ctx = (ggml_backend_rknpu_buffer_context*)src0_buffer->context;

            // In Protocol v3, we need to handle relative offsets for B-matrix too
            uint8_t* src0_sys_ptr = (uint8_t*)ggml_rknpu_get_system_ptr(src0);
            if (!src0_sys_ptr) return GGML_STATUS_FAILED;

            size_t src0_base_offset_in_dma = 0;
            if (src0_sys_ptr >= (uint8_t*)src0_buf_ctx->dma_buf.virt_addr &&
                src0_sys_ptr < (uint8_t*)src0_buf_ctx->dma_buf.virt_addr + src0_buf_ctx->dma_buf.size) {
                src0_base_offset_in_dma = src0_sys_ptr - (uint8_t*)src0_buf_ctx->dma_buf.virt_addr;
            } else {
                GGML_LOG_ERROR("[%s] src0 system pointer %p is outside DMA buffer %p (size %zu)\n",
                               __func__, src0_sys_ptr, src0_buf_ctx->dma_buf.virt_addr, src0_buf_ctx->dma_buf.size);
                return GGML_STATUS_FAILED;
            }

            size_t type_size_packed;
            if (op_support->npu_type_a == rknpu2_configuration::NPU_TYPE_FP16) type_size_packed = 2;
            else if (op_support->npu_type_a == rknpu2_configuration::NPU_TYPE_INT8) type_size_packed = 1;
            else type_size_packed = 0; // INT4

            size_t current_offset_in_tensor = 0;
            for (const auto& seg : all_segments) {
                for (size_t i = 0; i < num_active_segments; ++i) {
                    if (active_segments[i].offset_n == seg.offset_n) {
                        auto& matmul_ctx = matmul_ctxs[i];
                        size_t segment_size_bytes = matmul_ctx->io_attr.B.size;
                        size_t total_offset = src0_base_offset_in_dma + current_offset_in_tensor;

                        if (total_offset + segment_size_bytes > src0_buf_ctx->dma_buf.size) {
                            GGML_LOG_ERROR("[%s] B segment out of bounds: offset %zu + size %zu > buffer size %zu\n",
                                           __func__, total_offset, segment_size_bytes, src0_buf_ctx->dma_buf.size);
                            return GGML_STATUS_FAILED;
                        }

                        auto cache_key = std::make_tuple(src0_buf_ctx->dma_buf.fd, total_offset, segment_size_bytes);
                        std::lock_guard<std::mutex> lock(backend_ctx->mutex);
                        auto it = backend_ctx->b_mem_handle_cache.find(cache_key);

                        if (it != backend_ctx->b_mem_handle_cache.end()) {
                            mem_B_segments[i] = it->second;
                        } else {
                            rknn_tensor_mem* mem = rknn_create_mem_from_fd(matmul_ctx->ctx, src0_buf_ctx->dma_buf.fd, src0_buf_ctx->dma_buf.virt_addr, segment_size_bytes, total_offset);
                            if (!mem) return GGML_STATUS_FAILED;

                            auto deleter = [matmul_ctx](rknn_tensor_mem* m) { if (m) rknn_destroy_mem(matmul_ctx->ctx, m); };
                            mem_B_segments[i] = std::shared_ptr<rknn_tensor_mem>(mem, deleter);
                            backend_ctx->b_mem_handle_cache[cache_key] = mem_B_segments[i];
                        }
                        RKNN_CHECK(rknn_matmul_set_io_mem(matmul_ctx->ctx, mem_B_segments[i].get(), &matmul_ctx->io_attr.B), "set_io_mem B segment");
                        break;
                    }
                }
                current_offset_in_tensor += type_size_packed > 0 ? (size_t)seg.size_n * K_op * type_size_packed : (size_t)seg.size_n * K_op / 2;
            }
        }

        // ===========================================
        // ========== 3. Preparing A-matrix ==========
        // ===========================================
        std::vector<float> scales_A(M);
        float scale_B = 1.0f;
        {
            auto cache_key = std::make_tuple(M, K, (int)w_type);
            auto& matmul_ctx_0 = matmul_ctxs[0];
            mem_A_shared = get_or_create_npu_buffer(backend_ctx, matmul_ctx_0, matmul_ctx_0->io_attr.A.size, cache_key, backend_ctx->a_buffer_cache);
            if (!mem_A_shared) return GGML_STATUS_FAILED;

            const void* src1_data = ggml_rknpu_get_system_ptr(src1);
            if (!src1_data) return GGML_STATUS_FAILED;

            // Sync src1 to CPU if it's coming from RKNPU but was modified
            if (src1->buffer && src1->buffer->iface.get_base == ggml_backend_rknpu_buffer_get_base) {
                auto * src1_buf_ctx = (ggml_backend_rknpu_buffer_context*)src1->buffer->context;
                // Since RKNPU memory is DMA, we might need a sync from device to ensure CPU sees it
                // Actually rknpu_get_system_ptr just returns the virtual address.
                // We should sync BEFORE reading it on CPU if it was written by NPU.
                rknn_tensor_mem mem = {};
                mem.virt_addr = src1_buf_ctx->dma_buf.virt_addr;
                mem.fd = src1_buf_ctx->dma_buf.fd;
                mem.size = src1_buf_ctx->dma_buf.size;
                RKNN_CHECK(rknn_mem_sync(get_rknpu_memory_context().get_ctx(), &mem, RKNN_MEMORY_SYNC_FROM_DEVICE), "sync src1 FROM_DEVICE");
            }

            const int row_stride = (int)(src1->nb[1] / ggml_element_size(src1));
            void* dst_base = mem_A_shared->virt_addr;

            switch (op_support->npu_type_a) {
                case rknpu2_configuration::NPU_TYPE_FP16: {
                    uint16_t* dst_ptr = (uint16_t*)dst_base;
                    #pragma omp parallel for
                    for (int m = 0; m < M; ++m) {
                        if (src1->type == GGML_TYPE_F32) {
                            const float* src_row = (const float*)src1_data + (size_t)m * row_stride;
                            uint16_t* dst_row = dst_ptr + (size_t)m * K;
                            rknpu2_quantization::convert_fp32_to_fp16(src_row, dst_row, K);
                        } else if (src1->type == GGML_TYPE_F16) {
                            const uint16_t* src_row = (const uint16_t*)src1_data + (size_t)m * row_stride;
                            uint16_t* dst_row = dst_ptr + (size_t)m * K;
                            memcpy(dst_row, src_row, K * sizeof(uint16_t));
                        }
                    }
                    break;
                }

                case rknpu2_configuration::NPU_TYPE_INT8: {
                    int8_t* dst_ptr = (int8_t*)dst_base;
                    #pragma omp parallel for
                    for (int m = 0; m < M; ++m) {
                        float amax_m = 0.0f;
                        int8_t* dst_row = dst_ptr + (size_t)m * K;

                        if (src1->type == GGML_TYPE_F32) {
                            const float* src_row = (const float*)src1_data + (size_t)m * row_stride;
                            for (int k = 0; k < K; ++k) amax_m = std::max(amax_m, std::abs(src_row[k]));
                            scales_A[m] = amax_m / 127.0f;
                            rknpu2_quantization::quantize_fp32_to_int8(src_row, dst_row, K, scales_A[m]);
                        } else if (src1->type == GGML_TYPE_F16) {
                            const uint16_t* src_row = (const uint16_t*)src1_data + (size_t)m * row_stride;
                            std::vector<float> tmp_row(K);
                            rknpu2_quantization::convert_fp16_to_fp32(src_row, tmp_row.data(), K);
                            for (int k = 0; k < K; ++k) amax_m = std::max(amax_m, std::abs(tmp_row[k]));
                            scales_A[m] = amax_m / 127.0f;
                            rknpu2_quantization::quantize_fp32_to_int8(tmp_row.data(), dst_row, K, scales_A[m]);
                        }
                    }
                    break;
                }

                case rknpu2_configuration::NPU_TYPE_INT4: {
                    uint8_t* dst_ptr = (uint8_t*)dst_base;
                    ggml_backend_buffer_t src0_buffer = src0->buffer;
                    auto* src0_buf_ctx = (ggml_backend_rknpu_buffer_context*)src0_buffer->context;
                    std::vector<float> s_vec = get_hadamard_s_vector(src0_buf_ctx, src0, K_op);

                    #pragma omp parallel
                    {
                        std::vector<float> tmp_row(K_op, 0.0f);
                        #pragma omp for
                        for (int m = 0; m < M; ++m) {
                            if (src1->type == GGML_TYPE_F32) {
                                const float* src_row = (const float*)src1_data + (size_t)m * row_stride;
                                memset(tmp_row.data(), 0, K_op * sizeof(float));
                                memcpy(tmp_row.data(), src_row, K * sizeof(float));
                            } else if (src1->type == GGML_TYPE_F16) {
                                const uint16_t* src_row = (const uint16_t*)src1_data + (size_t)m * row_stride;
                                memset(tmp_row.data(), 0, K_op * sizeof(float));
                                rknpu2_quantization::convert_fp16_to_fp32(src_row, tmp_row.data(), K);
                            }
                            
                            // Multiply by random sign vector
                            for (int k = 0; k < K_op; ++k) tmp_row[k] *= s_vec[k];
                            // Fast Walsh-Hadamard Transform
                            rknpu2_calibration::hadamard_transform(tmp_row.data(), tmp_row.data(), K, K_op);

                            float amax_m = 0.0f;
                            for (int k = 0; k < K_op; ++k) amax_m = std::max(amax_m, std::abs(tmp_row[k]));
                            scales_A[m] = amax_m / 7.0f;

                            uint8_t* dst_row = dst_ptr + (size_t)m * (K_op / 2);
                            rknpu2_quantization::quantize_fp32_to_int4_packed(tmp_row.data(), dst_row, K_op, scales_A[m]);
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
                scale_B = get_quantized_scale(src0_buf_ctx, src0);
            }

            RKNN_CHECK(rknn_mem_sync(matmul_ctxs[0]->ctx, mem_A_shared.get(), RKNN_MEMORY_SYNC_TO_DEVICE), "sync A TO_DEVICE");

            for (size_t i = 0; i < num_active_segments; i++) {
                RKNN_CHECK(rknn_matmul_set_io_mem(matmul_ctxs[i]->ctx, mem_A_shared.get(), &matmul_ctxs[i]->io_attr.A), "set_io_mem A for core");
            }
        }

        // ===========================================
        // ========== 4. Preparing C-matrix ==========
        // ===========================================
        // LOG_DBG("[%s] Node %d Step 4: Preparing C-matrix\n", __func__, i);
        {
            for (size_t i = 0; i < num_active_segments; i++) {
                auto& matmul_ctx = matmul_ctxs[i];
                auto& seg = active_segments[i];
                auto cache_key = std::make_tuple(M, seg.size_n, (int)w_type);

                mem_C_segments[i] = get_or_create_npu_buffer(backend_ctx, matmul_ctx, matmul_ctx->io_attr.C.size, cache_key, backend_ctx->c_buffer_cache);
                if (!mem_C_segments[i]) return GGML_STATUS_FAILED;
                RKNN_CHECK(rknn_matmul_set_io_mem(matmul_ctx->ctx, mem_C_segments[i].get(), &matmul_ctx->io_attr.C), "set_io_mem C segment");
            }
        }

        // ==========================================
        // ========== 5. Running operation ==========
        // ==========================================
        // LOG_DBG("[%s] Node %d Step 5: Running operation\n", __func__, i);
        {
            #pragma omp parallel for num_threads(num_active_segments)
            for (size_t i = 0; i < num_active_segments; i++) {
                int ret = rknn_matmul_run(matmul_ctxs[i]->ctx);
                if (ret != RKNN_SUCC) {
                    // Handle error
                }
            }
        }

        // ===========================================
        // ========== 6. Collecting results ==========
        // ===========================================
        {
            float* dst_data = (float*)dst->data;

            for (size_t i = 0; i < num_active_segments; i++) {
                RKNN_CHECK(rknn_mem_sync(matmul_ctxs[i]->ctx, mem_C_segments[i].get(), RKNN_MEMORY_SYNC_FROM_DEVICE), "sync C FROM_DEVICE");
            }

            #pragma omp parallel for
            for (int m = 0; m < M; m++) {
                switch (op_support->npu_type_c) {
                    case rknpu2_configuration::NPU_TYPE_FP32: {
                        for (size_t i = 0; i < num_active_segments; i++) {
                            int N_offset = active_segments[i].offset_n;
                            int N_segment = active_segments[i].size_n;
                            float* src_segment_base = (float*)mem_C_segments[i]->virt_addr;
                            memcpy(dst_data + (size_t)m * N + N_offset,
                                src_segment_base + (size_t)m * N_segment,
                                N_segment * sizeof(float));
                        }
                        break;
                    }

                    case rknpu2_configuration::NPU_TYPE_INT32: {
                        float dequant_scale = scales_A[m] * scale_B;
                        if (op_support->npu_type_a == rknpu2_configuration::NPU_TYPE_INT4) {
                            dequant_scale /= (float)K_op;
                        }

                        for (size_t i = 0; i < num_active_segments; i++) {
                            int N_offset = active_segments[i].offset_n;
                            int N_segment = active_segments[i].size_n;
                            float* dst_ptr = dst_data + (size_t)m * N + N_offset;
                            int32_t* src_segment_base = (int32_t*)mem_C_segments[i]->virt_addr;
                            int32_t* src_ptr = src_segment_base + (size_t)m * N_segment;
                            rknpu2_quantization::dequantize_int32_to_fp32(src_ptr, dst_ptr, N_segment, dequant_scale);
                        }
                        break;
                    }

                    case rknpu2_configuration::NPU_TYPE_INT16: {
                        float dequant_scale = scales_A[m] * scale_B;
                        if (op_support->npu_type_a == rknpu2_configuration::NPU_TYPE_INT4) {
                            dequant_scale /= (float)K_op;
                        }

                        for (size_t i = 0; i < num_active_segments; i++) {
                            int N_offset = active_segments[i].offset_n;
                            int N_segment = active_segments[i].size_n;
                            float* dst_ptr = dst_data + (size_t)m * N + N_offset;
                            int16_t* src_segment_base = (int16_t*)mem_C_segments[i]->virt_addr;
                            int16_t* src_ptr = src_segment_base + (size_t)m * N_segment;
                            rknpu2_quantization::dequantize_int16_to_fp32(src_ptr, dst_ptr, N_segment, dequant_scale);
                        }
                        break;
                    }

                    default:
                        // This should not be reached if config is correct
                        break;
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
    ggml_backend_rknpu_buffer_context * ctx = (ggml_backend_rknpu_buffer_context *)buffer->context;
    return ctx->dma_buf.virt_addr;
}

static enum ggml_status ggml_backend_rknpu_buffer_init_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor) {
    ggml_backend_rknpu_buffer_context * ctx = (ggml_backend_rknpu_buffer_context *)buffer->context;
    if (tensor->view_src != nullptr) {
        tensor->data = (uint8_t *)tensor->view_src->data + tensor->view_offs;
        return GGML_STATUS_SUCCESS;
    }
    tensor->data = (uint8_t *)ctx->dma_buf.virt_addr + ctx->base_ptr_offset;
    ctx->base_ptr_offset += ggml_nbytes(tensor);
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_rknpu_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    auto * ctx = (ggml_backend_rknpu_buffer_context *) buffer->context;
    if (ctx == nullptr) {
        GGML_LOG_ERROR("[%s] buffer context is null!\n", __func__);
        return;
    }

    uint8_t* tensor_dma_ptr = (uint8_t*)ggml_rknpu_get_system_ptr(tensor);
    if (!tensor_dma_ptr) {
        return;
    }

    // Safety check against DMA buffer bounds
    uint8_t* dma_base = (uint8_t*)ctx->dma_buf.virt_addr;
    if (tensor_dma_ptr < dma_base || tensor_dma_ptr + offset + size > dma_base + ctx->dma_buf.size) {
        GGML_LOG_ERROR("[%s] out of bounds: ptr=%p, offset=%zu, size=%zu, dma_base=%p, dma_size=%zu\n",
                       __func__, (void*)tensor_dma_ptr, offset, size, (void*)dma_base, ctx->dma_buf.size);
        return;
    }
    // GGML_LOG_INFO("[%s] tensor_dma_ptr: %p, size: %zu, dma_buf.size: %zu\n", __func__, (void*)tensor_dma_ptr, size, ctx->dma_buf.size);

    // Getting the current device configuration to drive the packing logic
    auto& config_manager = rknpu2_configuration::Rknpu2ConfigManager::get_instance();
    const auto& config = config_manager.get_current_config();
    if (config.device_name == "NONE") {
        GGML_LOG_ERROR("[%s] No device configuration found!\n", __func__);
        return;
    }
    // If there is a specific packing function defined for this tensor type, it's a weight matrix
    // We assume that the packing is the same regardless of activation type (true for currently supported RK3588 ops)
    const auto* op_support = config.find_op_support(tensor->type, GGML_TYPE_F32);
    if (!op_support) op_support = config.find_op_support(tensor->type, GGML_TYPE_F16);

    if (op_support && op_support->pack_func) {
        const int K = (int)tensor->ne[0];
        const int N = (int)tensor->ne[1];

        // Mark as packed (use base tensor for the status)
        {
            const struct ggml_tensor * base = tensor;
            while (base->view_src != nullptr) base = base->view_src;
            std::lock_guard<std::mutex> lock(ctx->mutex);
            ctx->packed_tensors.insert(base);
        }

        if (tensor->type == GGML_TYPE_F16) {
            auto segments = compute_matrix_segments(N, config.core_count, op_support->n_align);

            uint8_t* current_write_ptr = tensor_dma_ptr;
            std::vector<uint8_t> packed_data_temp;

            for (const auto& seg : segments) {
                if (seg.size_n == 0) continue;
                size_t segment_packed_size_bytes = (size_t)seg.size_n * K * sizeof(uint16_t);
                packed_data_temp.resize(segment_packed_size_bytes);

                op_support->pack_func(
                    packed_data_temp.data(),
                    (const uint8_t*)data,
                    K, N, seg.offset_n, seg.size_n
                );
                memcpy(current_write_ptr, packed_data_temp.data(), segment_packed_size_bytes);
                current_write_ptr += segment_packed_size_bytes;
            }
        // All quantized types mapping to FP16
        } else if (op_support->npu_type_a == rknpu2_configuration::NPU_TYPE_FP16 && ggml_is_quantized(tensor->type)) {
            const size_t n_elements = (size_t)K * N;
            auto traits = ggml_get_type_traits(tensor->type);
            const size_t row_size_bytes = ggml_row_size(tensor->type, K);

            // Dequantizing directly row-by-row to FP16
            std::vector<uint16_t> fp16_data(n_elements);
            #pragma omp parallel
            {
                std::vector<float> tmp_row(K);
                #pragma omp for
                for (int n = 0; n < N; ++n) {
                    traits->to_float((const uint8_t*)data + (size_t)n * row_size_bytes, tmp_row.data(), K);
                    rknpu2_quantization::convert_fp32_to_fp16(tmp_row.data(), fp16_data.data() + (size_t)n * K, K);
                }
            }

            // Packing into native format by segments
            auto segments = compute_matrix_segments(N, config.core_count, op_support->n_align);
            uint8_t* current_write_ptr = tensor_dma_ptr;
            std::vector<uint8_t> packed_data_temp;

            for (const auto& seg : segments) {
                if (seg.size_n == 0) continue;
                size_t segment_packed_size = (size_t)seg.size_n * K * sizeof(uint16_t);
                packed_data_temp.resize(segment_packed_size);

                op_support->pack_func(
                    packed_data_temp.data(),
                    (const uint8_t*)fp16_data.data(),
                    K, N, seg.offset_n, seg.size_n
                );
                memcpy(current_write_ptr, packed_data_temp.data(), segment_packed_size);
                current_write_ptr += segment_packed_size;
            }
        // Native NPU INT8 for Q8_0
        } else if (tensor->type == GGML_TYPE_Q8_0 && op_support->npu_type_a == rknpu2_configuration::NPU_TYPE_INT8) {
            const block_q8_0* src_blocks = (const block_q8_0*)data;
            const size_t n_elements = (size_t)K * N;

            // Finding the global scale
            float amax = 0.0f;
            #pragma omp parallel
            {
                std::vector<float> tmp_row(K);
                float amax_local = 0.0f;
                #pragma omp for
                for (int n = 0; n < N; ++n) {
                    dequantize_row_q8_0(src_blocks + (size_t)n * (K / QK8_0), tmp_row.data(), K);
                    for (int k = 0; k < K; ++k) amax_local = std::max(amax_local, std::abs(tmp_row[k]));
                }
                #pragma omp critical
                {
                    amax = std::max(amax, amax_local);
                }
            }

            const float global_scale_b = amax / 127.0f;

            // Storing it in the buffer context cache
            {
                std::lock_guard<std::mutex> lock(ctx->mutex);
                ctx->quantized_tensor_scales[tensor] = global_scale_b;
            }

            // Dequantizing and re-quantizing directly row-by-row
            std::vector<int8_t> requantized_data(n_elements);
            #pragma omp parallel
            {
                std::vector<float> tmp_row(K);
                #pragma omp for
                for (int n = 0; n < N; ++n) {
                    dequantize_row_q8_0(src_blocks + (size_t)n * (K / QK8_0), tmp_row.data(), K);
                    rknpu2_quantization::quantize_fp32_to_int8(tmp_row.data(), requantized_data.data() + (size_t)n * K, K, global_scale_b);
                }
            }

            // Packing into native format by segments
            auto segments = compute_matrix_segments(N, config.core_count, op_support->n_align);
            uint8_t* current_write_ptr = tensor_dma_ptr;
            std::vector<uint8_t> packed_data_temp;

            for (const auto& seg : segments) {
                if (seg.size_n == 0) continue;
                size_t segment_packed_size = (size_t)seg.size_n * K;
                packed_data_temp.resize(segment_packed_size);

                op_support->pack_func(
                    packed_data_temp.data(),
                    (const uint8_t*)requantized_data.data(),
                    K, N, seg.offset_n, seg.size_n
                );
                memcpy(current_write_ptr, packed_data_temp.data(), segment_packed_size);
                current_write_ptr += segment_packed_size;
            }
        // Native NPU INT4 for Q4_0 with Hadamard
        } else if (tensor->type == GGML_TYPE_Q4_0 && op_support->npu_type_a == rknpu2_configuration::NPU_TYPE_INT4) {
             const block_q4_0* src_blocks = (const block_q4_0*)data;

            // Dequantizing original weights to FP32
            const size_t n_elements_original = (size_t)K * N;
            std::vector<float> dequantized_data(n_elements_original);
            #pragma omp parallel for
            for (int n = 0; n < N; ++n) {
                dequantize_row_q4_0(src_blocks + (size_t)n * (K / QK4_0), dequantized_data.data() + (size_t)n * K, K);
            }

            // Padding K and generating the random sign vector 's'
            const int padded_K = rknpu2_calibration::next_power_of_two(K);
            std::vector<float> s_vec = get_hadamard_s_vector(ctx, tensor, padded_K);

            // Applying Hadamard Transform
            std::vector<float> hadamard_data((size_t)padded_K * N, 0.0f);
            #pragma omp parallel
            {
                std::vector<float> tmp_row(padded_K, 0.0f);
                #pragma omp for
                for (int n = 0; n < N; ++n) {
                    memset(tmp_row.data(), 0, padded_K * sizeof(float));
                    memcpy(tmp_row.data(), dequantized_data.data() + (size_t)n * K, K * sizeof(float));
                    // Multiply by random sign vector
                    for (int k = 0; k < padded_K; ++k) tmp_row[k] *= s_vec[k];
                    // Fast Walsh-Hadamard Transform
                    rknpu2_calibration::hadamard_transform(tmp_row.data(), tmp_row.data(), K, padded_K);
                    memcpy(hadamard_data.data() + (size_t)n * padded_K, tmp_row.data(), padded_K * sizeof(float));
                }
            }

            // Finding optimal scale using entropy-based calibration (KL-divergence)
            const float amax = rknpu2_calibration::calculate_entropy_amax(hadamard_data.data(), (size_t)padded_K * N);
            const float global_scale_b = amax / 7.0f;

            // Storing the scale
            {
                std::lock_guard<std::mutex> lock(ctx->mutex);
                ctx->quantized_tensor_scales[tensor] = global_scale_b;
            }

            // Quantizing to INT4
            std::vector<uint8_t> quantized_data((size_t)padded_K * N / 2);
            #pragma omp parallel for
            for (int n = 0; n < N; ++n) {
                rknpu2_quantization::quantize_fp32_to_int4_packed(hadamard_data.data() + (size_t)n * padded_K, quantized_data.data() + (size_t)n * padded_K / 2, padded_K, global_scale_b);
            }

            // Packing into native format
            auto segments = compute_matrix_segments(N, config.core_count, op_support->n_align);
            uint8_t* current_write_ptr = tensor_dma_ptr;
            std::vector<uint8_t> packed_data_temp;

            for (const auto& seg : segments) {
                if (seg.size_n == 0) continue;
                size_t segment_packed_size = (size_t)seg.size_n * padded_K / 2;
                packed_data_temp.resize(segment_packed_size);

                op_support->pack_func(
                    packed_data_temp.data(),
                    (const uint8_t*)quantized_data.data(),
                    padded_K, N, seg.offset_n, seg.size_n
                );
                memcpy(current_write_ptr, packed_data_temp.data(), segment_packed_size);
                current_write_ptr += segment_packed_size;
            }
        }
    // Other tensor types
    } else {
        memcpy(tensor_dma_ptr + offset, data, size);
    }
}

static void ggml_backend_rknpu_buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    UNUSED(buffer);
    uint8_t* tensor_dma_ptr = (uint8_t*)ggml_rknpu_get_system_ptr(tensor);
    if (!tensor_dma_ptr) {
        return;
    }
    memcpy(data, tensor_dma_ptr + offset, size);
}

static void ggml_backend_rknpu_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_rknpu_buffer_context * ctx = (ggml_backend_rknpu_buffer_context *)buffer->context;
    memset(ctx->dma_buf.virt_addr, value, ctx->dma_buf.size);
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
    if (dma_buf.fd < 0) {
        return NULL;
    }

    ggml_backend_rknpu_buffer_context * ctx = new ggml_backend_rknpu_buffer_context;
    ctx->dma_buf = dma_buf;
    ctx->name = "rknpu_dma_buffer";
    ctx->base_ptr = dma_buf.virt_addr;
    ctx->base_ptr_offset = 0;

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
    return 4096;
}

static size_t ggml_backend_rknpu_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor) {
    UNUSED(buft);

    if (tensor->view_src != nullptr) {
        return ggml_nbytes(tensor);
    }

    // Getting the current device configuration
    const auto& config = rknpu2_configuration::Rknpu2ConfigManager::get_instance().get_current_config();
    const auto* op_support = config.find_op_support(tensor->type, GGML_TYPE_F32);
    if (!op_support) op_support = config.find_op_support(tensor->type, GGML_TYPE_F16);

    if (op_support && op_support->pack_func) {
        const int K = (int)tensor->ne[0];
        const int N = (int)tensor->ne[1];

        auto segments = compute_matrix_segments(N, config.core_count, op_support->n_align);
        size_t total_size = 0;
        for (const auto& seg : segments) {
            if (seg.size_n > 0) {
                if (op_support->mm_type == RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32) {
                    total_size += (size_t)seg.size_n * K * 2;
                } else if (op_support->npu_type_a == rknpu2_configuration::NPU_TYPE_INT8) {
                    total_size += (size_t)seg.size_n * K;
                } else if (op_support->npu_type_a == rknpu2_configuration::NPU_TYPE_INT4) {
                    total_size += (size_t)seg.size_n * K / 2;
                }
            }
        }
        return total_size;
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
#ifdef __linux__
    struct sysinfo si;
    if (sysinfo(&si) == 0) {
        *total = (size_t)si.totalram * si.mem_unit;
        *free = (size_t)si.freeram * si.mem_unit;
    } else {
        *free = 0;
        *total = 0;
    }
#else
    *free = 0;
    *total = 0;
#endif
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
            return true;

        case GGML_OP_MUL_MAT: {
            const struct ggml_tensor * src0 = op->src[0]; // Weights
            const struct ggml_tensor * src1 = op->src[1]; // Activations

            // Finding if there is a supported operation for the given weight type and activation type
            bool found_op = false;
            for (const auto& rk_op : config.supported_ops) {
                if (rk_op.type_w == src0->type && rk_op.type_a == src1->type) {
                    // Checking for K alignment
                    if (src0->ne[0] % rk_op.k_align != 0) {
                        continue;
                    }

                    // Checking for N alignment
                    if (src0->ne[1] % rk_op.n_align != 0) {
                        continue;
                    }

                    found_op = true;
                    break;
                }
            }

            if (!found_op) {
                return false;
            }

            // Checking for exact dimensions
            if (src1->ne[0] != src0->ne[0]) {
                 return false;
            }

            // Checking contiguous memory
            // Note: src1 might be a view (KV cache), so we check the rows specifically during execution.
            // But for RKNPU, we currently require contiguous memory for the whole tensor if we offload it.
            // Actually, MUL_MAT on RKNPU works on M x K activations.
            // If it's a view, ggml_is_contiguous(src1) might be false even if it's "mostly" contiguous.
            // Let's allow non-contiguous src1 and handle it in graph_compute.
            if (!ggml_is_contiguous(src0)) {
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
