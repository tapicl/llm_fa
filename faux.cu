// fa4mimic_v2/faux_attn.cu — STEP 2 of the ladder: FAUX attention.
//
// FAUX (per memory feedback_faux_attention_definition.md):
//   TMA loads -> QK MMA -> cvt(S fp32 -> bf16) -> STTM-P -> PV MMA
//   -> LDTM-O fp32 -> cvt(no scale) -> STSM-O to sO -> TMA-store sO -> gmem
//
// NO softmax math (no rowmax, no scale_log2 fma, no exp2, no row_sum, no sScale)
// NO correction rescale (no FMUL2 on O accumulator)
// NO final 1/row_sum scale
//
// What CHANGES vs tma_mma.cu (minimal):
//   1. SM warps: add LD-S (32x32b.x32 x 4 chunks = 128 fp32/lane), cvt to bf16x2
//      packed (64 b32/lane, NO scale subtract), STTM-P (32x32b.x32 x 2 chunks).
//   2. COR warps: after mainloop, per pair wait o_acc_full, LDTM-O 128 fp32/lane
//      via 4 chunks, cvt fp32->bf16x2 (NO multiply by inv_l), sts.v4.b32 into sO,
//      fence.proxy.async.shared, arrive o_epi_full.
//   3. EPI warp: per pair wait o_epi_full, cp.async.bulk.tensor.2d.global.shared
//      from sO to gO, commit + wait_group.
//   4. Launcher: allocate real O tensor, encode O_tmap (SWIZZLE_NONE 2D), pass
//      O_gmem ptr and O_tmap into kernel.
//
// SMEM layout / mbar count / warp roles / MMA issue order / TMA swizzle for
// Q/K/V all match tma_mma.cu (= FA4).
//
// All experiment flags removed: no toggles, single canonical path.

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda/barrier>
#include <torch/extension.h>
#include <vector>

#ifndef ENABLE_PROF
#define ENABLE_PROF 0
#endif
#if ENABLE_PROF
// 128 slots × 16 warps × 12 B = 24 KB static SMEM; leaves room for 232 KB
// dynamic on the 256 KB SM cap. Trace shapes used by the dumper (n_kv ≤ ~30)
// emit < 50 events per warp, well under this bound.
#define PROF_PER_WARP_SLOTS 128
#include "profiler.cuh"
#else
// No-op PROF macros for non-trace builds.
struct ProfShRec { uint64_t _; };
struct ProfEvent { uint32_t _; };
#define PROF_BIND_SM(x) (void)(x)
#define PROF_INIT_BASELINE() do {} while (0)
#define PROF_BEGIN_R(name) do {} while (0)
#define PROF_END_R(name, evid) do {} while (0)
#define PROF_BEGIN_X(evid) do {} while (0)
#define PROF_END_X(evid) do {} while (0)
#define PROF_FLUSH(p, c) do {} while (0)
constexpr int PROF_MAX_EVENTS = 1;
#endif

enum : int {
    EV_TMA_Q       = 0,
    EV_TMA_WAIT_E  = 1,   // TMA wait on kv_empty (slot reuse)
    EV_TMA_K       = 2,
    EV_TMA_V       = 3,
    EV_MMA_WAIT_Q  = 4,
    EV_MMA_WAIT_K  = 5,
    EV_MMA_WAIT_V  = 6,
    EV_MMA_QK_P0   = 7,
    EV_MMA_QK_P1   = 8,
    EV_MMA_PV_P0   = 9,
    EV_MMA_PV_P1   = 10,
    EV_MMA_WAIT_SPE0 = 11, // unified gate: wait s_p_o_empty(0) before PV(0)
    EV_MMA_WAIT_SPE1 = 12, // unified gate: wait s_p_o_empty(1) before PV(1)
    EV_SM_WAIT_S   = 13,
    EV_SM_RM       = 14,  // LD-S + rowmax + acc_scale + STS sScale + cvt phase 0 + STTM#0 + bar.arrive sm_stats
    EV_COR_WAIT_O  = 16,  // COR pair-0 region (wait o_acc_full(0) or warmup → rescale → arrive)
    EV_EPI_WAIT    = 17,  // EPI wait o_epi_full (both pairs)
    EV_COR_WAIT_O1 = 18,  // COR pair-1 region
    EV_SM_STP      = 19,  // cvt phase 1 + STTM#1 + update_row_sum + arrive p_lastsplit + s_p_o_empty
};

// ============================================================================
// FA4-precise constants (matches flash_fwd_sm100.py for D=128, BF16, single-head)
// ============================================================================
constexpr int M_PER_CTA      = 128;
constexpr int M_JOINT        = M_PER_CTA * 2;       // 256: cga2 MMA tiler M
constexpr int N_M_PAIRS      = 2;                   // FA4 q_stage=2
constexpr int M_PER_CLUSTER  = M_JOINT * N_M_PAIRS; // 512 rows per cga2 cluster
constexpr int N_DIM          = 128;
constexpr int BLOCK_K        = 128;
constexpr int MMA_K          = 16;
constexpr int CTA_GROUP      = 2;
constexpr int NUM_WARPS      = 16;
constexpr int TB_SIZE        = NUM_WARPS * 32;

constexpr int Q_STAGE  = 2;
constexpr int KV_STAGE = 6;
constexpr int S_STAGE  = 2;
constexpr int N_KV_PAIRS_INFLIGHT = KV_STAGE / 2;

constexpr int Q_BYTES_PER_STAGE  = M_PER_CTA * N_DIM * 2;
constexpr int Q_BYTES_TOTAL      = Q_STAGE * Q_BYTES_PER_STAGE;
#if ENABLE_PROF
// Trace mode: drop sO scratch to make room for the per-warp profiler shmem
// rings. EPI TMA-store is gated off under ENABLE_PROF below; output O will be
// garbage but the timing capture is what we need for the .pftrace.
constexpr int O_BYTES_PER_STAGE  = 0;
constexpr int O_BYTES_TOTAL      = 0;
#else
constexpr int O_BYTES_PER_STAGE  = M_PER_CTA * N_DIM * 2;
constexpr int O_BYTES_TOTAL      = Q_STAGE * O_BYTES_PER_STAGE;
#endif
constexpr int KV_BYTES_PER_STAGE = (BLOCK_K * N_DIM * 2) / CTA_GROUP;
constexpr int KV_BYTES_TOTAL     = KV_STAGE * KV_BYTES_PER_STAGE;
constexpr int SCALE_BYTES_TOTAL  = Q_STAGE * M_PER_CTA * 2 * 4;
constexpr int DYN_SMEM_BYTES     = Q_BYTES_TOTAL + O_BYTES_TOTAL +
                                   KV_BYTES_TOTAL + SCALE_BYTES_TOTAL;
static_assert(DYN_SMEM_BYTES <= 232 * 1024, "SMEM cap");

constexpr int SOFTMAX0_LO = 0;             constexpr int SOFTMAX0_HI = 4;
constexpr int SOFTMAX1_LO = 4;             constexpr int SOFTMAX1_HI = 8;
constexpr int CORR_LO     = 8;             constexpr int CORR_HI     = 12;
constexpr int MMA_WARP    = 12;
constexpr int EPI_WARP    = 13;
constexpr int LOAD_WARP   = 14;
constexpr int EMPTY_WARP  = 15;

constexpr int REGS_SOFTMAX = 192;
constexpr int REGS_CORR    = 80;
constexpr int REGS_OTHER   = 48;

constexpr int Sm100MmaPeerBitMask = 0xFEFFFFFF;

// Named-barrier IDs for SM<->COR sm_stats handshake (matches FA4 sm_stats_barrier).
// 8 unique handshakes (q_stage=2 stages x 4 softmax warps per pair).
// IDs 1..8 (id 0 is reserved by hw). Per handshake: 1 SM warp + 1 COR warp = 64 threads.
constexpr int BAR_SM_STATS_BASE  = 1;
constexpr int SM_STATS_BAR_COUNT = 64;

// ============================================================================
// PTX helpers
// ============================================================================
template <int N> __device__ __forceinline__ void smnr_dec() {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" ::"n"(N));
}
template <int N> __device__ __forceinline__ void smnr_inc() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" ::"n"(N));
}
__device__ __forceinline__ uint32_t cluster_ctarank() {
    uint32_t r;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(r));
    return r;
}
__device__ __forceinline__ uint32_t elect_sync_one() {
    uint32_t pred = 0;
    asm volatile("{\n\t.reg .pred %%px;\n\t"
                 "elect.sync _|%%px, %1;\n\t"
                 "@%%px mov.s32 %0, 1;\n\t}"
                 : "+r"(pred) : "r"(0xFFFFFFFFu));
    return pred;
}
__device__ __forceinline__ void mbarrier_init(int m, int c) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(m), "r"(c));
}
__device__ __forceinline__ void mbarrier_wait(int mbar_addr, int phase) {
    uint32_t ticks = 0x989680;
    asm volatile("{\n\t.reg .pred P1;\n\t"
                 "LAB_WAIT:\n\t"
                 "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1, %2;\n\t"
                 "@P1 bra.uni DONE;\n\t"
                 "bra.uni LAB_WAIT;\n\t"
                 "DONE:\n\t}" ::"r"(mbar_addr), "r"(phase), "r"(ticks));
}
// Faster wait for mbars whose producer is expected to already be ready by the
// time we wait. Uses mbarrier.test_wait (no built-in 100-cycle sleep hint that
// try_wait has). Compiles to SYNCS.PHASECHK.TRANS64 (no .TRYWAIT) in SASS.
// Use for MMA's s_p_o_empty waits in post-reorder schedule where the 2-MMA
// overlap window makes SM/COR done before MMA gets here.
__device__ __forceinline__ void mbarrier_wait_fast(int mbar_addr, int phase) {
    asm volatile("{\n\t.reg .pred P1;\n\t"
                 "LAB_WAIT_F:\n\t"
                 "mbarrier.test_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1;\n\t"
                 "@P1 bra.uni DONE_F;\n\t"
                 "bra.uni LAB_WAIT_F;\n\t"
                 "DONE_F:\n\t}" ::"r"(mbar_addr), "r"(phase));
}
__device__ __forceinline__ void mbarrier_arrive(int mbar_addr) {
    asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" ::"r"(mbar_addr) : "memory");
}
// Named-barrier sync (BAR.SYNC.DEFER_BLOCKING in SASS). Matches FA4's
// `sm_stats_barrier`: SM warp fires `arrive`, COR warp `sync` (= arrive_and_wait).
// Fewer SASS instructions than mbar try_wait polling.
__device__ __forceinline__ void named_barrier_arrive(int id, int count) {
    asm volatile("bar.arrive %0, %1;" :: "r"(id), "r"(count));
}
__device__ __forceinline__ void named_barrier_sync(int id, int count) {
    asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(count));
}
__device__ __forceinline__ void tma_3d_local(
    int dst, const void* tmap, int x, int y, int z, int mbar) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global"
        ".mbarrier::complete_tx::bytes.cta_group::1 "
        "[%0], [%1, {%3, %4, %5}], [%2];" ::"r"(dst),
        "l"(tmap), "r"(mbar), "r"(x), "r"(y), "r"(z) : "memory");
}
__device__ __forceinline__ void tma_3d_cga2(
    int dst, const void* tmap, int x, int y, int z, int mbar) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global"
        ".mbarrier::complete_tx::bytes.cta_group::2 "
        "[%0], [%1, {%3, %4, %5}], [%2];" ::"r"(dst),
        "l"(tmap), "r"(mbar), "r"(x), "r"(y), "r"(z) : "memory");
}
// 3D TMA STORE: sO -> gO. Matches FA4's K_SW128 sO atom layout. Validated
// against a known pattern in micro_sw128.cu (LAYOUT=1, 100% correct).
__device__ __forceinline__ void tma_store_3d(int src_smem, const void* tmap, int x, int y, int z) {
    asm volatile(
        "cp.async.bulk.tensor.3d.global.shared::cta.bulk_group "
        "[%0, {%2, %3, %4}], [%1];" ::
        "l"(tmap), "r"(src_smem), "r"(x), "r"(y), "r"(z) : "memory");
}
// K_SW128 atom-based SMEM phys offset for logical (row, col_bf16).
__device__ __forceinline__ int sw128_sO_offset(int row, int col_bf16) {
    const int atom_row     = row >> 3;
    const int atom_col     = col_bf16 >> 6;
    const int row_in_8     = row & 7;
    const int col_in_atom  = (col_bf16 & 63) << 1;       // bytes 0..126
    const int within_atom  = (row_in_8 << 7) + (col_in_atom ^ (row_in_8 << 4));
    return ((atom_row + atom_col * 16) << 10) + within_atom;
}
__device__ __forceinline__ void tma_store_commit() {
    asm volatile("cp.async.bulk.commit_group;");
}
__device__ __forceinline__ void tma_store_wait() {
    asm volatile("cp.async.bulk.wait_group 0;");
}
__device__ __forceinline__ constexpr uint64_t desc_encode(uint64_t x) {
    return (x & 0x3'FFFFULL) >> 4ULL;
}
__device__ __forceinline__ uint64_t make_desc_qk(int addr) {
    constexpr int SBO = 8 * 128;
    return desc_encode((uint64_t)(uint32_t)addr) |
           (desc_encode(SBO) << 32ULL) |
           (1ULL << 46ULL) | (2ULL << 61ULL);
}
__device__ __forceinline__ uint64_t make_desc_v_pv(int addr) {
    constexpr int LBO = 128;
    constexpr int SBO = 8 * 128;
    return desc_encode((uint64_t)(uint32_t)addr) |
           (desc_encode(LBO) << 16ULL) |
           (desc_encode(SBO) << 32ULL) |
           (1ULL << 46ULL) | (2ULL << 61ULL);
}
__device__ __forceinline__ void cga2_mma_ss(
    int taddr, uint64_t a_desc, uint64_t b_desc, uint32_t i_desc, int en_d) {
    asm volatile(
        "{\n\t.reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n\t}" ::
            "r"(taddr), "l"(a_desc), "l"(b_desc), "r"(i_desc), "r"(en_d));
}
__device__ __forceinline__ void cga2_mma_ts(
    int taddr, uint32_t tmem_a, uint64_t b_desc, uint32_t i_desc, int en_d) {
    asm volatile(
        "{\n\t.reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::2.kind::f16 [%0], [%1], %2, %3, p;\n\t}" ::
            "r"(taddr), "r"(tmem_a), "l"(b_desc), "r"(i_desc), "r"(en_d));
}
// NOTE: We tested the FA4/v8 split-desc pattern (HI=0x40004040 baked as PTX
// literal, only LO varies per cga2). It REGRESSED by ~40 µs at q=20480 sk=131072.
// ptxas already optimizes the implicit make_desc form well — the explicit
// mov.b64 {%r, literal} per cga2 ends up costing more than the addr-add it
// replaces. Helped v8 but not this kernel. Keeping the simple make_desc form.

// ============================================================================
// Kernel
// ============================================================================
__global__ void __cluster_dims__(CTA_GROUP, 1, 1) __launch_bounds__(TB_SIZE, 1)
fa4_faux_attn_kernel(
    const __grid_constant__ CUtensorMap Q_tmap,
    const __grid_constant__ CUtensorMap K_tmap,
    const __grid_constant__ CUtensorMap V_tmap,
    const __grid_constant__ CUtensorMap O_tmap,
    int n_kv,
    ProfEvent* prof_buf,
    int* prof_count) {
    const int tid     = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const uint32_t cta_rank = cluster_ctarank();

    const int q_cluster_offset = (int)blockIdx.y * M_PER_CLUSTER;
    const int q_pair_offset[N_M_PAIRS] = {
        q_cluster_offset,
        q_cluster_offset + M_JOINT,
    };

#if ENABLE_PROF
    constexpr int PROF_SM_SIZE = PROF_NUM_WARPS * PROF_PER_WARP_SLOTS;
#else
    constexpr int PROF_SM_SIZE = 1;
#endif
    __shared__ ProfShRec __prof_sm_storage[PROF_SM_SIZE];
    PROF_BIND_SM(__prof_sm_storage);
    PROF_INIT_BASELINE();
    (void)prof_buf; (void)prof_count;

    extern __shared__ __align__(1024) char smem_ptr[];
    const int smem    = (int)__cvta_generic_to_shared(smem_ptr);
    const int sQ_p0   = smem;
    const int sQ_p1   = sQ_p0   + Q_BYTES_PER_STAGE;
    const int sO_p0   = sQ_p1   + Q_BYTES_PER_STAGE;
    const int sO_p1   = sO_p0   + O_BYTES_PER_STAGE;
    const int sKV_base= sO_p1   + O_BYTES_PER_STAGE;
    const int sScale  = sKV_base+ KV_BYTES_TOTAL;
    float* const sScale_ptr =
        reinterpret_cast<float*>(smem_ptr + (Q_BYTES_TOTAL + O_BYTES_TOTAL + KV_BYTES_TOTAL));
#define Q_smem(p)  ((p) == 0 ? sQ_p0 : sQ_p1)
#define O_smem(p)  ((p) == 0 ? sO_p0 : sO_p1)
#define KV_smem(s) (sKV_base + (s) * KV_BYTES_PER_STAGE)

#pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ uint64_t mbars[37];
    __shared__ int tmem_addr_smem[1];
    const int mb = (int)__cvta_generic_to_shared(mbars);
#define q_full(s)              (mb + ( 0 + (s)) * 8)
#define q_empty(s)             (mb + ( 2 + (s)) * 8)
#define kv_full(s)             (mb + ( 4 + (s)) * 8)
#define kv_empty(s)            (mb + (10 + (s)) * 8)
#define s_p_o_full(s)          (mb + (16 + (s)) * 8)
#define s_p_o_empty(s)         (mb + (18 + (s)) * 8)
#define p_lastsplit_full(s)    (mb + (20 + (s)) * 8)
#define p_lastsplit_empty(s)   (mb + (22 + (s)) * 8)
#define o_acc_full(s)          (mb + (24 + (s)) * 8)
#define o_acc_empty(s)         (mb + (26 + (s)) * 8)
#define sm_stats_full(s)       (mb + (28 + (s)) * 8)
#define sm_stats_empty(s)      (mb + (30 + (s)) * 8)
#define o_epi_full(s)          (mb + (32 + (s)) * 8)
#define o_epi_empty(s)         (mb + (34 + (s)) * 8)
#define tmem_dealloc           (mb + 36 * 8)

    if (warp_id == 0 && elect_sync_one()) {
        for (int s = 0; s < Q_STAGE; s++) {
            mbarrier_init(q_full(s),  1);
            mbarrier_init(q_empty(s), 1);
        }
        for (int s = 0; s < KV_STAGE; s++) {
            mbarrier_init(kv_full(s),  CTA_GROUP);
            mbarrier_init(kv_empty(s), 1);
        }
        for (int s = 0; s < Q_STAGE; s++) {
            mbarrier_init(s_p_o_full(s),  1);
            mbarrier_init(s_p_o_empty(s), 8);
        }
        for (int s = 0; s < Q_STAGE; s++) {
            mbarrier_init(p_lastsplit_full(s),  4);
            mbarrier_init(p_lastsplit_empty(s), 1);
        }
        for (int s = 0; s < Q_STAGE; s++) {
            mbarrier_init(o_acc_full(s),  1);
            mbarrier_init(o_acc_empty(s), 4);
        }
        for (int s = 0; s < Q_STAGE; s++) {
            mbarrier_init(sm_stats_full(s),  4);
            mbarrier_init(sm_stats_empty(s), 4);
        }
        for (int s = 0; s < Q_STAGE; s++) {
            mbarrier_init(o_epi_full(s),  4);
            mbarrier_init(o_epi_empty(s), 1);
        }
        mbarrier_init(tmem_dealloc, 1);
        asm volatile("fence.mbarrier_init.release.cluster;");
    } else if (warp_id == 1) {
        const int addr = (int)__cvta_generic_to_shared(tmem_addr_smem);
        asm volatile(
            "tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], 512;" ::"r"(addr));
    }
    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");
    const int taddr = tmem_addr_smem[0];

    const int slot_taddr_p0 = taddr;
    const int slot_taddr_p1 = taddr + 128;
    const int O_acc_p0      = taddr + 256;
    const int O_acc_p1      = taddr + 384;

    constexpr uint16_t cta_mask = (1u << CTA_GROUP) - 1u;

    if (warp_id < SOFTMAX1_HI) {
        smnr_inc<REGS_SOFTMAX>();
    } else if (warp_id < CORR_HI) {
        smnr_dec<REGS_CORR>();
    } else {
        smnr_dec<REGS_OTHER>();
    }

    constexpr uint32_t qk_i_desc =
        (1U << 4U) | (1U << 7U) | (1U << 10U) |
        ((uint32_t)N_DIM >> 3U << 17U) |
        ((uint32_t)M_JOINT >> 4U << 24U);
    static_assert(qk_i_desc == 0x10200490u, "qk_i_desc");
    constexpr uint32_t pv_i_desc =
        (1U << 4U) | (1U << 7U) | (1U << 10U) | (1U << 16U) |
        ((uint32_t)N_DIM >> 3U << 17U) |
        ((uint32_t)M_JOINT >> 4U << 24U);
    static_assert(pv_i_desc == 0x10210490u, "pv_i_desc");

    constexpr int Q_PANEL_OFFSET = M_PER_CTA * 64 * 2;
    constexpr int K_PANEL_OFFSET = (BLOCK_K / CTA_GROUP) * 64 * 2;
    constexpr int V_K_BYTES_PER_KC = MMA_K * (N_DIM / CTA_GROUP) * 2;

    auto issue_qk = [&](int pair, int s_kv) {
        asm volatile("tcgen05.fence::after_thread_sync;");
        PROF_BEGIN_R(qk);
        const int slot_taddr_pair = (pair == 0) ? slot_taddr_p0 : slot_taddr_p1;
        const int Q_smem_pair     = Q_smem(pair);
        for (int k2 = 0; k2 < 4; k2++) {
            cga2_mma_ss(slot_taddr_pair,
                        make_desc_qk(Q_smem_pair + k2 * 32),
                        make_desc_qk(KV_smem(s_kv) + k2 * 32),
                        qk_i_desc, k2 == 0 ? 0 : 1);
        }
        for (int k2 = 0; k2 < 4; k2++) {
            cga2_mma_ss(slot_taddr_pair,
                        make_desc_qk(Q_smem_pair + Q_PANEL_OFFSET + k2 * 32),
                        make_desc_qk(KV_smem(s_kv) + K_PANEL_OFFSET + k2 * 32),
                        qk_i_desc, 1);
        }
        asm volatile(
            "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster"
            ".multicast::cluster.b64 [%0], %1;" ::
                "r"(s_p_o_full(pair)), "h"(cta_mask) : "memory");
        PROF_END_R(qk, pair == 0 ? EV_MMA_QK_P0 : EV_MMA_QK_P1);
    };
    auto issue_pv = [&](int pair, int s_v, bool first_for_O) {
        asm volatile("tcgen05.fence::after_thread_sync;");
        PROF_BEGIN_R(pv);
        const int slot_taddr_pair = (pair == 0) ? slot_taddr_p0 : slot_taddr_p1;
        const int O_acc_pair      = (pair == 0) ? O_acc_p0      : O_acc_p1;
        const uint64_t v_desc_base = make_desc_v_pv(KV_smem(s_v));
        constexpr uint64_t v_desc_step = (uint64_t)V_K_BYTES_PER_KC >> 4ULL;
        const int p_base = slot_taddr_pair + 64;
        constexpr int p_step = MMA_K / 2;
        for (int kc = 0; kc < BLOCK_K / MMA_K; kc++) {
            cga2_mma_ts(O_acc_pair,
                        p_base + kc * p_step,
                        v_desc_base + (uint64_t)kc * v_desc_step,
                        pv_i_desc,
                        (first_for_O && kc == 0) ? 0 : 1);
        }
        asm volatile(
            "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster"
            ".multicast::cluster.b64 [%0], %1;" ::
                "r"(o_acc_full(pair)), "h"(cta_mask) : "memory");
        PROF_END_R(pv, pair == 0 ? EV_MMA_PV_P0 : EV_MMA_PV_P1);
    };

    // ===================================================================
    // Warp dispatch
    // ===================================================================

    if (warp_id < SOFTMAX1_HI) {
        // SOFTMAX (warps 0-7) — FAUX+CORR:
        //   LD-S + row_max + acc_scale = exp2((prev - new) * LOG2_E)
        //        + STS sScale + cvt-only + STTM-P + update_row_sum (cost-fairness).
        const bool is_p0 = (warp_id < SOFTMAX1_LO);
        const int pair   = is_p0 ? 0 : 1;
        const int slot_taddr_pair = is_p0 ? slot_taddr_p0 : slot_taddr_p1;
        const int warp_id_local   = is_p0 ? warp_id : warp_id - SOFTMAX1_LO;
        const int row_addr        = slot_taddr_pair + ((warp_id_local * 32) << 16);
        const int my_sScale_off   = pair * M_PER_CTA + warp_id_local * 32 + lane_id;
        // FA4 default softmax_scale = 1/sqrt(D); scale_log2 = softmax_scale * LOG2_E.
        // For D=128: LOG2_E / sqrt(128) = 1.4426950408889634 / 11.31370849898476.
        constexpr float SOFTMAX_SCALE_LOG2 = 0.12753134917511246f;
        float prev_row_max = -INFINITY;  // init for j==0 (acc_scale=0; not read by COR warmup)
        float row_sum_reg  = 0.0f;        // dead state (FA4 fills sScale only at epi, which we skip)
        int phase_sr = 0;
        for (int j = 0; j < n_kv; j++) {
            PROF_BEGIN_X(EV_SM_WAIT_S);
            mbarrier_wait(s_p_o_full(pair), phase_sr);
            PROF_END_X(EV_SM_WAIT_S);
            phase_sr ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            PROF_BEGIN_X(EV_SM_RM);

            // LD-S: 128 fp32 per lane (4 chunks of 32 cols each).
            float s[128];
            #pragma unroll
            for (int chunk = 0; chunk < 4; chunk++) {
                const int addr = row_addr + chunk * 32;
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,"
                    "%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,"
                    "%28,%29,%30,%31}, [%32];"
                    : "=f"(s[chunk*32+0]),  "=f"(s[chunk*32+1]),  "=f"(s[chunk*32+2]),  "=f"(s[chunk*32+3]),
                      "=f"(s[chunk*32+4]),  "=f"(s[chunk*32+5]),  "=f"(s[chunk*32+6]),  "=f"(s[chunk*32+7]),
                      "=f"(s[chunk*32+8]),  "=f"(s[chunk*32+9]),  "=f"(s[chunk*32+10]), "=f"(s[chunk*32+11]),
                      "=f"(s[chunk*32+12]), "=f"(s[chunk*32+13]), "=f"(s[chunk*32+14]), "=f"(s[chunk*32+15]),
                      "=f"(s[chunk*32+16]), "=f"(s[chunk*32+17]), "=f"(s[chunk*32+18]), "=f"(s[chunk*32+19]),
                      "=f"(s[chunk*32+20]), "=f"(s[chunk*32+21]), "=f"(s[chunk*32+22]), "=f"(s[chunk*32+23]),
                      "=f"(s[chunk*32+24]), "=f"(s[chunk*32+25]), "=f"(s[chunk*32+26]), "=f"(s[chunk*32+27]),
                      "=f"(s[chunk*32+28]), "=f"(s[chunk*32+29]), "=f"(s[chunk*32+30]), "=f"(s[chunk*32+31])
                    : "r"(addr));
            }
            asm volatile("tcgen05.wait::ld.sync.aligned;");

            // ---- Row-max (DSL fmax_reduce sm_100 pattern) + acc_scale + STS sScale ----
            // 4 independent accumulator chains with 3-input FMNMX3 per step.
            // prev_row_max folded into m0 via 3-input fmax (no extra critical-path op).
            // Critical-path depth: 15 (loop) + 2 (tree-reduce) = 17, vs 42 for a
            // sequential chain. Matches DSL utils.fmax_reduce line 281-298 byte-for-byte.
            float m0 = fmaxf(prev_row_max, fmaxf(s[0], s[1]));  // FMNMX3 (3-input)
            float m1 = fmaxf(s[2], s[3]);
            float m2 = fmaxf(s[4], s[5]);
            float m3 = fmaxf(s[6], s[7]);
            #pragma unroll
            for (int i = 8; i < 128; i += 8) {
                m0 = fmaxf(m0, fmaxf(s[i+0], s[i+1]));  // FMNMX3
                m1 = fmaxf(m1, fmaxf(s[i+2], s[i+3]));  // FMNMX3 (independent)
                m2 = fmaxf(m2, fmaxf(s[i+4], s[i+5]));  // FMNMX3 (independent)
                m3 = fmaxf(m3, fmaxf(s[i+6], s[i+7]));  // FMNMX3 (independent)
            }
            m0 = fmaxf(m0, m1);                          // tree-reduce
            const float new_row_max = fmaxf(m0, fmaxf(m2, m3));  // FMNMX3 final
            // acc_scale: j==0 -> exp2(-INF)=0 (unused; COR warmup ignores);
            //            j>=1 -> exp2((prev - new) * LOG2_E).
            const float acc_scale = exp2f((prev_row_max - new_row_max) * SOFTMAX_SCALE_LOG2);
            prev_row_max = new_row_max;
            // STS acc_scale (matches FA4 layout: sScale[stage * 128 + tidx]).
            sScale_ptr[my_sScale_off] = acc_scale;

            // bar.arrive BEFORE any STTM-P (matches FA4 pattern at line 2112 of
            // flash_fwd_sm100_faux.py). Lets COR start as soon as sScale is
            // visible — the 64 cvts + 2 STTMs that follow run concurrent with
            // COR's wait+rescale on the previous iter. Reg pressure from holding
            // 64 cvt-result regs (pp[0..63]) fits in REGS_SOFTMAX=192.
            named_barrier_arrive(BAR_SM_STATS_BASE + pair * 4 + warp_id_local,
                                 SM_STATS_BAR_COUNT);
            PROF_END_X(EV_SM_RM);

            PROF_BEGIN_X(EV_SM_STP);
            // Single cvt loop (64 cvts -> 64 bf16x2) then STTM.x16 x 4 chunks
            // (matches DSL's St32x32bOp(Repetition(16)) — 32 cols per chunk).
            // Each STTM writes 16 b32 (= 32 bf16 cols). Total 128 cols across
            // 4 STTMs. Finer split enables future split_P_arrive optimization
            // and tracks DSL's exact PTX shape.
            uint32_t pp[64];
            #pragma unroll
            for (int i = 0; i < 64; i++) {
                asm volatile("cvt.rn.bf16x2.f32 %0, %2, %1;"
                    : "=r"(pp[i]) : "f"(s[2*i]), "f"(s[2*i+1]));
            }
            #define STTM_X16(addr_off, base) \
                asm volatile( \
                    "tcgen05.st.sync.aligned.32x32b.x16.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8," \
                    "%9,%10,%11,%12,%13,%14,%15,%16};" \
                    ::"r"(row_addr + (addr_off)), \
                    "r"(pp[(base)+0]),  "r"(pp[(base)+1]),  "r"(pp[(base)+2]),  "r"(pp[(base)+3]), \
                    "r"(pp[(base)+4]),  "r"(pp[(base)+5]),  "r"(pp[(base)+6]),  "r"(pp[(base)+7]), \
                    "r"(pp[(base)+8]),  "r"(pp[(base)+9]),  "r"(pp[(base)+10]), "r"(pp[(base)+11]), \
                    "r"(pp[(base)+12]), "r"(pp[(base)+13]), "r"(pp[(base)+14]), "r"(pp[(base)+15]))
            STTM_X16(64,       0);  // cols 0-31
            STTM_X16(64 + 16, 16);  // cols 32-63
            STTM_X16(64 + 32, 32);  // cols 64-95
            STTM_X16(64 + 48, 48);  // cols 96-127
            #undef STTM_X16
            asm volatile("tcgen05.fence::before_thread_sync;");
            // update_row_sum: rescale prior sum by acc_scale, add this iter's
            // local sum. Kept in register (FA4 only stores at the epi, which
            // we skip). Cost-fairness vs DSL update_row_sum (~64 FFMA2).
            {
                float rs = row_sum_reg * acc_scale;
                #pragma unroll
                for (int i = 0; i < 128; i++) {
                    rs = rs + s[i];
                }
                row_sum_reg = rs;
            }
            if (lane_id == 0) {
                mbarrier_arrive(p_lastsplit_full(pair));
                mbarrier_arrive(s_p_o_empty(pair));
            }
            PROF_END_X(EV_SM_STP);
        }
        (void)row_sum_reg;  // silence unused (dead state for cost-fairness)

    } else if (warp_id < CORR_HI) {
        // CORRECTION (warps 8-11) — FAUX+CORR:
        //   warmup: consume sm_stats(j=0) (no rescale; acc_scale(0) ignored).
        //   loop j=1..n_kv-1: wait PV[j-1] done, consume sm_stats(j),
        //     LDS acc_scale, FMUL2 rescale O[pair], arrive o_acc_empty(pair).
        //   final: drain PV[n_kv-1] o_acc_full.
        //   then existing LDTM-O + cvt + STS-sO + TMA-store path.
        const int warp_id_local = warp_id - CORR_LO;
        const int o_addr_p0 = O_acc_p0 + ((warp_id_local * 32) << 16);
        const int o_addr_p1 = O_acc_p1 + ((warp_id_local * 32) << 16);
        const int my_sScale_off = warp_id_local * 32 + lane_id;
        int phase_oa0 = 0, phase_oa1 = 0;

        // Rescale O[pair] by acc_scale: 8 chunks of LDTM.x16 + 8 FMUL2 + STTM.x16.
        // Matches FA4 correction_rescale (corr_tile_size=16, frg_count=8). Adapted
        // from redist_v7's tested rescale_O lambda.
        auto rescale_O = [&](int o_addr, float scale) {
            uint64_t scale_pair;
            asm volatile("mov.b64 %0, {%1, %2};" : "=l"(scale_pair) : "f"(scale), "f"(scale));
            #pragma unroll
            for (int c = 0; c < 8; c++) {
                const int addr = o_addr + c * 16;
                float o[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x16.b32 "
                    "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
                    : "=f"(o[0]),  "=f"(o[1]),  "=f"(o[2]),  "=f"(o[3]),
                      "=f"(o[4]),  "=f"(o[5]),  "=f"(o[6]),  "=f"(o[7]),
                      "=f"(o[8]),  "=f"(o[9]),  "=f"(o[10]), "=f"(o[11]),
                      "=f"(o[12]), "=f"(o[13]), "=f"(o[14]), "=f"(o[15])
                    : "r"(addr));
                // No per-chunk tcgen05.wait::ld — matches FA4's correction_rescale
                // which relies on the SASS scoreboard to gate mul.f32x2 on the
                // per-register LDTM completion. Lets ptxas overlap chunk c+1's
                // LDTM issue with chunk c's mul.f32x2/STTM.
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    uint64_t p, r;
                    asm volatile("mov.b64 %0, {%1, %2};" : "=l"(p) : "f"(o[2*i]), "f"(o[2*i+1]));
                    asm volatile("mul.f32x2 %0, %1, %2;" : "=l"(r) : "l"(p), "l"(scale_pair));
                    asm volatile("mov.b64 {%0, %1}, %2;" : "=f"(o[2*i]), "=f"(o[2*i+1]) : "l"(r));
                }
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x16.b32 [%0], "
                    "{%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16};"
                    :: "r"(addr),
                       "f"(o[0]),  "f"(o[1]),  "f"(o[2]),  "f"(o[3]),
                       "f"(o[4]),  "f"(o[5]),  "f"(o[6]),  "f"(o[7]),
                       "f"(o[8]),  "f"(o[9]),  "f"(o[10]), "f"(o[11]),
                       "f"(o[12]), "f"(o[13]), "f"(o[14]), "f"(o[15]));
            }
            asm volatile("tcgen05.wait::st.sync.aligned;");
        };

        // ---- Warmup: consume sm_stats(j=0) without rescaling ----
        // Recorded as Wco0_0 / Wco1_0 (the COR slice gating MMA's PV[0]).
        PROF_BEGIN_X(EV_COR_WAIT_O);
        named_barrier_sync(BAR_SM_STATS_BASE + 0 * 4 + warp_id_local, SM_STATS_BAR_COUNT);
        if (lane_id == 0) {
            mbarrier_arrive(s_p_o_empty(0));
        }
        PROF_END_X(EV_COR_WAIT_O);
        PROF_BEGIN_X(EV_COR_WAIT_O1);
        named_barrier_sync(BAR_SM_STATS_BASE + 1 * 4 + warp_id_local, SM_STATS_BAR_COUNT);
        if (lane_id == 0) {
            mbarrier_arrive(s_p_o_empty(1));
        }
        PROF_END_X(EV_COR_WAIT_O1);

        // ---- Rescale loop: j=1..n_kv-1 ----
        for (int j = 1; j < n_kv; j++) {
            // Pair 0: wait PV[j-1] done, sync sm_stats(j) for pair 0, FMUL2 rescale.
            PROF_BEGIN_X(EV_COR_WAIT_O);
            mbarrier_wait(o_acc_full(0), phase_oa0); phase_oa0 ^= 1;
            named_barrier_sync(BAR_SM_STATS_BASE + 0 * 4 + warp_id_local, SM_STATS_BAR_COUNT);
            asm volatile("tcgen05.fence::after_thread_sync;");
            const float scale_p0 = sScale_ptr[0 * M_PER_CTA + my_sScale_off];
            rescale_O(o_addr_p0, scale_p0);
            asm volatile("tcgen05.fence::before_thread_sync;");
            // s_p_o_empty(0) arrive = "O rescaled" half of MMA's unified gate.
            if (lane_id == 0) {
                mbarrier_arrive(s_p_o_empty(0));
            }
            PROF_END_X(EV_COR_WAIT_O);

            // Pair 1: same for pair 1.
            PROF_BEGIN_X(EV_COR_WAIT_O1);
            mbarrier_wait(o_acc_full(1), phase_oa1); phase_oa1 ^= 1;
            named_barrier_sync(BAR_SM_STATS_BASE + 1 * 4 + warp_id_local, SM_STATS_BAR_COUNT);
            asm volatile("tcgen05.fence::after_thread_sync;");
            const float scale_p1 = sScale_ptr[1 * M_PER_CTA + my_sScale_off];
            rescale_O(o_addr_p1, scale_p1);
            asm volatile("tcgen05.fence::before_thread_sync;");
            if (lane_id == 0) {
                mbarrier_arrive(s_p_o_empty(1));
            }
            PROF_END_X(EV_COR_WAIT_O1);
        }

        // ---- Final: drain PV[n_kv-1] o_acc_full for both pairs (no rescale) ----
        mbarrier_wait(o_acc_full(0), phase_oa0); phase_oa0 ^= 1;
        mbarrier_wait(o_acc_full(1), phase_oa1); phase_oa1 ^= 1;
        // After mainloop: LDTM-O + cvt(NO scale) + sts.b128 at K_SW128 swizzled
        // addresses. Matches FA4 DSL SASS pattern exactly:
        //   LDTM.x16 (16 fp32) -> F2FP.BF16.F32.PACK_AB (8 b32) -> STS.128 (2)
        //   x 8 iters per warp per pair (covers 32 rows x 128 cols).
        // Interleaved (NOT batched) so peak live regs stay small (~24 b32),
        // matching DSL's pattern and avoiding spilling.
        #pragma unroll
        for (int pair = 0; pair < 2; pair++) {
            asm volatile("tcgen05.fence::after_thread_sync;");

            const int row_addr = (pair == 0) ? o_addr_p0 : o_addr_p1;
#if !ENABLE_PROF
            const int sO_pair  = (pair == 0) ? sO_p0     : sO_p1;
            const int my_row   = warp_id_local * 32 + lane_id;
#endif

            // 8 iters of 16 fp32 cols each (= 128 fp32 cols total per lane).
            #pragma unroll
            for (int it = 0; it < 8; it++) {
                const int addr = row_addr + it * 16;
                float o[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,"
                    "%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
                    : "=f"(o[ 0]), "=f"(o[ 1]), "=f"(o[ 2]), "=f"(o[ 3]),
                      "=f"(o[ 4]), "=f"(o[ 5]), "=f"(o[ 6]), "=f"(o[ 7]),
                      "=f"(o[ 8]), "=f"(o[ 9]), "=f"(o[10]), "=f"(o[11]),
                      "=f"(o[12]), "=f"(o[13]), "=f"(o[14]), "=f"(o[15])
                    : "r"(addr));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                // cvt 16 fp32 -> 8 b32 (= 16 bf16 = 2 b128 worth).
                uint32_t b0, b1, b2, b3, b4, b5, b6, b7;
                asm volatile("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(b0) : "f"(o[ 0]), "f"(o[ 1]));
                asm volatile("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(b1) : "f"(o[ 2]), "f"(o[ 3]));
                asm volatile("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(b2) : "f"(o[ 4]), "f"(o[ 5]));
                asm volatile("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(b3) : "f"(o[ 6]), "f"(o[ 7]));
                asm volatile("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(b4) : "f"(o[ 8]), "f"(o[ 9]));
                asm volatile("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(b5) : "f"(o[10]), "f"(o[11]));
                asm volatile("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(b6) : "f"(o[12]), "f"(o[13]));
                asm volatile("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(b7) : "f"(o[14]), "f"(o[15]));
#if !ENABLE_PROF
                // 2 STS.128 covering 16 bf16 cols at it*16..it*16+15.
                const int col_bf16_0 = it * 16;
                const int sts0 = sO_pair + sw128_sO_offset(my_row, col_bf16_0);
                asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};" ::
                             "r"(sts0), "r"(b0), "r"(b1), "r"(b2), "r"(b3));
                const int sts1 = sO_pair + sw128_sO_offset(my_row, col_bf16_0 + 8);
                asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};" ::
                             "r"(sts1), "r"(b4), "r"(b5), "r"(b6), "r"(b7));
#else
                (void)b0; (void)b1; (void)b2; (void)b3;
                (void)b4; (void)b5; (void)b6; (void)b7;
#endif
            }
            asm volatile("fence.proxy.async.shared::cta;");
            __syncwarp();
            if (lane_id == 0) {
                mbarrier_arrive(o_epi_full(pair));
            }
        }
        (void)o_addr_p0; (void)o_addr_p1; (void)my_sScale_off;

    } else if (warp_id == MMA_WARP) {
        if (cta_rank == 0 && elect_sync_one()) {
            int phase_q       = 0;
            int phase_kv_full = 0;
            // s_p_o_empty unifies "P ready" (SM arrives after STTM-P) and
            // "O rescaled" (COR arrives after rescale STTM). count=8 fills per
            // iter (4 SM + 4 COR). COR warmup pre-arrives 4 for iter 0.
            // Mirrors FA4 pipeline_s_p_o.producer_acquire pattern.
            int phase_spe     = 0;

            PROF_BEGIN_X(EV_MMA_WAIT_Q);
            mbarrier_wait(q_full(0), (phase_q >> 0) & 1); phase_q ^= 1 << 0;
            mbarrier_wait(q_full(1), (phase_q >> 1) & 1); phase_q ^= 1 << 1;
            PROF_END_X(EV_MMA_WAIT_Q);
            asm volatile("tcgen05.fence::after_thread_sync;");

            // Pipelined MMA schedule (ported from redist_v7 MMA_REORDER=1):
            //   prologue: wait K_0 -> Q0K_0
            //   step S in [0..n_kv): P1V_{S-1} (S>=1), Q1K_S, wait V_S, P0V_S,
            //                        Q0K_{S+1} (if S+1<n_kv)
            //   epilogue: P1V_{n_kv-1}
            // Pair 0: Q0K_S issued at step S-1 -> P0V_S at step S (2-MMA overlap).
            // Pair 1: Q1K_S issued at step S   -> P1V_S at step S+1 (2-MMA overlap).
            // Slot lifetimes (KV_STAGE=6, 3 K+V pairs in flight):
            //   K_S in slot (2S)%6   used by Q0K_S (step S-1) + Q1K_S (step S);
            //                        release at end of step S.
            //   V_S in slot (2S+1)%6 used by P0V_S (step S) + P1V_S (step S+1);
            //                        release at end of step S+1.

            // Prologue: Q0K_0
            {
                const int s_K0 = 0;
                PROF_BEGIN_X(EV_MMA_WAIT_K);
                mbarrier_wait(kv_full(s_K0), (phase_kv_full >> s_K0) & 1);
                phase_kv_full ^= 1 << s_K0;
                PROF_END_X(EV_MMA_WAIT_K);
                issue_qk(0, s_K0);
            }

            for (int S = 0; S < n_kv; S++) {
                const int s_K_S    = (2 * S) % KV_STAGE;
                const int s_V_S    = (2 * S + 1) % KV_STAGE;
                const int s_V_prev = (S >= 1) ? ((2 * (S - 1) + 1) % KV_STAGE) : -1;

                // 1. P1V_{S-1} (if S>=1). V_{S-1} was waited at step S-1.
                if (S >= 1) {
                    PROF_BEGIN_X(EV_MMA_WAIT_SPE1);
                    mbarrier_wait_fast(s_p_o_empty(1), (phase_spe >> 1) & 1);
                    phase_spe ^= 1 << 1;
                    PROF_END_X(EV_MMA_WAIT_SPE1);
                    issue_pv(1, s_V_prev, false);
                    // Release V_{S-1}: both P0V_{S-1} (step S-1) and P1V_{S-1} (now) done.
                    asm volatile(
                        "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster"
                        ".multicast::cluster.b64 [%0], %1;" ::
                            "r"(kv_empty(s_V_prev)), "h"(cta_mask) : "memory");
                }

                // 2. Q1K_S (K_S already waited; prologue for S=0, end of step S-1 for S>=1)
                issue_qk(1, s_K_S);
                // Release K_S: both Q0K_S and Q1K_S committed.
                asm volatile(
                    "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster"
                    ".multicast::cluster.b64 [%0], %1;" ::
                        "r"(kv_empty(s_K_S)), "h"(cta_mask) : "memory");

                // 3. P0V_S — wait V_S, wait pair-0 s_p_o_empty (SM done writing P[S]
                //   pair 0 + COR done rescaling O[0]), then issue.
                PROF_BEGIN_X(EV_MMA_WAIT_V);
                mbarrier_wait_fast(kv_full(s_V_S), (phase_kv_full >> s_V_S) & 1);
                phase_kv_full ^= 1 << s_V_S;
                PROF_END_X(EV_MMA_WAIT_V);
                PROF_BEGIN_X(EV_MMA_WAIT_SPE0);
                mbarrier_wait_fast(s_p_o_empty(0), (phase_spe >> 0) & 1);
                phase_spe ^= 1 << 0;
                PROF_END_X(EV_MMA_WAIT_SPE0);
                issue_pv(0, s_V_S, S == 0);

                // 4. Q0K_{S+1} (if not last iter) — wait K_{S+1}, then issue.
                if (S + 1 < n_kv) {
                    const int s_K_next = (2 * (S + 1)) % KV_STAGE;
                    PROF_BEGIN_X(EV_MMA_WAIT_K);
                    mbarrier_wait_fast(kv_full(s_K_next), (phase_kv_full >> s_K_next) & 1);
                    phase_kv_full ^= 1 << s_K_next;
                    PROF_END_X(EV_MMA_WAIT_K);
                    issue_qk(0, s_K_next);
                }
            }

            // Epilogue: P1V_{n_kv-1}
            {
                const int s_V_last = (2 * (n_kv - 1) + 1) % KV_STAGE;
                PROF_BEGIN_X(EV_MMA_WAIT_SPE1);
                mbarrier_wait(s_p_o_empty(1), (phase_spe >> 1) & 1);
                phase_spe ^= 1 << 1;
                PROF_END_X(EV_MMA_WAIT_SPE1);
                issue_pv(1, s_V_last, n_kv == 1);
                asm volatile(
                    "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster"
                    ".multicast::cluster.b64 [%0], %1;" ::
                        "r"(kv_empty(s_V_last)), "h"(cta_mask) : "memory");
            }
            // Final tcgen05 commit drains all in-flight MMAs.
            asm volatile(
                "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster"
                ".multicast::cluster.b64 [%0], %1;" ::
                    "r"(tmem_dealloc), "h"(cta_mask) : "memory");
        }

    } else if (warp_id == EPI_WARP) {
        // EPILOGUE (warp 13) — per-pair TMA store sO -> gO.
        if (elect_sync_one()) {
            #pragma unroll
            for (int pair = 0; pair < 2; pair++) {
                PROF_BEGIN_X(EV_EPI_WAIT);
                mbarrier_wait(o_epi_full(pair), 0);
                PROF_END_X(EV_EPI_WAIT);
#if !ENABLE_PROF
                const int base_row = (int)blockIdx.y * M_PER_CLUSTER
                                   + pair * M_JOINT
                                   + (int)cta_rank * M_PER_CTA;
                const int sO_pair = (pair == 0) ? sO_p0 : sO_p1;
                tma_store_3d(sO_pair, &O_tmap, 0, base_row, 0);
                tma_store_commit();
#endif
            }
#if !ENABLE_PROF
            tma_store_wait();
#endif
        }

    } else if (warp_id == LOAD_WARP) {
        if (elect_sync_one()) {
            for (int p = 0; p < Q_STAGE; p++) {
                PROF_BEGIN_X(EV_TMA_Q);
                asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;" ::
                             "r"(q_full(p)), "r"(Q_BYTES_PER_STAGE) : "memory");
                tma_3d_local(Q_smem(p), &Q_tmap, 0,
                             q_pair_offset[p] + (int)cta_rank * M_PER_CTA, 0,
                             q_full(p));
                PROF_END_X(EV_TMA_Q);
            }
            int phase_kv_empty = 0;
            for (int j = 0; j < n_kv; j++) {
                const int s_K = (2*j    ) % KV_STAGE;
                const int s_V = (2*j + 1) % KV_STAGE;

                if (j >= N_KV_PAIRS_INFLIGHT) {
                    PROF_BEGIN_X(EV_TMA_WAIT_E);
                    mbarrier_wait_fast(kv_empty(s_K), (phase_kv_empty >> s_K) & 1);
                    PROF_END_X(EV_TMA_WAIT_E);
                    phase_kv_empty ^= 1 << s_K;
                }
                {
                    PROF_BEGIN_X(EV_TMA_K);
                    const int kf_peer0 = kv_full(s_K) & Sm100MmaPeerBitMask;
                    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;" ::
                                 "r"(kf_peer0),
                                 "r"(KV_BYTES_PER_STAGE) : "memory");
                    // Iterate K tiles in REVERSE order (n_kv-1 down to 0) to match
                    // FA4 non-causal softmax_loop. Required for rescale-order parity
                    // (running_max history differs by iteration order).
                    const int n_block = n_kv - 1 - j;
                    const int k_y = n_block * BLOCK_K + (int)cta_rank * (BLOCK_K / CTA_GROUP);
                    tma_3d_cga2(KV_smem(s_K), &K_tmap, 0, k_y, 0, kf_peer0);
                    PROF_END_X(EV_TMA_K);
                }

                if (j >= N_KV_PAIRS_INFLIGHT) {
                    PROF_BEGIN_X(EV_TMA_WAIT_E);
                    mbarrier_wait_fast(kv_empty(s_V), (phase_kv_empty >> s_V) & 1);
                    PROF_END_X(EV_TMA_WAIT_E);
                    phase_kv_empty ^= 1 << s_V;
                }
                {
                    PROF_BEGIN_X(EV_TMA_V);
                    const int kf_peer0 = kv_full(s_V) & Sm100MmaPeerBitMask;
                    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;" ::
                                 "r"(kf_peer0),
                                 "r"(KV_BYTES_PER_STAGE) : "memory");
                    const int v_block = n_kv - 1 - j;
                    tma_3d_cga2(KV_smem(s_V), &V_tmap, 0, v_block * BLOCK_K, (int)cta_rank, kf_peer0);
                    PROF_END_X(EV_TMA_V);
                }
            }
        }
    }

    // Tail: wait MMA done, dealloc TMEM.
    __syncthreads();
    mbarrier_wait(tmem_dealloc, 0);
    if (warp_id == 1) {
        asm volatile(
            "tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, 512;" ::"r"(taddr));
    }
    PROF_FLUSH(prof_buf, prof_count);
}

// ============================================================================
// Host launch
// ============================================================================
std::vector<torch::Tensor> fa4_faux_attn_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda(), "cuda only");
    TORCH_CHECK(Q.dtype() == torch::kBFloat16, "bf16");
    TORCH_CHECK(Q.dim() == 2 && Q.size(1) == N_DIM, "Q [queries, 128]");
    TORCH_CHECK(Q.size(0) % M_PER_CLUSTER == 0,
                "Q rows must be multiple of M_PER_CLUSTER (512)");
    TORCH_CHECK(K.dim() == 2 && K.size(1) == N_DIM && V.dim() == 2 && V.size(1) == N_DIM,
                "K/V [Sk, 128]");
    int Sk = K.size(0);
    TORCH_CHECK(Sk == V.size(0) && Sk % BLOCK_K == 0 && Sk >= 2 * BLOCK_K,
                "Sk match, multiple of 128, >= 256");

    int n_kv = Sk / BLOCK_K;
    int queries = Q.size(0);
    int num_clusters = queries / M_PER_CLUSTER;

    auto opts = torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA);
    auto O = torch::empty({queries, N_DIM}, opts);

    auto encode = [](CUtensorMap* tmap, const nv_bfloat16* ptr,
                     uint64_t global_height, uint32_t shared_height, uint32_t box_z) {
        constexpr uint32_t rank = 3;
        uint64_t globalDim[rank]      = {64, global_height, (uint64_t)N_DIM / 64};
        uint64_t globalStrides[rank-1] = {(uint64_t)N_DIM * 2, 128};
        uint32_t boxDim[rank]         = {64, shared_height, box_z};
        uint32_t elementStrides[rank] = {1, 1, 1};
        auto err = cuTensorMapEncodeTiled(tmap, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, rank,
            (void*)ptr, globalDim, globalStrides, boxDim, elementStrides,
            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        TORCH_CHECK(err == CUDA_SUCCESS, "tmap encode fail");
    };
    auto encode_o = [](CUtensorMap* tmap, const nv_bfloat16* ptr, uint64_t global_height) {
        // 3D K_SW128 atom layout (validated in micro_sw128.cu LAYOUT=1).
        // Inner 64 bf16 = 128 bytes (= 1 swizzle stripe), middle = rows,
        // outer = N_DIM/64 = 2 stripes per row.
        constexpr uint32_t rank = 3;
        uint64_t globalDim[rank]       = {64, global_height, (uint64_t)N_DIM / 64};
        uint64_t globalStrides[rank-1] = {(uint64_t)N_DIM * 2, 128};
        uint32_t boxDim[rank]          = {64, (uint32_t)M_PER_CTA, 2};
        uint32_t elementStrides[rank]  = {1, 1, 1};
        auto err = cuTensorMapEncodeTiled(tmap, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, rank,
            (void*)ptr, globalDim, globalStrides, boxDim, elementStrides,
            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        TORCH_CHECK(err == CUDA_SUCCESS, "O tmap encode fail");
    };
    CUtensorMap Q_tmap{}, K_tmap{}, V_tmap{}, O_tmap{};
    encode(&Q_tmap, reinterpret_cast<const nv_bfloat16*>(Q.data_ptr()),
           (uint64_t)queries, M_PER_CTA, 2);
    encode(&K_tmap, reinterpret_cast<const nv_bfloat16*>(K.data_ptr()), (uint64_t)Sk,
           (uint32_t)(BLOCK_K / CTA_GROUP), 2);
    encode(&V_tmap, reinterpret_cast<const nv_bfloat16*>(V.data_ptr()), (uint64_t)Sk,
           (uint32_t)BLOCK_K, 1);
    encode_o(&O_tmap, reinterpret_cast<const nv_bfloat16*>(O.data_ptr()), (uint64_t)queries);

    cudaFuncSetAttribute(fa4_faux_attn_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, DYN_SMEM_BYTES);

    auto i32_opts = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    auto prof_buf   = torch::zeros({PROF_MAX_EVENTS * 6}, i32_opts);
    auto prof_count = torch::zeros({1}, i32_opts);

    dim3 grid(CTA_GROUP, num_clusters, 1);
    fa4_faux_attn_kernel<<<grid, TB_SIZE, DYN_SMEM_BYTES>>>(
        Q_tmap, K_tmap, V_tmap, O_tmap, n_kv,
        reinterpret_cast<ProfEvent*>(prof_buf.data_ptr()),
        reinterpret_cast<int*>(prof_count.data_ptr()));
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "launch failed: ", cudaGetErrorString(err));
    return {O, prof_buf, prof_count};
}
