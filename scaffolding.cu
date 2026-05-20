// redist_v6.cu — FA4-precise SMEM layout + mbarriers on top of redist_v4 base.
//
// What changes from redist_v4:
//   * SMEM layout matches FA4's flash_fwd_sm100.py exactly:
//       sQ  : q_stage=2 × 32 KB = 64 KB
//       sO  : q_stage=2 × 32 KB = 64 KB                     (NEW vs v4)
//       sK│V: kv_stage=6 × 16 KB per CTA = 96 KB            (ALIASED, NEW vs v4)
//             — K and V share the same physical SMEM range; each ring stage
//               holds *either* K[j] or V[j] depending on which TMA filled it.
//             — FA4 trick:  sV = recast_ptr(sK, sV_layout.inner)
//             — kv_stage=6 (vs v4's 5) because the alias halves K+V footprint.
//       sScale: q_stage * m * 2 * 4 = 2 KB
//     Total ≈ 226 KB dyn SMEM (matches FA4's NCU-reported 232.45 KB
//     dynamic SMEM after alignment).
//
//   * Mbarriers match FA4's 7 pipelines (37 mbars total) vs v4's 13:
//       q_full[2]  q_empty[2]               — pipeline_q     (q_stage*2 = 4)
//       kv_full[6] kv_empty[6]              — pipeline_kv    (kv_stage*2 = 12)
//       s_p_o_full[2] s_p_o_empty[2]        — pipeline_s_p_o (q_stage*2 = 4)
//       p_lastsplit_full[2] p_lastsplit_empty[2] — pipeline_p_lastsplit
//       o_acc_full[2] o_acc_empty[2]        — pipeline_o_acc (q_stage*2 = 4)
//       sm_stats_full[2] sm_stats_empty[2]  — pipeline_sm_stats (q_stage*2 = 4)
//       o_epi_full[2] o_epi_empty[2]        — pipeline_o_epi (q_stage*2 = 4)
//       tmem_dealloc                        — cluster sync
//
//   * Warp role split matches FA4 exactly:
//       warps 0-3   = softmax0  (WG0)  REGS_SM=192 (inc)
//       warps 4-7   = softmax1  (WG1)  REGS_SM=192 (inc)
//       warps 8-11  = correction(WG2)  REGS_COR=80 (dec)
//       warp  12    = MMA              REGS_OTHER=48 (dec)
//       warp  13    = epilogue         REGS_OTHER=48 (dec)
//       warp  14    = TMA load         REGS_OTHER=48 (dec)
//       warp  15    = empty            REGS_OTHER=48 (dec)
//
//   * KV pipeline protocol (FA4): K[j] and V[j] occupy SEPARATE pipeline_kv
//     stages. With kv_stage=6 and TMA order K0,V0,K1,V1,K2,V2,K3,…, K[j] lives
//     at stage (2j) % 6 and V[j] at stage (2j+1) % 6. This gives 3 K/V pairs
//     in flight at any time, but slot reuse can advance K[j+3] into K[0]'s slot
//     as soon as QK_j released it — finer-grained than v4's combined 5 slots.
//
// What we KEEP from redist_v4 (stub-but-signal pattern):
//   * Softmax (WG0/1) and correction (WG2) participate in mbarrier handshakes
//     but DO NO compute — they exercise the FA4 protocol so the latency
//     measurement reflects mbar overhead, not softmax compute.
//   * Sentinel epilogue from TMEM (no real TMA store of sO yet — sO is allocated
//     to match FA4's layout but unused in this revision).

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#ifndef ENABLE_PROF
#define ENABLE_PROF 0
#endif
// SKIP_SM_FULL=1: WG0/1, WG2, warp 13 fully idle. MMA bypasses s_p_o/o_acc
//   waits. True TMA+MMA SOL — useful for measuring the pipeline floor.
#ifndef SKIP_SM_FULL
#define SKIP_SM_FULL 0
#endif
#define PROF_PER_WARP_SLOTS 256
#include "profiler.cuh"

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
    EV_MMA_WAIT_PR0= 11,
    EV_MMA_WAIT_PR1= 12,
    EV_SM_WAIT_S   = 13,
    EV_SM_WORK     = 14,
    EV_COR_WAIT_O  = 15,
    EV_EPI_WAIT    = 16,
};

// =============================================================================
// FA4-precise constants (matches flash_fwd_sm100.py for D=128, BF16, single-head)
// =============================================================================
constexpr int M_PER_CTA      = 128;        // FA4 m_block_size
constexpr int M_JOINT        = M_PER_CTA * 2;       // 256: cga2 MMA tiler M
constexpr int N_M_PAIRS      = 2;          // FA4 q_stage=2: 2 M-tiles per CTA
constexpr int M_PER_CLUSTER  = M_JOINT * N_M_PAIRS; // 512 rows per cga2 cluster
constexpr int N_DIM          = 128;        // head_dim_padded
constexpr int BLOCK_K        = 128;        // n_block_size
constexpr int MMA_K          = 16;
constexpr int CTA_GROUP      = 2;          // cga2
constexpr int NUM_WARPS      = 16;
constexpr int TB_SIZE        = NUM_WARPS * 32;

// Pipeline depths (FA4)
constexpr int Q_STAGE  = 2;                // FA4 q_stage
constexpr int KV_STAGE = 6;                // FA4 kv_stage = (224K - 128K) / 16K = 6
constexpr int S_STAGE  = 2;                // FA4 s_stage
constexpr int N_KV_PAIRS_INFLIGHT = KV_STAGE / 2;  // 3 K+V pairs

// SMEM byte counts (per CTA)
constexpr int Q_BYTES_PER_STAGE  = M_PER_CTA * N_DIM * 2;            // 32 KB per Q tile
constexpr int Q_BYTES_TOTAL      = Q_STAGE * Q_BYTES_PER_STAGE;      // 64 KB
constexpr int O_BYTES_PER_STAGE  = M_PER_CTA * N_DIM * 2;            // 32 KB per O tile
constexpr int O_BYTES_TOTAL      = Q_STAGE * O_BYTES_PER_STAGE;      // 64 KB
constexpr int KV_BYTES_PER_STAGE = (BLOCK_K * N_DIM * 2) / CTA_GROUP; // 16 KB per CTA
constexpr int KV_BYTES_TOTAL     = KV_STAGE * KV_BYTES_PER_STAGE;    // 96 KB
constexpr int SCALE_BYTES_TOTAL  = Q_STAGE * M_PER_CTA * 2 * 4;      // 2 KB
constexpr int DYN_SMEM_BYTES     = Q_BYTES_TOTAL + O_BYTES_TOTAL +
                                   KV_BYTES_TOTAL + SCALE_BYTES_TOTAL;
static_assert(DYN_SMEM_BYTES <= 232 * 1024,
              "Dynamic SMEM exceeds 232 KB cap (B200 opt-in max).");

// Warp role layout (matches FA4 NamedBarrierFwd assignment)
constexpr int SOFTMAX0_LO = 0;             constexpr int SOFTMAX0_HI = 4;
constexpr int SOFTMAX1_LO = 4;             constexpr int SOFTMAX1_HI = 8;
constexpr int CORR_LO     = 8;             constexpr int CORR_HI     = 12;
constexpr int MMA_WARP    = 12;
constexpr int EPI_WARP    = 13;
constexpr int LOAD_WARP   = 14;
constexpr int EMPTY_WARP  = 15;

// Register tiers (FA4 D=128 BF16 with enable_ex2_emu=True, paged_kv_non_tma=False)
constexpr int REGS_SOFTMAX = 192;          // setmaxregister_increase
constexpr int REGS_CORR    = 80;           // setmaxregister_decrease
constexpr int REGS_OTHER   = 48;           // setmaxregister_decrease (MMA, epi, load, empty)

// cga2 peer-mbar mask (strips cta-rank bit so multicast TMA arrives on both peers)
constexpr int Sm100MmaPeerBitMask = 0xFEFFFFFF;

// =============================================================================
// PTX helpers (mirror of v4 — keep identical to inherit working SASS)
// =============================================================================
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
__device__ __forceinline__ void mbarrier_arrive(int mbar_addr) {
    asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" ::"r"(mbar_addr) : "memory");
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

// =============================================================================
// Kernel
// =============================================================================
__global__ void __cluster_dims__(CTA_GROUP, 1, 1) __launch_bounds__(TB_SIZE, 1)
redist_v6_kernel(
    const __grid_constant__ CUtensorMap Q_tmap,
    const __grid_constant__ CUtensorMap K_tmap,
    const __grid_constant__ CUtensorMap V_tmap,
    float* O_gmem,
    int n_kv,
    ProfEvent* prof_buf,
    int* prof_count) {
    const int tid     = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const uint32_t cta_rank = cluster_ctarank();

    // Each cga2 cluster handles N_M_PAIRS M-tile pairs = M_PER_CLUSTER rows.
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

    // -------- SMEM layout (matches FA4 exactly) ---------------------------
    // [sQ_p0 32KB][sQ_p1 32KB][sO_p0 32KB][sO_p1 32KB][KV_ring 6×16KB][sScale 2KB]
    extern __shared__ __align__(1024) char smem_ptr[];
    const int smem    = (int)__cvta_generic_to_shared(smem_ptr);
    const int sQ_p0   = smem;
    const int sQ_p1   = sQ_p0   + Q_BYTES_PER_STAGE;
    const int sO_p0   = sQ_p1   + Q_BYTES_PER_STAGE;
    const int sO_p1   = sO_p0   + O_BYTES_PER_STAGE;
    const int sKV_base= sO_p1   + O_BYTES_PER_STAGE;
    const int sScale  = sKV_base+ KV_BYTES_TOTAL;
    (void)sO_p0; (void)sO_p1; (void)sScale;  // allocated; unused in stub epi
#define Q_smem(p)  ((p) == 0 ? sQ_p0 : sQ_p1)
#define O_smem(p)  ((p) == 0 ? sO_p0 : sO_p1)
#define KV_smem(s) (sKV_base + (s) * KV_BYTES_PER_STAGE)

    // -------- Mbarrier layout (matches FA4: 7 pipelines + tmem_dealloc) ---
#pragma nv_diag_suppress static_var_with_dynamic_init
    // 4+12+4+4+4+4+4 + 1 = 37 mbars per CTA.
    __shared__ uint64_t mbars[37];
    __shared__ int tmem_addr_smem[1];
    const int mb = (int)__cvta_generic_to_shared(mbars);
    // Index assignments (8 bytes each):
    //   [0..1]   q_full       (q_stage=2)
    //   [2..3]   q_empty      (q_stage=2)
    //   [4..9]   kv_full      (kv_stage=6)
    //   [10..15] kv_empty     (kv_stage=6)
    //   [16..17] s_p_o_full   (q_stage=2)
    //   [18..19] s_p_o_empty  (q_stage=2)
    //   [20..21] p_lastsplit_full (q_stage=2)
    //   [22..23] p_lastsplit_empty
    //   [24..25] o_acc_full
    //   [26..27] o_acc_empty
    //   [28..29] sm_stats_full
    //   [30..31] sm_stats_empty
    //   [32..33] o_epi_full
    //   [34..35] o_epi_empty
    //   [36]     tmem_dealloc
#define q_full(s)         (mb + ( 0 + (s)) * 8)
#define q_empty(s)        (mb + ( 2 + (s)) * 8)
#define kv_full(s)        (mb + ( 4 + (s)) * 8)
#define kv_empty(s)       (mb + (10 + (s)) * 8)
#define s_p_o_full(s)     (mb + (16 + (s)) * 8)
#define s_p_o_empty(s)    (mb + (18 + (s)) * 8)
#define p_lastsplit_full(s)  (mb + (20 + (s)) * 8)
#define p_lastsplit_empty(s) (mb + (22 + (s)) * 8)
#define o_acc_full(s)     (mb + (24 + (s)) * 8)
#define o_acc_empty(s)    (mb + (26 + (s)) * 8)
#define sm_stats_full(s)  (mb + (28 + (s)) * 8)
#define sm_stats_empty(s) (mb + (30 + (s)) * 8)
#define o_epi_full(s)     (mb + (32 + (s)) * 8)
#define o_epi_empty(s)    (mb + (34 + (s)) * 8)
#define tmem_dealloc      (mb + 36 * 8)

    if (warp_id == 0 && elect_sync_one()) {
        // pipeline_q: q_full count = 1 (single TMA fires it via expect_tx).
        //              q_empty count = 1 (consumer release from MMA leader).
        for (int s = 0; s < Q_STAGE; s++) {
            mbarrier_init(q_full(s),  1);
            mbarrier_init(q_empty(s), 1);
        }
        // pipeline_kv: kv_full count = CTA_GROUP=2. cga2 TMA fires the mbar
        //   ONCE PER PEER's issuance (each peer's TMA call decrements count by 1
        //   on the masked-address shared mbar). With BOTH peers issuing per stage,
        //   count=2 → 0. Matches v5's pattern (count=2, 1 cga2 TMA per stage).
        //               kv_empty count = 1 (MMA leader fires via tcgen05.commit
        //   cluster multicast → fires both peer mbars; each peer's TMA waits its local).
        for (int s = 0; s < KV_STAGE; s++) {
            mbarrier_init(kv_full(s),  CTA_GROUP);
            mbarrier_init(kv_empty(s), 1);
        }
        // pipeline_s_p_o: s_p_o_full count = 1 (multicast cluster from MMA leader).
        //                  s_p_o_empty count = 8 per pair: 4 softmax warps for that
        //   pair + 4 correction warps (correction releases both pairs each iter).
        for (int s = 0; s < Q_STAGE; s++) {
            mbarrier_init(s_p_o_full(s),  1);
            mbarrier_init(s_p_o_empty(s), 8);
        }
        // pipeline_p_lastsplit: producer = softmax warps (4 per stage), consumer = MMA leader.
        for (int s = 0; s < Q_STAGE; s++) {
            mbarrier_init(p_lastsplit_full(s),  4);
            mbarrier_init(p_lastsplit_empty(s), 1);
        }
        // pipeline_o_acc: producer = MMA leader (multicast), consumer = correction (4 warps).
        for (int s = 0; s < Q_STAGE; s++) {
            mbarrier_init(o_acc_full(s),  1);
            mbarrier_init(o_acc_empty(s), 4);
        }
        // pipeline_sm_stats: producer = softmax (4 warps per stage), consumer = correction (4 warps).
        for (int s = 0; s < Q_STAGE; s++) {
            mbarrier_init(sm_stats_full(s),  4);
            mbarrier_init(sm_stats_empty(s), 4);
        }
        // pipeline_o_epi: producer = correction (4 warps), consumer = epi (1 warp).
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

    // TMEM partition (matches FA4 tmem_s_offset/tmem_p_offset/tmem_o_offset):
    //   slot_taddr_p0  cols   0..127 (S0 = 128 cols, P0 starts at col 64 = +64 in TMEM bytes)
    //   slot_taddr_p1  cols 128..255
    //   O_acc_p0       cols 256..383
    //   O_acc_p1       cols 384..511
    const int slot_taddr_p0 = taddr;
    const int slot_taddr_p1 = taddr + 128;
    const int O_acc_p0      = taddr + 256;
    const int O_acc_p1      = taddr + 384;

    constexpr uint16_t cta_mask = (1u << CTA_GROUP) - 1u;

    // -------- Register tier setup (FA4 exact: 192/80/48) ----------------
    if (warp_id < SOFTMAX1_HI) {
        smnr_inc<REGS_SOFTMAX>();
    } else if (warp_id < CORR_HI) {
        smnr_dec<REGS_CORR>();
    } else {
        smnr_dec<REGS_OTHER>();
    }

    // -------- MMA descriptor templates (same as v4) ---------------------
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

    // ===================================================================
    // MMA driver lambdas
    // ===================================================================
    // issue_qk: 8 cga2 QK MMAs (4 low panel, 4 high panel) on Q[pair] × K from
    // kv_stage `s`. Caller is responsible for pipeline_kv.consumer_wait on s
    // beforehand. Commits via tcgen05.commit → s_p_o_full (multicast cluster).
    auto issue_qk = [&](int pair, int s_kv) {
        asm volatile("tcgen05.fence::after_thread_sync;");
        PROF_BEGIN_X(pair == 0 ? EV_MMA_QK_P0 : EV_MMA_QK_P1);
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
        // Commit S → s_p_o_full[pair] multicast.
        asm volatile(
            "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster"
            ".multicast::cluster.b64 [%0], %1;" ::
                "r"(s_p_o_full(pair)), "h"(cta_mask) : "memory");
        PROF_END_X(pair == 0 ? EV_MMA_QK_P0 : EV_MMA_QK_P1);
    };

    // issue_pv: 8 cga2 PV MMAs (BLOCK_K/MMA_K = 8 inner k-chunks) on
    // P[pair] (TMEM) × V from kv_stage `s_v`. Caller waits pipeline_kv on s_v
    // and pipeline_p_lastsplit on `pair` first. Commits via tcgen05.commit →
    // o_acc_full[pair] multicast. `first_for_O` = (j == 0) — controls accum reset.
    auto issue_pv = [&](int pair, int s_v, bool first_for_O) {
        asm volatile("tcgen05.fence::after_thread_sync;");
        PROF_BEGIN_X(pair == 0 ? EV_MMA_PV_P0 : EV_MMA_PV_P1);
        const int slot_taddr_pair = (pair == 0) ? slot_taddr_p0 : slot_taddr_p1;
        const int O_acc_pair      = (pair == 0) ? O_acc_p0      : O_acc_p1;
        const uint64_t v_desc_base = make_desc_v_pv(KV_smem(s_v));
        constexpr uint64_t v_desc_step = (uint64_t)V_K_BYTES_PER_KC >> 4ULL;
        const int p_base = slot_taddr_pair + 64;  // P at TMEM offset +64 from S
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
        PROF_END_X(pair == 0 ? EV_MMA_PV_P0 : EV_MMA_PV_P1);
    };

    // ===================================================================
    // Warp dispatch
    // ===================================================================

    if (warp_id < SOFTMAX1_HI) {
        // ============================================================
        // SOFTMAX (warps 0-7 = WG0+WG1) — FAUX attention: cast S→bf16→TMEM
        // ============================================================
        // Each WG handles its own M-pair. WG0 → pair 0, WG1 → pair 1.
        // Each warp owns 32 rows; each lane within owns 1 row of 128 fp32.
        // Register scheme matches FA4's `float row[128]` (per-lane) — drives
        // setmaxnreg.inc<192> to engage (real R≥128 use, not DCE'd).
#if !SKIP_SM_FULL
        const bool is_p0 = (warp_id < SOFTMAX1_LO);
        const int pair   = is_p0 ? 0 : 1;
        const int slot_taddr_pair = is_p0 ? slot_taddr_p0 : slot_taddr_p1;
        const int warp_id_local   = is_p0 ? warp_id : warp_id - SOFTMAX1_LO;
        const int row_addr        = slot_taddr_pair + ((warp_id_local * 32) << 16);
        int phase_sr = 0;
        for (int j = 0; j < n_kv; j++) {
            PROF_BEGIN_X(EV_SM_WAIT_S);
            mbarrier_wait(s_p_o_full(pair), phase_sr);
            PROF_END_X(EV_SM_WAIT_S);
            phase_sr ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            PROF_BEGIN_X(EV_SM_WORK);

            // Read S row (128 fp32) from TMEM into registers — keeps R≥128.
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

            // FAUX softmax (rowmax stage): compute m = max(s).
            // 9-way chunked accumulator. Each chunk reduces 9 inputs via
            // 4 FMNMX3 (3 inner max3 + 1 outer max3), accumulates into m.
            // Avoids wide-tree intermediate arrays.
            float m = -INFINITY;
            #pragma unroll
            for (int i = 0; i < 14; i++) {
                float c = fmaxf(
                    fmaxf(
                        fmaxf(fmaxf(s[9*i+0], s[9*i+1]), s[9*i+2]),
                        fmaxf(fmaxf(s[9*i+3], s[9*i+4]), s[9*i+5])
                    ),
                    fmaxf(fmaxf(s[9*i+6], s[9*i+7]), s[9*i+8])
                );
                m = fmaxf(m, c);
            }
            m = fmaxf(m, fmaxf(s[126], s[127]));

            // Combined subtract+cast: per iteration consume 2 fp32 from s[] and
            // produce 1 uint32 in pp[]. Lets ptxas free s[2i]/s[2i+1] right
            // after the cvt, eliminating peak overlap of s[128]+pp[64]=192R.
            uint64_t mvec;
            asm volatile("mov.b64 %0, {%1, %2};" : "=l"(mvec) : "f"(m), "f"(m));
            uint32_t pp[64];
            #pragma unroll
            for (int i = 0; i < 64; i++) {
                uint64_t sp;
                asm volatile("mov.b64 %0, {%1, %2};"
                    : "=l"(sp) : "f"(s[2*i]), "f"(s[2*i+1]));
                uint64_t sub_res;
                asm volatile("sub.f32x2 %0, %1, %2;"
                    : "=l"(sub_res) : "l"(sp), "l"(mvec));
                float a, b;
                asm volatile("mov.b64 {%0, %1}, %2;"
                    : "=f"(a), "=f"(b) : "l"(sub_res));
                asm volatile("cvt.rn.bf16x2.f32 %0, %2, %1;"
                    : "=r"(pp[i]) : "f"(a), "f"(b));
            }

            // Write P to TMEM at offset +64 from S region (FA4 tmem_p_offset).
            #pragma unroll
            for (int chunk = 1; chunk >= 0; chunk--) {
                const int addr = row_addr + 64 + chunk * 32;
                uint32_t* base = &pp[chunk * 32];
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x32.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8,"
                    "%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,"
                    "%27,%28,%29,%30,%31,%32};"
                    ::"r"(addr),
                    "r"(base[0]),  "r"(base[1]),  "r"(base[2]),  "r"(base[3]),
                    "r"(base[4]),  "r"(base[5]),  "r"(base[6]),  "r"(base[7]),
                    "r"(base[8]),  "r"(base[9]),  "r"(base[10]), "r"(base[11]),
                    "r"(base[12]), "r"(base[13]), "r"(base[14]), "r"(base[15]),
                    "r"(base[16]), "r"(base[17]), "r"(base[18]), "r"(base[19]),
                    "r"(base[20]), "r"(base[21]), "r"(base[22]), "r"(base[23]),
                    "r"(base[24]), "r"(base[25]), "r"(base[26]), "r"(base[27]),
                    "r"(base[28]), "r"(base[29]), "r"(base[30]), "r"(base[31]));
            }
            asm volatile("tcgen05.wait::st.sync.aligned;");
            asm volatile("tcgen05.fence::before_thread_sync;");

            // 4 softmax warps per pair each arrive once on these (lane 0 only):
            if (lane_id == 0) {
                mbarrier_arrive(p_lastsplit_full(pair));
                mbarrier_arrive(sm_stats_full(pair));
                mbarrier_arrive(s_p_o_empty(pair));
            }
            PROF_END_X(EV_SM_WORK);
        }
#endif

    } else if (warp_id < CORR_HI) {
        // ============================================================
        // CORRECTION (warps 8-11 = WG2, stub-but-signal)
        // ============================================================
        // Per iter:  wait sm_stats_full[both pairs] (4 arrivals each) +
        //            wait o_acc_full → arrive o_acc_empty + s_p_o_empty +
        //            sm_stats_empty + (last iter only: o_epi_full)
#if !SKIP_SM_FULL
        int phase_oa0 = 0, phase_oa1 = 0;
        int phase_ss0 = 0, phase_ss1 = 0;
        for (int j = 0; j < n_kv; j++) {
            PROF_BEGIN_X(EV_COR_WAIT_O);
            mbarrier_wait(o_acc_full(0),    phase_oa0);
            mbarrier_wait(o_acc_full(1),    phase_oa1);
            mbarrier_wait(sm_stats_full(0), phase_ss0);
            mbarrier_wait(sm_stats_full(1), phase_ss1);
            PROF_END_X(EV_COR_WAIT_O);
            phase_oa0 ^= 1; phase_oa1 ^= 1;
            phase_ss0 ^= 1; phase_ss1 ^= 1;
            // 4 correction warps each arrive once on these:
            if (lane_id == 0) {
                mbarrier_arrive(o_acc_empty(0));
                mbarrier_arrive(o_acc_empty(1));
                mbarrier_arrive(s_p_o_empty(0));
                mbarrier_arrive(s_p_o_empty(1));
                mbarrier_arrive(sm_stats_empty(0));
                mbarrier_arrive(sm_stats_empty(1));
            }
        }
        // After mainloop: signal epi for both M-pairs.
        if (lane_id == 0) {
            mbarrier_arrive(o_epi_full(0));
            mbarrier_arrive(o_epi_full(1));
        }
#endif

    } else if (warp_id == MMA_WARP) {
        // ============================================================
        // MMA driver (warp 12, leader-CTA only — cga2 instructions fire on both)
        // ============================================================
        if (cta_rank == 0 && elect_sync_one()) {
            // Phase trackers as packed bitmaps (1 bit per stage). Avoids
            // ptxas spilling per-stage int arrays to local mem (was 24+8+8
            // bytes of spill in the array form).
            int phase_q       = 0;  // 2 bits used (q_stage=2)
            int phase_kv_full = 0;  // 6 bits used (kv_stage=6)
            int phase_pls     = 0;  // 2 bits used (q_stage=2)

            // Wait both Q tiles loaded.
            PROF_BEGIN_X(EV_MMA_WAIT_Q);
            mbarrier_wait(q_full(0), (phase_q >> 0) & 1); phase_q ^= 1 << 0;
            mbarrier_wait(q_full(1), (phase_q >> 1) & 1); phase_q ^= 1 << 1;
            PROF_END_X(EV_MMA_WAIT_Q);
            asm volatile("tcgen05.fence::after_thread_sync;");

            // Per iter j: K[j] @ stage (2j)%6, V[j] @ stage (2j+1)%6.
            // Order:  wait K_j → QK_p0 + QK_p1 → wait V_j → wait p_lastsplit_p0
            //         → PV_p0 → wait p_lastsplit_p1 → PV_p1
            // Slot release: kv_empty(K_stage) after QK done, kv_empty(V_stage) after PV done.
            for (int j = 0; j < n_kv; j++) {
                const int s_K = (2*j    ) % KV_STAGE;
                const int s_V = (2*j + 1) % KV_STAGE;

                PROF_BEGIN_X(EV_MMA_WAIT_K);
                mbarrier_wait(kv_full(s_K), (phase_kv_full >> s_K) & 1);
                phase_kv_full ^= 1 << s_K;
                PROF_END_X(EV_MMA_WAIT_K);

                issue_qk(0, s_K);
                issue_qk(1, s_K);

                // Release K stage as soon as both QK MMAs are committed.
                // tcgen05.commit waits for prior tcgen05 ops to drain, then
                // fires kv_empty(s_K) on BOTH peer CTAs' mbars (multicast).
                asm volatile(
                    "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster"
                    ".multicast::cluster.b64 [%0], %1;" ::
                        "r"(kv_empty(s_K)), "h"(cta_mask) : "memory");

                PROF_BEGIN_X(EV_MMA_WAIT_V);
                mbarrier_wait(kv_full(s_V), (phase_kv_full >> s_V) & 1);
                phase_kv_full ^= 1 << s_V;
                PROF_END_X(EV_MMA_WAIT_V);

                // Wait softmax done with P[0] and P[1] (stub fires immediately
                // after S is signaled, so this is mostly free in stub mode).
#if !SKIP_SM_FULL
                PROF_BEGIN_X(EV_MMA_WAIT_PR0);
                mbarrier_wait(p_lastsplit_full(0), (phase_pls >> 0) & 1); phase_pls ^= 1 << 0;
                PROF_END_X(EV_MMA_WAIT_PR0);
#endif
                issue_pv(0, s_V, j == 0);
#if !SKIP_SM_FULL
                PROF_BEGIN_X(EV_MMA_WAIT_PR1);
                mbarrier_wait(p_lastsplit_full(1), (phase_pls >> 1) & 1); phase_pls ^= 1 << 1;
                PROF_END_X(EV_MMA_WAIT_PR1);
#endif
                issue_pv(1, s_V, j == 0);

                // Release V stage after both PVs committed.
                asm volatile(
                    "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster"
                    ".multicast::cluster.b64 [%0], %1;" ::
                        "r"(kv_empty(s_V)), "h"(cta_mask) : "memory");
            }
            // Final tcgen05 commit drains all in-flight MMAs.
            asm volatile(
                "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster"
                ".multicast::cluster.b64 [%0], %1;" ::
                    "r"(tmem_dealloc), "h"(cta_mask) : "memory");
        }

    } else if (warp_id == EPI_WARP) {
        // ============================================================
        // EPILOGUE (warp 13, stub: wait o_epi, sentinel store)
        // ============================================================
#if !SKIP_SM_FULL
        PROF_BEGIN_X(EV_EPI_WAIT);
        mbarrier_wait(o_epi_full(0), 0);
        mbarrier_wait(o_epi_full(1), 0);
        PROF_END_X(EV_EPI_WAIT);
#endif

    } else if (warp_id == LOAD_WARP) {
        // ============================================================
        // TMA LOAD (warp 14)
        // ============================================================
        if (elect_sync_one()) {
            // Q TMAs (2 stages, separate mbars).
            PROF_BEGIN_X(EV_TMA_Q);
            for (int p = 0; p < Q_STAGE; p++) {
                asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;" ::
                             "r"(q_full(p)), "r"(Q_BYTES_PER_STAGE) : "memory");
                tma_3d_local(Q_smem(p), &Q_tmap, 0,
                             q_pair_offset[p] + (int)cta_rank * M_PER_CTA, 0,
                             q_full(p));
            }
            PROF_END_X(EV_TMA_Q);

            // K|V interleaved ring. K[j] @ stage (2j)%6, V[j] @ stage (2j+1)%6.
            // Slot reuse: wait kv_empty(stage) before issuing TMA into it.
            int phase_kv_empty = 0;  // 6 bits used (kv_stage=6) — bitmap, not array
            for (int j = 0; j < n_kv; j++) {
                const int s_K = (2*j    ) % KV_STAGE;
                const int s_V = (2*j + 1) % KV_STAGE;

                // K
                if (j >= N_KV_PAIRS_INFLIGHT) {
                    PROF_BEGIN_X(EV_TMA_WAIT_E);
                    mbarrier_wait(kv_empty(s_K), (phase_kv_empty >> s_K) & 1);
                    phase_kv_empty ^= 1 << s_K;
                    PROF_END_X(EV_TMA_WAIT_E);
                }
                PROF_BEGIN_X(EV_TMA_K);
                {
                    const int kf_peer0 = kv_full(s_K) & Sm100MmaPeerBitMask;
                    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;" ::
                                 "r"(kf_peer0),
                                 "r"(KV_BYTES_PER_STAGE) : "memory");
                    const int k_y = j * BLOCK_K + (int)cta_rank * (BLOCK_K / CTA_GROUP);
                    tma_3d_cga2(KV_smem(s_K), &K_tmap, 0, k_y, 0, kf_peer0);
                }
                PROF_END_X(EV_TMA_K);

                // V
                if (j >= N_KV_PAIRS_INFLIGHT) {
                    PROF_BEGIN_X(EV_TMA_WAIT_E);
                    mbarrier_wait(kv_empty(s_V), (phase_kv_empty >> s_V) & 1);
                    phase_kv_empty ^= 1 << s_V;
                    PROF_END_X(EV_TMA_WAIT_E);
                }
                PROF_BEGIN_X(EV_TMA_V);
                {
                    const int kf_peer0 = kv_full(s_V) & Sm100MmaPeerBitMask;
                    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;" ::
                                 "r"(kf_peer0),
                                 "r"(KV_BYTES_PER_STAGE) : "memory");
                    tma_3d_cga2(KV_smem(s_V), &V_tmap, 0, j * BLOCK_K, (int)cta_rank, kf_peer0);
                }
                PROF_END_X(EV_TMA_V);
            }
        }
    }
    // (warp 15 = empty: just dec regs at top, then fall through to tail)

    // -------- Tail: wait MMA done, sentinel epi from TMEM, dealloc --------
    __syncthreads();
    mbarrier_wait(tmem_dealloc, 0);
    asm volatile("tcgen05.fence::after_thread_sync;");

    // Sentinel epi: one fp32 from each O accumulator → gmem (no real TMA store).
    if (warp_id < 4) {
        const int row_addr_p0 = O_acc_p0 + ((warp_id * 32) << 16);
        const int row_addr_p1 = O_acc_p1 + ((warp_id * 32) << 16);
        float v0, v1;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];"
                     : "=f"(v0) : "r"(row_addr_p0));
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];"
                     : "=f"(v1) : "r"(row_addr_p1));
        asm volatile("tcgen05.wait::ld.sync.aligned;");
        if (lane_id == 0) {
            O_gmem[(int)cta_rank * 32 + warp_id]      = v0;
            O_gmem[(int)cta_rank * 32 + 16 + warp_id] = v1;
        }
    }
    __syncthreads();
    if (warp_id == 1) {
        asm volatile(
            "tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, 512;" ::"r"(taddr));
    }
    PROF_FLUSH(prof_buf, prof_count);
}

// =============================================================================
// Host launch  (mirrors v4: same TMA tensormap encoding for Q/K/V)
// =============================================================================
std::vector<torch::Tensor> redist_v6_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
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
    CUtensorMap Q_tmap{}, K_tmap{}, V_tmap{};
    encode(&Q_tmap, reinterpret_cast<const nv_bfloat16*>(Q.data_ptr()),
           (uint64_t)queries, M_PER_CTA, 2);
    encode(&K_tmap, reinterpret_cast<const nv_bfloat16*>(K.data_ptr()), (uint64_t)Sk,
           (uint32_t)(BLOCK_K / CTA_GROUP), 2);
    encode(&V_tmap, reinterpret_cast<const nv_bfloat16*>(V.data_ptr()), (uint64_t)Sk,
           (uint32_t)BLOCK_K, 1);

    cudaFuncSetAttribute(redist_v6_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, DYN_SMEM_BYTES);

    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto O = torch::zeros({(int64_t)num_clusters * 64}, opts);

    auto i32_opts = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    auto prof_buf = torch::zeros({PROF_MAX_EVENTS * 6}, i32_opts);
    auto prof_count = torch::zeros({1}, i32_opts);

    dim3 grid(CTA_GROUP, num_clusters, 1);
    redist_v6_kernel<<<grid, TB_SIZE, DYN_SMEM_BYTES>>>(
        Q_tmap, K_tmap, V_tmap,
        reinterpret_cast<float*>(O.data_ptr()), n_kv,
        reinterpret_cast<ProfEvent*>(prof_buf.data_ptr()),
        reinterpret_cast<int*>(prof_count.data_ptr()));
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "launch failed: ", cudaGetErrorString(err));
    cudaDeviceSynchronize();
    return {O, prof_buf, prof_count};
}
