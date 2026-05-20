// llm_fa.cu — extends redist_v6 with FULL online softmax + correction.
//
// What's new vs redist_v6:
//   * Real online softmax in WG0/1: track per-lane m_i (running rowmax) and
//     l_i (running sum of exp2). Cast P = exp2((S - m_new) * log2(e)/sqrt(D))
//     to bf16. Send acc_scale = exp2((m_old - m_new) * log2(e)/sqrt(D)) to
//     correction via sScale smem (matches FA4's sm_stats path).
//   * Real correction in WG2: receive acc_scale, rescale O accumulator in
//     TMEM (LD → fmul → ST). On final iter, divide by l_i to produce
//     final O = softmax(QK^T / sqrt(D)) @ V.
//   * Real TMA-store epi (warp 13): O TMEM → sO smem → gmem via cga2 TMA.
//
// FA4 references:
//   * softmax_loop:    flash_fwd_sm100.py:1688
//   * correction_loop: flash_fwd_sm100.py:2163
//
// What changes from redist_v4 (inherited from v6):
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
#ifndef EPI_FULL
// 1 = real epilogue: writes full Sq × D bf16 output via TMEM → reg → gmem.
// 0 = legacy sentinel: writes 8 fp32 per CTA for quick verification only.
// Default ON. Legacy path retained behind the flag for apples-to-apples
// benchmarks against the pre-epi numbers.
#define EPI_FULL 1
#endif
#ifndef MMA_REORDER
// 0 = legacy schedule (per iter j: Q0K_j, Q1K_j, P0V_j, P1V_j sequential).
// 1 = pipelined schedule (DEFAULT — saves ~1ms at worst case):
//        prologue: Q0K_0
//        step S in [0..n_kv): P1V_{S-1} (S>=1), Q1K_S, P0V_S, Q0K_{S+1} (S+1<n_kv)
//        epilogue: P1V_{n_kv-1}
//     Q0K is issued one step before the matching P0V → softmax of pair-0
//     gets two MMA-issues of overlap window before its PV waits.
//     Same for pair 1 via the across-step Q1K → P1V offset.
#define MMA_REORDER 1
#endif
#ifndef SKIP_SM_EXP
// 1 = SM warpgroup skips ALL math (LD S, rowmax, exp2 loop, TMEM-store P)
// but still drives the mbar protocol so COR runs normally. Used to isolate
// the latency contribution of the COR rescale path. Result will be wrong;
// only the timing is meaningful. Defaults to 0 (real softmax).
#define SKIP_SM_EXP 0
#endif
#ifndef USE_EX2_EMU
// 0 = MUFU.EX2 (XU pipe). 1 = polynomial emulation (FMA pipe — FFMA2/FADD2).
// 2 = HYBRID matching FA4 ex2_emu_freq=12,res=4 (12.5% poly on FMA pipe, 87.5%
//     MUFU.EX2 on XU pipe). Offloads ~33% of XU pressure to FMA without
//     overloading FMA. Default = 2 (FA4-faithful, −130 µs at worst case).
//     Set USE_EX2_EMU=0 to disable hybrid; USE_EX2_EMU=1 to use 100% poly (regresses).
#define USE_EX2_EMU 2
#endif
// SKIP_SM_FULL=1: WG0/1, WG2, warp 13 fully idle. MMA bypasses s_p_o/o_acc
//   waits. True TMA+MMA SOL — useful for measuring the pipeline floor.
#ifndef SKIP_SM1_EXP
#define SKIP_SM1_EXP 0
#endif
#ifndef LSUM_4STREAM
#define LSUM_4STREAM 1
#endif
#ifndef PINGPONG
#define PINGPONG 0
#endif
#ifndef SPLIT_P_ARRIVE
#define SPLIT_P_ARRIVE 0
#endif
#ifndef STREAM_STP
#define STREAM_STP 1
#endif

// SM_LEVEL: incremental bisection on top of SKIP_SM_FULL=1 (= pure SOL).
//   0 = pure SOL (SM body fully disabled)
//   1 = + LD-S + rowmax + acc_scale + sScale write
//   2 = + Phase A FFMA2 scale_subtract (register-only)
//   3 = + Phase B EX2 + cvt + 4-stream FADD2 l_sum + l_state update
//        (register-only — STTM still gated off)
//   4 = + STTM to TMEM + mbar arrives (= full integration; race-prone in SOL)
// SKIP_SM_FULL=0 ignores SM_LEVEL (always runs full kernel).
#ifndef SM_LEVEL
#define SM_LEVEL 0
#endif

// Use FA4-style per-warp-pair named-bar handshake for SM→COR sm_stats signaling
// instead of mbarrier_arrive/mbarrier_wait. Lighter weight: bar.arrive is ~10
// cycles, mbarrier_arrive is ~30 cycles + SMEM atomic + parity flip.
#ifndef USE_NAMED_BAR_SM_STATS
#define USE_NAMED_BAR_SM_STATS 1
#endif

// FA4 pattern (flash_fwd_sm100.py:2158 + softmax.py:218): perform the l_partial
// (row_sum) reduction AS A SEPARATE PHASE after EX2+cvt+STTM, not interleaved
// with the EX2 loop. Keeps fp32 exp2 values alive in registers across STTM,
// then reduces them at the end. Removes FMA-pipe pressure from the EX2 phase.
// Accepts higher register pressure (may spill — FA4 has 57 spill ops here).
#ifndef LSUM_AT_END
#define LSUM_AT_END 1
#endif

// FA4 sm_stats back-pressure wait (flash_fwd_sm100.py:2157
// pipeline_sm_stats.producer_acquire_w_index_phase). SM waits at end of each
// iter for COR to have consumed previous-iter stats. Default ON: matches FA4
// and wins ~40 µs at worst case in current context (post SPLIT_P_PV).
#ifndef BACKPRESSURE_SM_STATS
#define BACKPRESSURE_SM_STATS 1
#endif

// FA4 pipeline_p_lastsplit producer_group = softmax_warps_cluster (4 warps × cta_group=2 = 8).
// In v8 this defaults to a per-CTA mbar with count=4 (each peer waits independently).
// Cluster-shared form: count = 4 * CTA_GROUP = 8, each softmax warp arrives on BOTH local
// and peer mbar (via mapa.shared::cluster). MMA leader on each CTA waits on its local mbar
// which now counts 4 self + 4 peer arrives. Matches FA4 exactly.
// 0 = per-CTA count=4 (legacy v8), 1 = cluster-shared count=8 (FA4-faithful).
#ifndef P_LASTSPLIT_CLUSTER
#define P_LASTSPLIT_CLUSTER 0
#endif

// FA4 split_P_arrive (blackwell_helpers.py:529-590 gemm_ptx_partial with
// mbar_ptr): SM fires s_p_o_empty after first STTM chunk (early — partial P
// ready); MMA's PV issues first 6 of 8 cga2 MMAs (read first 75% of P), waits
// inline on p_lastsplit_full mbar (= "all P done"), then issues last 2 MMAs.
// Enables MMA to start consuming P before SM finishes the full P-store.
#ifndef SPLIT_P_PV
#define SPLIT_P_PV 1
#endif

// PER_PAIR_DRAIN: FA4-style per-pair tcgen05.commit after each pair's final PV
// (last iter), instead of a single tmem_dealloc commit draining BOTH pairs.
// Lets the tail epi for pair 0 (warps 0-3, LDTM + scale + cvt + st.gmem.b128)
// start as soon as pair 0's last PV drains — overlapping with pair 1's PV
// issue + drain on the MMA critical path. Pair 1 epi (warps 4-7) gated on
// o_acc_full(1). Replaces the single tmem_dealloc wait with per-pair waits.
// Default OFF (opt-in). Re-uses existing o_acc_full[Q_STAGE=2] mbars (count=1).
#ifndef PER_PAIR_DRAIN
#define PER_PAIR_DRAIN 0
#endif
constexpr int PV_MMA_TOTAL  = 8;   // BLOCK_K / MMA_K = 128 / 16
// Our STREAM_STP writes P in 2 chunks of 32 b32 cols each (first chunk = cols
// 64..95 = first 64 bf16 elements; second = cols 96..127 = last 64 bf16).
// Each PV MMA reads MMA_K/2=8 b32 cols, so MMAs 0..3 consume first STTM chunk,
// MMAs 4..7 consume second. SPLIT=4 = "first chunk fully ready" sync point.
constexpr int PV_MMA_SPLIT  = 4;

// Phase enable gates — each phase activates when SM_LEVEL reaches it OR full kernel.
#define SM_BODY_ON         (!SKIP_SM_FULL || (SM_LEVEL >= 1))  // LD-S + rowmax
#define SM_PHASE_A_ON      (!SKIP_SM_FULL || (SM_LEVEL >= 2))  // Phase A FFMA2
#define SM_PHASE_B_ON      (!SKIP_SM_FULL || (SM_LEVEL >= 3))  // Phase B EX2 + l_sum
#define SM_INTEGRATION_ON  (!SKIP_SM_FULL || (SM_LEVEL >= 4))  // STTM + mbar arrives

// SM0/SM1 ping-pong critical-section barriers (FA4 paper §3.1.2):
// Bar 4 = "SM0 done with EX2", bar 5 = "SM1 done with EX2".
// count = 256 = 8 SM warps × 32 lanes. Free ids (SM↔COR uses 8-15).
constexpr int BAR_SM0_DONE  = 4;
constexpr int BAR_SM1_DONE  = 5;
constexpr int PP_BAR_COUNT  = 256;
// SM↔COR sm_stats handshake via named bars (FA4 pattern, flash_fwd_sm100.py:940).
// 8 barriers: ids 8..15 = base + pair*4 + warp_id_local. count=64 = 1 SM warp +
// 1 COR warp (2 warps × 32 lanes). Per-warp-pair fine-grained handshake replaces
// our heavier mbarrier_arrive (atomic counter + parity flip).
constexpr int BAR_SM_STATS_BASE  = 8;
constexpr int SM_STATS_BAR_COUNT = 64;
#ifndef SKIP_SM_FULL
#define SKIP_SM_FULL 0
#endif
#define PROF_PER_WARP_SLOTS 320
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
    EV_SM_RM       = 14,  // softmax phase 1: LD S → rowmax/FMNMX3 → acc_scale ex2 + tau → write sScale → EARLY arrive sm_stats (COR trigger)
    EV_SM_EX2      = 15,  // softmax phase 2: 128× ex2.approx + ffma + cvt to bf16x2 + accumulate l_new (the actual exp math; runs in parallel with COR)
    EV_COR_WAIT_O  = 16,  // COR pair-0 region (wait sm_stats(0) → rescale O for pair 0 → fire)
    EV_EPI_WAIT    = 17,
    EV_COR_WAIT_O1 = 18,  // COR pair-1 region (wait sm_stats(1) → rescale O for pair 1 → fire)
    EV_SM_STP      = 19,  // softmax phase 3: tcgen05.st write P to TMEM → arrive p_lastsplit + s_p_o_empty
    EV_PP_WAIT     = 20,  // PINGPONG: SM0/SM1 waiting on partner's bar/mbar (gap before EX2 enters CS)
    EV_PP_ARRIVE   = 21,  // PINGPONG: SM0/SM1 arrive (release partner). Tiny marker.
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

// SMEM byte counts (per CTA). When ENABLE_PROF=1, we skip the sO buffer
// (allocated but unused in current sentinel epi) to free 64 KB for the prof
// scratch (~50 KB needed for 16 warps × PROF_PER_WARP_SLOTS records).
constexpr int Q_BYTES_PER_STAGE  = M_PER_CTA * N_DIM * 2;            // 32 KB per Q tile
constexpr int Q_BYTES_TOTAL      = Q_STAGE * Q_BYTES_PER_STAGE;      // 64 KB
// V8: epi writes TMEM → reg → gmem directly (no sO staging). Free this 64 KB
// for deeper KV pipelining.
constexpr int O_BYTES_PER_STAGE  = 0;
constexpr int O_BYTES_TOTAL      = 0;
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
#ifndef MBAR_WAIT_TESTWAIT
#define MBAR_WAIT_TESTWAIT 0
#endif
#if MBAR_WAIT_TESTWAIT
    // mbarrier.test_wait — synchronous test, no internal sleep hint. ptxas
    // emits SYNCS.PHASECHK.TRANS64 (no .TRYWAIT) and no BSSY/BSYNC scoping
    // around the spin. Use when the producer is expected to be ready already
    // (avoids the 100-cycle TRYWAIT latency that causes long_scoreboard stalls
    // on the predicate-dependent branch).
    asm volatile("{\n\t.reg .pred P1;\n\t"
                 "LAB_WAIT:\n\t"
                 "mbarrier.test_wait.parity.shared::cta.b64 P1, [%0], %1;\n\t"
                 "@P1 bra.uni DONE;\n\t"
                 "bra.uni LAB_WAIT;\n\t"
                 "DONE:\n\t}" ::"r"(mbar_addr), "r"(phase));
#else
    // FA4-faithful: 0x989680 (10 ms) suspendTimeHint — matches blackwell_helpers.py:522.
    // FA4-faithful: no .acquire.cta ordering (relaxed). TMEM ordering handled
    // by explicit tcgen05.fence; SMEM ordering for sm_stats handshake handled
    // by bar.sync (named bar) when USE_NAMED_BAR_SM_STATS=1.
    uint32_t ticks = 0x989680;
    asm volatile("{\n\t.reg .pred P1;\n\t"
                 "LAB_WAIT:\n\t"
                 "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, %2;\n\t"
                 "@P1 bra.uni DONE;\n\t"
                 "bra.uni LAB_WAIT;\n\t"
                 "DONE:\n\t}" ::"r"(mbar_addr), "r"(phase), "r"(ticks));
#endif
}
__device__ __forceinline__ void mbarrier_arrive(int mbar_addr) {
    asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" ::"r"(mbar_addr) : "memory");
}
// Cluster-multicast arrive: arrive on peer CTA's mbar (same offset in their SMEM).
// Used by pipeline_p_lastsplit when P_LASTSPLIT_CLUSTER=1 (FA4 producer_group=softmax_warps_cluster).
__device__ __forceinline__ void mbarrier_arrive_peer(int mbar_addr) {
    uint32_t peer_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(peer_rank));
    peer_rank ^= 1u;
    uint32_t peer_addr;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                 : "=r"(peer_addr) : "r"(mbar_addr), "r"(peer_rank));
    asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];"
                 ::"r"(peer_addr) : "memory");
}

// Named barrier helpers. FA4 uses asymmetric arrive/wait:
//   producer (SM): bar.arrive  — fire-and-forget, no block
//   consumer (COR): bar.sync   — arrive-and-wait
// Barriers 8..15 reserved for SM↔COR (8-11 = pair0 warps 0-3, 12-15 = pair1).
// tcgen05.st of 32×32 b32 values (32 cols × 32 rows = 1024 b32 per warp).
// `addr` = TMEM target address. `p` = pointer to 32 b32 values in registers.
__device__ __forceinline__ void tcgen05_st_32x32b_x32(int addr, const uint32_t* p) {
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x32.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8,"
        "%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,"
        "%27,%28,%29,%30,%31,%32};"
        ::"r"(addr),
        "r"(p[0]),  "r"(p[1]),  "r"(p[2]),  "r"(p[3]),
        "r"(p[4]),  "r"(p[5]),  "r"(p[6]),  "r"(p[7]),
        "r"(p[8]),  "r"(p[9]),  "r"(p[10]), "r"(p[11]),
        "r"(p[12]), "r"(p[13]), "r"(p[14]), "r"(p[15]),
        "r"(p[16]), "r"(p[17]), "r"(p[18]), "r"(p[19]),
        "r"(p[20]), "r"(p[21]), "r"(p[22]), "r"(p[23]),
        "r"(p[24]), "r"(p[25]), "r"(p[26]), "r"(p[27]),
        "r"(p[28]), "r"(p[29]), "r"(p[30]), "r"(p[31]));
}

// FA4 uses STTM.x16 for P writes (apply_exp2_convert stores fragment-by-fragment).
// Each x16 call stores 16 b32 per lane = 16 fp32-equivalents (32 bf16 packed).
__device__ __forceinline__ void tcgen05_st_32x32b_x16(int addr, const uint32_t* p) {
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x16.b32 [%0], "
        "{%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16};"
        ::"r"(addr),
        "r"(p[0]),  "r"(p[1]),  "r"(p[2]),  "r"(p[3]),
        "r"(p[4]),  "r"(p[5]),  "r"(p[6]),  "r"(p[7]),
        "r"(p[8]),  "r"(p[9]),  "r"(p[10]), "r"(p[11]),
        "r"(p[12]), "r"(p[13]), "r"(p[14]), "r"(p[15]));
}

__device__ __forceinline__ void named_barrier_arrive(int barrier_id, int count) {
    asm volatile("bar.arrive %0, %1;" :: "r"(barrier_id), "r"(count) : "memory");
}
__device__ __forceinline__ void named_barrier_sync(int barrier_id, int count) {
    asm volatile("bar.sync %0, %1;" :: "r"(barrier_id), "r"(count) : "memory");
}

// FA4 polynomial exp2 emulation (degree 3 Remez), packed f32x2.
// Replaces 2× MUFU.EX2 (XU pipe, ~4 cycles each, contended across 8 SM warps)
// with packed fp32x2 ops on the FMA pipe (FFMA2/FADD2 SASS — 2 fp32 ops/cycle).
// Source: fa4 utils.py:606 ex2_emulation_2 / e2e_asm2.
//   exp2(x) = 2^floor(x) * poly(frac(x))
//   poly(f) = ((0.0770*f + 0.2274)*f + 0.6948)*f + 1.0   for f in [0, 1)
// Operates in-place on a pair (a, b). Single asm block — ptxas can still
// reorder different invocations across the unrolled loop.
__device__ __forceinline__ void ex2_poly_pair(float& a, float& b) {
    asm(
        "{\n\t"
        ".reg .f32 fa, fb;\n\t"
        ".reg .b64 lin, lmag, lflrm, lflr, lfrac, lpoly, lc1, lc2, lc3, lone;\n\t"
        ".reg .s32 ri_a, ri_b, rp_a, rp_b, rs_a, rs_b, ro_a, ro_b;\n\t"
        "max.ftz.f32 fa, %0, 0fC2FE0000;\n\t"
        "max.ftz.f32 fb, %1, 0fC2FE0000;\n\t"
        "mov.b64 lin, {fa, fb};\n\t"
        "mov.b64 lmag, {0f4B400000, 0f4B400000};\n\t"
        "add.rm.ftz.f32x2 lflrm, lin, lmag;\n\t"
        "sub.rn.ftz.f32x2 lflr,  lflrm, lmag;\n\t"
        "sub.rn.ftz.f32x2 lfrac, lin,   lflr;\n\t"
        "mov.b64 lc3, {0f3D9DF09D, 0f3D9DF09D};\n\t"
        "mov.b64 lc2, {0f3E6906A4, 0f3E6906A4};\n\t"
        "mov.b64 lc1, {0f3F31F519, 0f3F31F519};\n\t"
        "mov.b64 lone,{0f3F800000, 0f3F800000};\n\t"
        "fma.rn.ftz.f32x2 lpoly, lfrac, lc3, lc2;\n\t"
        "fma.rn.ftz.f32x2 lpoly, lpoly, lfrac, lc1;\n\t"
        "fma.rn.ftz.f32x2 lpoly, lpoly, lfrac, lone;\n\t"
        "mov.b64 {ri_a, ri_b}, lflrm;\n\t"
        "mov.b64 {rp_a, rp_b}, lpoly;\n\t"
        "shl.b32 rs_a, ri_a, 23;\n\t"
        "shl.b32 rs_b, ri_b, 23;\n\t"
        "add.s32 ro_a, rs_a, rp_a;\n\t"
        "add.s32 ro_b, rs_b, rp_b;\n\t"
        "mov.b32 %0, ro_a;\n\t"
        "mov.b32 %1, ro_b;\n\t"
        "}"
        : "+f"(a), "+f"(b)
    );
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
// FA4 CuTeDSL split-desc helper (blackwell_helpers.py:174-180). The descriptor
// HI 32 bits (SBO=0x40, base_offset=1, layout=SWIZZLE_128B → 0x40004040) is a
// compile-time constant; bake it as a hex literal inside the asm string so the
// only runtime input is the LO 32 bits (encoded start_address). ptxas treats
// the LO as a 32-bit "r" register and avoids the R→UR transfer that the full
// 64-bit "l" descriptor would require.
// HI 32 bits for QK descriptor = 0x40004040:
//   bit 32-45: SBO encoded = (8*128 & 0x3FFFF) >> 4 = 0x40 → HI[0:13] = 0x40
//   bit 46:    base_offset = 1                       → HI[14]   = 1
//   bits 61-63: layout_type = 2 (SWIZZLE_128B)       → HI[29:31]= 010
// Combined: 0x40 | (1<<14) | (2<<29) = 0x40004040.
__device__ __forceinline__ uint32_t desc_lo_qk(int addr) {
    return (uint32_t)desc_encode((uint64_t)(uint32_t)addr);
}
__device__ __forceinline__ void cga2_mma_ss(
    int taddr, uint64_t a_desc, uint64_t b_desc, uint32_t i_desc, int en_d) {
    asm volatile(
        "{\n\t.reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n\t}" ::
            "r"(taddr), "l"(a_desc), "l"(b_desc), "r"(i_desc), "r"(en_d));
}
// FA4 CuTeDSL-style split-desc: HI baked as 0x40004040 literal in PTX; only LO
// passed as 32-bit "r" input. Closes 16-21% of sub-1-wave gap vs full 64-bit
// "l" desc. Used by issue_qk (QK MMAs use SBO=8*128 with no LBO).
__device__ __forceinline__ void cga2_mma_ss_split(
    int taddr, uint32_t a_lo, uint32_t b_lo, uint32_t i_desc, int en_d) {
    asm volatile(
        "{\n\t.reg .pred p;\n\t"
        ".reg .b64 da, db;\n\t"
        "mov.b64 da, {%1, 0x40004040};\n\t"
        "mov.b64 db, {%2, 0x40004040};\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::2.kind::f16 [%0], da, db, %3, p;\n\t}" ::
            "r"(taddr), "r"(a_lo), "r"(b_lo), "r"(i_desc), "r"(en_d));
}
__device__ __forceinline__ void cga2_mma_ts(
    int taddr, uint32_t tmem_a, uint64_t b_desc, uint32_t i_desc, int en_d) {
    asm volatile(
        "{\n\t.reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::2.kind::f16 [%0], [%1], %2, %3, p;\n\t}" ::
            "r"(taddr), "r"(tmem_a), "l"(b_desc), "r"(i_desc), "r"(en_d));
}
// PV split-desc v2: caller passes LO with LBO already baked in (computed once
// per issue_pv via UIADD3 stride). The asm just does mov.b64. No OR per call.
__device__ __forceinline__ void cga2_mma_ts_split(
    int taddr, uint32_t tmem_a, uint32_t b_lo, uint32_t i_desc, int en_d) {
    asm volatile(
        "{\n\t.reg .pred p;\n\t"
        ".reg .b64 db;\n\t"
        "mov.b64 db, {%2, 0x40004040};\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::2.kind::f16 [%0], [%1], db, %3, p;\n\t}" ::
            "r"(taddr), "r"(tmem_a), "r"(b_lo), "r"(i_desc), "r"(en_d));
}
__device__ __forceinline__ uint32_t desc_addr_enc(int addr) {
    return (uint32_t)desc_encode((uint64_t)(uint32_t)addr);
}
// LBO=128 encoded = (128 >> 4) = 8, placed at bits 16-29 of LO = 0x80000.
__device__ __forceinline__ uint32_t desc_lo_v_pv_base(int addr) {
    return ((uint32_t)desc_encode((uint64_t)(uint32_t)addr)) | 0x80000u;
}

// PER_PAIR_DRAIN helper: fire a cluster-multicast tcgen05.commit that arrives
// on the given mbar AFTER all prior tcgen05 ops on this issue queue drain.
// Used after each pair's final PV (last iter) to signal pair-local O drain so
// the tail epi for that pair can start as soon as its MMAs finish, even while
// the other pair's PV is still in-flight.
#define MMA_PAIR_DRAIN_COMMIT(mbar) do {                                  \
    asm volatile(                                                         \
        "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster" \
        ".multicast::cluster.b64 [%0], %1;" ::                            \
            "r"(mbar), "h"(cta_mask) : "memory");                         \
} while (0)

// =============================================================================
// Mbar slot layout (depends only on Q_STAGE / KV_STAGE — hoisted from kernel
// body so file-scope emit_qk / emit_pv can address mbars by offset).
// =============================================================================
#define MB_Q_FULL_OFF       0
#define MB_Q_EMPTY_OFF      (MB_Q_FULL_OFF       + Q_STAGE)
#define MB_KV_FULL_OFF      (MB_Q_EMPTY_OFF      + Q_STAGE)
#define MB_KV_EMPTY_OFF     (MB_KV_FULL_OFF      + KV_STAGE)
#define MB_SPO_FULL_OFF     (MB_KV_EMPTY_OFF     + KV_STAGE)
#define MB_SPO_EMPTY_OFF    (MB_SPO_FULL_OFF     + Q_STAGE)
#define MB_PLS_FULL_OFF     (MB_SPO_EMPTY_OFF    + Q_STAGE)
#define MB_PLS_EMPTY_OFF    (MB_PLS_FULL_OFF     + Q_STAGE)
#define MB_OACC_FULL_OFF    (MB_PLS_EMPTY_OFF    + Q_STAGE)
#define MB_OACC_EMPTY_OFF   (MB_OACC_FULL_OFF    + Q_STAGE)
#define MB_SMSTATS_FULL_OFF (MB_OACC_EMPTY_OFF   + Q_STAGE)
#define MB_SMSTATS_EMPTY_OFF (MB_SMSTATS_FULL_OFF + Q_STAGE)
#define MB_OEPI_FULL_OFF    (MB_SMSTATS_EMPTY_OFF + Q_STAGE)
#define MB_OEPI_EMPTY_OFF   (MB_OEPI_FULL_OFF    + Q_STAGE)
#define MB_TMEM_DEALLOC_OFF (MB_OEPI_EMPTY_OFF   + Q_STAGE)
#define MB_PP_SEQ_OFF       (MB_TMEM_DEALLOC_OFF + 1)
#define MB_TOTAL_SLOTS      (MB_PP_SEQ_OFF       + 2)

// =============================================================================
// MMA driver helpers (file-scope, ex-lambda)
// =============================================================================
// File-scope constexpr i_desc bit packings (depend only on file-scope consts).
// qk: A_layout=swizzle128, B_layout=swizzle128, neg_a=0, neg_b=0, a_transp=0,
//     b_transp=0, N=N_DIM/8, M=M_JOINT/16. pv: same + accum_C flag at bit 16.
constexpr uint32_t QK_I_DESC =
    (1U << 4U) | (1U << 7U) | (1U << 10U) |
    ((uint32_t)N_DIM >> 3U << 17U) |
    ((uint32_t)M_JOINT >> 4U << 24U);
constexpr uint32_t PV_I_DESC =
    (1U << 4U) | (1U << 7U) | (1U << 10U) | (1U << 16U) |
    ((uint32_t)N_DIM >> 3U << 17U) |
    ((uint32_t)M_JOINT >> 4U << 24U);
static_assert(QK_I_DESC == 0x10200490u, "QK_I_DESC");
static_assert(PV_I_DESC == 0x10210490u, "PV_I_DESC");

constexpr int Q_PANEL_OFFSET_FS = M_PER_CTA * 64 * 2;
constexpr int K_PANEL_OFFSET_FS = (BLOCK_K / CTA_GROUP) * 64 * 2;
constexpr int V_K_BYTES_PER_KC_FS = MMA_K * (N_DIM / CTA_GROUP) * 2;

// emit_qk: 8 cga2 QK MMAs (4 low panel, 4 high panel) on Q[pair] × K from kv
// stage `s_kv`. All capture-state passed as explicit args; no [&] closure.
__device__ __forceinline__ void emit_qk(
    int pair, int s_kv,
    int slot_taddr_p0, int slot_taddr_p1,
    int sQ_p0_, int sQ_p1_, int sKV_base_,
    int mb_, uint16_t cta_mask_) {
    asm volatile("tcgen05.fence::after_thread_sync;");
    const int slot_taddr_pair = (pair == 0) ? slot_taddr_p0 : slot_taddr_p1;
    const int Q_smem_pair     = (pair == 0) ? sQ_p0_ : sQ_p1_;
    const int K_smem_pair     = sKV_base_ + s_kv * KV_BYTES_PER_STAGE;
    const uint32_t q_lo_base    = desc_lo_qk(Q_smem_pair);
    const uint32_t k_lo_base    = desc_lo_qk(K_smem_pair);
    const uint32_t q_lo_panel   = desc_lo_qk(Q_smem_pair + Q_PANEL_OFFSET_FS);
    const uint32_t k_lo_panel   = desc_lo_qk(K_smem_pair + K_PANEL_OFFSET_FS);
    #pragma unroll
    for (int k2 = 0; k2 < 4; k2++) {
        cga2_mma_ss_split(slot_taddr_pair,
                          q_lo_base + (uint32_t)(k2 * 2),
                          k_lo_base + (uint32_t)(k2 * 2),
                          QK_I_DESC, k2 == 0 ? 0 : 1);
    }
    #pragma unroll
    for (int k2 = 0; k2 < 4; k2++) {
        cga2_mma_ss_split(slot_taddr_pair,
                          q_lo_panel + (uint32_t)(k2 * 2),
                          k_lo_panel + (uint32_t)(k2 * 2),
                          QK_I_DESC, 1);
    }
    asm volatile(
        "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster"
        ".multicast::cluster.b64 [%0], %1;" ::
            "r"(mb_ + (MB_SPO_FULL_OFF + pair) * 8), "h"(cta_mask_) : "memory");
}

// emit_pv: 8 cga2 PV MMAs (BLOCK_K/MMA_K=8 inner k-chunks) on P[pair] × V.
// `first_for_O = (j == 0)` — controls accumulator-disable on the first MMA.
__device__ __forceinline__ void emit_pv(
    int pair, int s_v, bool first_for_O, int pls_phase,
    int slot_taddr_p0, int slot_taddr_p1,
    int O_acc_p0_, int O_acc_p1_,
    int sKV_base_, int mb_) {
    asm volatile("tcgen05.fence::after_thread_sync;");
    const int slot_taddr_pair = (pair == 0) ? slot_taddr_p0 : slot_taddr_p1;
    const int O_acc_pair      = (pair == 0) ? O_acc_p0_     : O_acc_p1_;
    const int v_smem_addr     = sKV_base_ + s_v * KV_BYTES_PER_STAGE;
    const uint64_t v_desc_base = make_desc_v_pv(v_smem_addr);
    constexpr uint64_t v_desc_step = (uint64_t)V_K_BYTES_PER_KC_FS >> 4ULL;
    const int p_base = slot_taddr_pair + 64;
    constexpr int p_step = MMA_K / 2;
#if SPLIT_P_PV
    #pragma unroll
    for (int kc = 0; kc < PV_MMA_SPLIT; kc++) {
        cga2_mma_ts(O_acc_pair,
                    p_base + kc * p_step,
                    v_desc_base + (uint64_t)kc * v_desc_step,
                    PV_I_DESC,
                    (first_for_O && kc == 0) ? 0 : 1);
    }
    mbarrier_wait(mb_ + (MB_PLS_FULL_OFF + pair) * 8, pls_phase);
    #pragma unroll
    for (int kc = PV_MMA_SPLIT; kc < PV_MMA_TOTAL; kc++) {
        cga2_mma_ts(O_acc_pair,
                    p_base + kc * p_step,
                    v_desc_base + (uint64_t)kc * v_desc_step,
                    PV_I_DESC,
                    1);
    }
#else
    (void)pls_phase;
    #pragma unroll
    for (int kc = 0; kc < PV_MMA_TOTAL; kc++) {
        cga2_mma_ts(O_acc_pair,
                    p_base + kc * p_step,
                    v_desc_base + (uint64_t)kc * v_desc_step,
                    PV_I_DESC,
                    (first_for_O && kc == 0) ? 0 : 1);
    }
#endif
}

// emit_rescale_O: reverted to baseline. Tried mirroring FA4 (8× LDTM.x16 +
// 8 FMUL2 + STTM.x16) on 2026-05-12 — neutral perf (rescale only runs when
// the ballot mask fires, which is rare in steady state). The barrier-stall
// gap to FA4 lives at the COR sm_stats bar.sync, not in the rescale path.
__device__ __forceinline__ void emit_rescale_O(int o_addr, float scale) {
    uint64_t scale_pair;
    asm volatile("mov.b64 %0, {%1, %2};" : "=l"(scale_pair) : "f"(scale), "f"(scale));
    #pragma unroll
    for (int c = 0; c < 4; c++) {
        const int addr = o_addr + c * 32;
        float o[32];
        asm volatile(
            "tcgen05.ld.sync.aligned.32x32b.x32.b32 "
            "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,"
            "%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,"
            "%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,"
            "%30,%31}, [%32];"
            : "=f"(o[0]),  "=f"(o[1]),  "=f"(o[2]),  "=f"(o[3]),
              "=f"(o[4]),  "=f"(o[5]),  "=f"(o[6]),  "=f"(o[7]),
              "=f"(o[8]),  "=f"(o[9]),  "=f"(o[10]), "=f"(o[11]),
              "=f"(o[12]), "=f"(o[13]), "=f"(o[14]), "=f"(o[15]),
              "=f"(o[16]), "=f"(o[17]), "=f"(o[18]), "=f"(o[19]),
              "=f"(o[20]), "=f"(o[21]), "=f"(o[22]), "=f"(o[23]),
              "=f"(o[24]), "=f"(o[25]), "=f"(o[26]), "=f"(o[27]),
              "=f"(o[28]), "=f"(o[29]), "=f"(o[30]), "=f"(o[31])
            : "r"(addr));
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            asm volatile(
                "{\n\t"
                ".reg .b64 p, r;\n\t"
                "mov.b64 p, {%0, %1};\n\t"
                "mul.f32x2 r, p, %2;\n\t"
                "mov.b64 {%0, %1}, r;\n\t"
                "}"
                : "+f"(o[2*i]), "+f"(o[2*i+1])
                : "l"(scale_pair));
        }
        asm volatile(
            "tcgen05.st.sync.aligned.32x32b.x32.b32 [%0], "
            "{%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,"
            "%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,"
            "%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,"
            "%31,%32};"
            :: "r"(addr),
               "f"(o[0]),  "f"(o[1]),  "f"(o[2]),  "f"(o[3]),
               "f"(o[4]),  "f"(o[5]),  "f"(o[6]),  "f"(o[7]),
               "f"(o[8]),  "f"(o[9]),  "f"(o[10]), "f"(o[11]),
               "f"(o[12]), "f"(o[13]), "f"(o[14]), "f"(o[15]),
               "f"(o[16]), "f"(o[17]), "f"(o[18]), "f"(o[19]),
               "f"(o[20]), "f"(o[21]), "f"(o[22]), "f"(o[23]),
               "f"(o[24]), "f"(o[25]), "f"(o[26]), "f"(o[27]),
               "f"(o[28]), "f"(o[29]), "f"(o[30]), "f"(o[31]));
    }
    asm volatile("tcgen05.wait::st.sync.aligned;");
}

// =============================================================================
// Kernel
// =============================================================================
__global__ void __cluster_dims__(CTA_GROUP, 1, 1) __launch_bounds__(TB_SIZE, 1)
llm_fa_kernel(
    const __grid_constant__ CUtensorMap Q_tmap,
    const __grid_constant__ CUtensorMap K_tmap,
    const __grid_constant__ CUtensorMap V_tmap,
#if EPI_FULL
    __nv_bfloat16* O_gmem,
#else
    float* O_gmem,
#endif
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
    __shared__ uint64_t mbars[39];  // +2 for SM0/SM1 pingpong sequence mbars
    __shared__ int tmem_addr_smem[1];
    const int mb = (int)__cvta_generic_to_shared(mbars);
    // Index assignments (8 bytes each):
    // mbar layout (each 8 bytes), KV_STAGE-parameterized for v8 deeper pipelining:
    //   q_full[Q_STAGE], q_empty[Q_STAGE]
    //   kv_full[KV_STAGE], kv_empty[KV_STAGE]
    //   s_p_o_full[Q_STAGE], s_p_o_empty[Q_STAGE]
    //   p_lastsplit_full/empty[Q_STAGE], o_acc_full/empty[Q_STAGE]
    //   sm_stats_full/empty[Q_STAGE], o_epi_full/empty[Q_STAGE]
    //   tmem_dealloc
    // MB_*_OFF macros hoisted to file scope above; mbar-addr accessors stay here.
#define q_full(s)         (mb + (MB_Q_FULL_OFF   + (s)) * 8)
#define q_empty(s)        (mb + (MB_Q_EMPTY_OFF  + (s)) * 8)
#define kv_full(s)        (mb + (MB_KV_FULL_OFF  + (s)) * 8)
#define kv_empty(s)       (mb + (MB_KV_EMPTY_OFF + (s)) * 8)
#define s_p_o_full(s)     (mb + (MB_SPO_FULL_OFF + (s)) * 8)
#define s_p_o_empty(s)    (mb + (MB_SPO_EMPTY_OFF + (s)) * 8)
#define p_lastsplit_full(s)  (mb + (MB_PLS_FULL_OFF + (s)) * 8)
#define p_lastsplit_empty(s) (mb + (MB_PLS_EMPTY_OFF + (s)) * 8)
#define o_acc_full(s)     (mb + (MB_OACC_FULL_OFF + (s)) * 8)
#define o_acc_empty(s)    (mb + (MB_OACC_EMPTY_OFF + (s)) * 8)
#define sm_stats_full(s)  (mb + (MB_SMSTATS_FULL_OFF + (s)) * 8)
#define sm_stats_empty(s) (mb + (MB_SMSTATS_EMPTY_OFF + (s)) * 8)
#define o_epi_full(s)     (mb + (MB_OEPI_FULL_OFF + (s)) * 8)
#define o_epi_empty(s)    (mb + (MB_OEPI_EMPTY_OFF + (s)) * 8)
#define tmem_dealloc      (mb + MB_TMEM_DEALLOC_OFF * 8)
// pp_seq[0] = SM0_done (SM1 waits, SM0 arrives). pp_seq[1] = SM1_done (SM0 waits).
#define pp_seq(i)         (mb + (MB_PP_SEQ_OFF + (i)) * 8)

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
        // P_LASTSPLIT_CLUSTER=1: matches FA4 producer_group=softmax_warps_cluster (4*cta_group=8);
        // each softmax warp arrives on BOTH local and peer mbar via mapa.shared::cluster.
        for (int s = 0; s < Q_STAGE; s++) {
            mbarrier_init(p_lastsplit_full(s),  P_LASTSPLIT_CLUSTER ? 4 * CTA_GROUP : 4);
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
        // pp_seq mbars: count=1 (single arrive flips parity).
        // FA4 pingpong: SM0 arrives pp_seq[0], SM1 waits pp_seq[0] (and vice versa).
        mbarrier_init(pp_seq(0), 1);
        mbarrier_init(pp_seq(1), 1);
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
    // MMA dispatch: lambdas eliminated, now file-scope __forceinline__
    // functions taking all state as explicit args (see emit_qk / emit_pv
    // before the kernel). Wrapper macros keep the call sites short.
    // ===================================================================
    #define ISSUE_QK(pair, s_kv) \
        emit_qk((pair), (s_kv), slot_taddr_p0, slot_taddr_p1, sQ_p0, sQ_p1, sKV_base, mb, cta_mask)
    #define ISSUE_PV(pair, s_v, first_for_O, pls_phase) \
        emit_pv((pair), (s_v), (first_for_O), (pls_phase), slot_taddr_p0, slot_taddr_p1, O_acc_p0, O_acc_p1, sKV_base, mb)

    // ===================================================================
    // Warp dispatch
    // ===================================================================

    if (warp_id < SOFTMAX1_HI) {
        // ============================================================
        // SOFTMAX (warps 0-7 = WG0+WG1) — REAL online softmax
        // ============================================================
        // Each WG handles its own M-pair. WG0 → pair 0, WG1 → pair 1.
        // Each warp owns 32 rows; each lane within owns 1 row of 128 fp32.
        // Per-lane state: m (running rowmax in scaled-log2 space) and
        // l (running row sum of exp2). Updated each iter:
        //   m_new      = max(m_old, max(S_j) * scale_log2)
        //   acc_scale  = exp2(m_old - m_new)
        //   P_unnorm   = exp2(S_j * scale_log2 - m_new)   ← bf16 cast
        //   l_new      = acc_scale * l_old + sum(P_unnorm)
        // Sends acc_scale to correction via sScale[stage*M + lane_row].
        // After mainloop, writes l to sScale[Q_STAGE*M + stage*M + lane_row].
        // SCALE_LOG2 = (1/sqrt(D)) * log2(e) = (1/sqrt(128)) * 1.4427 ≈ 0.12748.
#if SM_BODY_ON
        constexpr float SCALE_LOG2 = 0.12747904f;  // (1/sqrt(128)) * log2(e)
        const bool is_p0 = (warp_id < SOFTMAX1_LO);
        const int pair   = is_p0 ? 0 : 1;
        const int slot_taddr_pair = is_p0 ? slot_taddr_p0 : slot_taddr_p1;
        const int warp_id_local   = is_p0 ? warp_id : warp_id - SOFTMAX1_LO;
        const int row_addr        = slot_taddr_pair + ((warp_id_local * 32) << 16);
        const int sScale_row      = pair * M_PER_CTA + warp_id_local * 32 + lane_id;
        float* sScale_ptr         = (float*)(smem_ptr + (Q_BYTES_TOTAL + O_BYTES_TOTAL + KV_BYTES_TOTAL));
        float m_state = -1e30f;  // running rowmax in scaled-log2 space
        float l_state = 0.0f;    // running row sum of exp2
        int phase_sr = 0;
        int phase_sse = 0;       // sm_stats_empty back-pressure phase (FA4 pattern)
#if PINGPONG == 3
        // FA4 ping-pong init: SM0 starts blocking (phase=1, mbar parity=0 → wait
        // blocks); SM1 starts non-blocking (phase=0 → returns immediately).
        // Therefore SM1 enters Phase B first; SM0 follows once SM1's arrive
        // flips its mbar parity. After EX2, SM0 fires pp_seq[1] (SM1's wake) and
        // SM1 fires pp_seq[0] (SM0's wake).
        int pp_phase = is_p0 ? 1 : 0;
#endif
        for (int j = 0; j < n_kv; j++) {
            PROF_BEGIN_X(EV_SM_WAIT_S);
            mbarrier_wait(s_p_o_full(pair), phase_sr);
            PROF_END_X(EV_SM_WAIT_S);
            phase_sr ^= 1;
#if SKIP_SM1_EXP
            // SM1-only bypass: warps 4-7 skip RM/EX2/STP math but keep the mbar
            // choreography so MMA + COR + pair-0 dependencies are unaffected.
            // Tests contention hypothesis: 8 SM warps hitting MUFU.EX2 on XU pipe
            // simultaneously. With SM1 silent, only SM0's 4 warps drive EX2.
            // Ignore correctness.
            if (!is_p0) {
                PROF_BEGIN_X(EV_SM_RM);
                sScale_ptr[sScale_row] = 1.0f;  // best-case COR (rescale skipped)
                __syncwarp();
                if (lane_id == 0) {
                    mbarrier_arrive(sm_stats_full(pair));
                }
                PROF_END_X(EV_SM_RM);
                PROF_BEGIN_X(EV_SM_EX2);
                PROF_END_X(EV_SM_EX2);
                PROF_BEGIN_X(EV_SM_STP);
                asm volatile("tcgen05.fence::before_thread_sync;");
                if (lane_id == 0) {
                    mbarrier_arrive(p_lastsplit_full(pair));
#if P_LASTSPLIT_CLUSTER
                    mbarrier_arrive_peer(p_lastsplit_full(pair));
#endif
                    mbarrier_arrive(s_p_o_empty(pair));
                }
                PROF_END_X(EV_SM_STP);
                continue;
            }
#endif
            asm volatile("tcgen05.fence::after_thread_sync;");
            PROF_BEGIN_X(EV_SM_RM);
#if SKIP_SM_EXP
            // Strip ALL SM math; just drive mbar choreography so COR runs.
            // SKIP_SM_EXP=1: acc_scale = 0.5 → ballot fires every iter, rescale_O
            //                runs (worst-case COR — what the FIRST few iters look like).
            // SKIP_SM_EXP=2: acc_scale = 1.0 → ballot empty, rescale_O skips
            //                (best-case COR — what most iters look like after tau stabilizes).
  #if SKIP_SM_EXP == 2
            sScale_ptr[sScale_row] = 1.0f;
  #else
            sScale_ptr[sScale_row] = 0.5f;
  #endif
            __syncwarp();
            if (lane_id == 0) {
                mbarrier_arrive(sm_stats_full(pair));
            }
            PROF_END_X(EV_SM_RM);
            PROF_BEGIN_X(EV_SM_EX2);
            // (no exp math)
            PROF_END_X(EV_SM_EX2);
            PROF_BEGIN_X(EV_SM_STP);
            asm volatile("tcgen05.fence::before_thread_sync;");
            if (lane_id == 0) {
                mbarrier_arrive(p_lastsplit_full(pair));
#if P_LASTSPLIT_CLUSTER
                mbarrier_arrive_peer(p_lastsplit_full(pair));
#endif
                mbarrier_arrive(s_p_o_empty(pair));
            }
            PROF_END_X(EV_SM_STP);
            continue;
#endif

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

#if SPLIT_P_ARRIVE
            // FA4 split_P_arrive (paper §3.1.2 "stage out storing P"): fire
            // SM's s_p_o_empty arrive AS SOON AS S has been read into registers.
            // SM no longer needs the S region of TMEM, so we signal MMA early.
            // s_p_o_empty count=8 (4 SM + 4 COR); SM's 4 arrives land here,
            // mbar flips when COR's 4 also land (or vice versa).
            //
            // SAFETY: v8's TMEM layout has S+P overlapping in slot_p (S = cols
            // 0..127, P = cols 64..127). Firing s_p_o_empty early lets MMA's
            // next-iter QK potentially write S_{j+1} to cols 0..127 BEFORE our
            // current P-store completes. This is the race FA4 avoids with
            // S_STAGE=2 (separate TMEM slot per iter). We are testing whether
            // the natural schedule (kv_full(K_{j+1}) dependency, MMA pipeline
            // ordering) prevents the race in practice on our shapes.
            asm volatile("tcgen05.fence::before_thread_sync;");
            if (lane_id == 0) {
                mbarrier_arrive(s_p_o_empty(pair));
            }
#endif

            // ----- 1. Compute m_local (max of S) — FA4-style 4-stream parallel reduce -----
            // (utils.py:251 fmax_reduce). 4 independent accumulator chains; ptxas
            // emits FMNMX3 (3-input max) for `max(acc, s[i], s[i+1])`. Breaks the
            // serial dependency that a single accumulator would create.
            float ml0 = fmaxf(s[0], s[1]);
            float ml1 = fmaxf(s[2], s[3]);
            float ml2 = fmaxf(s[4], s[5]);
            float ml3 = fmaxf(s[6], s[7]);
            #pragma unroll
            for (int i = 8; i < 128; i += 8) {
                ml0 = fmaxf(ml0, fmaxf(s[i + 0], s[i + 1]));
                ml1 = fmaxf(ml1, fmaxf(s[i + 2], s[i + 3]));
                ml2 = fmaxf(ml2, fmaxf(s[i + 4], s[i + 5]));
                ml3 = fmaxf(ml3, fmaxf(s[i + 6], s[i + 7]));
            }
            float m_local = fmaxf(fmaxf(ml0, ml1), fmaxf(ml2, ml3));
            // Promote to scaled-log2 space (matches m_state)
            const float m_local_log2 = m_local * SCALE_LOG2;
            float m_new = fmaxf(m_state, m_local_log2);

            // ----- 2. acc_scale = exp2(m_old - m_new), with FA4 tau threshold -----
            // (NO_TAU_SKIP experiment reverted — confirmed barrier stalls drop
            //  -0.55 cycles/inst but added rescale work overshoots, +210 µs.)
            constexpr float TAU = 8.0f;
            const float acc_scale_log2 = m_state - m_new;
            float acc_scale;
            if (acc_scale_log2 >= -TAU) {
                m_new = m_state;
                acc_scale = 1.0f;
            } else {
                asm volatile("ex2.approx.ftz.f32 %0, %1;"
                    : "=f"(acc_scale) : "f"(acc_scale_log2));
            }

            // ----- 3. Write acc_scale to sScale (one row per lane) -----
            sScale_ptr[sScale_row] = acc_scale;
            // FA4 early arrive: fire sm_stats so COR can rescale O while we compute P.
            __syncwarp();
#if !SKIP_SM_FULL
            // sm_stats_full has no consumer in pure-SOL build, skip the arrive.
  #if USE_NAMED_BAR_SM_STATS
            // FA4 pattern (flash_fwd_sm100.py:940 + 2116): per-warp-pair named bar
            // arrive. Each SM warp i of pair p fires bar id 8 + p*4 + i (count 64,
            // = this SM warp + paired COR warp). Lighter than mbarrier_arrive.
            // ALL 32 lanes must execute the bar.arrive (no lane_id==0 gate).
            named_barrier_arrive(BAR_SM_STATS_BASE + pair * 4 + warp_id_local,
                                 SM_STATS_BAR_COUNT);
  #else
            if (lane_id == 0) {
                mbarrier_arrive(sm_stats_full(pair));
            }
  #endif
#endif
            // End of RM phase: this is the "early COR trigger" point.
            PROF_END_X(EV_SM_RM);

#if SM_LEVEL == 1 && SKIP_SM_FULL
            // SOL+RM build-up: rowmax done, skip Phase A/B/STTM/l_sum entirely.
            // No mbar arrives needed — all consumers (COR/EPI/MMA-side waits) off.
            continue;
#endif

#if PINGPONG == 1
            // SM0/SM1 ping-pong (wide CS: covers Phase A FFMA2 + Phase B EX2).
            // FA4 paper §3.1.2. Bar count=256 across all 8 SM warps.
            PROF_BEGIN_X(EV_PP_WAIT);
            if (is_p0) {
                if (j > 0) named_barrier_sync(BAR_SM1_DONE, PP_BAR_COUNT);
            } else {
                named_barrier_sync(BAR_SM0_DONE, PP_BAR_COUNT);
            }
            PROF_END_X(EV_PP_WAIT);
#endif

            // ----- 4. Compute P = exp2(S * scale_log2 - m_new) and l_partial -----
            // FA4 pattern (softmax.py:223 scale_subtract_rowmax + 237 apply_exp2_convert):
            //   Phase A: FFMA2-packed scale-subtract: s = s * SCALE_LOG2 - m_new (in place).
            //   Phase B: exp2 + cvt to bf16x2 + accumulate l_new.
            // Splitting phases lets ptxas vectorize phase A to FFMA2 SASS (2 fp32 ops/cycle
            // on FMA pipe) and gives the exp2 loop its own scheduling context.
            PROF_BEGIN_X(EV_SM_EX2);
            float l_new = acc_scale * l_state;
            uint32_t pp[64];
            const float neg_m_new = -m_new;
#if SM_PHASE_A_ON
            // Phase A: 64× fma.rn.ftz.f32x2 — scale all 128 elements in place.
            #pragma unroll
            for (int i = 0; i < 64; i++) {
                asm("{\n\t"
                    ".reg .b64 lin, lsc, lmn;\n\t"
                    "mov.b64 lin, {%0, %1};\n\t"
                    "mov.b64 lsc, {%2, %2};\n\t"
                    "mov.b64 lmn, {%3, %3};\n\t"
                    "fma.rn.ftz.f32x2 lin, lin, lsc, lmn;\n\t"
                    "mov.b64 {%0, %1}, lin;\n\t"
                    "}"
                    : "+f"(s[2*i]), "+f"(s[2*i+1])
                    : "f"(SCALE_LOG2), "f"(neg_m_new));
            }
#endif
#if !SM_PHASE_B_ON
            // SM_LEVEL=2 stop: Phase A done, skip Phase B/STTM/mbars.
            // Prevent ptxas from DCE'ing s[] (only output of Phase A).
            asm volatile("" : : "f"(s[0]), "f"(s[63]), "f"(s[127]) : "memory");
            PROF_END_X(EV_SM_EX2);
            PROF_BEGIN_X(EV_SM_STP);
            PROF_END_X(EV_SM_STP);
            continue;
#endif
#if PINGPONG == 2
            // SM0/SM1 ping-pong (narrow CS: ONLY Phase B EX2 + FADD2; Phase A FFMA2
            // stays parallel). FA4 source pattern (flash_fwd_sm100.py:2119-2138).
            PROF_BEGIN_X(EV_PP_WAIT);
            if (is_p0) {
                if (j > 0) named_barrier_sync(BAR_SM1_DONE, PP_BAR_COUNT);
            } else {
                named_barrier_sync(BAR_SM0_DONE, PP_BAR_COUNT);
            }
            PROF_END_X(EV_PP_WAIT);
#endif
#if PINGPONG == 3
            // Mbar-based pingpong (FA4 exact pattern, flash_fwd_sm100.py:2122).
            // SM0 waits pp_seq[0] (SM1 arrives there); SM1 waits pp_seq[1].
            // 1-thread arrive vs bar.sync's 256-thread rendezvous = ~10x lighter.
            PROF_BEGIN_X(EV_PP_WAIT);
            if (is_p0) {
                mbarrier_wait(pp_seq(0), pp_phase);
            } else {
                mbarrier_wait(pp_seq(1), pp_phase);
            }
            pp_phase ^= 1;
            PROF_END_X(EV_PP_WAIT);
#endif

            // Phase B: exp2 + cvt to bf16x2 + accumulate.
            //
            // USE_EX2_EMU selects how each EX2 pair is computed:
            //   0: 2× MUFU.EX2 per pair (XU pipe — 16 ops/clock/SM bottleneck).
            //   1: ex2_poly_pair per pair (FMA pipe — packed FFMA2 degree-3 Remez,
            //      ~5 ops/pair vs 2 XU ops; 4× more FMA work per pair).
            //   2: HYBRID matching FA4 ex2_emu_freq=12/res=4 pattern. Groups 0-3
            //      and 12-15 use all-MUFU; groups 4-11 mark pair-3 as poly (the
            //      "tail" of each group window mirrors FA4's k%12 >= 8 selection).
            //      Net: 8 poly pairs / 64 total = 12.5% FMA-pipe offload, with
            //      first+last fragments always pure MUFU to keep XU drain timing
            //      predictable.
            #define MUFU_EX2_PAIR(A, B) do {                                       \
                asm volatile("ex2.approx.ftz.f32 %0, %1;" : "+f"(A) : "f"(A));     \
                asm volatile("ex2.approx.ftz.f32 %0, %1;" : "+f"(B) : "f"(B));     \
            } while (0)
#if USE_EX2_EMU == 1
            #define LSUM_EX2_PAIR(A, B) ex2_poly_pair((A), (B))
#else
            #define LSUM_EX2_PAIR(A, B) MUFU_EX2_PAIR(A, B)
#endif
#if LSUM_4STREAM
            // 4-stream packed FADD2 accumulator (matches FA4 softmax.py:280-ish
            // l_partial reduction pattern, FA4 SASS lines 4015-4143).
            // Replaces 128-deep scalar FADD chain with 64 FADD2 across 4 independent
            // streams + 3-element reduction tree. Cuts FMA-pipe op count ~2× and
            // breaks the serial dep chain so MUFU.EX2 latency can overlap.
            float l_acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#if LSUM_AT_END
            // FA4-pattern bodies: EX2 + cvt only. Write back exp2(s) to s[] for
            // the late l_partial reduction. No inline LSUM_ADD2.
            #define LSUM_BODY(g_)                                                  \
                {                                                                  \
                    float a0 = s[(g_)*8 + 0], b0 = s[(g_)*8 + 1];                  \
                    float a1 = s[(g_)*8 + 2], b1 = s[(g_)*8 + 3];                  \
                    float a2 = s[(g_)*8 + 4], b2 = s[(g_)*8 + 5];                  \
                    float a3 = s[(g_)*8 + 6], b3 = s[(g_)*8 + 7];                  \
                    LSUM_EX2_PAIR(a0, b0);                                         \
                    LSUM_EX2_PAIR(a1, b1);                                         \
                    LSUM_EX2_PAIR(a2, b2);                                         \
                    LSUM_EX2_PAIR(a3, b3);                                         \
                    asm volatile("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(pp[(g_)*4+0]) : "f"(a0), "f"(b0));\
                    asm volatile("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(pp[(g_)*4+1]) : "f"(a1), "f"(b1));\
                    asm volatile("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(pp[(g_)*4+2]) : "f"(a2), "f"(b2));\
                    asm volatile("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(pp[(g_)*4+3]) : "f"(a3), "f"(b3));\
                    s[(g_)*8 + 0] = a0; s[(g_)*8 + 1] = b0;                        \
                    s[(g_)*8 + 2] = a1; s[(g_)*8 + 3] = b1;                        \
                    s[(g_)*8 + 4] = a2; s[(g_)*8 + 5] = b2;                        \
                    s[(g_)*8 + 6] = a3; s[(g_)*8 + 7] = b3;                        \
                }
            #define LSUM_BODY_MIXED(g_)                                            \
                {                                                                  \
                    float a0 = s[(g_)*8 + 0], b0 = s[(g_)*8 + 1];                  \
                    float a1 = s[(g_)*8 + 2], b1 = s[(g_)*8 + 3];                  \
                    float a2 = s[(g_)*8 + 4], b2 = s[(g_)*8 + 5];                  \
                    float a3 = s[(g_)*8 + 6], b3 = s[(g_)*8 + 7];                  \
                    MUFU_EX2_PAIR(a0, b0);                                         \
                    MUFU_EX2_PAIR(a1, b1);                                         \
                    MUFU_EX2_PAIR(a2, b2);                                         \
                    ex2_poly_pair(a3, b3);                                         \
                    asm volatile("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(pp[(g_)*4+0]) : "f"(a0), "f"(b0));\
                    asm volatile("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(pp[(g_)*4+1]) : "f"(a1), "f"(b1));\
                    asm volatile("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(pp[(g_)*4+2]) : "f"(a2), "f"(b2));\
                    asm volatile("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(pp[(g_)*4+3]) : "f"(a3), "f"(b3));\
                    s[(g_)*8 + 0] = a0; s[(g_)*8 + 1] = b0;                        \
                    s[(g_)*8 + 2] = a1; s[(g_)*8 + 3] = b1;                        \
                    s[(g_)*8 + 4] = a2; s[(g_)*8 + 5] = b2;                        \
                    s[(g_)*8 + 6] = a3; s[(g_)*8 + 7] = b3;                        \
                }
#else
            #define LSUM_BODY(g_)                                                  \
                {                                                                  \
                    float a0 = s[(g_)*8 + 0], b0 = s[(g_)*8 + 1];                  \
                    float a1 = s[(g_)*8 + 2], b1 = s[(g_)*8 + 3];                  \
                    float a2 = s[(g_)*8 + 4], b2 = s[(g_)*8 + 5];                  \
                    float a3 = s[(g_)*8 + 6], b3 = s[(g_)*8 + 7];                  \
                    LSUM_EX2_PAIR(a0, b0);                                         \
                    LSUM_EX2_PAIR(a1, b1);                                         \
                    LSUM_EX2_PAIR(a2, b2);                                         \
                    LSUM_EX2_PAIR(a3, b3);                                         \
                    LSUM_ADD2(l_acc[0], l_acc[1], a0, b0);                         \
                    LSUM_ADD2(l_acc[2], l_acc[3], a1, b1);                         \
                    LSUM_ADD2(l_acc[4], l_acc[5], a2, b2);                         \
                    LSUM_ADD2(l_acc[6], l_acc[7], a3, b3);                         \
                    asm volatile("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(pp[(g_)*4+0]) : "f"(a0), "f"(b0));\
                    asm volatile("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(pp[(g_)*4+1]) : "f"(a1), "f"(b1));\
                    asm volatile("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(pp[(g_)*4+2]) : "f"(a2), "f"(b2));\
                    asm volatile("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(pp[(g_)*4+3]) : "f"(a3), "f"(b3));\
                }
            // Hybrid body: 3 MUFU pairs + 1 poly pair (pair-3 = "tail" of group),
            // used in middle groups (4..11) under USE_EX2_EMU == 2.
            #define LSUM_BODY_MIXED(g_)                                            \
                {                                                                  \
                    float a0 = s[(g_)*8 + 0], b0 = s[(g_)*8 + 1];                  \
                    float a1 = s[(g_)*8 + 2], b1 = s[(g_)*8 + 3];                  \
                    float a2 = s[(g_)*8 + 4], b2 = s[(g_)*8 + 5];                  \
                    float a3 = s[(g_)*8 + 6], b3 = s[(g_)*8 + 7];                  \
                    MUFU_EX2_PAIR(a0, b0);                                         \
                    MUFU_EX2_PAIR(a1, b1);                                         \
                    MUFU_EX2_PAIR(a2, b2);                                         \
                    ex2_poly_pair(a3, b3);                                         \
                    LSUM_ADD2(l_acc[0], l_acc[1], a0, b0);                         \
                    LSUM_ADD2(l_acc[2], l_acc[3], a1, b1);                         \
                    LSUM_ADD2(l_acc[4], l_acc[5], a2, b2);                         \
                    LSUM_ADD2(l_acc[6], l_acc[7], a3, b3);                         \
                    asm volatile("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(pp[(g_)*4+0]) : "f"(a0), "f"(b0));\
                    asm volatile("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(pp[(g_)*4+1]) : "f"(a1), "f"(b1));\
                    asm volatile("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(pp[(g_)*4+2]) : "f"(a2), "f"(b2));\
                    asm volatile("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(pp[(g_)*4+3]) : "f"(a3), "f"(b3));\
                }
#endif
            #define LSUM_ADD2(LO, HI, A, B)                                        \
                asm("{ .reg .b64 p, r;\n\t"                                        \
                    "mov.b64 p, {%2, %3};\n\t"                                     \
                    "mov.b64 r, {%0, %1};\n\t"                                     \
                    "add.f32x2 r, r, p;\n\t"                                       \
                    "mov.b64 {%0, %1}, r;\n\t}"                                    \
                    : "+f"(LO), "+f"(HI) : "f"(A), "f"(B))
#if STREAM_STP
            // Stream P-store: split the 16-group EX2 loop into two 8-group halves
            // with a tcgen05.st fired between them. The first store can drain in
            // the TMEM async pipeline while groups 8..15 compute, and the second
            // store can drain in parallel with the reduction tree + l_state update.
            // Matches FA4 SASS pattern (MUFU.EX2 → F2FP.BF16.F32.PACK_AB → STTM.x16).
  #if USE_EX2_EMU == 2
            // Hybrid: groups 0-3 all-MUFU, 4-7 mixed (1 poly per group), then STTM,
            // 8-11 mixed, 12-15 all-MUFU, then STTM. Mirrors FA4's
            // start_frg=1, end_frg=frg_cnt-1 layout.
            #pragma unroll
            for (int g = 0;  g < 4;  g++) { LSUM_BODY(g); }
            #pragma unroll
            for (int g = 4;  g < 8;  g++) { LSUM_BODY_MIXED(g); }
    #if SM_INTEGRATION_ON
            tcgen05_st_32x32b_x32(row_addr + 64, &pp[0]);
      #if SPLIT_P_PV
            // FA4 split_P_arrive: fence the first STTM, fire s_p_o_empty early
            // (partial P ready for MMA's PV first 4 MMAs). count=8 (4 SM + 4 COR);
            // SM's 4 arrives land here, mbar flips when COR's 4 also land.
            asm volatile("tcgen05.fence::before_thread_sync;");
            if (lane_id == 0) {
                mbarrier_arrive(s_p_o_empty(pair));
            }
      #endif
    #endif
            #pragma unroll
            for (int g = 8;  g < 12; g++) { LSUM_BODY_MIXED(g); }
            #pragma unroll
            for (int g = 12; g < 16; g++) { LSUM_BODY(g); }
    #if SM_INTEGRATION_ON
            tcgen05_st_32x32b_x32(row_addr + 96, &pp[32]);
    #endif
  #else
            #pragma unroll
            for (int g = 0; g < 8; g++) { LSUM_BODY(g); }
    #if SM_INTEGRATION_ON
            // Fire chunk-0 STTM (pp[0..31] → row_addr + 64)
            tcgen05_st_32x32b_x32(row_addr + 64, &pp[0]);
      #if SPLIT_P_PV
            asm volatile("tcgen05.fence::before_thread_sync;");
            if (lane_id == 0) {
                mbarrier_arrive(s_p_o_empty(pair));
            }
      #endif
    #endif
            #pragma unroll
            for (int g = 8; g < 16; g++) { LSUM_BODY(g); }
    #if SM_INTEGRATION_ON
            // Fire chunk-1 STTM (pp[32..63] → row_addr + 96)
            tcgen05_st_32x32b_x32(row_addr + 96, &pp[32]);
    #endif
  #endif
#else
  #if USE_EX2_EMU == 2
            #pragma unroll
            for (int g = 0;  g < 4;  g++) { LSUM_BODY(g); }
            #pragma unroll
            for (int g = 4;  g < 12; g++) { LSUM_BODY_MIXED(g); }
            #pragma unroll
            for (int g = 12; g < 16; g++) { LSUM_BODY(g); }
  #else
            #pragma unroll
            for (int g = 0; g < 16; g++) { LSUM_BODY(g); }
  #endif
#endif
            #undef LSUM_BODY
            #undef LSUM_BODY_MIXED
            #undef LSUM_ADD2
            #undef LSUM_EX2_PAIR
            #undef MUFU_EX2_PAIR
            // 3-element FADD2 reduction tree: (s0+s1, s2+s3) → final packed pair.
            #define LSUM_ADD2(LO, HI, A, B)                                        \
                asm("{ .reg .b64 p, r;\n\t"                                        \
                    "mov.b64 p, {%2, %3};\n\t"                                     \
                    "mov.b64 r, {%0, %1};\n\t"                                     \
                    "add.f32x2 r, r, p;\n\t"                                       \
                    "mov.b64 {%0, %1}, r;\n\t}"                                    \
                    : "+f"(LO), "+f"(HI) : "f"(A), "f"(B))
#if LSUM_AT_END
            // FA4-pattern late l_partial reduction: read exp2 values back from s[]
            // (which now holds Phase B outputs) and reduce with 4 packed streams.
            // Matches FA4 fadd_reduce ([utils.py:302-339]) called from
            // update_row_sum at end of softmax_step.
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                LSUM_ADD2(l_acc[0], l_acc[1], s[i*8 + 0], s[i*8 + 1]);
                LSUM_ADD2(l_acc[2], l_acc[3], s[i*8 + 2], s[i*8 + 3]);
                LSUM_ADD2(l_acc[4], l_acc[5], s[i*8 + 4], s[i*8 + 5]);
                LSUM_ADD2(l_acc[6], l_acc[7], s[i*8 + 6], s[i*8 + 7]);
            }
#endif
            LSUM_ADD2(l_acc[0], l_acc[1], l_acc[2], l_acc[3]);
            LSUM_ADD2(l_acc[4], l_acc[5], l_acc[6], l_acc[7]);
            LSUM_ADD2(l_acc[0], l_acc[1], l_acc[4], l_acc[5]);
            #undef LSUM_ADD2
            // Scalar tail: l_new already = acc_scale * l_state (line 720).
            l_new = l_new + l_acc[0] + l_acc[1];
#else
            #pragma unroll
            for (int i = 0; i < 64; i++) {
                float a = s[2*i];
                float b = s[2*i+1];
                asm volatile("ex2.approx.ftz.f32 %0, %1;" : "+f"(a) : "f"(a));
                asm volatile("ex2.approx.ftz.f32 %0, %1;" : "+f"(b) : "f"(b));
                l_new += a + b;
                asm volatile("cvt.rn.bf16x2.f32 %0, %2, %1;"
                    : "=r"(pp[i]) : "f"(a), "f"(b));
            }
#endif
            m_state = m_new;
            l_state = l_new;
            PROF_END_X(EV_SM_EX2);

#if PINGPONG == 1 || PINGPONG == 2
            // Release our partner via named bar (count=256 rendezvous, heavyweight).
            PROF_BEGIN_X(EV_PP_ARRIVE);
            if (is_p0) {
                named_barrier_arrive(BAR_SM0_DONE, PP_BAR_COUNT);
            } else {
                named_barrier_arrive(BAR_SM1_DONE, PP_BAR_COUNT);
            }
            PROF_END_X(EV_PP_ARRIVE);
#endif
#if PINGPONG == 3
            // Release partner via mbar (1-thread arrive). FA4 fires from a single
            // elected lane; here we use lane 0 of any warp in the warpgroup —
            // lightweight since mbar arrive_count=1 and only one lane contributes.
            // SM0 arrives pp_seq[1] (SM1's wake); SM1 arrives pp_seq[0] (SM0's wake).
            PROF_BEGIN_X(EV_PP_ARRIVE);
            if (lane_id == 0 && warp_id_local == 0) {
                if (is_p0) {
                    mbarrier_arrive(pp_seq(1));
                } else {
                    mbarrier_arrive(pp_seq(0));
                }
            }
            PROF_END_X(EV_PP_ARRIVE);
#endif

            PROF_BEGIN_X(EV_SM_STP);
#if SM_INTEGRATION_ON
  #if !STREAM_STP
            // Write P to TMEM at offset +64 from S region (FA4 tmem_p_offset).
            #pragma unroll
            for (int chunk = 1; chunk >= 0; chunk--) {
                const int addr = row_addr + 64 + chunk * 32;
                uint32_t* base = &pp[chunk * 32];
                tcgen05_st_32x32b_x32(addr, base);
            }
  #endif
            // P stores fired (either batched here or streamed mid/end-EX2).
            asm volatile("tcgen05.wait::st.sync.aligned;");
            asm volatile("tcgen05.fence::before_thread_sync;");

            // sm_stats_full was fired EARLY (in EV_SM_RM region), don't re-fire here.
            // Arrive on remaining mbars: P done (p_lastsplit) and S consumed (s_p_o_empty).
            // With SPLIT_P_ARRIVE: s_p_o_empty was already fired right after LD-S.
            // With SPLIT_P_PV: s_p_o_empty was already fired after first STTM chunk
            // (in the LSUM_4STREAM block) — skip it here too.
            if (lane_id == 0) {
                mbarrier_arrive(p_lastsplit_full(pair));
  #if P_LASTSPLIT_CLUSTER
                mbarrier_arrive_peer(p_lastsplit_full(pair));
  #endif
  #if !SPLIT_P_ARRIVE && !SPLIT_P_PV
                mbarrier_arrive(s_p_o_empty(pair));
  #endif
            }
#else
            // SOL+SM math build-up: prevent ptxas from DCE'ing the SM math loop
            // outputs (l_state and pp[]) that nothing else reads.
            asm volatile("" : : "f"(l_state), "r"(pp[0]), "r"(pp[31]), "r"(pp[63]) : "memory");
#endif
            PROF_END_X(EV_SM_STP);

#if !SKIP_SM_FULL && BACKPRESSURE_SM_STATS
            // FA4 pattern (flash_fwd_sm100.py:2157):
            // pipeline_sm_stats.producer_acquire_w_index_phase — back-pressure on
            // sm_stats slot. SM waits for COR to have consumed previous-iter stats
            // before this iter ends.
            mbarrier_wait(sm_stats_empty(pair), phase_sse);
            phase_sse ^= 1;
#endif
        }
        // ----- After mainloop: write final l_state to sScale for EPI/COR -----
        // Layout: sScale[Q_STAGE * M_PER_CTA + pair * M_PER_CTA + lane_row]
        sScale_ptr[Q_STAGE * M_PER_CTA + sScale_row] = l_state;
        asm volatile("fence.proxy.async;");  // make smem write visible
#endif

    } else if (warp_id < CORR_HI) {
        // ============================================================
        // CORRECTION (warps 8-11 = WG2) — REAL rescale of O accumulator
        // ============================================================
        // Per iter, for each pair: read scale from sScale[pair*M + lane_row],
        // LD/multiply/ST O TMEM in 8 chunks of 16 cols (matches cuDNN's
        // LDTM.x8 + FMUL2 + STTM.x8 pattern with paired LDs per chunk).
#if !SKIP_SM_FULL
        const int warp_id_local = warp_id - CORR_LO;  // 0..3
        const int o_addr_p0 = O_acc_p0 + ((warp_id_local * 32) << 16);
        const int o_addr_p1 = O_acc_p1 + ((warp_id_local * 32) << 16);
        const int sScale_row = warp_id_local * 32 + lane_id;
        float* sScale_ptr = (float*)(smem_ptr + (Q_BYTES_TOTAL + O_BYTES_TOTAL + KV_BYTES_TOTAL));
        int phase_oa0 = 0, phase_oa1 = 0;
        int phase_ss0 = 0, phase_ss1 = 0;

        // rescale_O lambda eliminated → emit_rescale_O (file-scope helper).

        for (int j = 0; j < n_kv; j++) {
            // FA4 pattern (flash_fwd_sm100.py:2253-2254 + 2276 commented):
            // wait sm_stats only — NOT o_acc_full. The implicit dependency is:
            // sm_stats_full only fires after softmax done with S, which only
            // happens after MMA's QK commit, which drains prior PV[j-1]'s
            // writes to O TMEM. So O[j-1] is safely visible when COR runs.
            // Waiting on o_acc_full would create circular dep with our PV
            // gating on s_p_o_empty.
            // PAIR 0 region: wait sm_stats(0), rescale O for pair 0, fire pair-0 mbars.
            PROF_BEGIN_X(EV_COR_WAIT_O);
#if USE_NAMED_BAR_SM_STATS
            // Per-warp named-bar handshake with SM warp_id_local of pair 0.
            named_barrier_sync(BAR_SM_STATS_BASE + 0 * 4 + warp_id_local,
                               SM_STATS_BAR_COUNT);
#else
            mbarrier_wait(sm_stats_full(0), phase_ss0); phase_ss0 ^= 1;
#endif
            asm volatile("tcgen05.fence::after_thread_sync;");
            const float scale_p0 = sScale_ptr[0 * M_PER_CTA + sScale_row];
            const unsigned mask_p0 = __ballot_sync(0xFFFFFFFFu, scale_p0 < 1.0f);
            if (mask_p0) emit_rescale_O(o_addr_p0, scale_p0);
            asm volatile("tcgen05.fence::before_thread_sync;");
            if (lane_id == 0) {
                mbarrier_arrive(s_p_o_empty(0));
#if BACKPRESSURE_SM_STATS
                mbarrier_arrive(sm_stats_empty(0));
#endif
            }
            PROF_END_X(EV_COR_WAIT_O);

            // PAIR 1 region: wait sm_stats(1), rescale O for pair 1, fire pair-1 mbars.
            PROF_BEGIN_X(EV_COR_WAIT_O1);
#if USE_NAMED_BAR_SM_STATS
            named_barrier_sync(BAR_SM_STATS_BASE + 1 * 4 + warp_id_local,
                               SM_STATS_BAR_COUNT);
#else
            mbarrier_wait(sm_stats_full(1), phase_ss1); phase_ss1 ^= 1;
#endif
            asm volatile("tcgen05.fence::after_thread_sync;");
            const float scale_p1 = sScale_ptr[1 * M_PER_CTA + sScale_row];
            const unsigned mask_p1 = __ballot_sync(0xFFFFFFFFu, scale_p1 < 1.0f);
            if (mask_p1) emit_rescale_O(o_addr_p1, scale_p1);
            asm volatile("tcgen05.fence::before_thread_sync;");
            if (lane_id == 0) {
                mbarrier_arrive(s_p_o_empty(1));
#if BACKPRESSURE_SM_STATS
                mbarrier_arrive(sm_stats_empty(1));
#endif
            }
            PROF_END_X(EV_COR_WAIT_O1);
            (void)phase_oa0; (void)phase_oa1;  // unused now
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
            // ptxas spilling per-stage int arrays to local mem.
            int phase_q       = 0;  // 2 bits used (q_stage=2)
            int phase_kv_full = 0;  // 6 bits used (kv_stage=6)
            int phase_pls     = 0;  // 2 bits used (q_stage=2)
            int phase_spe     = 0;  // 2 bits used — s_p_o_empty (COR rescale done)

            // Wait both Q tiles loaded.
            PROF_BEGIN_X(EV_MMA_WAIT_Q);
            mbarrier_wait(q_full(0), (phase_q >> 0) & 1); phase_q ^= 1 << 0;
            mbarrier_wait(q_full(1), (phase_q >> 1) & 1); phase_q ^= 1 << 1;
            PROF_END_X(EV_MMA_WAIT_Q);
            asm volatile("tcgen05.fence::after_thread_sync;");

#if MMA_REORDER == 0
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

                PROF_BEGIN_X(EV_MMA_QK_P0);
                ISSUE_QK(0, s_K);
                PROF_END_X(EV_MMA_QK_P0);
                PROF_BEGIN_X(EV_MMA_QK_P1);
                ISSUE_QK(1, s_K);
                PROF_END_X(EV_MMA_QK_P1);

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

                // Wait for THIS iter's softmax (P ready) AND correction (O
                // rescaled) before PV. s_p_o_empty fires after both softmax (4
                // arrives) and COR (4 arrives) per pair = count 8 → 0. Reading
                // the RESCALED O before PV adds new contribution gives correct
                // online softmax math (FA4 flash_fwd_sm100.py:1648 pattern).
#if !SKIP_SM_FULL
                PROF_BEGIN_X(EV_MMA_WAIT_PR0);
                mbarrier_wait(s_p_o_empty(0),     (phase_spe >> 0) & 1); phase_spe ^= 1 << 0;
  #if !SPLIT_P_PV
                mbarrier_wait(p_lastsplit_full(0), (phase_pls >> 0) & 1); phase_pls ^= 1 << 0;
  #endif
                PROF_END_X(EV_MMA_WAIT_PR0);
#endif
                {
                    int pls_p0 = (phase_pls >> 0) & 1;
#if SPLIT_P_PV && !SKIP_SM_FULL
                    phase_pls ^= 1 << 0;
#endif
                    PROF_BEGIN_X(EV_MMA_PV_P0);
                    ISSUE_PV(0, s_V, j == 0, pls_p0);
                    PROF_END_X(EV_MMA_PV_P0);
                }
#if PER_PAIR_DRAIN
                if (j + 1 == n_kv) {
                    MMA_PAIR_DRAIN_COMMIT(o_acc_full(0));
                }
#endif
#if !SKIP_SM_FULL
                PROF_BEGIN_X(EV_MMA_WAIT_PR1);
                mbarrier_wait(s_p_o_empty(1),     (phase_spe >> 1) & 1); phase_spe ^= 1 << 1;
  #if !SPLIT_P_PV
                mbarrier_wait(p_lastsplit_full(1), (phase_pls >> 1) & 1); phase_pls ^= 1 << 1;
  #endif
                PROF_END_X(EV_MMA_WAIT_PR1);
#endif
                {
                    int pls_p1 = (phase_pls >> 1) & 1;
#if SPLIT_P_PV && !SKIP_SM_FULL
                    phase_pls ^= 1 << 1;
#endif
                    PROF_BEGIN_X(EV_MMA_PV_P1);
                    ISSUE_PV(1, s_V, j == 0, pls_p1);
                    PROF_END_X(EV_MMA_PV_P1);
                }
#if PER_PAIR_DRAIN
                if (j + 1 == n_kv) {
                    MMA_PAIR_DRAIN_COMMIT(o_acc_full(1));
                }
#endif

                // Release V stage after both PVs committed.
                asm volatile(
                    "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster"
                    ".multicast::cluster.b64 [%0], %1;" ::
                        "r"(kv_empty(s_V)), "h"(cta_mask) : "memory");
            }
#elif MMA_REORDER == 1
            // Pipelined schedule:
            //   prologue: wait K_0 → Q0K_0
            //   step S in [0..n_kv): P1V_{S-1} (S>=1), Q1K_S, P0V_S, Q0K_{S+1}
            //   epilogue: P1V_{n_kv-1}, release V_{n_kv-1}
            // Pair 0: Q0K_S issued at step S-1 → P0V_S at step S.
            //          Softmax has 2 MMA-issues of overlap window.
            // Pair 1: Q1K_S issued at step S → P1V_S at step S+1.
            //          Same 2-MMA overlap window.
            // Slot lifetimes:
            //   K_S  in slot (2S)%6     — used by Q0K_S (step S-1) and Q1K_S (step S),
            //                            release at end of step S.
            //   V_S  in slot (2S+1)%6   — used by P0V_S (step S) and P1V_S (step S+1),
            //                            release at end of step S+1.
            //   With KV_STAGE=6 (3 K+V pairs in flight), no slot collisions.

            // Prologue: Q0K_0 + Q1K_0 (FA4 pattern — both pair softmaxes start in parallel)
            {
                const int s_K0 = 0;
                PROF_BEGIN_X(EV_MMA_WAIT_K);
                mbarrier_wait(kv_full(s_K0), (phase_kv_full >> s_K0) & 1);
                phase_kv_full ^= 1 << s_K0;
                PROF_END_X(EV_MMA_WAIT_K);
                PROF_BEGIN_X(EV_MMA_QK_P0);
                ISSUE_QK(0, s_K0);
                PROF_END_X(EV_MMA_QK_P0);
                PROF_BEGIN_X(EV_MMA_QK_P1);
                ISSUE_QK(1, s_K0);
                PROF_END_X(EV_MMA_QK_P1);
                // Release K_0 — both Q0K_0 and Q1K_0 committed via tcgen05.commit.
                asm volatile(
                    "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster"
                    ".multicast::cluster.b64 [%0], %1;" ::
                        "r"(kv_empty(s_K0)), "h"(cta_mask) : "memory");
            }

            for (int S = 0; S < n_kv; S++) {
                const int s_K_S      = (2 * S) % KV_STAGE;
                const int s_V_S      = (2 * S + 1) % KV_STAGE;
                const int s_V_prev   = (S >= 1) ? ((2 * (S - 1) + 1) % KV_STAGE) : -1;

                // 1. P1V_{S-1} (if S>=1). V_{S-1} was waited at step S-1.
                if (S >= 1) {
#if !SKIP_SM_FULL
                    PROF_BEGIN_X(EV_MMA_WAIT_PR1);
                    mbarrier_wait(s_p_o_empty(1),     (phase_spe >> 1) & 1); phase_spe ^= 1 << 1;
  #if !SPLIT_P_PV
                    mbarrier_wait(p_lastsplit_full(1), (phase_pls >> 1) & 1); phase_pls ^= 1 << 1;
  #endif
                    PROF_END_X(EV_MMA_WAIT_PR1);
#endif
                    int pls_p1 = (phase_pls >> 1) & 1;
#if SPLIT_P_PV && !SKIP_SM_FULL
                    phase_pls ^= 1 << 1;
#endif
                    PROF_BEGIN_X(EV_MMA_PV_P1);
                    ISSUE_PV(1, s_V_prev, false, pls_p1);
                    PROF_END_X(EV_MMA_PV_P1);
                    // Release V_{S-1} after both PVs (P0V_{S-1} from step S-1, P1V_{S-1} now).
                    asm volatile(
                        "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster"
                        ".multicast::cluster.b64 [%0], %1;" ::
                            "r"(kv_empty(s_V_prev)), "h"(cta_mask) : "memory");
                }

                // 2. Q1K_S (skip S=0 — Q1K_0 was done in prologue along with K_0 release).
                if (S >= 1) {
                    PROF_BEGIN_X(EV_MMA_QK_P1);
                    ISSUE_QK(1, s_K_S);
                    PROF_END_X(EV_MMA_QK_P1);
                    // Release K_S after both Q0K_S (step S-1) and Q1K_S (now) committed.
                    asm volatile(
                        "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster"
                        ".multicast::cluster.b64 [%0], %1;" ::
                            "r"(kv_empty(s_K_S)), "h"(cta_mask) : "memory");
                }

                // 3. P0V_S — wait V_S, wait pair-0 softmax+rescale, then issue.
                PROF_BEGIN_X(EV_MMA_WAIT_V);
                mbarrier_wait(kv_full(s_V_S), (phase_kv_full >> s_V_S) & 1);
                phase_kv_full ^= 1 << s_V_S;
                PROF_END_X(EV_MMA_WAIT_V);
#if !SKIP_SM_FULL
                PROF_BEGIN_X(EV_MMA_WAIT_PR0);
                mbarrier_wait(s_p_o_empty(0),     (phase_spe >> 0) & 1); phase_spe ^= 1 << 0;
  #if !SPLIT_P_PV
                mbarrier_wait(p_lastsplit_full(0), (phase_pls >> 0) & 1); phase_pls ^= 1 << 0;
  #endif
                PROF_END_X(EV_MMA_WAIT_PR0);
#endif
                int pls_p0 = (phase_pls >> 0) & 1;
#if SPLIT_P_PV && !SKIP_SM_FULL
                phase_pls ^= 1 << 0;
#endif
                PROF_BEGIN_X(EV_MMA_PV_P0);
                ISSUE_PV(0, s_V_S, S == 0, pls_p0);
                PROF_END_X(EV_MMA_PV_P0);
#if PER_PAIR_DRAIN
                // Pair 0 final PV (last loop iter) → fire o_acc_full(0) so tail
                // epi for pair 0 (warps 0-3) can start while pair 1's PV is
                // still in-flight in the epilogue block below.
                if (S + 1 == n_kv) {
                    MMA_PAIR_DRAIN_COMMIT(o_acc_full(0));
                }
#endif

                // 4. Q0K_{S+1} (if not last iter) — wait K_{S+1} then issue.
                if (S + 1 < n_kv) {
                    const int s_K_next = (2 * (S + 1)) % KV_STAGE;
                    PROF_BEGIN_X(EV_MMA_WAIT_K);
                    mbarrier_wait(kv_full(s_K_next), (phase_kv_full >> s_K_next) & 1);
                    phase_kv_full ^= 1 << s_K_next;
                    PROF_END_X(EV_MMA_WAIT_K);
                    PROF_BEGIN_X(EV_MMA_QK_P0);
                    ISSUE_QK(0, s_K_next);
                    PROF_END_X(EV_MMA_QK_P0);
                }
            }

            // Epilogue: P1V_{n_kv-1}
            {
                const int s_V_last = (2 * (n_kv - 1) + 1) % KV_STAGE;
#if !SKIP_SM_FULL
                PROF_BEGIN_X(EV_MMA_WAIT_PR1);
                mbarrier_wait(s_p_o_empty(1),     (phase_spe >> 1) & 1); phase_spe ^= 1 << 1;
  #if !SPLIT_P_PV
                mbarrier_wait(p_lastsplit_full(1), (phase_pls >> 1) & 1); phase_pls ^= 1 << 1;
  #endif
                PROF_END_X(EV_MMA_WAIT_PR1);
#endif
                int pls_p1 = (phase_pls >> 1) & 1;
#if SPLIT_P_PV && !SKIP_SM_FULL
                phase_pls ^= 1 << 1;
#endif
                PROF_BEGIN_X(EV_MMA_PV_P1);
                ISSUE_PV(1, s_V_last, n_kv == 1, pls_p1);
                PROF_END_X(EV_MMA_PV_P1);
#if PER_PAIR_DRAIN
                // Pair 1 final PV → o_acc_full(1). This is also the all-MMA
                // drain (last tcgen05 op on the queue). Combined with the V_last
                // kv_empty commit below, no separate tmem_dealloc commit is
                // needed in the PER_PAIR_DRAIN path.
                MMA_PAIR_DRAIN_COMMIT(o_acc_full(1));
#endif
                asm volatile(
                    "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster"
                    ".multicast::cluster.b64 [%0], %1;" ::
                        "r"(kv_empty(s_V_last)), "h"(cta_mask) : "memory");
            }
#elif MMA_REORDER == 2
            // FA4-faithful per-stage schedule (flash_fwd_sm100.py:1577-1632).
            // Within iter i, per-stage interleave: PV_s, K_wait (s=0 only),
            // QK_s_{i+1}, V_release (s=1 only), commit.
            //   prologue: Q0K_0, Q1K_0 (both pair softmaxes start in parallel)
            //   loop iter i in [0, n_kv-1):
            //     wait V_i
            //     stage 0: PV_0_i, wait K_{i+1}, Q0K_{i+1}
            //     stage 1: PV_1_i, release V_i, Q1K_{i+1}
            //     release K_{i+1}
            //   epilogue: wait V_{n_kv-1}, PV_0_{n_kv-1}, PV_1_{n_kv-1}, release V_{n_kv-1}
            {
                const int s_K0 = 0;
                PROF_BEGIN_X(EV_MMA_WAIT_K);
                mbarrier_wait(kv_full(s_K0), (phase_kv_full >> s_K0) & 1);
                phase_kv_full ^= 1 << s_K0;
                PROF_END_X(EV_MMA_WAIT_K);
                PROF_BEGIN_X(EV_MMA_QK_P0);
                ISSUE_QK(0, s_K0);
                PROF_END_X(EV_MMA_QK_P0);
                PROF_BEGIN_X(EV_MMA_QK_P1);
                ISSUE_QK(1, s_K0);
                PROF_END_X(EV_MMA_QK_P1);
                asm volatile(
                    "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster"
                    ".multicast::cluster.b64 [%0], %1;" ::
                        "r"(kv_empty(s_K0)), "h"(cta_mask) : "memory");
            }

            // Main loop: n_kv - 1 iters. Each issues 2 PVs (current iter) + 2 QKs (next iter).
            for (int S = 0; S + 1 < n_kv; S++) {
                const int s_V_S    = (2 * S + 1) % KV_STAGE;
                const int s_K_next = (2 * (S + 1)) % KV_STAGE;

                PROF_BEGIN_X(EV_MMA_WAIT_V);
                mbarrier_wait(kv_full(s_V_S), (phase_kv_full >> s_V_S) & 1);
                phase_kv_full ^= 1 << s_V_S;
                PROF_END_X(EV_MMA_WAIT_V);

                // stage 0: PV_0_S, wait K_{S+1}, Q0K_{S+1}
#if !SKIP_SM_FULL
                PROF_BEGIN_X(EV_MMA_WAIT_PR0);
                mbarrier_wait(s_p_o_empty(0),     (phase_spe >> 0) & 1); phase_spe ^= 1 << 0;
  #if !SPLIT_P_PV
                mbarrier_wait(p_lastsplit_full(0), (phase_pls >> 0) & 1); phase_pls ^= 1 << 0;
  #endif
                PROF_END_X(EV_MMA_WAIT_PR0);
#endif
                {
                    int pls_p0 = (phase_pls >> 0) & 1;
#if SPLIT_P_PV && !SKIP_SM_FULL
                    phase_pls ^= 1 << 0;
#endif
                    PROF_BEGIN_X(EV_MMA_PV_P0);
                    ISSUE_PV(0, s_V_S, S == 0, pls_p0);
                    PROF_END_X(EV_MMA_PV_P0);
                }
                PROF_BEGIN_X(EV_MMA_WAIT_K);
                mbarrier_wait(kv_full(s_K_next), (phase_kv_full >> s_K_next) & 1);
                phase_kv_full ^= 1 << s_K_next;
                PROF_END_X(EV_MMA_WAIT_K);
                PROF_BEGIN_X(EV_MMA_QK_P0);
                ISSUE_QK(0, s_K_next);
                PROF_END_X(EV_MMA_QK_P0);

                // stage 1: PV_1_S, release V_S, Q1K_{S+1}
#if !SKIP_SM_FULL
                PROF_BEGIN_X(EV_MMA_WAIT_PR1);
                mbarrier_wait(s_p_o_empty(1),     (phase_spe >> 1) & 1); phase_spe ^= 1 << 1;
  #if !SPLIT_P_PV
                mbarrier_wait(p_lastsplit_full(1), (phase_pls >> 1) & 1); phase_pls ^= 1 << 1;
  #endif
                PROF_END_X(EV_MMA_WAIT_PR1);
#endif
                {
                    int pls_p1 = (phase_pls >> 1) & 1;
#if SPLIT_P_PV && !SKIP_SM_FULL
                    phase_pls ^= 1 << 1;
#endif
                    PROF_BEGIN_X(EV_MMA_PV_P1);
                    ISSUE_PV(1, s_V_S, S == 0, pls_p1);
                    PROF_END_X(EV_MMA_PV_P1);
                }
                // Release V_S after both PVs
                asm volatile(
                    "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster"
                    ".multicast::cluster.b64 [%0], %1;" ::
                        "r"(kv_empty(s_V_S)), "h"(cta_mask) : "memory");
                PROF_BEGIN_X(EV_MMA_QK_P1);
                ISSUE_QK(1, s_K_next);
                PROF_END_X(EV_MMA_QK_P1);
                // Release K_{S+1} after both QKs
                asm volatile(
                    "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster"
                    ".multicast::cluster.b64 [%0], %1;" ::
                        "r"(kv_empty(s_K_next)), "h"(cta_mask) : "memory");
            }

            // Epilogue: PV_0 and PV_1 for the last iter (n_kv-1), no QK
            {
                const int s_V_last = (2 * (n_kv - 1) + 1) % KV_STAGE;
                PROF_BEGIN_X(EV_MMA_WAIT_V);
                mbarrier_wait(kv_full(s_V_last), (phase_kv_full >> s_V_last) & 1);
                phase_kv_full ^= 1 << s_V_last;
                PROF_END_X(EV_MMA_WAIT_V);

                // stage 0
#if !SKIP_SM_FULL
                PROF_BEGIN_X(EV_MMA_WAIT_PR0);
                mbarrier_wait(s_p_o_empty(0),     (phase_spe >> 0) & 1); phase_spe ^= 1 << 0;
  #if !SPLIT_P_PV
                mbarrier_wait(p_lastsplit_full(0), (phase_pls >> 0) & 1); phase_pls ^= 1 << 0;
  #endif
                PROF_END_X(EV_MMA_WAIT_PR0);
#endif
                {
                    int pls_p0 = (phase_pls >> 0) & 1;
#if SPLIT_P_PV && !SKIP_SM_FULL
                    phase_pls ^= 1 << 0;
#endif
                    PROF_BEGIN_X(EV_MMA_PV_P0);
                    ISSUE_PV(0, s_V_last, n_kv == 1, pls_p0);
                    PROF_END_X(EV_MMA_PV_P0);
                }
#if PER_PAIR_DRAIN
                MMA_PAIR_DRAIN_COMMIT(o_acc_full(0));
#endif

                // stage 1
#if !SKIP_SM_FULL
                PROF_BEGIN_X(EV_MMA_WAIT_PR1);
                mbarrier_wait(s_p_o_empty(1),     (phase_spe >> 1) & 1); phase_spe ^= 1 << 1;
  #if !SPLIT_P_PV
                mbarrier_wait(p_lastsplit_full(1), (phase_pls >> 1) & 1); phase_pls ^= 1 << 1;
  #endif
                PROF_END_X(EV_MMA_WAIT_PR1);
#endif
                {
                    int pls_p1 = (phase_pls >> 1) & 1;
#if SPLIT_P_PV && !SKIP_SM_FULL
                    phase_pls ^= 1 << 1;
#endif
                    PROF_BEGIN_X(EV_MMA_PV_P1);
                    ISSUE_PV(1, s_V_last, n_kv == 1, pls_p1);
                    PROF_END_X(EV_MMA_PV_P1);
                }
#if PER_PAIR_DRAIN
                MMA_PAIR_DRAIN_COMMIT(o_acc_full(1));
#endif
                // Release V_last
                asm volatile(
                    "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster"
                    ".multicast::cluster.b64 [%0], %1;" ::
                        "r"(kv_empty(s_V_last)), "h"(cta_mask) : "memory");
            }
#endif
#if !PER_PAIR_DRAIN
            // Final tcgen05 commit drains all in-flight MMAs.
            asm volatile(
                "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster"
                ".multicast::cluster.b64 [%0], %1;" ::
                    "r"(tmem_dealloc), "h"(cta_mask) : "memory");
#endif
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

    // -------- Tail: wait MMA done, full epi from TMEM, dealloc --------
#if PER_PAIR_DRAIN
    // Per-pair drain: warps 0-3 wait o_acc_full(0), warps 4-7 wait o_acc_full(1).
    // Pair 0 epi can start while MMA is still issuing pair 1 PV → tail latency
    // collapses by ~(pair-1 MMA time) on critical path. tcgen05.dealloc is
    // gated by an all-warps __syncthreads() AFTER both per-pair epis complete.
    if (warp_id < 4) {
        mbarrier_wait(o_acc_full(0), 0);
    } else if (warp_id < 8) {
        mbarrier_wait(o_acc_full(1), 0);
    }
    asm volatile("tcgen05.fence::after_thread_sync;");
#else
    __syncthreads();
    mbarrier_wait(tmem_dealloc, 0);
    asm volatile("tcgen05.fence::after_thread_sync;");
#endif

#if EPI_FULL
    // Full epilogue: TMEM → reg → gmem. Warps 0-3 handle pair 0 (rows 0..127
    // of pair 0 across all 4 warps × 32 lanes). Warps 4-7 handle pair 1.
    // Each lane k of warp w:
    //   1) tcgen05.ld 32×32b.x32 to pull 128 fp32 cols of its own row from TMEM
    //   2) read l_state from sScale, compute 1/l_state via rcp.approx
    //   3) mul each fp32 col by 1/l, cvt to bf16, pack 8 bf16 into uint128
    //   4) 16× st.global.b128 writes the 128 bf16 row to gmem
    // Global row: blockIdx.y * 512 + pair * 256 + cta_rank * 128 + warp_local*32 + lane.
    if (warp_id < 8) {
        const int pair          = (warp_id < 4) ? 0 : 1;
        const int warp_id_local = (warp_id < 4) ? warp_id : warp_id - 4;
        const int O_acc_pair    = (pair == 0) ? O_acc_p0 : O_acc_p1;
        const int row_addr      = O_acc_pair + ((warp_id_local * 32) << 16);

        // 1/l_state for this lane's row
        float* sScale_ptr = (float*)(smem_ptr + (Q_BYTES_TOTAL + O_BYTES_TOTAL + KV_BYTES_TOTAL));
        const int row_in_pair = warp_id_local * 32 + lane_id;
        const float l = sScale_ptr[Q_STAGE * M_PER_CTA + pair * M_PER_CTA + row_in_pair];
        float inv_l;
        asm volatile("rcp.approx.f32 %0, %1;" : "=f"(inv_l) : "f"(l));

        const int m_row_global =
            (int)blockIdx.y * M_PER_CLUSTER + pair * M_JOINT
            + (int)cta_rank * M_PER_CTA + warp_id_local * 32 + lane_id;
        __nv_bfloat16* row_ptr = O_gmem + (size_t)m_row_global * N_DIM;

        // Chunked load+scale+cast+store: 4 chunks × 32 cols each.
        // Keeps live register pressure to ~32 fp32 instead of 128 (avoids spills).
        #pragma unroll
        for (int chunk = 0; chunk < 4; chunk++) {
            const int addr = row_addr + chunk * 32;
            float o[32];
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,"
                "%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,"
                "%28,%29,%30,%31}, [%32];"
                : "=f"(o[0]),  "=f"(o[1]),  "=f"(o[2]),  "=f"(o[3]),
                  "=f"(o[4]),  "=f"(o[5]),  "=f"(o[6]),  "=f"(o[7]),
                  "=f"(o[8]),  "=f"(o[9]),  "=f"(o[10]), "=f"(o[11]),
                  "=f"(o[12]), "=f"(o[13]), "=f"(o[14]), "=f"(o[15]),
                  "=f"(o[16]), "=f"(o[17]), "=f"(o[18]), "=f"(o[19]),
                  "=f"(o[20]), "=f"(o[21]), "=f"(o[22]), "=f"(o[23]),
                  "=f"(o[24]), "=f"(o[25]), "=f"(o[26]), "=f"(o[27]),
                  "=f"(o[28]), "=f"(o[29]), "=f"(o[30]), "=f"(o[31])
                : "r"(addr));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            // 32 fp32 → 4 uint4 stores (b128 each = 8 bf16).
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                uint32_t b0, b1, b2, b3;
                const float a0 = o[8*i+0] * inv_l;
                const float a1 = o[8*i+1] * inv_l;
                const float a2 = o[8*i+2] * inv_l;
                const float a3 = o[8*i+3] * inv_l;
                const float a4 = o[8*i+4] * inv_l;
                const float a5 = o[8*i+5] * inv_l;
                const float a6 = o[8*i+6] * inv_l;
                const float a7 = o[8*i+7] * inv_l;
                asm volatile("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(b0) : "f"(a0), "f"(a1));
                asm volatile("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(b1) : "f"(a2), "f"(a3));
                asm volatile("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(b2) : "f"(a4), "f"(a5));
                asm volatile("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(b3) : "f"(a6), "f"(a7));
                uint4 v{b0, b1, b2, b3};
                reinterpret_cast<uint4*>(row_ptr)[chunk * 4 + i] = v;
            }
        }
    }
#else
    // Legacy sentinel epi: writes 4 fp32 per warp per CTA for quick verify.
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
            float* sScale_ptr = (float*)(smem_ptr + (Q_BYTES_TOTAL + O_BYTES_TOTAL + KV_BYTES_TOTAL));
            const int row = warp_id * 32;
            const float l0 = sScale_ptr[Q_STAGE * M_PER_CTA + 0 * M_PER_CTA + row];
            const float l1 = sScale_ptr[Q_STAGE * M_PER_CTA + 1 * M_PER_CTA + row];
            const int base = (int)blockIdx.y * 64 + (int)cta_rank * 32;
            O_gmem[base + warp_id]      = v0 / l0;
            O_gmem[base + 16 + warp_id] = v1 / l1;
        }
    }
#endif
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
std::vector<torch::Tensor> llm_fa_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
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

    cudaFuncSetAttribute(llm_fa_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, DYN_SMEM_BYTES);

#if EPI_FULL
    auto opts = torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA);
    auto O = torch::zeros({(int64_t)queries, (int64_t)N_DIM}, opts);
#else
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto O = torch::zeros({(int64_t)num_clusters * 64}, opts);
#endif

    auto i32_opts = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    auto prof_buf = torch::zeros({PROF_MAX_EVENTS * 6}, i32_opts);
    auto prof_count = torch::zeros({1}, i32_opts);

    dim3 grid(CTA_GROUP, num_clusters, 1);
    llm_fa_kernel<<<grid, TB_SIZE, DYN_SMEM_BYTES>>>(
        Q_tmap, K_tmap, V_tmap,
#if EPI_FULL
        reinterpret_cast<__nv_bfloat16*>(O.data_ptr()), n_kv,
#else
        reinterpret_cast<float*>(O.data_ptr()), n_kv,
#endif
        reinterpret_cast<ProfEvent*>(prof_buf.data_ptr()),
        reinterpret_cast<int*>(prof_count.data_ptr()));
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "launch failed: ", cudaGetErrorString(err));
    cudaDeviceSynchronize();
    return {O, prof_buf, prof_count};
}
