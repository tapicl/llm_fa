// profiler.cuh — low-overhead intra-kernel profiler.
//
// Design (vs. earlier global-atomic version):
//   * One shmem ring buffer per warp; lane 0 writes its own slots.
//   * Per-warp slot index lives in a register (lane-0-uniform), not shmem.
//   * No atomicAdd anywhere on the recording hot path.
//   * Records 32-bit cycle deltas relative to a per-warp baseline; halves
//     the shmem store width and avoids a clock64 -> packed-uint64 split.
//   * Bulk gmem flush at kernel exit (off the critical path).
//
// The host-side ProfEvent layout (4 u32: warp_id, event_id, start_lo, start_hi,
// end_lo, end_hi packed into a 6×u32 ring) is preserved so that existing
// trace_dump_v*.py scripts keep working — we just convert per-warp 32-bit
// deltas back to absolute u64 cycles inside the flush.
//
// Usage in kernel:
//   PROF_INIT();                         // declares thread-local state
//   {
//     PROF_BEGIN(EV_FOO);
//     ... work ...
//     PROF_END(EV_FOO);
//   }
//   PROF_FLUSH(prof_buf, prof_count);    // ONE call near kernel exit
//
// The kernel must allocate `__shared__ ProfShRec prof_sm[16 * PROF_PER_WARP_SLOTS]`
// and pass it via PROF_BIND(prof_sm). Per-warp baseline + slot count are local
// register state captured in PROF_INIT.

#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

constexpr int PROF_MAX_EVENTS = 16384;    // host-side buffer cap (unchanged)
constexpr int PROF_NUM_WARPS = 16;        // warps per recording CTA
// Per-warp event capacity — default 96; kernels can #define before including
// profiler.cuh to capture longer traces (at the cost of more static smem:
// 16 warps * SLOTS * 12 B; e.g. 256 → 48 KB static).
#ifndef PROF_PER_WARP_SLOTS
#define PROF_PER_WARP_SLOTS 96
#endif

// Shmem record: evid + 32-bit start delta + 32-bit end delta = 12 B.
struct ProfShRec {
    uint32_t evid;
    uint32_t start_d;
    uint32_t end_d;
};

// Host-visible record (written by flush).
struct ProfEvent {
    uint32_t warp_id;
    uint32_t event_id;
    uint64_t start;
    uint64_t end;
};

__device__ inline uint64_t prof_clock() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t));
    return t;
}

// Lane-0-only recorder. `count` is a register held in the caller.
__device__ inline void prof_record_sm_(int warp_id, int evid, uint32_t start_d, uint32_t end_d,
                                       int lane_id, int& count, ProfShRec* sm_buf) {
    if (lane_id != 0)
        return;
    if (blockIdx.x | blockIdx.y | blockIdx.z)
        return;
    if (count >= PROF_PER_WARP_SLOTS)
        return;
    ProfShRec r;
    r.evid = (uint32_t)evid;
    r.start_d = start_d;
    r.end_d = end_d;
    sm_buf[warp_id * PROF_PER_WARP_SLOTS + count] = r;
    count++;
}

// Flush all per-warp shmem rings to the global ring buffer used by the host.
// Called once near kernel exit. Each warp's lane 0 copies its slots, then
// arrives on a single atomicAdd to claim a contiguous gmem region for its
// own warp — one atomic per warp instead of one per event.
__device__ inline void prof_flush_(int warp_id, int lane_id, uint64_t baseline, int count,
                                   ProfShRec* sm_buf, ProfEvent* prof_buf, int* prof_count) {
    if (lane_id != 0)
        return;
    if (blockIdx.x | blockIdx.y | blockIdx.z)
        return;
    if (count <= 0)
        return;
    int base_idx = atomicAdd(prof_count, count);
    if (base_idx >= PROF_MAX_EVENTS)
        return;
    int n = count;
    if (base_idx + n > PROF_MAX_EVENTS)
        n = PROF_MAX_EVENTS - base_idx;
    ProfShRec* src = sm_buf + warp_id * PROF_PER_WARP_SLOTS;
    for (int k = 0; k < n; k++) {
        ProfEvent e;
        e.warp_id = (uint32_t)warp_id;
        e.event_id = src[k].evid;
        e.start = baseline + (uint64_t)src[k].start_d;
        e.end = baseline + (uint64_t)src[k].end_d;
        prof_buf[base_idx + k] = e;
    }
}

// -------- macros (call inside the kernel) --------
//
// PROF_INIT() -> declares __prof_count (register), __prof_base (register),
//                and assumes __prof_sm and warp_id, lane_id are in scope.
// PROF_BIND(buf) -> sets __prof_sm to the shmem buffer pointer.
// PROF_BEGIN(id) -> reads clock64, stores start delta in __prof_t_<id>.
// PROF_END(id)   -> reads clock64, computes deltas, writes shmem record.
// PROF_FLUSH(buf, cnt) -> emits per-warp record range to gmem.

// All PROF_* macros below compile to nothing when ENABLE_PROF=0 — no clock
// reads, no register pressure, no static smem allocation.

#if ENABLE_PROF

#define PROF_INIT_BASELINE()                                                                       \
    uint64_t __prof_base = prof_clock();                                                           \
    int __prof_count = 0

#define PROF_BIND_SM(_sm_ptr) ProfShRec* __prof_sm = (_sm_ptr)

#define PROF_BEGIN_X(id)                                                                           \
    uint32_t __prof_t_##id = (uint32_t)(prof_clock() - __prof_base)
#define PROF_END_X(id)                                                                             \
    do {                                                                                           \
        uint32_t __end_d = (uint32_t)(prof_clock() - __prof_base);                                 \
        prof_record_sm_(warp_id, (id), __prof_t_##id, __end_d, lane_id, __prof_count,              \
                        __prof_sm);                                                                \
    } while (0)

#define PROF_BEGIN_R(tag)                                                                          \
    uint32_t __prof_t_##tag = (uint32_t)(prof_clock() - __prof_base)
#define PROF_END_R(tag, id)                                                                        \
    do {                                                                                           \
        uint32_t __end_d = (uint32_t)(prof_clock() - __prof_base);                                 \
        prof_record_sm_(warp_id, (id), __prof_t_##tag, __end_d, lane_id, __prof_count,             \
                        __prof_sm);                                                                \
    } while (0)

#define PROF_FLUSH(_pbuf, _pcnt)                                                                   \
    do {                                                                                           \
        prof_flush_(warp_id, lane_id, __prof_base, __prof_count, __prof_sm, (_pbuf), (_pcnt));     \
    } while (0)

#else  // ENABLE_PROF == 0 — fully compile out

#define PROF_INIT_BASELINE()        ((void)0)
#define PROF_BIND_SM(_sm_ptr)       ((void)0)
#define PROF_BEGIN_X(id)            ((void)0)
#define PROF_END_X(id)              ((void)0)
#define PROF_BEGIN_R(tag)           ((void)0)
#define PROF_END_R(tag, id)         ((void)0)
#define PROF_FLUSH(_pbuf, _pcnt)    ((void)0)

#endif
