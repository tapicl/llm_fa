# redist_v8d — FlashAttention forward for NVIDIA B200 (sm_100a)

A C++/inline-PTX implementation of FlashAttention forward (BF16, head_dim=128,
non-causal). On a B200 at NCU-locked clocks, **`redist_v8d` trails the stock
CuTeDSL FA4 reference by ~160 ns per main-loop iteration (ratio 0.94)** at
single-wave shapes — i.e. ~6% slower than FA4 while staying entirely in
public PTX. Built directly against `tcgen05.mma` / `cp.async.bulk.tensor` /
`setmaxnreg`.

The repo includes three kernels showing the design at three stages of
completeness:

| File | What it does | Output |
|---|---|---|
| `scaffolding.cu` | TMA + cga2 MMA + 7-pipeline mbarrier choreography. Softmax warps and the correction warp drive the barrier protocol but skip all compute. | **garbage** (intentional) |
| `faux.cu` | + 4-stream row-max + `acc_scale = exp2((m_old − m_new)·log2(e)/√D)` STS + COR FMUL2 rescale of O via TMEM LD→FMUL→ST. Still no softmax exp or final divide. | **garbage** (intentional) |
| `redist_v8d.cu` | + real online softmax (per-lane m_i / l_i) + `cvt.bf16x2` of `exp2(S − m_new)` for P + final 1/l divide. | **correct** (matches torch SDPA to bf16 precision) |

Each adds one phase to the previous; the wallclock cost of each phase shows
where the time goes.

## Latency comparison (NCU clock-locked, single-wave shapes)

NCU `gpu__time_duration.sum` metric, median of 10 runs after 3 warmups.
B200 has 74 cga2 cluster slots; q ≤ 32768 (= 64 clusters) stays single-wave.

| Shape (q, sk) | n_kv | scaffolding µs | faux µs | **redist_v8d µs** | FA4 µs | FA4/v8d | ns/iter |
|--------------:|----:|---:|---:|---:|---:|---:|---:|
| 16384, 16384  |  128 |  344.6 |  319.1 |  **385.3** |  360.6 | 0.936 | −192.6 |
| 16384, 32768  |  256 |  658.0 |  606.7 |  **733.2** |  686.6 | 0.936 | −182.4 |
| 16384, 65536  |  512 | 1242.2 | 1141.0 | **1380.6** | 1294.6 | 0.938 | −167.9 |
| 16384, 131072 | 1024 | 2363.6 | 2138.3 | **2626.2** | 2451.7 | 0.934 | −170.4 |
| 32768, 16384  |  128 |  393.9 |  371.6 |  **438.7** |  412.4 | 0.940 | −205.9 |
| 32768, 32768  |  256 |  743.0 |  696.1 |  **821.1** |  776.1 | 0.945 | −175.9 |
| 32768, 65536  |  512 | 1375.3 | 1283.2 | **1515.3** | 1432.2 | 0.945 | −162.4 |
| 32768, 131072 | 1024 | 2538.8 | 2358.3 | **2809.2** | 2645.3 | 0.942 | −160.1 |

Notes:
- `FA4/v8d` is the latency ratio FA4 µs ÷ v8d µs (<1 means v8d trails FA4).
- `ns/iter = (FA4 − v8d) µs × 1000 / n_kv`; negative means v8d is slower per main-loop iter.
- The per-iter gap plateaus at **−160 to −180 ns/iter** at single-wave shapes;
  it does *not* keep amortizing toward zero with larger n_kv.

## Where the remaining ~6% gap lives

At q=32768/sk=131072 single-wave, the gap is ~163 µs / 1024 iters ≈ 160 ns
per main-loop iteration. That maps to roughly:

- ~50-100 cycles per iter on `s_p_o_full` `mbarrier.try_wait.parity` — the
  SM warp's first wait each iter, waiting for MMA's QK^T commit. FA4 does
  more LDS-heavy work per iter, so MMA's commit fires *within* its SM iter
  body; ours is leaner → finishes faster → stalls.
- Multi-wave shapes (q > 32768) widen this to ~−300 ns/iter because v8d
  runs ≥1.73 waves while FA4 is persistent (1 wave).

## SMEM and warp layout (per CTA)

```
sQ      :  2 stages × 32 KB           = 64 KB
sO      :  2 stages × 32 KB           = 64 KB
sK | sV :  6 stages × 16 KB (aliased) = 96 KB
sScale  :                               2 KB
─────────────────────────────────────────────────
Total dynamic SMEM                    ≈ 226 KB

37 mbarriers across 7 pipelines:
  pipeline_q          (q_stage × 2 = 4 mbars)
  pipeline_kv         (kv_stage × 2 = 12)
  pipeline_s_p_o      (q_stage × 2 = 4)
  pipeline_p_lastsplit                (4)
  pipeline_o_acc      (q_stage × 2 = 4)
  pipeline_sm_stats   (q_stage × 2 = 4)
  pipeline_o_epi      (q_stage × 2 = 4)
  tmem_dealloc                         (1)

16 warps × 32 lanes = 512 threads/CTA:
  warps 0-7   softmax (WG0/1, 192 regs each via setmaxnreg.inc.u32 192)
  warps 8-11  correction        (80  regs via setmaxnreg.dec.u32 80)
  warp 12     MMA driver        (48  regs)
  warp 13     epilogue          (48  regs)
  warp 14     TMA load          (48  regs)
  warp 15     empty             (48  regs)
  total       8·192 + 4·80 + 4·48 = 1728 regs/warp × 32 lanes/warp
              ≈ 55296 regs/CTA, well under 64 K reg file
```

## Key optimizations in `redist_v8d`

- **MMA reorder** — Q0K is issued one main-loop step BEFORE the matching
  P0V (and Q1K → P1V offset). Saves ~1 ms at worst case. See `MMA_REORDER`
  in the source; default is on.
- **FFMA2 split-phase `scale_subtract_rowmax`** — FA4's pattern of
  `s' = s · scale − m · scale` via 64× `fma.rn.ftz.f32x2`. Compiles to
  64 FFMA2 SASS ops, 0 spills.
- **4-stream parallel row-max** — FA4's `fmax_reduce`-style 4-way fanout
  of FMNMX3.FTZ. Breaks the 127-deep serial dep chain a naive reduction
  produces.
- **Hybrid `ex2_emu`** — FA4's `ex2_emu_freq=12, res=4` pattern (12.5%
  polynomial on the FMA pipe, 87.5% MUFU.EX2 on the XU pipe). Offloads
  ~33% of XU pressure to the FMA pipe without overloading it.
- **`fence.acq_rel.cta` bar.arrive pin** — pins the SM→COR named-bar
  `bar.arrive` at the source position so ptxas can't reorder it past the
  ~100 SASS instructions of post-acc_scale compute (`fma.rn.f32x2`,
  `ex2.approx`, `cvt.bf16x2`, STTM). Saves ~230 µs.
- **FA4 `split_P_arrive`** — SM fires `s_p_o_empty` after the first STTM
  chunk; MMA's PV issues 6 of 8 cga2 MMAs (consuming the first 75% of P),
  waits inline on `p_lastsplit_full`, then issues the last 2.
- **K/V SMEM aliasing** — K and V share the same SMEM ring stages (6
  stages, each holding *either* K[j] or V[j]). FA4 trick:
  `sV = recast_ptr(sK, sV_layout.inner)`. Halves K+V footprint.

The CuTeDSL FA4 source these patterns mirror is at
`flash_attn/cute/flash_fwd_sm100.py` and
`flash_attn/cute/blackwell_helpers.py` in the `flash-attn` Python package.

## Requirements

- **Hardware**: NVIDIA Blackwell B200 (sm_100a). The kernel uses
  `tcgen05.mma.cta_group::2`, `cp.async.bulk.tensor.cluster.cta_group::2`,
  `setmaxnreg.{inc,dec}.sync.aligned.u32`, and `mbarrier.try_wait.parity`,
  none of which run on sm_90 or earlier.
- **CUDA toolkit**: 13.0+ (the launcher uses NVCC's `-gencode arch=compute_100a,code=sm_100a`).
- **Python**: 3.10+ with `torch` and `ninja`. Tested on torch 2.6 / CUDA 13.0.
- **For bench only**: `flash-attn>=4.0` with the CuTeDSL backend, and `ncu`
  in `$PATH` (or one of `/usr/local/cuda*/bin/ncu`).

```bash
pip install torch ninja
pip install flash-attn>=4.0     # only required for bench.py
```

## Build + validate

```bash
python validate.py --q 4096 --sk 4096
```

Compiles all three kernels, runs each at the given shape, and checks
`redist_v8d` against `torch.nn.functional.scaled_dot_product_attention`.
Expected: `max_abs ≤ 1e-3`, `status: PASS`.

First build of each kernel takes 60-180 s (mostly ptxas). Subsequent runs
use the cached `.so` under `~/.cache/torch_extensions/`.

## Bench

```bash
python bench.py                     # full table over single-wave shapes
python bench.py --quick             # just q=32768/sk=131072
python bench.py --kernels redist_v8d FA4
```

`bench.py` runs `ncu` for each (kernel, shape) pair and extracts
`gpu__time_duration.sum`. NCU automatically locks the SM clock to its base
(~1.07 GHz on B200), so the wallclock from `cudaEvent` would be ~1.5× faster
but apples-to-apples comparison requires the locked-clock metric.

## Programmatic use

```python
import torch
import redist_v8d

Q = torch.randn(32768, 128, device="cuda", dtype=torch.bfloat16)
K = torch.randn(131072, 128, device="cuda", dtype=torch.bfloat16)
V = torch.randn(131072, 128, device="cuda", dtype=torch.bfloat16)
O = redist_v8d.forward(Q, K, V)   # (32768, 128) bf16
```

Shape constraints: `q` multiple of 512, `sk` multiple of 128 with `sk ≥ 256`,
head dim fixed at 128.

## File layout

```
redist_v8d_release/
├── README.md          ← you are here
├── profiler.cuh       intra-kernel per-warp shmem ring (only used when ENABLE_PROF=1)
├── scaffolding.cu / .py    TMA + MMA + mbarrier scaffold (stubbed softmax)
├── faux.cu / .py           + row-max + acc_scale + COR rescale (still stubbed exp)
├── redist_v8d.cu / .py     full online softmax + final divide
├── validate.py        build all 3, check redist_v8d vs torch SDPA
└── bench.py           NCU sweep with per-shape comparison table
```

## Provenance

The three kernels are lifted from the [`fa_b200`](https://github.com/dxyz/fa_b200)
research repo:

- `scaffolding.cu`  ← `setmaxn_test/redist_v6.cu`
- `faux.cu`         ← `fa4mimic_v2/faux_attn.cu`
- `redist_v8d.cu`   ← `setmaxn_test/redist_v8d.cu`

`profiler.cuh` is from `final_kernels/`. Vendored verbatim; the only changes
in this release directory are renamed `.py` launchers and the README.
