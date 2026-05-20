"""llm_fa — FA forward on B200 (sm_100a), full softmax + correction + final divide.

The fastest C++/PTX full-attention kernel in this repo. Trails stock FA4 by
~143 ns per main-loop iter (ratio 0.947) at q=32768/sk=131072 single-wave.
Output matches torch.nn.functional.scaled_dot_product_attention to bf16 precision.

What's inside (per cga2 cluster):
  * sQ + sO + sK/V (aliased 6-stage ring) + sScale ≈ 226 KB dynamic SMEM
  * 37 mbarriers across 7 pipelines (q, kv, s_p_o, p_lastsplit, o_acc,
    sm_stats, o_epi) — matches FA4's flash_fwd_sm100.py
  * 16 warps × 32 lanes = 512 threads/CTA:
      WG0/1 softmax (warps 0-7)   → REGS_SM = 192 (via setmaxnreg.inc)
      WG2  correction (warps 8-11)→ REGS_COR = 80  (via setmaxnreg.dec)
      warp 12 MMA driver / 13 epi / 14 TMA load / 15 empty → 48 R each
  * Real online softmax: per-lane m_i (running rowmax) and l_i (running sum
    of exp2). Cast P = exp2((S − m_new)·log2(e)/√D) to bf16. Sends
    acc_scale = exp2((m_old − m_new)·log2(e)/√D) to correction via sScale.
  * Correction warp rescales O accumulator in TMEM (LD → FMUL2 → ST) per
    iter; final iter divides by l_i via rcp.approx.
  * MMA reorder (Q0K issued one step before matching P0V) for ~1 ms savings
  * FFMA2 split-phase scale_subtract_rowmax (FA4-faithful)
  * 4-stream parallel rowmax (FMNMX3 reduce tree)
  * Hybrid ex2_emu (FA4's freq=12, res=4 pattern: 12.5% poly, 87.5% MUFU.EX2)
  * FA4 split_P_arrive (SM fires s_p_o_empty after first STTM chunk;
    MMA's PV consumes 6/8 cga2 MMAs, waits on p_lastsplit_full, then 2/8).

Shape constraints: q ≥ 512 multiple of 512, sk ≥ 256 multiple of 128, D = 128.
"""
from pathlib import Path
import os
import torch
from torch.utils.cpp_extension import load_inline

HERE = Path(__file__).parent
ENABLE_PROF = int(os.environ.get("ENABLE_PROF", "0"))

mod = load_inline(
    name=f"llm_fa{'_prof' if ENABLE_PROF else ''}",
    cpp_sources=(
        '#include <torch/extension.h>\n'
        '#include <vector>\n'
        'std::vector<torch::Tensor> llm_fa_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V);'
    ),
    cuda_sources=(HERE / "llm_fa.cu").read_text(),
    functions=["llm_fa_forward"],
    extra_cuda_cflags=[
        "-O3", "-gencode=arch=compute_100a,code=sm_100a",
        "--use_fast_math", "-std=c++17", "-lineinfo", "-Xptxas=-v",
        f"-DENABLE_PROF={ENABLE_PROF}",
        f"-I{HERE}",
    ],
    extra_ldflags=["-lcuda"],
    verbose=False,
)


def forward(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """Full attention forward: O = softmax(Q Kᵀ / √D) @ V.

    Q: (q,  128) bf16, q multiple of 512.
    K: (sk, 128) bf16, sk multiple of 128, sk >= 256.
    V: (sk, 128) bf16.
    Returns (q, 128) bf16 matching torch SDPA to bf16 precision (max_abs ≲ 1e-3).
    """
    return mod.llm_fa_forward(Q, K, V)[0]


def run(q: int = 32768, sk: int = 131072, seed: int = 0):
    """Build random Q/K/V at the given shape, run the kernel, return its
    raw output. When ENABLE_PROF=1, the kernel returns a 3-tuple
    (O, prof_buf, prof_count); when ENABLE_PROF=0 the same 3-tuple is
    returned but prof_buf and prof_count are placeholders.
    """
    torch.manual_seed(seed)
    Q = torch.randn(q,  128, device="cuda", dtype=torch.bfloat16)
    K = torch.randn(sk, 128, device="cuda", dtype=torch.bfloat16)
    V = torch.randn(sk, 128, device="cuda", dtype=torch.bfloat16)
    return mod.llm_fa_forward(Q, K, V)


if __name__ == "__main__":
    import sys
    q  = int(sys.argv[1]) if len(sys.argv) > 1 else 32768
    sk = int(sys.argv[2]) if len(sys.argv) > 2 else 131072
    out = run(q=q, sk=sk)
    O = out[0] if isinstance(out, (list, tuple)) else out
    print(f"llm_fa OK  q={q}  sk={sk}  O.shape={tuple(O.shape)}  "
          f"sum={O.float().sum().item():.3e}")
