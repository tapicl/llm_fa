"""faux — TMA + QK + (row_max + acc_scale STS) + cvt + STTM-P + PV + COR FMUL2 rescale + STSM-O + TMA-store-O.

Faux+correction attention: exercises the full SM↔COR handshake protocol but
omits the inner softmax math (no exp2 of S, no row_sum tracking, no final
1/row_sum divide). Output O is INTENTIONALLY incorrect; only the latency is
meaningful. Sits between scaffolding (no softmax at all) and full attention
(real online softmax + correction + final divide).

What this measures: the COR rescale pipeline cost when the row-max & acc_scale
production path is real (4-stream FMNMX3 reduce + MUFU.EX2 of (m_old − m_new)
+ STS to sScale + sm_stats bar.sync), without the dominant softmax math.
"""
from pathlib import Path
import os
import torch
from torch.utils.cpp_extension import load_inline

HERE = Path(__file__).parent
ENABLE_PROF = int(os.environ.get("ENABLE_PROF", "0"))

mod = load_inline(
    name=f"faux{'_prof' if ENABLE_PROF else ''}",
    cpp_sources=(
        '#include <torch/extension.h>\n'
        '#include <vector>\n'
        'std::vector<torch::Tensor> fa4_faux_attn_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V);'
    ),
    cuda_sources=(HERE / "faux.cu").read_text(),
    functions=["fa4_faux_attn_forward"],
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
    """Run the faux+corr kernel. Output is INTENTIONALLY incorrect.

    Q: (q,  128) bf16    K: (sk, 128) bf16    V: (sk, 128) bf16
    Returns (q, 128) bf16, values are garbage (no real softmax).
    """
    return mod.fa4_faux_attn_forward(Q, K, V)[0]


def run(q: int = 32768, sk: int = 131072, seed: int = 0):
    torch.manual_seed(seed)
    Q = torch.randn(q,  128, device="cuda", dtype=torch.bfloat16)
    K = torch.randn(sk, 128, device="cuda", dtype=torch.bfloat16)
    V = torch.randn(sk, 128, device="cuda", dtype=torch.bfloat16)
    return forward(Q, K, V)


if __name__ == "__main__":
    import sys
    q  = int(sys.argv[1]) if len(sys.argv) > 1 else 32768
    sk = int(sys.argv[2]) if len(sys.argv) > 2 else 131072
    out = run(q=q, sk=sk)
    print(f"faux OK  q={q}  sk={sk}  O.shape={tuple(out.shape)}")
