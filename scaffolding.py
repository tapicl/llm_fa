"""scaffolding — FA4-precise SMEM layout + 37-mbarrier pipeline with stubbed softmax.

Measures the pure TMA + MMA + barrier-choreography floor on B200. Softmax warps
(WG0, WG1) and the correction warp (WG2) participate in mbarrier handshakes but
do no math — output O is INTENTIONALLY incorrect; only the latency is meaningful.

SMEM layout (per CTA):
  sQ      : 2 stages × 32 KB = 64 KB
  sO      : 2 stages × 32 KB = 64 KB
  sK | sV : 6 stages × 16 KB (K and V aliased per stage) = 96 KB
  sScale  : 2 KB
  total   ≈ 226 KB dynamic SMEM
"""
from pathlib import Path
import os
import torch
from torch.utils.cpp_extension import load_inline

HERE = Path(__file__).parent
ENABLE_PROF  = int(os.environ.get("ENABLE_PROF", "0"))
SKIP_SM_FULL = int(os.environ.get("SKIP_SM_FULL", "0"))

mod = load_inline(
    name=f"scaffolding{'_prof' if ENABLE_PROF else ''}",
    cpp_sources=(
        '#include <torch/extension.h>\n'
        '#include <vector>\n'
        'std::vector<torch::Tensor> redist_v6_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V);'
    ),
    cuda_sources=(HERE / "scaffolding.cu").read_text(),
    functions=["redist_v6_forward"],
    extra_cuda_cflags=[
        "-O3", "-gencode=arch=compute_100a,code=sm_100a",
        "--use_fast_math", "-std=c++17", "-lineinfo", "-Xptxas=-v",
        f"-DENABLE_PROF={ENABLE_PROF}",
        f"-DSKIP_SM_FULL={SKIP_SM_FULL}",
        f"-I{HERE}",
    ],
    extra_ldflags=["-lcuda"],
    verbose=False,
)


def forward(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """Run the scaffolding kernel. Output is INTENTIONALLY incorrect (stub softmax).

    Q: (q,  128) bf16    K: (sk, 128) bf16    V: (sk, 128) bf16
    Returns (q, 128) bf16, but the values are garbage.
    """
    return mod.redist_v6_forward(Q, K, V)[0]


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
    print(f"scaffolding OK  q={q}  sk={sk}  O.shape={tuple(out.shape)}")
