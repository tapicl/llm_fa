"""Build all three kernels and validate llm_fa against torch SDPA.

Scaffolding and faux kernels intentionally produce incorrect output (no real
softmax math); we just verify they BUILD AND RUN. llm_fa's output is
verified against torch.nn.functional.scaled_dot_product_attention to
bf16 precision (max_abs ≲ 1e-3, mean_abs ≲ 1e-4).
"""
import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))


def torch_sdpa(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """Reference: softmax(QKᵀ/√D) @ V via PyTorch SDPA (non-causal)."""
    q, D = Q.shape
    sk = K.shape[0]
    scale = 1.0 / math.sqrt(D)
    return F.scaled_dot_product_attention(
        Q.view(1, 1, q, D), K.view(1, 1, sk, D), V.view(1, 1, sk, D),
        is_causal=False, scale=scale,
    ).view(q, D)


def report(name: str, out: torch.Tensor, ref: torch.Tensor | None) -> None:
    print(f"\n=== {name} ===")
    print(f"  output shape : {tuple(out.shape)}")
    print(f"  output sum   : {out.float().sum().item():.4e}")
    if ref is None:
        print(f"  (no reference — kernel intentionally produces garbage output)")
        return
    diff = (out.float() - ref.float()).abs()
    print(f"  ref sum      : {ref.float().sum().item():.4e}")
    print(f"  max_abs vs ref : {diff.max().item():.3e}")
    print(f"  mean_abs vs ref: {diff.mean().item():.3e}")
    ok = diff.max().item() < 5e-3
    print(f"  status       : {'PASS' if ok else 'FAIL'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--q",  type=int, default=4096)
    p.add_argument("--sk", type=int, default=4096)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    print(f"shape q={args.q}  sk={args.sk}  D=128  bf16  non-causal")
    print(f"building all three kernels (first build can take 60-180 s) …")

    import scaffolding, faux, llm_fa  # noqa: E402

    torch.manual_seed(args.seed)
    Q = torch.randn(args.q,  128, device="cuda", dtype=torch.bfloat16)
    K = torch.randn(args.sk, 128, device="cuda", dtype=torch.bfloat16)
    V = torch.randn(args.sk, 128, device="cuda", dtype=torch.bfloat16)

    ref = torch_sdpa(Q, K, V)

    report("scaffolding",   scaffolding.forward(Q, K, V), ref=None)
    report("faux",          faux.forward(Q, K, V),        ref=None)
    report("llm_fa",        llm_fa.forward(Q, K, V),      ref=ref)


if __name__ == "__main__":
    main()
