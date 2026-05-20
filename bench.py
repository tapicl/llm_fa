"""NCU clock-locked bench: scaffolding / faux / llm_fa vs stock FA4.

Single-wave shapes (q ≤ 32768 → ≤ 64 cga2 cluster slots on B200's 74 slots).
Prints one row per (kernel, shape) with median µs over `iters` runs after
`warmup` warmups.

Requires:
  * nvidia-cuda-toolkit (ncu in $PATH or /usr/local/cuda*/bin/ncu)
  * flash_attn>=4.0 with CuTeDSL backend for FA4 reference
  * torch, ninja (for load_inline)

Usage:
  python bench.py                  # full table
  python bench.py --quick          # 1 shape only (q=32768/sk=131072)
  python bench.py --kernels llm_fa   # subset
"""
import argparse
import csv
import io
import os
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
PY = sys.executable

# Locate ncu
def find_ncu() -> str:
    for path in [shutil.which("ncu"),
                 "/usr/local/cuda-13.0/bin/ncu",
                 "/usr/local/cuda-12.5/bin/ncu",
                 "/usr/local/cuda/bin/ncu"]:
        if path and Path(path).exists():
            return path
    raise SystemExit("could not find ncu — install nvidia-cuda-toolkit")

NCU = find_ncu()


# kernel_re below matches the kernel symbol emitted by load_inline.
# When the .cu source is renamed but the __global__ function keeps its
# original name, the regex looks for that original name.
KERNELS = {
    # name            : (worker script path,   kernel_re,                         build_warmup_call)
    "scaffolding": ("scaffolding",  "regex:redist_v6_kernel"),
    "faux":        ("faux",         "regex:fa4_faux_attn_kernel"),
    "llm_fa":      ("llm_fa",       "regex:llm_fa_kernel"),
    "FA4":         ("fa4",          "regex:.*FlashAttentionForwardSm100[^F].*"),
}


# Single-wave shapes: q × (16 KB Q + 16 KB O) ≤ 226 KB × 74 cga2 slots →
# q ≤ 64 · 512 = 32768 (single-wave). sk ranges from 16384 to 131072.
DEFAULT_SHAPES = [
    (16384, 16384), (16384, 32768), (16384, 65536), (16384, 131072),
    (32768, 16384), (32768, 32768), (32768, 65536), (32768, 131072),
]


WORKER_TMPL = '''\
"""Per-kernel NCU worker. argv: q sk iters."""
import sys
q, sk, iters = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
sys.argv = ["_worker"]
sys.path.insert(0, "{here}")
import torch
{import_line}
torch.manual_seed(0)
{shape_setup}
for _ in range(iters):
    {call_line}
torch.cuda.synchronize()
'''


def write_worker(name: str) -> Path:
    workers = {
        "scaffolding": dict(
            import_line="from scaffolding import mod",
            shape_setup=(
                'Q = torch.randn(q,  128, device="cuda", dtype=torch.bfloat16).contiguous()\n'
                'K = torch.randn(sk, 128, device="cuda", dtype=torch.bfloat16).contiguous()\n'
                'V = torch.randn(sk, 128, device="cuda", dtype=torch.bfloat16).contiguous()'),
            call_line="mod.redist_v6_forward(Q, K, V)",
        ),
        "faux": dict(
            import_line="from faux import mod",
            shape_setup=(
                'Q = torch.randn(q,  128, device="cuda", dtype=torch.bfloat16).contiguous()\n'
                'K = torch.randn(sk, 128, device="cuda", dtype=torch.bfloat16).contiguous()\n'
                'V = torch.randn(sk, 128, device="cuda", dtype=torch.bfloat16).contiguous()'),
            call_line="mod.fa4_faux_attn_forward(Q, K, V)",
        ),
        "llm_fa": dict(
            import_line="from llm_fa import mod",
            shape_setup=(
                'Q = torch.randn(q,  128, device="cuda", dtype=torch.bfloat16).contiguous()\n'
                'K = torch.randn(sk, 128, device="cuda", dtype=torch.bfloat16).contiguous()\n'
                'V = torch.randn(sk, 128, device="cuda", dtype=torch.bfloat16).contiguous()'),
            call_line="mod.llm_fa_forward(Q, K, V)",
        ),
        "FA4": dict(
            import_line="from flash_attn.cute import flash_attn_func",
            shape_setup=(
                'Q = torch.randn(1, q,  1, 128, device="cuda", dtype=torch.bfloat16).contiguous()\n'
                'K = torch.randn(1, sk, 1, 128, device="cuda", dtype=torch.bfloat16).contiguous()\n'
                'V = torch.randn(1, sk, 1, 128, device="cuda", dtype=torch.bfloat16).contiguous()'),
            call_line="flash_attn_func(Q, K, V, causal=False)",
        ),
    }
    src = WORKER_TMPL.format(here=HERE, **workers[name])
    path = HERE / f"_worker_{name}.py"
    path.write_text(src)
    return path


def parse_durations_us(csv_text: str) -> list[float]:
    durations_ns = []
    rows = list(csv.reader(io.StringIO(csv_text)))
    if not rows: return []
    h = rows[0]
    try:
        ui = h.index("Metric Unit")
        vi = h.index("Metric Value")
        mi = h.index("Metric Name")
    except ValueError:
        return []
    factor = {"ns": 1.0, "us": 1e3, "ms": 1e6, "s": 1e9}
    for r in rows[1:]:
        if len(r) <= max(ui, vi, mi): continue
        if r[mi] != "gpu__time_duration.sum": continue
        try: v = float(r[vi])
        except ValueError: continue
        durations_ns.append(factor.get(r[ui], 1.0) * v)
    return [x / 1e3 for x in durations_ns]


def run_ncu(worker: Path, kernel_re: str, q: int, sk: int,
            warmup: int, iters: int, timeout: float = 1500.0) -> list[float]:
    env = os.environ.copy()
    cmd = [
        NCU, "--csv", "--target-processes", "all",
        "--metrics", "gpu__time_duration.sum",
        "--kernel-name", kernel_re,
        "--launch-count", str(warmup + iters),
        PY, str(worker), str(q), str(sk), str(warmup + iters),
    ]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
    if p.returncode != 0:
        raise RuntimeError(p.stderr[-400:])
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith('"') and '","' in ln]
    if not lines:
        raise RuntimeError("no CSV rows in ncu output")
    header = ('"ID","Process ID","Process Name","Host Name","Kernel Name",'
              '"Context","Stream","Block Size","Grid Size","Device","CC",'
              '"Section Name","Metric Name","Metric Unit","Metric Value"')
    return parse_durations_us(header + "\n" + "\n".join(lines))


def summarize(durs: list[float], warmup: int, iters: int):
    if len(durs) < warmup + iters: return None
    measured = durs[warmup : warmup + iters]
    return {"median": statistics.median(measured),
            "min": min(measured),
            "max": max(measured)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters",  type=int, default=10)
    ap.add_argument("--quick", action="store_true",
                    help="only q=32768/sk=131072")
    ap.add_argument("--kernels", nargs="+",
                    default=list(KERNELS.keys()),
                    choices=list(KERNELS.keys()))
    args = ap.parse_args()

    shapes = [(32768, 131072)] if args.quick else DEFAULT_SHAPES

    print(f"# NCU clock-locked latency, kernel duration only (ncu metric "
          f"gpu__time_duration.sum)")
    print(f"# median of {args.iters} measurements after {args.warmup} warmups, µs")
    print(f"# single-wave shapes on B200 (74 cga2 cluster slots)")
    print()

    # Build workers
    workers = {name: write_worker(name) for name in args.kernels}

    # Run all (shape × kernel)
    results: dict[tuple, dict] = {}
    for (q, sk) in shapes:
        for name in args.kernels:
            kernel_re = KERNELS[name][1]
            print(f"  running {name:<12} q={q:>5} sk={sk:>6} …", end="", flush=True)
            try:
                d = run_ncu(workers[name], kernel_re, q, sk, args.warmup, args.iters)
                s = summarize(d, args.warmup, args.iters)
                if s is None:
                    print(f" no data ({len(d)} samples)")
                    continue
                results[(q, sk, name)] = s
                print(f" {s['median']:>8.1f} µs")
            except Exception as e:
                print(f" FAIL: {str(e)[:200]}")

    # Print table
    print("\n" + "=" * 92)
    print(f"{'shape':<18} {'n_kv':>5} " +
          " ".join(f"{n:>12}" for n in args.kernels) +
          f" {'llm_fa/FA4':>10} {'ns/iter':>10}")
    print("-" * 92)
    for (q, sk) in shapes:
        n_kv = sk // 128
        row = f"q{q}_sk{sk:<8}"[:18]
        cells = []
        for name in args.kernels:
            s = results.get((q, sk, name))
            cells.append(f"{s['median']:>12.1f}" if s else f"{'-':>12}")
        cs = " ".join(cells)
        # ratio + ns/iter for llm_fa vs FA4 (if both present)
        ours = results.get((q, sk, "llm_fa"))
        fa4 = results.get((q, sk, "FA4"))
        if ours and fa4:
            ratio = fa4["median"] / ours["median"]
            dpi = (fa4["median"] - ours["median"]) * 1000 / n_kv
            print(f"{row:<18} {n_kv:>5} {cs} {ratio:>10.3f} {dpi:>+8.1f}ns")
        else:
            print(f"{row:<18} {n_kv:>5} {cs} {'-':>10} {'-':>10}")

    print()
    print("# ratio = FA4 / llm_fa   (1.0 means tied; <1 means we trail FA4)")
    print("# ns/iter = (FA4_us − llm_fa_us) × 1000 / n_kv   (negative = trail)")


if __name__ == "__main__":
    main()
