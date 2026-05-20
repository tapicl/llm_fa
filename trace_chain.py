"""trace_chain.py — print the transitive flow chain feeding into a slice.

Usage:
  python trace_chain.py P0V7
  python trace_chain.py P0V7 --trace traces/trace_llm_fa_pp1_q512_sk512.pftrace
  python trace_chain.py P0V7 --depth 6   # cap walk depth
  python trace_chain.py P0V7 --forward   # walk outgoing instead of incoming
"""
import argparse
from pathlib import Path
from perfetto.trace_processor import TraceProcessor

HERE = Path(__file__).parent
DEFAULT_TRACE = HERE / "traces" / "trace_llm_fa_pp1_q512_sk512.pftrace"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("slice_name")
    ap.add_argument("--trace", default=str(DEFAULT_TRACE))
    ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--forward", action="store_true",
                    help="walk outgoing flows (descendants) instead of incoming (ancestors)")
    args = ap.parse_args()

    tp = TraceProcessor(trace=args.trace)

    # Cross-warp edges from the flow table.
    rows = list(tp.query("""
      SELECT s_src.id AS src_id, s_src.name AS src_name, s_src.ts AS src_ts,
             s_dst.id AS dst_id, s_dst.name AS dst_name, s_dst.ts AS dst_ts
      FROM flow f
      JOIN slice s_src ON s_src.id = f.slice_out
      JOIN slice s_dst ON s_dst.id = f.slice_in
    """))

    # Same-track temporal edges: previous slice on the same thread track.
    # These represent "this slice can't start until the previous one on
    # this warp finished" — they're implicit on the timeline but invisible
    # to the flow table.
    same_rows = list(tp.query("""
      WITH ordered AS (
        SELECT s.id, s.name, s.ts, s.track_id,
               LAG(s.id)   OVER (PARTITION BY s.track_id ORDER BY s.ts) AS prev_id,
               LAG(s.name) OVER (PARTITION BY s.track_id ORDER BY s.ts) AS prev_name,
               LAG(s.ts)   OVER (PARTITION BY s.track_id ORDER BY s.ts) AS prev_ts
        FROM slice s
      )
      SELECT id AS dst_id, name AS dst_name, ts AS dst_ts,
             prev_id AS src_id, prev_name AS src_name, prev_ts AS src_ts
      FROM ordered WHERE prev_id IS NOT NULL
    """))

    if args.forward:
        graph = {}
        for r in rows:
            graph.setdefault(r.src_id, []).append((r.dst_id, r.dst_name, r.dst_ts, "flow"))
        for r in same_rows:
            graph.setdefault(r.src_id, []).append((r.dst_id, r.dst_name, r.dst_ts, "track"))
    else:
        graph = {}
        for r in rows:
            graph.setdefault(r.dst_id, []).append((r.src_id, r.src_name, r.src_ts, "flow"))
        for r in same_rows:
            graph.setdefault(r.dst_id, []).append((r.src_id, r.src_name, r.src_ts, "track"))

    # Find starting slice ID(s).
    starts = list(tp.query(
        f"SELECT id, name, ts FROM slice WHERE name = '{args.slice_name}'"))
    if not starts:
        print(f"no slice named {args.slice_name!r}")
        tp.close()
        return

    direction = "outgoing (descendants)" if args.forward else "incoming (ancestors)"
    print(f"Chain {direction} from {args.slice_name!r} (depth ≤ {args.depth}):\n")

    for start in starts:
        print(f"[start] {start.name}@ts={start.ts}  id={start.id}")
        visited = set()

        def walk(node_id, depth, prefix):
            if depth > args.depth or node_id in visited:
                return
            visited.add(node_id)
            for (nid, nname, nts, kind) in graph.get(node_id, []):
                arrow = "<-" if not args.forward else "->"
                tag = "[F]" if kind == "flow" else "[T]"  # F=flow, T=same-track
                print(f"{prefix}{arrow} {tag} {nname}@ts={nts}  id={nid}")
                walk(nid, depth + 1, prefix + "  ")

        walk(start.id, 1, "  ")
        print()

    tp.close()


if __name__ == "__main__":
    main()
