"""trace_dump.py — Perfetto PROTOBUF (.pftrace) dumper for llm_fa.

Adapted from trace_dump_v7_pb.py. Adds rendering for the new SM0/SM1
ping-pong events (EV_PP_WAIT=20, EV_PP_ARRIVE=21) on the SM warp tracks.

Output: traces/trace_llm_fa_pp{PP}_q{Q}_sk{SK}.pftrace

Open in https://ui.perfetto.dev (drag-and-drop).

Capacity note (n_kv=1024 worst case):
  Each SM warp emits 6 events / iter (WS, RM, PP_WAIT, EX2, PP_ARRIVE, STP).
  Per-warp ring buffer is PROF_PER_WARP_SLOTS=256 slots. So per-warp the
  trace truncates at iter ~42. Dumper detects this and reports the actual
  captured iteration count.
"""
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ["ENABLE_PROF"] = "1"
# Pick up PINGPONG / LSUM_4STREAM from caller env (default PP=1, l4s on)
os.environ.setdefault("PINGPONG", "1")
os.environ.setdefault("LSUM_4STREAM", "1")

import torch  # noqa: E402
from perfetto.trace_builder.proto_builder import TraceProtoBuilder
from perfetto.protos.perfetto.trace.perfetto_trace_pb2 import TrackEvent

HERE = Path(__file__).parent
TRACES_DIR = HERE / "traces"
TRACES_DIR.mkdir(exist_ok=True)
sys.path.insert(0, str(HERE))

EVENT_NAMES = {
    0:  "TMA_Q",       1: "TMA_wait_kv_empty", 2: "TMA_K",      3: "TMA_V",
    4:  "MMA_wait_Q",  5: "MMA_wait_K",        6: "MMA_wait_V",
    7:  "MMA_QK_p0",   8: "MMA_QK_p1",         9: "MMA_PV_p0",  10: "MMA_PV_p1",
    11: "MMA_wait_PR0", 12: "MMA_wait_PR1",
    13: "SM_wait_S",   14: "SM_RM",            15: "SM_EX2",
    16: "COR_wait_O0", 17: "EPI_wait",         18: "COR_wait_O1",
    19: "SM_STP",
    20: "PP_wait",     21: "PP_arrive",
}

WARP_ROLE = {
    0: "SM0_w0", 1: "SM0_w1", 2: "SM0_w2", 3: "SM0_w3",
    4: "SM1_w0", 5: "SM1_w1", 6: "SM1_w2", 7: "SM1_w3",
    8: "COR_w0", 9: "COR_w1", 10: "COR_w2", 11: "COR_w3",
    12: "MMA", 13: "EPI", 14: "LOAD", 15: "EMPTY",
}

# Once-only events (no iter index)
NO_ITER_EVIDS = {4, 17}

# Wait-type events. Rendered as 80 ns markers at END so flow arrows point in
# the natural producer→consumer dependency direction. EV_PP_WAIT (20) IS NOT
# included here on purpose: we want to SEE the full ping-pong stall as a
# colored bar — that's the whole point of this trace. The PP wait does not
# carry user-chain flow edges, so making it long does not break navigation.
WAIT_EVIDS = {1, 4, 5, 6, 11, 12, 13, 16, 17, 18}
WAIT_MARKER_NS = 80


def compact_label(evid, warp, k):
    p = 0 if warp < 4 else 1
    pi_us = "" if k is None else f"_{k}"
    if evid == 0:   return f"TQ{k}"        if k is not None else "TQ"
    if evid == 1:
        N_INFLIGHT = 3
        if k is None: return "Wke"
        stage = "K" if k % 2 == 0 else "V"
        iter_num = (k // 2) + N_INFLIGHT
        return f"Wke{iter_num}{stage}"
    if evid == 2:   return f"Tk{k}"
    if evid == 3:   return f"Tv{k}"
    if evid == 4:   return "WQ"
    if evid == 5:   return f"Wk{k}"
    if evid == 6:   return f"Wv{k}"
    if evid == 7:   return f"Q0K{k}"
    if evid == 8:   return f"Q1K{k}"
    if evid == 9:   return f"P0V{k}"
    if evid == 10:  return f"P1V{k}"
    if evid == 11:  return f"WPr0{pi_us}"
    if evid == 12:  return f"WPr1{pi_us}"
    if evid == 13:  return f"WS{p}{pi_us}"
    if evid == 14:  return f"Rm{p}{pi_us}"
    if evid == 15:  return f"Ex{p}{pi_us}"
    if evid == 16:  return f"Wco0_{k}"
    if evid == 17:  return "EW"
    if evid == 18:  return f"Wco1_{k}"
    if evid == 19:  return f"StP{p}_{k}"
    if evid == 20:  return f"PPw{p}_{k}"   # PP wait (the visible stagger)
    if evid == 21:  return f"PPa{p}_{k}"   # PP arrive (short marker)
    return f"ev{evid}"


def main():
    q  = int(sys.argv[1]) if len(sys.argv) > 1 else 65536
    sk = int(sys.argv[2]) if len(sys.argv) > 2 else 131072
    PP = int(os.environ.get("PINGPONG", "1"))
    L4 = int(os.environ.get("LSUM_4STREAM", "1"))
    GHZ = 1.965
    NS_PER_CYCLE = 1.0 / GHZ  # nanoseconds

    import llm_fa

    # Warm-up (JIT compile + caches), then measure.
    for _ in range(3):
        llm_fa.run(q=q, sk=sk)
    torch.cuda.synchronize()

    O, prof_buf, prof_count = llm_fa.run(q=q, sk=sk)
    torch.cuda.synchronize()
    n = int(prof_count.cpu().item())
    cap = prof_buf.numel() // 6
    print(f"recorded {n} events (host cap {cap})  PINGPONG={PP}  LSUM_4STREAM={L4}")
    if n == 0:
        print("WARNING: no events recorded")
        return
    if n >= cap:
        print(f"WARNING: host buffer saturated (n=={cap}); per-warp truncation also expected")

    raw = prof_buf.cpu().tolist()[: n * 6]
    def u32_pair(lo, hi):
        return (lo & 0xFFFFFFFF) | ((hi & 0xFFFFFFFF) << 32)
    starts = [u32_pair(raw[i*6 + 2], raw[i*6 + 3]) for i in range(n)]
    ends   = [u32_pair(raw[i*6 + 4], raw[i*6 + 5]) for i in range(n)]
    t0 = min(starts)

    # Build events list. Each event also gets a unique index for flow lookup.
    events = []
    by_we = defaultdict(list)
    we_counter = defaultdict(int)
    for i in range(n):
        warp = int(raw[i*6 + 0])
        evid = int(raw[i*6 + 1])
        s_cyc = int(starts[i] - t0)
        e_cyc = int(ends[i]   - t0)
        s_ns_orig  = int(s_cyc * NS_PER_CYCLE)
        e_ns_orig  = max(s_ns_orig + 1, int(e_cyc * NS_PER_CYCLE))
        s_ns, e_ns = s_ns_orig, e_ns_orig
        if evid in WAIT_EVIDS:
            s_ns = max(s_ns, e_ns - WAIT_MARKER_NS)
        if evid in NO_ITER_EVIDS:
            k = None
        else:
            k = we_counter[(warp, evid)]
            we_counter[(warp, evid)] += 1
        idx = len(events)
        events.append({
            "idx": idx, "warp": warp, "evid": evid, "k": k,
            "s_ns": s_ns, "e_ns": e_ns,
            "name": compact_label(evid, warp, k),
            "full_name": EVENT_NAMES.get(evid, f"ev{evid}") + (f"[k={k}]" if k is not None else ""),
        })
        by_we[(warp, evid)].append(idx)

        # Un-shrunk visualization slice for COR pair work (carries no flows).
        if evid == 16 or evid == 18:
            pair_id = 0 if evid == 16 else 1
            idx_v = len(events)
            events.append({
                "idx": idx_v, "warp": warp, "evid": -1, "k": k,
                "s_ns": s_ns_orig, "e_ns": e_ns_orig,
                "name": f"Cor{pair_id}_{k}",
                "full_name": f"COR_pair{pair_id}_work[k={k}]",
            })

    # Determine effective n_kv per warp role (use the SM warps which are the
    # main fan-out and the ones whose ring will saturate first).
    sm_iters_per_warp = [we_counter.get((w, 13), 0) for w in range(8)]  # WS counter
    mma_iters = we_counter.get((12, 7), 0)  # Q0K counter
    n_kv_used = min([x for x in sm_iters_per_warp if x > 0] + [mma_iters])
    print(f"SM warp iters captured: {sm_iters_per_warp}")
    print(f"MMA Q0K iters captured: {mma_iters}")
    print(f"Effective n_kv for edge encoding: {n_kv_used}")

    # Flow ID bookkeeping.
    outgoing = defaultdict(list)
    incoming = defaultdict(list)
    flow_id_counter = [1]

    def add_edge(src_warp, src_evid, src_idx, dst_warp, dst_evid, dst_idx):
        srcs = by_we.get((src_warp, src_evid), [])
        dsts = by_we.get((dst_warp, dst_evid), [])
        if src_idx >= len(srcs) or dst_idx >= len(dsts):
            return False
        s_ev = events[srcs[src_idx]]
        d_ev = events[dsts[dst_idx]]
        if s_ev["s_ns"] >= d_ev["e_ns"]:
            return False
        fid = flow_id_counter[0]
        flow_id_counter[0] += 1
        outgoing[s_ev["idx"]].append(fid)
        incoming[d_ev["idx"]].append(fid)
        return True

    n_kv = n_kv_used
    LOAD_W = 14
    MMA_W = 12
    COR_WS = (8, 9, 10, 11)
    SM_WGS = {0: (0, 1, 2, 3), 1: (4, 5, 6, 7)}
    WCO_EVID = {0: 16, 1: 18}
    COR_LEAD = COR_WS[0]
    COR_LEAD_FOR_RM = COR_WS[0]

    # 1. Tk → Wk, Tv → Wv
    for j in range(n_kv):
        add_edge(LOAD_W, 2, j, MMA_W, 5, j)
        add_edge(LOAD_W, 3, j, MMA_W, 6, j)
    # 2. Q<p>K → WS<p>
    for j in range(n_kv):
        for pair in (0, 1):
            add_edge(MMA_W, 7 + pair, j, SM_WGS[pair][0], 13, j)
    # 3. Rm<p> → Wco<p>  (per-pair COR wait — early trigger). All 4 SM warps.
    for j in range(n_kv):
        for pair in (0, 1):
            for w in SM_WGS[pair]:
                add_edge(w, 14, j, COR_LEAD_FOR_RM, WCO_EVID[pair], j)
    # 4. StP<p> → WPr<p>  (MMA wait_PR proceeds when SM stores P)
    for j in range(n_kv):
        for pair in (0, 1):
            add_edge(SM_WGS[pair][0], 19, j, MMA_W, 11 + pair, j)
    # 5. P<p>V → Wke[j+LAG] V-stage. LAG = N_INFLIGHT = 3.
    LAG = 3
    N_INFLIGHT = 3
    for j in range(n_kv - LAG):
        for pair in (0, 1):
            target_iter = j + LAG
            t_idx = 2 * (target_iter - N_INFLIGHT) + 1
            add_edge(MMA_W, 9 + pair, j, LOAD_W, 1, t_idx)
    # 6. Wco<p> → WPr<p>  (COR rescale done releases MMA)
    for j in range(n_kv):
        for pair in (0, 1):
            add_edge(COR_LEAD, WCO_EVID[pair], j, MMA_W, 11 + pair, j)

    # 7. User-clickable shortcut edges. All-warp fan-in for SM-side slices.
    for j in range(n_kv):
        for pair in (0, 1):
            # P<p>V depends on Wco<p>
            add_edge(COR_LEAD, WCO_EVID[pair], j, MMA_W, 9 + pair, j)
            for w in SM_WGS[pair]:
                # P<p>V depends on every warp's Ex<p>
                add_edge(w, 15, j, MMA_W, 9 + pair, j)
                # Every warp's Rm<p> depends on Q<p>K
                add_edge(MMA_W, 7 + pair, j, w, 14, j)

    # ----- Protobuf build -----
    builder = TraceProtoBuilder()
    PID = 1
    SEQ = 1
    PROCESS_UUID = 1

    pkt = builder.add_packet()
    pkt.track_descriptor.uuid = PROCESS_UUID
    pkt.track_descriptor.process.pid = PID
    pkt.track_descriptor.process.process_name = (
        f"llm_fa q={q} sk={sk} PP={PP} L4={L4} n_kv_capt={n_kv}"
    )
    pkt.trusted_packet_sequence_id = SEQ
    pkt.first_packet_on_sequence = True
    pkt.previous_packet_dropped = True
    pkt.sequence_flags = 1

    warp_track_uuid = {}
    for warp in range(16):
        uuid = 100 + warp
        warp_track_uuid[warp] = uuid
        pkt = builder.add_packet()
        pkt.track_descriptor.uuid = uuid
        pkt.track_descriptor.parent_uuid = PROCESS_UUID
        pkt.track_descriptor.thread.pid = PID
        pkt.track_descriptor.thread.tid = warp
        pkt.track_descriptor.thread.thread_name = WARP_ROLE.get(warp, f"warp{warp}")
        pkt.trusted_packet_sequence_id = SEQ

    # END-anchor any flows whose producer.BEGIN >= consumer.BEGIN.
    incoming_at_end = defaultdict(list)
    for ev in events:
        kept = []
        for fid in incoming[ev["idx"]]:
            src_evs = [e for e in events if fid in outgoing[e["idx"]]]
            inverted = any(s["s_ns"] >= ev["s_ns"] for s in src_evs)
            if inverted:
                incoming_at_end[ev["idx"]].append(fid)
            else:
                kept.append(fid)
        incoming[ev["idx"]] = kept

    timeline = []
    for ev in events:
        timeline.append((ev["s_ns"], "B", ev))
        timeline.append((ev["e_ns"], "E", ev))
    def sort_key(t):
        order = {"B": 0, "E": 1}
        return (t[0], order[t[1]])
    timeline.sort(key=sort_key)

    for ts, kind, ev in timeline:
        warp = ev["warp"]
        pkt = builder.add_packet()
        pkt.timestamp = ts
        pkt.track_event.track_uuid = warp_track_uuid[warp]
        if kind == "B":
            pkt.track_event.type = TrackEvent.TYPE_SLICE_BEGIN
            pkt.track_event.name = ev["name"]
            for fid in outgoing[ev["idx"]]:
                pkt.track_event.flow_ids.append(fid)
            for fid in incoming[ev["idx"]]:
                pkt.track_event.terminating_flow_ids.append(fid)
        else:
            pkt.track_event.type = TrackEvent.TYPE_SLICE_END
            for fid in incoming_at_end[ev["idx"]]:
                pkt.track_event.terminating_flow_ids.append(fid)
        pkt.trusted_packet_sequence_id = SEQ

    out_path = TRACES_DIR / f"trace_llm_fa_pp{PP}_q{q}_sk{sk}.pftrace"
    with open(out_path, "wb") as f:
        f.write(builder.serialize())

    n_x    = len(events)
    n_flow = sum(len(v) for v in outgoing.values())
    span_us = max(e["e_ns"] for e in events) / 1000.0
    print(f"wrote {out_path}  ({n_x} slices, {n_flow} flow arrows, span {span_us:.1f} µs)")

    # Per-event totals
    import re
    iter_suffix = re.compile(r"(_?\d+)$")
    by_event = {}
    for ev in events:
        name = ev["name"]
        if ev["k"] is not None:
            name = iter_suffix.sub("", name)
        cnt, total = by_event.get(name, (0, 0.0))
        by_event[name] = (cnt + 1, total + (ev["e_ns"] - ev["s_ns"]) / 1000.0)
    print("\nper-event totals (sorted by total µs descending):")
    for name, (cnt, total) in sorted(by_event.items(), key=lambda kv: -kv[1][1]):
        print(f"  {name:8s}  {cnt:5d}  avg={total/cnt:7.3f} µs   total={total:8.2f} µs")

    # Quick PP stagger summary: median start time of EX2 per SM pair, per iter.
    print("\nPP stagger summary (per-iter, first few iters):")
    print(f"  {'j':>3s}  {'SM0_EX2.beg':>12s}  {'SM1_EX2.beg':>12s}  {'Δ ns':>8s}")
    sm0_ex2 = sorted(
        [e for e in events if e["warp"] in (0,1,2,3) and e["evid"] == 15],
        key=lambda e: (e["k"] or 0, e["warp"])
    )
    sm1_ex2 = sorted(
        [e for e in events if e["warp"] in (4,5,6,7) and e["evid"] == 15],
        key=lambda e: (e["k"] or 0, e["warp"])
    )
    # Bucket by iter and take min start time across the 4 warps in the pair.
    from collections import defaultdict as dd
    sm0_by_j = dd(list); sm1_by_j = dd(list)
    for e in sm0_ex2: sm0_by_j[e["k"]].append(e["s_ns"])
    for e in sm1_ex2: sm1_by_j[e["k"]].append(e["s_ns"])
    js = sorted(set(sm0_by_j.keys()) & set(sm1_by_j.keys()))
    for j in js[:20]:
        s0 = min(sm0_by_j[j]); s1 = min(sm1_by_j[j])
        print(f"  {j:3d}  {s0:12d}  {s1:12d}  {s1-s0:8d}")
    if js:
        deltas = [min(sm1_by_j[j]) - min(sm0_by_j[j]) for j in js]
        print(f"  median Δ (SM1 - SM0): {sorted(deltas)[len(deltas)//2]} ns  "
              f"(n={len(deltas)}; positive → SM1 starts AFTER SM0)")


if __name__ == "__main__":
    main()
