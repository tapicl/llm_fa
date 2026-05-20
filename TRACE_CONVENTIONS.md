# Perfetto Trace Conventions

All future kernel traces (redist_v8, v9, ...) MUST follow these conventions.
Reference implementation: `trace_dump.py`.

## Goal

A `.pftrace` (Perfetto protobuf) file the user can drag into
[ui.perfetto.dev](https://ui.perfetto.dev) and walk the kernel's dependency
chain by clicking slices. The "Preceding flows" panel must show the upstream
producers in dependency order, single-hop per click.

Target chain (must work for every iteration `j`):

```
P<p>V_j   ──preceding──▶  {Wco<p>_j, Ex<p>_j}
Wco<p>_j  ──preceding──▶  {Rm<p>_j}
Rm<p>_j ──preceding──▶  {Q<p>K_j}
Q<p>K_j ──preceding──▶  {Wk_j, ...}
```

## Compact Event Naming

Use these short names. Iteration index `j` is appended; pair index `p ∈ {0,1}`
identifies the M-pair (warps 0–3 vs 4–7 for a 2mpair kernel).

| evid | role          | name             | meaning |
|-----:|---------------|------------------|---------|
| 0    | LOAD          | `TQ`             | TMA Q load (once per CTA) |
| 1    | LOAD          | `Wke<j>K` / `Wke<j>V` | wait_kv_empty (slot reuse) — split K/V stage |
| 2    | LOAD          | `Tk<j>`          | TMA K load |
| 3    | LOAD          | `Tv<j>`          | TMA V load |
| 4    | MMA           | `WQ`             | MMA wait_Q (once) |
| 5    | MMA           | `Wk<j>`          | MMA wait_K |
| 6    | MMA           | `Wv<j>`          | MMA wait_V |
| 7    | MMA           | `Q0K<j>`         | UTCHMMA QK^T pair-0 |
| 8    | MMA           | `Q1K<j>`         | UTCHMMA QK^T pair-1 |
| 9    | MMA           | `P0V<j>`         | UTCHMMA PV pair-0 |
| 10   | MMA           | `P1V<j>`         | UTCHMMA PV pair-1 |
| 11   | MMA           | `WPr0<j>`        | wait_p_ready (s_p_ready_0) |
| 12   | MMA           | `WPr1<j>`        | wait_p_ready (s_p_ready_1) |
| 13   | SM            | `WS<p>_<j>`      | SM wait_S (S tile from QK arrived) |
| 14   | SM            | `Rm<p>_<j>`      | SM softmax phase 1: LD S → rowmax/FMNMX3 → acc_scale ex2 + tau → write sScale → **EARLY arrive sm_stats** (COR trigger) |
| 15   | SM            | `Ex<p>_<j>`      | SM softmax phase 2: 128× ex2.approx + ffma + cvt to bf16x2 + accumulate l_new — **the actual exp math, runs in parallel with COR** |
| 19   | SM            | `StP<p>_<j>`     | SM softmax phase 3: tcgen05.st write P to TMEM → arrive p_lastsplit + s_p_o_empty |
| 16   | COR           | `Wco0_<j>`       | COR **pair-0** region (wait sm_stats(0) → rescale O for pair 0 → fire). **Shrunk to 80 ns marker** at END for flow rendering. |
| 18   | COR           | `Wco1_<j>`       | COR **pair-1** region. **Shrunk to 80 ns marker** at END. |
| —    | COR (synth.)  | `Cor<p>_<j>`     | Pure-visualization slice spanning the original (un-shrunk) COR span so the user can see COR/Ex overlap. No flow attachments — emitted by the dumper, not the kernel. |
| 17   | EPI           | `EW`             | epilogue wait |

**Wke split rule** (TMA empty wait): the `wait_empty` mbar fires twice per iter
(once for K release, once for V release). When the kernel records it as a
single evid=1 event, the dumper labels alternating instances `Wke<i>K` and
`Wke<i>V` where `i = (counter // 2) + N_INFLIGHT` (N_INFLIGHT = KV_STAGE / 2
in the FA4-precise topology, typically 3).

**Track names** (`WARP_ROLE` map):
- `SM<p>_w<i>` for SM warpgroup pair `p`, warp `i` (0–3 or 4–7).
- `COR_w<i>` for the COR warpgroup (4 warps).
- `MMA`, `EPI`, `LOAD`, `EMPTY` for the dispatcher warps.

## Hard Rules for the Dumper

### Rule 1 — Wait-marker shrinking

**For every wait-type event, emit a tiny 80 ns marker at its END:**

```python
WAIT_EVIDS = {1, 4, 5, 6, 11, 12, 13, 16, 17, 18}  # Wke, WQ, Wk, Wv, WPr0/1, WS, Wco0, EW, Wco1
WAIT_MARKER_NS = 80
if evid in WAIT_EVIDS:
    s_ns = max(s_ns, e_ns - WAIT_MARKER_NS)
```

Why: Perfetto renders flow arrows from the slice with the **earlier BEGIN** to
the slice with the **later BEGIN**, regardless of which side carries
`flow_ids` vs `terminating_flow_ids`. Long wait bars BEGIN before their
producers, so the producer ends up in the wait's *Following* panel instead
of *Preceding*. Shrinking the wait so its BEGIN ≈ END (the moment of the
producer's signal) puts the producer earlier and makes the chain render
in the user's mental dependency direction.

What's lost: the visual stall length of a wait. END timestamps are unchanged,
so producer→consumer relative timing is fully accurate; only the duration of
camping-on-mbarrier is no longer visible as a long bar.

### Rule 2 — Encode flow edges for the user's mental chain

The kernel's actual mbarrier topology has many more edges than the user
wants to navigate. Encode **only** the edges the user clicks through, plus
the natural pipeline edges. Required edges per iter `j`:

| edge                                   | producer → consumer         |
|----------------------------------------|------------------------------|
| TMA → MMA wait_K/V                     | `Tk<j> → Wk<j>`, `Tv<j> → Wv<j>` |
| MMA QK → SM wait_S                     | `Q<p>K<j> → WS<p>_<j>` (warp 0 of pair) |
| SM rowmax → COR wait (per-pair)        | `Rm<p>_<j> → Wco<p>_<j>` for **all 4** SM warps in the pair |
| SM TMEM-store-P → MMA wait_PR          | `StP<p>_<j> → WPr<p>_<j>` (warp 0 of pair) — p_lastsplit_full fires at end of StP |
| MMA PV → TMA wait_empty (lagged)       | `P<p>V<j> → Wke<j+LAG>V` (LAG = N_INFLIGHT) |
| COR → MMA wait_PR (per-pair)           | `Wco<p>_<j> → WPr<p>_<j>` |
| **User-chain shortcut edges**          | (added on top so the panel walks 1-hop) |
| COR → P*V (per-pair)                   | `Wco<p>_<j> → P<p>V<j>` |
| SM exp2 → P*V                          | `Ex<p>_<j> → P<p>V<j>` for **all 4** warps |
| MMA QK → SM rowmax                     | `Q<p>K<j> → Rm<p>_<j>` for **all 4** warps |

**All-warp fan-in for shared-name slices.** SM/COR slices that share a name
across warps (e.g., 4 separate `Rm0_<j>` slices on tracks `SM0_w0..w3`) must
have the cross-warp incoming edge encoded to **all** of them. Otherwise
clicking "Rm0_<j>" in the Preceding-flows panel can land on a different
warp's slice that lacks the upstream edge, and the chain breaks. The
user-chain shortcut edges (Q*K → Rm, Ex → P*V, Rm → Wco) all need this
fan-in.

### Rule 3 — END-anchored terminating_flow_ids for inverted-time edges

Perfetto's `trace_processor` matches `flow_ids` and `terminating_flow_ids`
in **packet stream order** (not timestamp order). Putting `terminating_flow_ids`
on a consumer's BEGIN packet drops the flow if the consumer's BEGIN comes
before the producer's BEGIN in the packet stream.

After Rule 1 (wait shrinking) most edges are no longer inverted. For any
that remain:

```python
incoming_at_end = defaultdict(list)
for ev in events:
    kept = []
    for fid in incoming[ev["idx"]]:
        src_evs = [e for e in events if fid in outgoing[e["idx"]]]
        if any(s["s_ns"] >= ev["s_ns"] for s in src_evs):
            incoming_at_end[ev["idx"]].append(fid)
        else:
            kept.append(fid)
    incoming[ev["idx"]] = kept
```

In the writing loop, attach `incoming_at_end[ev]` to the consumer's
`SLICE_END` packet (no name) — Perfetto associates it with the closing
slice without producing phantom slices.

### Rule 4 — Single trusted_packet_sequence_id

Use one `trusted_packet_sequence_id` for the whole trace. Per-warp
sequences regress to ~80 flows because cross-sequence flow matching still
requires global packet-stream order; per-warp adds nothing and removes the
incremental-state-cleared anchoring.

```python
SEQ = 1
# First packet sets sequence_flags = 1 (SEQ_INCREMENTAL_STATE_CLEARED)
# and first_packet_on_sequence = True.
```

### Rule 5 — Packet stream order

Emit packets in **timestamp-sorted** order, with BEGIN < END at the same
timestamp:

```python
def sort_key(t):
    order = {"B": 0, "E": 1}
    return (t[0], order[t[1]])
timeline.sort(key=sort_key)
```

Out-of-order timestamps cause Perfetto to silently drop packets.

## Verification Checklist

After regenerating, run these queries against the trace via `TraceProcessor`.
**All must pass**:

1. **Flow count** — should match the encoded edges.

   ```python
   r = tp.query("SELECT COUNT(*) FROM flow")
   ```

2. **No phantom / empty-name slices**.

   ```python
   tp.query("SELECT COUNT(*) FROM slice WHERE name IS NULL OR name = ''")
   # → 0
   ```

3. **Click-chain works for every iter** — pick `j ∈ {0, mid, n_kv-1}`:

   ```python
   def preceding(name):
       r = tp.query(f"""
         SELECT s_src.name AS src
         FROM flow f
         JOIN slice s_src ON s_src.id = f.slice_out
         JOIN slice s_dst ON s_dst.id = f.slice_in
         WHERE s_dst.name = '{name}'
       """)
       return sorted({row.src for row in r})
   ```

   Required:
   - `preceding(f"P0V{j}")` ⊇ `{f"Wco0_{j}", f"Ex0_{j}"}`
   - `preceding(f"P1V{j}")` ⊇ `{f"Wco1_{j}", f"Ex1_{j}"}`
   - `preceding(f"Wco0_{j}")` ⊇ `{f"Rm0_{j}"}`
   - `preceding(f"Wco1_{j}")` ⊇ `{f"Rm1_{j}"}`
   - `preceding(f"Rm<p>_{j}")` ⊇ `{f"Q<p>K{j}"}` for `p ∈ {0, 1}`

4. **trace_chain.py walks the chain end-to-end**:

   ```bash
   python trace_chain.py P0V<j> --depth 6
   # Should reach Q0K<j> on the [F] flow path (not just [T] track-temporal).
   ```

## How to Regenerate

```bash
PATH=$PWD/.venv/bin:$PATH .venv/bin/python trace_dump.py 1024 2048
# Output: traces/trace_redist_v8d_pp1_q1024_sk2048.pftrace
```

For a new kernel `redist_vN`:
1. Copy `trace_dump.py` to `trace_dump_vN_pb.py`.
2. Update `EVENT_NAMES`, `WARP_ROLE`, `compact_label`, and the kernel module
   import to match the new kernel's profiler buffer layout.
3. Update the encoded-edges section (n_kv loop) for any new pipeline edges.
4. Keep Rules 1–5 above byte-identical.
5. Run the verification checklist before declaring success.

## How to Walk the Chain

Two ways:

- **Perfetto UI**: drag the .pftrace into ui.perfetto.dev → click any
  `P<p>V<j>` slice → "Preceding flows" panel → click `Wco<p>_<j>` → click
  `Rm<p>_<j>` → click `Q<p>K<j>`. End of chain.

- **Programmatic** (full transitive closure):

  ```bash
  python trace_chain.py P0V7 --depth 6
  python trace_chain.py P0V7 --forward    # walk descendants
  ```

  `[F]` = encoded cross-warp flow edge, `[T]` = same-track temporal predecessor.

## Frequency Conversion

Profiler counters are GPU clock cycles. The dumper converts to ns at boost:

```python
GHZ = 1.965
NS_PER_CYCLE = 1.0 / GHZ  # ≈ 0.509 ns/cycle
```

Update if the GPU clock policy changes.
