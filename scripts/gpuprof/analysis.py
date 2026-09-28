"""Decision-layer analysis: kernel lint, floors, levers, residency, boundaries.

Everything here is computed from data the capture already collects — launch
configs, transfer sizes, per-stage idle splits — and turns accounting
("where time went") into decisions ("what is recoverable, what to fix").
"""

from util import fmt_ms, short_kernel_name

# RTX 4090 (sm_89) static limits for theoretical-occupancy math.
SM89 = {
    "sms": 128,
    "warp": 32,
    "threads_per_sm": 1536,
    "regs_per_sm": 65536,
    "smem_per_sm": 102400,
    "blocks_per_sm": 24,
}

# Conservative per-launch driver overhead used for the launch floor.
LAUNCH_OVERHEAD_US = 4.0


# ── kernel lint ──────────────────────────────────────────────────────────────


def kernel_configs(kernels):
    agg = {}
    for k in kernels:
        grid = tuple(v or 1 for v in (k.get("grid") or [1, 1, 1]))
        block = tuple(v or 1 for v in (k.get("block") or [1, 1, 1]))
        key = (
            k["name"], grid, block,
            k.get("registers_per_thread") or 0,
            k.get("local_memory_per_thread") or 0,
            (k.get("static_shared_memory") or 0) + (k.get("dynamic_shared_memory") or 0),
        )
        a = agg.setdefault(key, {"count": 0, "ms": 0.0})
        a["count"] += 1
        a["ms"] += (k["end"] - k["start"]) / 1e6
    return agg


def theoretical_occupancy(threads, regs, smem):
    if threads <= 0:
        return 1.0, "none"
    limits = {
        "threads": SM89["threads_per_sm"] // threads,
        "blocks": SM89["blocks_per_sm"],
    }
    if regs:
        limits["registers"] = SM89["regs_per_sm"] // max(regs * threads, 1)
    if smem:
        limits["shared_memory"] = SM89["smem_per_sm"] // smem
    limiter = min(limits, key=limits.get)
    occ = min(1.0, max(limits[limiter], 0) * threads / SM89["threads_per_sm"])
    return occ, (limiter if occ < 1.0 else "none")


def lint_kernels(kernels):
    """Static red flags from launch configs alone (no counters needed)."""
    findings = []
    for key, a in sorted(kernel_configs(kernels).items(), key=lambda kv: -kv[1]["ms"]):
        name, grid, block, regs, local_mem, smem = key
        threads = block[0] * block[1] * block[2]
        blocks = grid[0] * grid[1] * grid[2]
        base = {
            "kernel": short_kernel_name(name),
            "grid": list(grid),
            "block": list(block),
            "registers": regs,
            "launches": a["count"],
            "total_ms": round(a["ms"], 2),
        }
        if local_mem > 0:
            findings.append({**base, "severity": "high", "rule": "register-spill",
                             "detail": f"{local_mem} B/thread spilled to local memory"})
        if threads % SM89["warp"] != 0:
            findings.append({**base, "severity": "warn", "rule": "block-not-warp-multiple",
                             "detail": f"block of {threads} threads is not a multiple of {SM89['warp']}"})
        occ, limiter = theoretical_occupancy(threads, regs, smem)
        if occ < 0.5 and a["ms"] >= 0.5:
            findings.append({**base, "severity": "warn", "rule": "low-occupancy",
                             "detail": f"theoretical occupancy {occ:.0%}, limited by {limiter}"})
        if a["ms"] >= 0.5 and blocks * threads < SM89["sms"] * SM89["threads_per_sm"] // 2:
            findings.append({**base, "severity": "warn", "rule": "grid-underfill",
                             "detail": f"{blocks} blocks x {threads} threads cannot fill {SM89['sms']} SMs"})
    return findings


def print_lint(findings, limit=12):
    print("KERNEL LINT (static launch-config checks; sm_89 limits)")
    print("-------------------------------------------------------")
    if not findings:
        print("clean: no spills, occupancy caps, or underfilled grids detected")
        print()
        return
    for f in findings[:limit]:
        print(f"[{f['severity']:>4}] {f['rule']:<24} {f['kernel']:<34} "
              f"{f['total_ms']:>8.1f}ms/{f['launches']:<4} {f['detail']}")
    if len(findings) > limit:
        print(f"... {len(findings) - limit} more findings (full list in the JSON 'lint' field)")
    print()


# ── floors + Amdahl levers ───────────────────────────────────────────────────


def measured_bandwidth_mb_per_ms(memcpys, kind):
    """Achievable link bandwidth from this run's own >=1MB copies (p90)."""
    rates = sorted(
        c["bytes"] / max(c["end"] - c["start"], 1)  # bytes/ns == GB/s == MB/ms
        for c in memcpys
        if c["copy_kind"] == kind and c["bytes"] >= (1 << 20)
    )
    if not rates:
        return 20.0
    return rates[max(0, int(len(rates) * 0.9) - 1)]


def build_levers(stages, online_cuda_ms, memcpys):
    """Rank leaf stages by recoverable ms against a physics floor.

    floor = kernel busy (treated as the floor until ncu counters are
    unlocked) + transfers at this run's achievable bandwidth + launch
    overhead. recoverable = wall - floor; the Amdahl projection assumes the
    GPU chain is serial, which holds until whole-fold command streams land.
    """
    bw = {kind: measured_bandwidth_mb_per_ms(memcpys, kind) for kind in (1, 2, 3)}
    leaves = [
        label for label in stages
        if not any(other != label and other.startswith(label + ".") for other in stages)
    ]
    # Synthetic windows duplicate the sub-stages they span; ranking them
    # would double-count the last fold.
    synthetic = {"finalize.terminal_fold"}
    levers = []
    for label in leaves:
        if label in synthetic or not (label.startswith("fold") or label.startswith("finalize")):
            continue
        t = stages[label]
        wall = t.get("wall_gpu", 0.0)
        if wall < 1.0:
            continue
        busy = t.get("gpu_ms", 0.0)
        xfer_floor = (
            t.get("h2d_mb", 0.0) / bw[1]
            + t.get("d2h_mb", 0.0) / bw[2]
            + t.get("dtod_mb", 0.0) / bw[3]
        )
        launch_floor = t.get("launches", 0.0) * LAUNCH_OVERHEAD_US / 1e3
        floor = busy + xfer_floor + launch_floor
        recoverable = max(0.0, wall - floor)
        lever = {
            "stage": label,
            "wall_ms": wall,
            "busy_ms": busy,
            "xfer_floor_ms": xfer_floor,
            "launch_floor_ms": launch_floor,
            "floor_ms": floor,
            "recoverable_ms": recoverable,
            "idle_sync_ms": t.get("idle_sync_ms", 0.0),
            "idle_api_ms": t.get("idle_api_ms", 0.0),
            "idle_host_ms": t.get("idle_host_ms", 0.0),
        }
        if online_cuda_ms:
            lever["projected_online_ms"] = max(0.0, online_cuda_ms - recoverable)
            lever["online_gain_pct"] = 100.0 * recoverable / online_cuda_ms
        levers.append(lever)
    return sorted(levers, key=lambda x: -x["recoverable_ms"])


def print_levers(levers, online_cuda_ms, limit=8):
    print("TOP LEVERS (leaf stages ranked by recoverable ms vs physics floor)")
    print("------------------------------------------------------------------")
    if not levers:
        print("no fold/finalize leaf stages above 1ms")
        print()
        return
    hdr = (
        f"{'stage':<44}{'wall':>8}{'floor':>8}{'recov':>8}"
        f"{'cause s/a/h':>14}{'-> online':>11}{'gain':>7}"
    )
    print(hdr)
    print("-" * len(hdr))
    for lv in levers[:limit]:
        name = lv["stage"].split("superneo.")[-1]
        cause = (f"{lv['idle_sync_ms']:.0f}/{lv['idle_api_ms']:.0f}"
                 f"/{lv['idle_host_ms']:.0f}")
        projected = fmt_ms(lv.get("projected_online_ms", 0.0)) if online_cuda_ms else "."
        gain = f"-{lv.get('online_gain_pct', 0.0):.1f}%" if online_cuda_ms else "."
        print(
            f"{name:<44}{fmt_ms(lv['wall_ms']):>8}{fmt_ms(lv['floor_ms']):>8}"
            f"{fmt_ms(lv['recoverable_ms']):>8}{cause:>14}{projected:>11}{gain:>7}"
        )
    print("-" * len(hdr))
    print("floor = kernel busy (pending ncu counters) + transfers at this run's")
    print("achievable bandwidth + 4us/launch; '-> online' projects the online prove")
    print("wall if the stage hit its floor (serial-chain Amdahl approximation).")
    print()


# ── residency gate ───────────────────────────────────────────────────────────

# The CURRENT architecture contract, per stage: (h2d_mb_max, d2h_mb_max);
# None = unconstrained today. Tighten these as slices land (e.g. sumcheck
# d2h -> ~0 after device-driven FS; oracle d2h -> 0 after resident F tables).
RESIDENCY_BUDGETS = {
    "session.structure": (None, 1.0),
    "fold.ingest.fresh": (None, 1.0),
    "fold.ingest.running": (1.0, 1.0),
    "fold.commit.fresh": (None, 1.0),
    # Device-driven FE/NC sumcheck may export small terminal proof/debug logs,
    # but it must not reintroduce per-round or table-sized D2H traffic.
    "fold.superneo.pi_ccs.sumcheck": (None, 0.25),
    "fold.superneo.pi_rlc.mix_witness": (None, 1.0),
}


def check_residency(stages):
    results, failures = [], []
    for stage, (h2d_max, d2h_max) in RESIDENCY_BUDGETS.items():
        s = stages.get(stage, {})
        for metric, cap in (("h2d_mb", h2d_max), ("d2h_mb", d2h_max)):
            if cap is None:
                continue
            measured = s.get(metric, 0.0)
            ok = measured <= cap
            results.append({
                "stage": stage, "metric": metric,
                "measured_mb": round(measured, 3), "budget_mb": cap, "ok": ok,
            })
            if not ok:
                failures.append(f"{stage} {metric} {measured:.1f}MB > budget {cap}MB")
    return results, failures


def structural_causes(a, b):
    """Explain a stage's wall change via structure: counts, sizes, kernels.

    Returns human-readable cause strings for facts that changed between run
    `a` (baseline) and run `b` (candidate) — launch counts, copy counts/MB,
    memsets, syncs, average kernel size, and per-kernel-name ms deltas.
    """
    causes = []

    def count_delta(key):
        av, bv = a.get(key, 0.0), b.get(key, 0.0)
        if abs(bv - av) > 0.5:
            causes.append(f"{key} {av:.0f} -> {bv:.0f}")

    def mb_delta(key):
        av, bv = a.get(key, 0.0), b.get(key, 0.0)
        if abs(bv - av) > 1.0:
            causes.append(f"{key} {av:.1f} -> {bv:.1f}")

    for key in ("launches", "h2d_copies", "d2h_copies", "dtod_copies", "memset_count", "syncs"):
        count_delta(key)
    for key in ("h2d_mb", "d2h_mb", "dtod_mb", "memset_mb"):
        mb_delta(key)

    def avg_kernel_us(row):
        launches = row.get("launches", 0.0)
        return 1000.0 * row.get("gpu_ms", 0.0) / launches if launches else 0.0

    if (a.get("launches") or b.get("launches")) and abs(avg_kernel_us(b) - avg_kernel_us(a)) > 20.0:
        causes.append(f"avg kernel {avg_kernel_us(a):.0f}us -> {avg_kernel_us(b):.0f}us")

    ak = a.get("kernels") or {}
    bk = b.get("kernels") or {}
    changed = sorted(set(ak) | set(bk), key=lambda n: -abs(bk.get(n, 0.0) - ak.get(n, 0.0)))
    for name in changed[:4]:
        av, bv = ak.get(name, 0.0), bk.get(name, 0.0)
        if abs(bv - av) > 0.5:
            tag = " (new)" if av == 0.0 else (" (gone)" if bv == 0.0 else "")
            causes.append(f"kernel {short_kernel_name(name)} {av:.1f} -> {bv:.1f}ms{tag}")
    return causes


def print_residency(results, failures):
    print("RESIDENCY GATE (measured transfers vs architecture budgets)")
    print("-----------------------------------------------------------")
    for r in results:
        mark = "ok" if r["ok"] else "FAIL"
        print(f"[{mark:>4}] {r['stage']:<44} {r['metric']:<8} "
              f"{r['measured_mb']:>8.2f}MB (budget {r['budget_mb']}MB)")
    verdict = "VIOLATED" if failures else "clean"
    print(f"residency: {len(results)} budgets checked, {len(failures)} violations — {verdict}")
    print()


# ── protocol-boundary scorecard ──────────────────────────────────────────────

BOUNDARY_STAGES = [
    ("fold.ingest", "input planes enter device"),
    ("fold.commit.fresh", "fresh Ajtai commitments"),
    ("fold.superneo.pi_ccs.oracle", "Pi_CCS oracle data"),
    ("fold.superneo.pi_ccs.sumcheck.fe", "FE row/tail sumcheck"),
    ("fold.superneo.pi_ccs.sumcheck.nc", "NC column/tail sumcheck"),
    ("fold.superneo.pi_ccs.output.y_prime", "Pi_CCS output surfaces"),
    ("fold.superneo.pi_rlc.combine_claims", "Pi_RLC claim/rho host shell"),
    ("fold.superneo.pi_rlc.mix_witness", "Pi_RLC mixed witness"),
    ("fold.superneo.pi_rlc.output.k_surfaces", "Pi_RLC K surfaces"),
    ("fold.superneo.pi_dec.open_children", "Pi_DEC child openings"),
    ("fold.superneo.pi_dec.emit.planes", "terminal witness export"),
    ("fold.superneo.pi_dec.emit.assemble", "host proof assembly"),
    ("fold.egress.export", "final proof/public export"),
]


def _cpu_owned_ms(stage):
    """Host work not explained by CUDA activity for a stage row."""
    wall = stage.get("wall_gpu", 0.0)
    device_work = (
        stage.get("gpu_ms", 0.0)
        + stage.get("api_ms", 0.0)
        + stage.get("sync_idle_ms", 0.0)
    )
    transfer_or_launch = (
        stage.get("launches", 0.0)
        + stage.get("h2d_copies", 0.0)
        + stage.get("d2h_copies", 0.0)
        + stage.get("dtod_copies", 0.0)
        + stage.get("memset_count", 0.0)
    )
    pure_host = wall if wall >= 0.05 and device_work == 0.0 and transfer_or_launch == 0.0 else 0.0
    return max(stage.get("idle_host_ms", 0.0), pure_host)


def build_boundary_scorecard(stages):
    rows = []
    for name, ownership in BOUNDARY_STAGES:
        s = stages.get(name, {})
        if not s:
            continue
        h2d_mb = s.get("h2d_mb", 0.0)
        d2h_mb = s.get("d2h_mb", 0.0)
        h2d_copies = s.get("h2d_copies", 0.0)
        d2h_copies = s.get("d2h_copies", 0.0)
        syncs = s.get("syncs", 0.0)
        cpu_owned_ms = _cpu_owned_ms(s)
        host_join_count = h2d_copies + d2h_copies + syncs
        rows.append({
            "stage": name,
            "ownership": ownership,
            "wall_ms": s.get("wall_gpu", 0.0),
            "gpu_busy_ms": s.get("gpu_ms", 0.0),
            "cpu_owned_ms": cpu_owned_ms,
            "host_join_count": host_join_count,
            "h2d_mb": h2d_mb,
            "d2h_mb": d2h_mb,
            "h2d_copies": h2d_copies,
            "d2h_copies": d2h_copies,
            "syncs": syncs,
            "launches": s.get("launches", 0.0),
        })
    return rows


def print_boundary_scorecard(rows, limit=14):
    print("BOUNDARY SCORECARD (CPU/GPU crossings still in the prover path)")
    print("----------------------------------------------------------------")
    if not rows:
        print("no boundary stages found")
        print()
        return
    hdr = (
        f"{'stage':<40}{'cpu ms':>8}{'H2D':>9}{'D2H':>9}"
        f"{'joins':>8}{'launch':>8}  owner"
    )
    print(hdr)
    print("-" * len(hdr))
    ranked = sorted(
        rows,
        key=lambda r: (
            -(r["cpu_owned_ms"] + 0.25 * (r["h2d_mb"] + r["d2h_mb"])),
            -r["host_join_count"],
        ),
    )
    for row in ranked[:limit]:
        short = row["stage"].removeprefix("fold.superneo.")
        print(
            f"{short:<40}"
            f"{fmt_ms(row['cpu_owned_ms']):>8}"
            f"{row['h2d_mb']:>8.1f}M"
            f"{row['d2h_mb']:>8.1f}M"
            f"{row['host_join_count']:>8.0f}"
            f"{row['launches']:>8.0f}  "
            f"{row['ownership']}"
        )
    print("-" * len(hdr))
    print("cpu ms = host-owned wall not explained by CUDA activity; joins = H2D + D2H + sync count.")
    print("Rows rank migration pressure, not performance findings.")
    print()


# ── SuperNeo protocol context ────────────────────────────────────────────────

SUPERN_STAGE_CONTEXT = [
    {
        "stage": "fold.ingest",
        "phase": "ingest",
        "role": "move fresh planes onto the device and retain running planes",
        "target_owner": "GPU after fresh upload",
        "transfer_contract": "fresh_input_h2d_allowed",
        "parallelism_model": "fresh H2D can overlap only with independent work from another fold or chain",
        "first_principles_question": "is this data new, or can it stay device-resident from the previous fold?",
    },
    {
        "stage": "fold.commit.fresh",
        "phase": "commit",
        "role": "Ajtai Commit(z_i) for fresh CCS instances",
        "target_owner": "GPU",
        "transfer_contract": "fresh_input_h2d_allowed",
        "parallelism_model": "matrix-vector work across rows/commitments; fresh H2D remains unavoidable",
        "first_principles_question": "are we copying only genuinely new witnesses, and are commits batched?",
    },
    {
        "stage": "fold.superneo.pi_ccs.oracle",
        "phase": "pi_ccs.oracle",
        "role": "build F/Eval/NC helper tables for Pi_CCS",
        "target_owner": "GPU",
        "transfer_contract": "device_resident_required",
        "parallelism_model": "F, Eval, NC, and tensor/eq prep are independent after public challenges",
        "first_principles_question": "are independent oracle tables enqueued as a DAG or serialized by CPU call order?",
    },
    {
        "stage": "fold.superneo.pi_ccs.sumcheck.fe",
        "phase": "pi_ccs.sumcheck.fe",
        "role": "FE row rounds plus proof-log ownership",
        "target_owner": "GPU with deferred proof carrier",
        "transfer_contract": "no_repeated_d2h",
        "parallelism_model": "rounds are Fiat-Shamir serial; fill cores by widening per-round work or overlapping independent work",
        "first_principles_question": "is row proof material consumed on device, or exported to satisfy a host proof object?",
    },
    {
        "stage": "fold.superneo.pi_ccs.sumcheck.nc",
        "phase": "pi_ccs.sumcheck.nc",
        "role": "NC digit/column rounds",
        "target_owner": "GPU",
        "transfer_contract": "device_resident_required",
        "parallelism_model": "rounds are transcript-ordered; digit/table prep can be independent before consumption",
        "first_principles_question": "is this near the kernel floor, or still gated by host joins?",
    },
    {
        "stage": "fold.superneo.pi_ccs.output",
        "phase": "pi_ccs.output",
        "role": "terminal Pi_CCS output surfaces and CE claims",
        "target_owner": "GPU surfaces, host materialization only at proof boundary",
        "transfer_contract": "device_intermediate",
        "parallelism_model": "surface mat-vecs are device work; claim object assembly should not gate repeated folds",
        "first_principles_question": "are outputs kept as device surfaces through RLC, or rebuilt as host claims?",
    },
    {
        "stage": "fold.superneo.pi_rlc.combine_claims",
        "phase": "pi_rlc.combine_claims",
        "role": "random linear combination claim shell algebra",
        "target_owner": "mostly device or boundary-only host shell",
        "transfer_contract": "host_shell_to_remove",
        "parallelism_model": "small algebra; avoid making it a per-fold host island",
        "first_principles_question": "does this stage need full host claims, or only a device-backed shell?",
    },
    {
        "stage": "fold.superneo.pi_rlc.mix_witness",
        "phase": "pi_rlc.mix_witness",
        "role": "Z_mix witness combination",
        "target_owner": "GPU",
        "transfer_contract": "device_resident_required",
        "parallelism_model": "parallel across witness words/lanes",
        "first_principles_question": "does Z_mix stay resident into DEC?",
    },
    {
        "stage": "fold.superneo.pi_dec.split",
        "phase": "pi_dec.split",
        "role": "split Z_mix into DEC children",
        "target_owner": "GPU",
        "transfer_contract": "device_schedule_required",
        "parallelism_model": "parallel across words/children, but compact active scheduling can force host joins",
        "first_principles_question": "is active-child metadata device-driven, or is the host joining to steer the schedule?",
    },
    {
        "stage": "fold.superneo.pi_dec.open_children",
        "phase": "pi_dec.open_children",
        "role": "child openings and commitments",
        "target_owner": "GPU",
        "transfer_contract": "device_fanout",
        "parallelism_model": "child openings/commits are fan-out candidates when they do not share a saturated mat-vec path",
        "first_principles_question": "can children or surfaces run on separate streams without contending on the same resident planes?",
    },
    {
        "stage": "fold.superneo.pi_dec.emit",
        "phase": "pi_dec.emit",
        "role": "materialize child claims/proof material",
        "target_owner": "deferred device-backed fold output",
        "transfer_contract": "proof_materialization_deferred",
        "parallelism_model": "not a core-fill problem; this is a proof-carrier boundary",
        "first_principles_question": "why does this fold need host CeClaim/NifsProof bytes before final/audit/verify?",
    },
    {
        "stage": "fold.accumulate",
        "phase": "accumulate",
        "role": "advance running state",
        "target_owner": "device-backed RunningInstance carrier",
        "transfer_contract": "device_running_carrier",
        "parallelism_model": "host state construction should not force device surfaces to leave the GPU",
        "first_principles_question": "can the next fold consume a running carrier instead of a host RunningInstance?",
    },
    {
        "stage": "fold.egress.export",
        "phase": "egress.export",
        "role": "proof/public export boundary",
        "target_owner": "final/audit-only host materialization",
        "transfer_contract": "final_or_audit_only",
        "parallelism_model": "egress is acceptable only at final proof/audit/verification boundaries",
        "first_principles_question": "is this real final egress, or a relocated per-fold export?",
    },
]

SUPERN_DAG_EDGES = [
    ("fold.ingest", "fold.commit.fresh", "data", False, "fresh planes must exist before fresh commitments"),
    ("fold.commit.fresh", "fold.superneo.pi_ccs.oracle", "data", False, "Pi_CCS sees committed fresh/running instances"),
    ("fold.superneo.pi_ccs.oracle", "fold.superneo.pi_ccs.sumcheck.fe", "data", True, "oracle F/Eval prep can expose independent table work"),
    ("fold.superneo.pi_ccs.sumcheck.fe", "fold.superneo.pi_ccs.sumcheck.nc", "fiat_shamir", False, "NC challenges follow the FE transcript spine"),
    ("fold.superneo.pi_ccs.sumcheck.nc", "fold.superneo.pi_ccs.output", "data", False, "Pi_CCS output needs finalized FE/NC state"),
    ("fold.superneo.pi_ccs.output", "fold.superneo.pi_rlc.combine_claims", "data", False, "RLC samples/mixes Pi_CCS outputs"),
    ("fold.superneo.pi_rlc.combine_claims", "fold.superneo.pi_rlc.mix_witness", "data", False, "mixing needs rho/claim shell"),
    ("fold.superneo.pi_rlc.mix_witness", "fold.superneo.pi_dec.split", "data", False, "DEC splits Z_mix"),
    ("fold.superneo.pi_dec.split", "fold.superneo.pi_dec.open_children", "data", True, "children/surfaces are fan-out candidates after split"),
    ("fold.superneo.pi_dec.open_children", "fold.superneo.pi_dec.emit", "proof_materialization", False, "host proof carrier currently forces materialization"),
    ("fold.superneo.pi_dec.emit", "fold.accumulate", "host_contract", False, "host RunningInstance/proof carrier is still the repeated-loop contract"),
    ("fold.accumulate", "fold.egress.export", "host_contract", False, "egress should be final/audit-only, not relocated per-fold export"),
]

OVERLAP_CANDIDATES = [
    {
        "candidate": "oracle.F + oracle.Eval + oracle.NC prep",
        "stages": [
            "fold.superneo.pi_ccs.oracle.F",
            "fold.superneo.pi_ccs.oracle.Eval",
            "fold.superneo.pi_ccs.oracle.NC",
        ],
        "legal": True,
        "reason": "after public challenges, table prep lanes are data-independent until FE/NC consumes them",
    },
    {
        "candidate": "DEC child openings/commit fan-out",
        "stages": [
            "fold.superneo.pi_dec.open_children.y_ring",
            "fold.superneo.pi_dec.open_children.y_zcol",
            "fold.superneo.pi_dec.commit_children",
        ],
        "legal": True,
        "reason": "children/surfaces can be independent, but prior stream splits lost when they contended on the same mat-vec path",
    },
    {
        "candidate": "independent chains on one CUDA context",
        "stages": ["fold"],
        "legal": True,
        "reason": "different proofs/chains have no Fiat-Shamir dependency and can fill underused SMs",
    },
]

BLOCKED_PARALLELISM = [
    {
        "candidate": "FE round i with FE round i+1",
        "stages": ["fold.superneo.pi_ccs.sumcheck.fe"],
        "legal": False,
        "reason": "round i+1 consumes the challenge sampled from round i coefficients",
    },
    {
        "candidate": "Y_eval / Ajtai tail overlapped with NC rounds",
        "stages": [
            "fold.superneo.pi_ccs.output.y_prime",
            "fold.superneo.pi_ccs.sumcheck.nc",
        ],
        "legal": False,
        "reason": "the measured y_prime label is the Ajtai Y_eval spine node; FE tail consumes it before NC transcript continuation",
    },
    {
        "candidate": "FE row proof-log export relocated to egress",
        "stages": [
            "fold.superneo.pi_ccs.sumcheck.fe.row_download",
            "fold.egress.export",
        ],
        "legal": False,
        "reason": "without a device proof consumer this only moves the D2H boundary later",
    },
]


def _stage_row(stages, name):
    return stages.get(name, {})


def _transfer_mb(row):
    return row.get("h2d_mb", 0.0) + row.get("d2h_mb", 0.0) + row.get("dtod_mb", 0.0)


def _transfer_contract(row, contract):
    h2d = row.get("h2d_mb", 0.0)
    d2h = row.get("d2h_mb", 0.0)
    dtod = row.get("dtod_mb", 0.0)
    allowed_h2d = h2d if contract == "fresh_input_h2d_allowed" else 0.0
    allowed_d2h = d2h if contract == "final_or_audit_only" else 0.0
    avoidable_h2d = max(0.0, h2d - allowed_h2d)
    avoidable_d2h = max(0.0, d2h - allowed_d2h)
    return {
        "contract": contract,
        "allowed_h2d_mb": allowed_h2d,
        "allowed_d2h_mb": allowed_d2h,
        "avoidable_h2d_mb": avoidable_h2d,
        "avoidable_d2h_mb": avoidable_d2h,
        "avoidable_transfer_mb": avoidable_h2d + avoidable_d2h,
        "device_local_transfer_mb": dtod,
    }


def _join_count(row):
    return (
        row.get("h2d_copies", 0.0)
        + row.get("d2h_copies", 0.0)
        + row.get("dtod_copies", 0.0)
        + row.get("syncs", 0.0)
    )


def _util_pct(row):
    wall = row.get("wall_gpu", 0.0)
    return 100.0 * row.get("gpu_ms", 0.0) / wall if wall > 0.0 else 0.0


def _kernel_rules_for_stage(row, lint_findings):
    kernels = {short_kernel_name(name) for name in (row.get("kernels") or {})}
    rules = []
    for finding in lint_findings:
        if finding.get("kernel") in kernels:
            rule = finding.get("rule")
            if rule and rule not in rules:
                rules.append(rule)
    return rules


def _stage_issue(row, lint_rules):
    if row.get("d2h_mb", 0.0) >= 1.0:
        return "device-to-host boundary"
    if row.get("h2d_mb", 0.0) >= 1.0:
        return "host-to-device input boundary"
    if _cpu_owned_ms(row) >= 2.0:
        return "host-owned work"
    if row.get("api_ms", 0.0) >= 2.0 and row.get("gpu_ms", 0.0) < 0.25 * max(row.get("wall_gpu", 0.0), 1e-9):
        return "CUDA API / host scheduling"
    if _join_count(row) >= 32 and row.get("wall_gpu", 0.0) >= 2.0:
        return "join-heavy schedule"
    if "grid-underfill" in lint_rules:
        return "underfilled CUDA work"
    if "low-occupancy" in lint_rules:
        return "register/occupancy limited"
    if row.get("launches", 0.0) >= 128:
        return "many tiny launches"
    if _util_pct(row) >= 80.0:
        return "near busy floor"
    return "low measured pressure"


def _dominant_issue(row, lint_rules, transfer):
    if transfer["avoidable_d2h_mb"] >= 1.0:
        return "avoidable D2H boundary"
    if transfer["avoidable_h2d_mb"] >= 1.0:
        return "avoidable H2D boundary"
    if transfer["allowed_h2d_mb"] >= 1.0:
        return "fresh H2D allowed"
    if transfer["allowed_d2h_mb"] >= 1.0:
        return "final D2H allowed"
    return _stage_issue(row, lint_rules)


def _migration_score(node):
    return (
        node["cpu_owned_ms"]
        + 0.35 * node["avoidable_transfer_mb"]
        + 0.02 * node["joins"]
        + (10.0 if node["dominant_issue"] == "avoidable D2H boundary" else 0.0)
    )


def build_superneo_context(stages, lint_findings):
    nodes = []
    for item in SUPERN_STAGE_CONTEXT:
        row = _stage_row(stages, item["stage"])
        lint_rules = _kernel_rules_for_stage(row, lint_findings)
        transfer = _transfer_contract(row, item["transfer_contract"])
        node = {
            **item,
            "wall_ms": row.get("wall_gpu", 0.0),
            "gpu_busy_ms": row.get("gpu_ms", 0.0),
            "gpu_util_pct": _util_pct(row),
            "cpu_owned_ms": _cpu_owned_ms(row),
            "transfer_mb": _transfer_mb(row),
            **transfer,
            "h2d_mb": row.get("h2d_mb", 0.0),
            "d2h_mb": row.get("d2h_mb", 0.0),
            "joins": _join_count(row),
            "launches": row.get("launches", 0.0),
            "kernel_lint_rules": lint_rules,
            "dominant_issue": _dominant_issue(row, lint_rules, transfer),
        }
        node["migration_score"] = _migration_score(node)
        nodes.append(node)
    edges = [
        {
            "from": src,
            "to": dst,
            "dependency": dep,
            "can_overlap": can_overlap,
            "reason": reason,
        }
        for src, dst, dep, can_overlap, reason in SUPERN_DAG_EDGES
    ]
    migration_queue = sorted(
        nodes,
        key=lambda n: (-n["migration_score"], n["stage"]),
    )
    return {
        "schema_version": 1,
        "rule": "schedule by SuperNeo dependency graph, not old CPU call order",
        "nodes": nodes,
        "edges": edges,
        "overlap_candidates": OVERLAP_CANDIDATES,
        "blocked_parallelism": BLOCKED_PARALLELISM,
        "migration_queue": migration_queue,
    }


def print_superneo_context(context, limit=10):
    print("SUPERNEO PROTOCOL CONTEXT")
    print("-------------------------")
    print("rule: do not serialize A -> B -> C -> D when the dependency graph has independent lanes")
    print("      optimize each stage for residency first, then legal CUDA parallelism.")
    print()
    hdr = (
        f"{'stage':<38}{'issue':<29}{'wall':>8}{'busy':>8}"
        f"{'avoid':>8}{'allow':>8}{'joins':>7}{'launch':>8}  next question"
    )
    print(hdr)
    print("-" * len(hdr))
    for node in context["migration_queue"][:limit]:
        short = node["stage"].removeprefix("fold.superneo.")
        print(
            f"{short:<38}{node['dominant_issue']:<29}"
            f"{fmt_ms(node['wall_ms']):>8}{fmt_ms(node['gpu_busy_ms']):>8}"
            f"{node['avoidable_transfer_mb']:>7.1f}M"
            f"{(node['allowed_h2d_mb'] + node['allowed_d2h_mb']):>7.1f}M"
            f"{node['joins']:>7.0f}"
            f"{node['launches']:>8.0f}  {node['first_principles_question']}"
        )
    print("-" * len(hdr))
    print("avoid = transfer that violates the stage contract; allow = expected fresh/final transfer.")
    print()
    print("LEGAL OVERLAP CANDIDATES")
    print("------------------------")
    for candidate in context["overlap_candidates"]:
        print(f"[ok] {candidate['candidate']}: {candidate['reason']}")
    print()
    print("BLOCKED / REJECTED PARALLELISM")
    print("------------------------------")
    for candidate in context["blocked_parallelism"]:
        print(f"[no] {candidate['candidate']}: {candidate['reason']}")
    print()
