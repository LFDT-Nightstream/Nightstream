# gpuscope — GPU diagnostics harness, spec v1

One-sentence spec: a reproducible GPU diagnostics harness where every
benchmark number is tied to a correctness artifact, every CUDA event is
mapped to protocol ownership, every lost millisecond has a cause bucket,
and every output is raw evidence, an exit-coded judgment, or a ranked
optimization lever.

Division of labor: NVIDIA tools (nsys, ncu, compute-sanitizer) acquire
facts. This harness interprets those facts against the SuperNeo protocol
taxonomy and turns them into decisions. It never re-implements acquisition.

The primary consumer is an AI optimization loop. Terminal tables are a
courtesy; the structured artifacts are the product.

## Value model — what this tool is and is not

Read this before extending the spec; it exists to keep scope honest.
gpuscope is one real profiler feature plus a lab notebook. It adds
exactly three pieces of measurement capability over the raw tools:

1. **Stage-scoped gap-cause attribution** (the flagship). Classifying
   each stage's non-busy time into host_gap / sync_wait / api /
   transfer buckets against the protocol taxonomy. No NVIDIA or
   open-source tool produces this, and it targets the live frontier:
   fold wall ~190ms vs kernel-busy ~110ms — the recoverable money is
   in gaps, not kernels. The reconciliation invariant is not
   bookkeeping; it is the coverage guarantee that gap attribution
   accounts for ALL time (what made the ~69ms replay barrier
   impossible to hide).
2. **Static kernel diagnostics** from the sqlite (registers, block
   shape, occupancy, grid underfill, spill) — real kernel diagnosis
   available today while ncu is permission-blocked.
3. **Enqueue-side attribution** via correlation IDs — the correctness
   property that keeps (1) true once async streams land.

Everything else — run identity, mode separation, check gates, parity
binding, ledgers — is structure, not profiling. That is deliberate and
justified empirically, not aesthetically: every structural rule traces
to a documented campaign failure where the facts were available and
judgment failed anyway:

- `tree_diff_hash` ← three weeks of uncommitted work sharing one
  git_head; history became unattributable.
- reconciliation-as-defect ← the ~69ms replay barrier hidden in
  unattributed time.
- mode separation + structural causes on every FAIL ← the cuMemFree
  316ms red herring (API-summary time read as critical path) and a
  false external 1.7GB-H2D claim.
- repeat medians + spread ← an iteration whose e2e numbers were
  noise-dominated (303–320ms band) and nearly drove a wrong call.
- versioned taxonomy + offline re-analysis ← AI sessions reset; ad-hoc
  SQL redefines "host_gap" per session and silently destroys trend
  comparability. Definition stability across sessions is the product
  in a longitudinal campaign.

The AI consumer makes the structure MORE necessary, not less: a
session can always improvise SQL, but exit-coded gates and closed
enums are what keep improvisation from anchoring on plausible-but-
wrong numbers.

Limits — no harness substitutes for facts that were never acquired:

- Kernel-internal truth needs ncu unlocked (one driver setting); until
  then counters mode honestly reports blocked. The mat_vec
  compute-bound verdict came from hand op-counting, not tooling.
- Host-side dark time needs Rust timers in the crate, not GPU tools.
- Stage-level residency budgets (stage × direction × MB caps, proven in
  gpuprof's `--assert-residency`) have a data source today and are the
  v1 gate. Only the per-buffer ledger needs crate-side buffer identity
  (a registry or NVTX copy annotations) that does not exist yet; until
  it does, per-buffer rows in buffers.json are schema without a data
  source.

Scope test for any new feature: it must be one of the three
measurement capabilities above, or structure traceable to a real,
named failure. Anything else — including features replaceable by a
convention in a markdown file — stays out.

## Goals

Answer, with evidence, per run:
- What ran? Where did time go? What moved between CPU/GPU?
- What blocked progress (critical path)? Which kernels are weak?
- Did correctness still hold? What changed vs the last accepted run?
- What is the next optimization, its cause, and its bounded payoff?

## Non-goals

- Not a benchmark-only script; not a replacement for nsys/ncu/sanitizer.
- No GUI, no custom trace viewer (perfetto/chrome export covers it).
- No prose-first findings: report.md is rendered from analysis JSONs and
  contains nothing that is not in them.
- No hardware abstraction layer: a device-profile table, first entry
  sm_89; new hardware = new entry, not new architecture.
- Parity/replay validation time is NEVER counted inside fast benchmark
  timing (see Correctness binding).
- No speculative analysis fields: a field enters `analysis/*.json` only
  if it is consumed by compare/check/trend, critical_path, levers, or the
  report renderer. Unused JSON is bloat with a schema.

## Prior art (evaluated, not adopted)

pytorch/kineto + HolisticTraceAnalysis: same acquisition source as nsys
(CUPTI) and a generic idle-breakdown analysis — overlaps only where we
are already covered, and its unique value (PyTorch op→kernel correlation)
has no counterpart here; our "ops" are the NVTX taxonomy. Not worth an
in-process C++ FFI dependency. Optional hook: a kineto-schema-compatible
trace export would let HTA serve as an independent second opinion on the
decomposition during differential validation.

zymtrace ("profile-guided AI optimization"): same thesis as this spec —
profiles as AI input — but at fleet altitude: eBPF sampling flamegraphs
of production services, exposed to AI assistants over an MCP server.
Sampling profiles cannot answer our timeline-causality questions (idle
attribution, transfer serialization), and MCP solves a data-distance
problem we don't have (our artifacts are local files). Lesson adopted:
their ~20x token reduction for AI-facing views — report.md is
token-budgeted per the artifact schema above.

## Naming & relationship to gpuprof

`gpuscope` is a NEW tool at `scripts/gpuscope/`, built in parallel while
`scripts/gpuprof/` remains the campaign's working instrument. Policy
during the build:

- gpuprof is feature-frozen: bugfixes only, no new capabilities. Its
  outputs are the oracle gpuscope validates against.
- The optimization loop keeps using gpuprof until cutover, then gpuscope
  only — never both as peers in one iteration.
- At cutover, gpuprof is DELETED (lean-codebase rule: temporary
  duplication is a bridge, not an end state).

Shared truth that makes the parallel build safe: the raw nsys sqlite.
gpuscope's `analyze` MUST run offline on gpuprof-era sqlite exports — that
is both its validation harness (no GPU time needed) and the history
bridge.

Module layout:

- `cli.py` — command parsing + orchestration
- `collect.py` — benchmark + external tool invocation, bundle assembly
- `parse.py` — nsys/ncu/sanitizer/stdout parsing
- `taxonomy.py` — the versioned taxonomy contract
- `analyze.py` — decomposition, floors, levers, lint, residency,
  critical path
- `compare.py` — diff / check / trend
- `render.py` — terminal tables, report.md, perfetto export

Behaviors ported from gpuprof, not reinvented: enqueue-side attribution
(correlation IDs — survives async command streams), the loud stale-binary
guard (perf-timers build freshness), schema-tolerant readers for old
JSONs.

## Measurement modes (mutually exclusive)

A collector changes what the run measures. One bundle = one mode.
NEVER mix: ncu replays and serializes kernels (its wall times are not
benchmarks); sanitizers instrument (same). Modes:

- `timing` — nsys + NVTX + repeat medians. The only mode whose wall
  numbers may enter history/trend/check.
- `counters` — ncu on selected kernels. Kernel-internal truth, no walls.
- `sanitize` — compute-sanitizer suites. Correctness of memory/races.
- `cpuprof` — host-side profile (perf, when available).

Bundles from different modes on the same tree link via metadata identity
(below), which is how "this kernel is bandwidth-bound AND costs 12ms/fold"
gets assembled without lying about either half.

## CLI surface

- `doctor` — environment probe: nsys/ncu present, ncu counter permission
  (detect ERR_NVGPUCTRPERM and print the one-line fix), NVTX lib
  dlopen-able, perf-timers binary fresh vs source mtimes, driver/CUDA
  versions, device profile known. Exit 1 on any blocker for the requested
  mode. Every failure states the next action.
- `run <gate> [--repeat N] [--assert-residency]` — timing mode bundle.
- `ncu <gate> [--kernel <name>]` — counters mode bundle.
- `sanitize <gate> [--tool ...]` — sanitizer mode bundle.
- `analyze <bundle-dir>` — (re)derive analysis/* from raw artifacts.
  Analysis MUST be re-runnable offline: improving the analyzer re-explains
  old runs without re-running the GPU.
- `compare <baseline-dir> <candidate-dir>` — exploratory structural diff.
- `check <candidate-dir> [--baseline <dir>]` — gate with exit codes;
  default baseline = last history entry with gate PASS + green parity link.
- `trend [--stage <label>]` — campaign trajectory from history.
- `history [--rejected]` — query run + experiment ledgers.

No `bundle` command: the run directory IS the self-contained bundle.

## Artifact schema (bundle layout, versioned)

`benchmark-results/gpu/<run-id>/` where run-id = `<utc-ts>-<slug>-<mode>`:

- `metadata.json` — identity + environment (schema below)
- `run.json` — online numbers, per-repeat values, median, spread
- `stdout.txt`, `stderr.txt`
- `nsys/report.sqlite` (+ `.nsys-rep`) — raw truth, always retained
- `trace/perfetto.json`
- `ncu/kernels.json` — real data or `{"status":"blocked","reason":...}`
- `sanitizer/*.log`
- `analysis/` — `stages.json`, `kernels.json`, `transfers.json`,
  `buffers.json`, `critical_path.json`, `levers.json`, `regression.json`
- `report.md` — deterministic rendering of analysis/*, nothing original.
  Token-budgeted for AI consumption: compact tabular text over raw JSON
  dumps, repeats collapsed, below-threshold rows culled only WITH an
  explicit dropped-row count (never silently)

`metadata.json` fields: `run_id`, `mode`, `schema_version`,
`taxonomy_version`, `git_head`, `tree_diff_hash`, `config_hash`, `bench`
(gate + args + repeat), `hw` (gpu name, sm arch, SM count, VRAM, driver,
CUDA), `tools` (nsys/ncu versions), `parity_run_id` (nullable),
timestamps.

`config_hash`: hash over benchmark args, cargo feature set, relevant
protocol params, device profile, and measurement-mode config. Two runs
compare/link only when both `tree_diff_hash` and `config_hash` match —
"same tree" alone can hide config drift.

`tree_diff_hash`: sha256 over `git diff` of tracked files plus content
hashes of `git ls-files -o --exclude-standard` untracked files. Motivated
by a real failure: three weeks of uncommitted work shared one git_head and
history became unattributable.

## Stage record — the decomposition contract

Every stage in `stages.json`:

```json
{
  "stage_id": "fold.superneo.pi_ccs.sumcheck.fe",
  "chain": "cuda",
  "wall_ms": 155.5,
  "gpu_busy_ms": 47.9,
  "sync_wait_ms": 0.2,
  "api_ms": 4.0,
  "host_gap_ms": 34.4,
  "transfer_wait_ms": 0.0,
  "unattributed_ms": 69.0,
  "launches": 552, "syncs": 32,
  "h2d_mb": 0.0, "d2h_mb": 0.0, "dtod_mb": 0.0,
  "top_kernels": ["fe_round_partials"],
  "source": {"host": ["crates/neo-prover-cuda/src/reduce/ccs/fe.rs"],
             "kernels": ["fe_round_partials"]},
  "instances": [{"fold": 0, "wall_ms": 31.1}]
}
```

`transfer_wait_ms` counts only copy time that serialized progress on the
critical path (compute waited on the copy). Total copy durations and
bytes live in the `h2d_*`/`d2h_*`/`dtod_*` fields — a perfectly
overlapped transfer contributes bytes but zero wait.

Invariant: `wall = gpu_busy + sync_wait + api + host_gap + transfer_wait
+ unattributed` (within epsilon). If buckets do not sum, that is a tooling
defect, not acceptable ambiguity. `unattributed_ms` is reported loudly and
treated as a defect metric to drive toward zero — the 2026-07-04 run hid
~69ms (the replay barrier) exactly there.

Async caveat: attribution is by enqueue, so busy may land outside the NVTX
window for fire-and-forget stages; such stages carry
`busy_outside_window_ms` explicitly instead of silently breaking the sum.

## Taxonomy contract (versioned)

The dotted NVTX stage labels are the interface between the Rust crate and
this tooling. `taxonomy.py` owns `TAXONOMY_VERSION` and the node table.

Rules:
- Additions are allowed freely.
- Renames/removals require a version bump plus a mapping entry so `trend`
  can compare across versions.
- Every node has stable ownership and a source mapping (crate file).
- Every NVTX label seen in a trace must resolve to a known node or be
  reported under `unknown.<label>` with a count — never silently dropped.
- Nodes MAY carry an `independent_of` annotation (protocol-level
  independence, e.g. NC digit-table init vs FE rounds) — this is the input
  that lets the critical-path layer compute stream-concurrency opportunity
  instead of guessing.

## Critical-path model (v1 — honest approximation)

nsys provides intervals and correlation IDs, not host dependency edges, so
v1 is gap classification, not a full DAG:

1. Build the device timeline per stream (kernel ∪ copy intervals).
2. Every gap in the union is a stall candidate. Classify each by what ends
   it and what API activity spans it:
   - ended by a kernel whose launch call sits inside the gap → `host_gap`
     (host was thinking/serializing)
   - spanned by a synchronize call → `sync_wait` (CPU blocked on GPU;
     reported from the CPU side)
   - copy activity without overlapping compute → `transfer_serialized`
3. Emit ordered segments:

```json
{"stage": "fold.superneo.pi_ccs.sumcheck.fe",
 "blocker": "host_gap_between_device_fs_segments",
 "cost_ms": 84.4, "fix_class": "device_fs_chain"}
```

`fix_class` is a closed enum: `host_eviction`, `device_fs_chain`,
`graph_capture`, `stream_overlap`, `transfer_elimination`,
`kernel_occupancy`, `kernel_fusion`, `unknown`. Closed on purpose — an
open string field decays into prose.

v2 (after streams land): cross-stream gap analysis + concurrency
opportunity computed against the taxonomy's `independent_of` annotations.

## Kernel diagnostics

Static (available now): registers/thread, block shape vs warp multiple,
theoretical occupancy + limiter, grid underfill vs SM count, local-memory
spill. From the device profile table, not hardcoded constants.

Counters (schema defined now, filled when ncu is unlocked): achieved
occupancy, SM/DRAM/L2 throughput %, warp stall reasons, branch divergence,
load/store efficiency. This is what separates "kernel was running" from
"cores were well used". Until unlocked, `ncu.status = "blocked"` with the
unlock command in the message.

## Residency ledger & budgets

`buffers.json`: per buffer — owner, producer stage, consumer stages,
intended residency, measured H2D/D2H/DtoD (MB + copies). Budgets are
gates: `--assert-residency` and `check` fail when a forbidden transfer
returns (e.g. mid-sumcheck D2H after slice 3). Budgets tighten
monotonically as slices land; loosening one requires an explicit note in
the run's regression.json.

## Floors, levers, ranking

`floor = gpu_busy + transfer_floor(measured p90 bandwidth) + launch_floor
(launches × 4µs)`; `recoverable = wall − floor`. Levers are emitted
machine-readable, each with `cause` and `fix_class` (same enum as critical
path), plus composable what-if: `projected_online([stages...])` so a
slice's expected payoff is computed, not estimated in prose. Distance to
floor is the blessed success metric — not occupancy, not utilization %.

## Regression layer (`check`)

Exit-coded verdicts on: online wall (tolerance + abs floor), per-stage
walls, H2D/D2H budgets, sync/launch count deltas, residency violations,
parity linkage present. Every FAIL line carries its structural cause
(what changed: launches, copies, MB, per-kernel ms) — a bare red number is
a spec violation. Baseline defaults to the last accepted history entry.

## Correctness binding

Two prover modes exist (see PLAN-full-gpu.md): fast (no host replay) and
parity (full replay + byte comparison). The harness times fast mode only;
parity is a separate run. A timing bundle is VALID only when linked:

```json
{"fast_run_id": "...", "parity_run_id": "...",
 "same_tree": true, "proof_bytes_match": true}
```

Link rule: the parity run may occur before or after the timing run, but
the two must share `git_head`, `tree_diff_hash`, and `config_hash`
(benchmark, protocol, and CUDA feature config). `same_tree: true` asserts
all three match, not just the diff hash.

`check` and `trend` flag unlinked timing runs. An unreproducible or
parity-less number cannot become a baseline.

## History & experiment ledger

`history.jsonl`: one line per run — run_id, mode, git_head,
tree_diff_hash, config_hash, hw, online numbers, gate status, parity
link, accepted flag. Appending never fails a run.

`accepted run` (the only runs eligible as auto-baselines) is defined
mechanically, not subjectively: `check` PASS + linked parity PASS + zero
residency violations. Bootstrap: while no accepted run exists, `check`
requires an explicit `--baseline`.

`experiments.jsonl`: hypothesis, toggle/diff summary, result, verdict
(kept/rejected), evidence run_ids. `history --rejected` surfaces the
do-not-retry list (E-grouped mat-vec, shared-memory tiles, NC digit
branching, ...) so no fresh session re-proposes a measured-slower idea.
Source-of-truth notes stay next to the kernels; the ledger indexes them.

## Success standard

The harness is good when an AI can state, without guessing:

"The next optimization is X because stage Y lost Z ms to cause C
(critical-path segment + decomposition evidence), the expected payoff is
bounded by N ms (floor math), the change did not regress (check PASS),
and the proof still matches CPU byte-for-byte (parity link)."

## Build order & cutover

Build order (each step usable on its own):

1. `collect` + bundle layout + metadata identity (tree_diff_hash) —
   restores history integrity for the uncommitted tree from day one.
2. `parse` + taxonomy port (versioned, unknown-label reporting).
3. `analyze` decomposition with the reconciliation contract — kill the
   unattributed gap by construction.
4. **Differential validation**: run `gpuscope analyze` over recent
   gpuprof sqlite exports; stage walls, counts, and MB must match gpuprof
   within epsilon wherever semantics coincide, with intentional deltas
   (e.g. the unattributed bucket) documented per field. This gate is what
   licenses trusting gpuscope at all.
5. Critical-path v1 + levers with fix_class.
6. `check` with auto-baseline + parity linkage; import
   `benchmark-results/gpuprof-history.jsonl` so `trend` spans both eras.
7. `doctor`; ncu schema stub emitting blocked-status; residency gates;
   experiment ledger + `history --rejected`.

Cutover criteria (all required):

- Differential validation green on ≥3 real campaign bundles.
- The optimization loop has used gpuscope `run`/`check`/`trend`
  exclusively for ≥5 consecutive iterations with no capability gap noted.
- History continuity verified (`trend` shows the full campaign).

Then delete `scripts/gpuprof/`.
