# Full-GPU campaign — continuation plan

Goal (user directive): the entire SuperNeo protocol runs on GPU — no idle
waits on the CPU, no mid-fold data movement. End state: the host enqueues a
whole fold as one command stream and downloads only the proof.

Benchmark contracts:

- `e2e_bench` is the full-audit guard. It measures the full lifecycle audit
  path on both CPU and CUDA, including terminal fold plus terminal
  private-witness export/materialization. This gate keeps full terminal
  witness parity covered.
- `e2e_gpu_fast_bench` is the GPU-resident prover target. It models the
  intended end state: terminal private planes stay on device for a later
  device consumer/decider, while the CPU receives only proof/public material
  needed by the fast prover boundary. It compares the claims-only audit
  representation and leaves full terminal-witness validation to `e2e_bench`
  outside the timed fast path.
- Setup/cold-start work (`synth`, `setup`, `cuda-init`, `cuda-prepare`) is
  printed separately and is not part of online prove.

Do not change either timed window in the same iteration as a claimed
performance win. If a new contract is needed, add a separate gate and baseline
it explicitly before using it as an optimization target.

The axiom (byte-identical proofs vs CPU) stays. Key unlock already proven:
byte-identity does NOT require a host-authoritative transcript — a bit-exact
device Poseidon2 preserves it, and that now exists and is gate-checked.
The full-audit e2e gate compares terminal running witnesses directly, so
device-resident witness work cannot narrow parity coverage while claiming a
fast-path prover speedup.

Current numbers live only in the tool output:

- Full-audit trajectory:
  `python3 scripts/gpuprof/gpuprof.py trend e2e_bench`.
- Fast-contract trajectory:
  `python3 scripts/gpuprof/gpuprof.py trend e2e_gpu_fast_bench`.
- Fresh fast median + levers:
  `python3 scripts/gpuprof/gpuprof.py run e2e_gpu_fast_bench --repeat 3 --assert-residency`.
- Fresh full-audit guard:
  `python3 scripts/gpuprof/gpuprof.py run e2e_bench --repeat 3 --assert-residency`.
- Merge gate:
  `python3 scripts/gpuprof/gpuprof.py check <accepted-baseline.json> <candidate.json>`.

## Done — slice 1: device Poseidon2 + transcript mirror (2026-07-02)

- `neo-ccs::crypto::poseidon2_goldilocks::round_constants()` regenerates the
  canonical permutation's constants from `SEED` via public p3 APIs; pinned by
  `crates/neo-ccs/tests/poseidon2_round_constants.rs`. New workspace dep
  alias `rand_p3` (rand 0.10, mirrors `rand_chacha_p3`).
- `src/kernels/poseidon2.rs` — permutation core (named register lanes) +
  `p2_permute_states` + single-thread `p2_transcript_ops` (OP_ABSORB /
  OP_CHALLENGE pairs, mirrors `Poseidon2Transcript` cursor semantics).
- `src/transcript.rs` — `DeviceTranscript`: seeds from the host snapshot
  shape (`state()` + `absorbed()`), runs op streams, exports the snapshot.
- `parity transcript` gate: 4096 permutations + 200-round mixed op stream
  bit-identical. Full suite + e2e green.

## Done early — resident-witness items pulled forward (2026-07-03)

- **Resident oracle F row tables**: built from resident witness planes,
  no per-fold table download (`oracle.F` D2H → 0).
- **Device-backed running witness / `DecOutputMode::ResidentOnly`**:
  mid-chain folds skip the child-plane download; full export only at
  chain boundaries (`emit` collapsed accordingly).

## Done — slice 2: device-driven sumcheck challenges (2026-07-03)

1. ✅ Op kernel extended: `p2_transcript_io_ops` with `OP_ABSORB_DEVICE` /
   `OP_CHALLENGE_DEVICE` triples — absorbs read a kernel-written device
   buffer, challenges land in a device buffer the fold kernels read.
   Gate: `transcript` asserts host+device challenge streams identical.
2. ✅ FE: `run_rows_device_transcript` drives whole FE row loops with
   device-sourced challenges (coeff + challenge logs, zero per-round D2H);
   integrated at the engine seam via the `FeSumcheckBackend` /
   `FeRowRoundTrace` hook. Gate: `ccs_fe` replays device-transcript rounds
   and asserts scalar/device challenge folds match.
3. ✅ Post-loop replay: one bulk D2H of the coeff/challenge logs, host
   transcript replay preserves byte-identity.
4. ✅ NC component path: `DeviceNcOracle::col_round_trace_from_transcript`
   mirrors the FE device-log pattern and sources column-round challenges on
   device.
5. ✅ Activation: the engine calls the trace hooks whenever the backends
   accept the snapshot (`optimized_engine/prove.rs` FE ~982, NC ~1230), and
   the adapter always threads the device backends — so `ccs_prove`, `nifs`,
   and `e2e_bench` already run device-FS with byte parity. Parity/debug
   mode self-checks by replaying the device log on the host; the timed CUDA
   path can now adopt the device transcript snapshot directly.

Hardening leftovers folded into slice 3's gate list below.

## Next — slice 3: device-owned Π_CCS FS chain (the orchestrator slice)

Why (verified in `neo-reductions/src/engines/optimized_engine/prove.rs`):
the FE row rounds run device-FS, but the sumcheck continues into the Ajtai
tail rounds, which are still host-driven per round (loop at ~1021, device
`Y_eval` injected at round == ell_n). Because the tail's absorbs must chain
onto the row absorbs, the engine downloads and replays the device log into
the host transcript immediately after the rows (~992-1016) — a hard
mid-phase barrier. Same shape at the NC prolog and before Π_RLC. Every
host-driven FS segment turns the preceding replay into a critical-path
sync, so "defer the replay" is impossible until the phase's whole FS chain
is device-owned. That barrier is most of the `sumcheck.fe` gap.

Scope: one backend call runs FE rows → Ajtai tail → NC prolog + cols →
output surfaces as a single enqueue. Fast mode continues directly from the
device transcript and downloads only proof/public material needed by the
caller. Parity mode performs a separate bulk log download and complete host
replay/validation against the CPU transcript.

New device work:
- Tail rounds on device: component-ready. `DeviceFeBackend` can keep the
  Ajtai `Y_eval` surface resident and compute tail-round coefficients from
  it on device; `parity ccs_fe` checks those coefficients against the CPU
  oracle field-for-field. The same gate also checks the slice-3 prerequisite
  path where `χ_r` is built from device-resident row challenges instead of a
  host-built table.
- Whole-FE trace hook: component-ready but not fast-path enabled.
  `FeSumcheckBackend::fe_phase_trace_from_transcript` and
  `DeviceFeBackend::full_fe_trace_from_transcript` can run FE rows plus
  Ajtai tail from a continuous device transcript and return the combined
  coeff/challenge log. It also carries the already-computed Ajtai `Y_eval`
  surface back to the host oracle, so the output path no longer recomputes
  the same ring mat-vec.
- Tail coefficient path: the catastrophic serial version is gone. The
  current implementation emits parallel Eval partials, reduces them through
  the shared sumcheck reducer, then runs a tiny final assembly kernel.
- Fast-vs-parity entrypoints are done. `BackendTranscriptMode::Replay`
  keeps full host replay for gates; `BackendTranscriptMode::DeviceSnapshot`
  restores the host transcript from the device snapshot without replaying
  the log online. The CUDA adapter uses `DeviceSnapshot`; parity gates keep
  `Replay`.

Decision: the current whole-FE / whole-Π_CCS command shape is parked as the
hot path. It is byte-identical and architecturally useful, but measured
diagnostics show it turns the host-submission gap into extra API, H2D, memset,
and underfilled device work. Do not promote it, graph it again, or retry
single-boundary FE fusions until the execution grain changes. The next
Π_CCS attempt must be a different schedule: persistent/fused row-round
submission, a coarser whole-fold command stream, or another design that
removes the row-round command ladder rather than rearranging its leaves.

RLC status: the standalone `pi_rlc.combine_claims` queue item is no longer
the old CPU-validation island. Π_CCS now owns a compact output digest for
rho binding, rho sampling runs on the device, commitment inputs are resident
when their host commitments match, rho materialization is lazy, and K-surface
output runs from resident Pi_CCS surfaces on a forked stream. Remaining RLC
work belongs inside the whole-fold transcript/command stream, not as another
standalone combine-claims cleanup.
- Current result: correct but not a major speedup. Use the accepted-baseline
  JSON above for the actual value. The entrypoint split by itself did not
  create a meaningful headline win. The current recovery came from moving
  one-time CUDA kernel loading out of the timed online prove path and into
  `CudaNifsProver::new`; cold-start still pays it, but the online benchmark
  now measures proving rather than lazy module load. The remaining FE gap is
  not mainly host replay; it is launch/API overhead plus small serial device
  transcript kernels.
- Small follow-up win: the ring-forms/y' buffer is now allocated without a
  redundant device memset because the CSR forms kernel writes the full output.
  This removed visible forms/y' memset waste.
- NC prolog absorbs are already in the device op stream. Do not repeat that
  work.
- Whole-FE without graph is now a separate gate from whole-FE through CUDA
  graph. It is byte-identical and useful for isolating ownership, but it is
  not accepted as the hot path: simply moving the Ajtai tail onto the device
  adds too many fine-grained kernels and transfers when measured at the
  current granularity.
- A whole-FE fast-transcript diagnostic confirmed that host replay is not the
  blocking cost for this path. Do not retry transcript-mode-only promotion of
  whole-FE; it needs a different execution shape.
- The default FE row-trace path now uses the retained `FePhaseWorkspace`
  for coeff logs, challenge logs, and transcript state. This keeps row-trace
  ownership aligned with the whole-FE path: the hot path no longer allocates
  separate per-prove FE log/transcript buffers while the graph/whole-phase
  path owns persistent ones.
- The reductions-level whole-phase seam is now consumed by the CUDA whole
  modes. `neo-reductions` owns the canonical protocol flow, calls an optional
  `PiCcsPhaseBackend`, and applies the returned FE + NC trace through a small
  host-side validator/bookkeeper. The CUDA adapter routes whole-FE and graph
  gates through `DevicePiCcsPhaseBackend`, not separate FE/NC hooks. The first
  single-witness fold still falls back to the row-trace path because it lacks
  the full FE + Ajtai phase shape; the default timed benchmark path remains
  row-trace until a repeat-3 gpuprof check justifies changing it.
- Pi_CCS public challenge sampling is now part of the whole-phase backend
  seam. `neo-reductions` still binds the canonical header/public instance/ME
  inputs, then optionally asks `PiCcsPhaseBackend` to sample
  `alpha`/`beta`/`gamma`/`beta_m` from a transcript snapshot. CUDA whole-phase
  modes sample those values on the device Poseidon2 transcript and replay mode
  validates them against the CPU transcript before any proof material is
  trusted. This removes another host-fed boundary from the whole-phase path;
  the default row-trace hot path keeps the existing CPU sampling until a
  larger command-scheduled path is accepted.
- The same device-sampled public challenges now stay resident as a compact
  CUDA buffer for whole-phase FE/NC setup. FE row points and NC
  `beta_a`/`gamma` are copied device-to-device from that buffer instead of
  being rebuilt from host challenge values and uploaded again. The host
  `Challenges` value remains the proof/audit surface; it is not the authority
  for the whole-phase device dataflow.
- Cached whole-FE / whole-Π_CCS graph replay is parked for now. Multiple
  designs are byte-identical, including the device-resident NC-tail gamma
  variant, but the graph lifecycle gates are still slower than the default
  row-trace path and `nsys` still cannot profile the repeated graph gate
  reliably. A later Y_eval workspace-ownership fix removed one graph-safety
  footgun, but did not change this conclusion. Do not keep trying small graph
  launch/cache/key variants.
- Per-round FE finalizer fusion is also parked. A design that reduced
  coeff block sums, wrote the proof log, absorbed the coeffs, and sampled
  the next challenge in one Poseidon2 kernel was byte-identical but used the
  wrong execution grain: it replaced launch count with an underfilled serial
  device kernel. Do not retry that shape as a standalone change.
- Pi_CCS now has a resident K-surface handoff into Pi_RLC on the default
  row-trace CUDA path. The FE backend retains the Ajtai `Y_eval` device
  surface it already returned to the CPU oracle, the NC backend retains the
  packed finalized `y_zcol` state it already returned to the CPU oracle, and
  the adapter packs those resident surfaces into `DevicePiCcsKSurfaces`.
  Pi_RLC then enqueues `rlc_combine_k_surfaces` from that device buffer
  instead of rebuilding/uploading the same K surfaces from host `CeClaim`s.
  The canonical `Proof.outputs` still exists for proof/audit/verification;
  it is no longer the authority for this CUDA data movement boundary.
- The whole-phase Pi_CCS path now exposes the same resident `Y_eval` and
  finalized NC surfaces to the adapter, so whole-FE diagnostics no longer
  fall back to the host `CeClaim` surface boundary before Pi_RLC. The latest
  diagnostics still do not promote whole-FE: see the resident-surface
  `e2e_bench` / `e2e_whole_fe_bench` snapshots in `benchmark-results/gpuprof`.
  The key finding is architectural, not numeric: the whole-phase backend can
  keep the downstream surfaces resident, and the remaining work is to preserve
  that ownership across the wider fold command stream instead of reintroducing
  a host proof-export boundary too early.
- Reductions-level terminal-summary seam is now present and the CUDA backend
  overrides it. `PiCcsPhaseBackend::summarize_pi_ccs_phase` returns compact
  FE/NC terminal state from device logs/surfaces instead of materializing every
  round polynomial as host `Vec<Vec<K>>`. Gate: `ccs_phase_summary` compares
  CPU terminal replay against the CUDA whole-phase summary path byte-for-byte,
  including final transcript state.
- The proof path can now use that same summary boundary and defer FE/NC
  proof-log export until egress. `neo-reductions` asks the phase backend for
  compact terminal state first; the adapter can hand Pi_CCS outputs/digest to
  Pi_RLC, run Pi_RLC/Pi_DEC, and only then export resident FE/NC coefficient
  logs while assembling `NifsProof`. This removes the earlier "full proof-log
  download before Pi_CCS terminal-state bookkeeping can advance" shape and the
  later "finish Pi_CCS proof before Pi_RLC can start" shape. Parity replay mode
  remains conservative and complete.
- Async proof-log D2H overlap is explicitly parked as a standalone shape. A
  forked-stream export from Pi_CCS exit to egress stayed byte-identical but did
  not pass the repeat-3 `gpuprof` check, and egress still paid host proof-log
  decode/assembly. Do not retry proof-log transfer overlap unless it is part of
  a larger device-owned proof assembly boundary.
- The deferred proof object has an explicit fallback contract: if CUDA owns the
  phase, proof assembly later asks the same phase backend for resident logs; if
  the backend declines and the CPU path runs, the object carries the owned CPU
  proof rounds instead of creating an unfinishable deferred proof.
- Immediate implementation focus: change the Pi_CCS execution grain, not the
  existing graph wrapper. Device should own FE rows, Ajtai tail, NC prolog/cols,
  transcript state, and proof-log buffers as one profitable schedule whose proof
  export is an egress action. CUDA graph capture is useful only after that
  schedule beats the row-trace path. Do not spend the next loop on standalone
  1-20ms leaves.
- The next Pi_CCS boundary attempt should split terminal outputs into a light
  host claim shell plus backend/device-owned K surfaces and proof-output
  materialization. The current proof-log deferral seam is only half the
  boundary: proof rounds can stay backend-owned, but full host `CeClaim`
  outputs are still materialized before Π_RLC. The intended direction is:
  Π_CCS produces resident output surfaces + recomputable digest, Π_RLC consumes
  the digest/shell/resident K surfaces, and full output claims materialize only
  at proof egress or verifier-facing boundaries.
- Latest audit: the remaining `k_surfaces.host_claims` branch is not the
  performance lever by itself. It appears once in whole-FE diagnostics, moves
  only about 9KB H2D, and costs about 0.2ms. Treat it as a symptom of the
  larger output-materialization seam, not as the next standalone target. A
  useful implementation must change when canonical Pi_CCS output claims/digest
  are materialized; merely forcing this fallback closed is not enough.
- Current audit decision: keep the existing whole-FE/whole-Pi_CCS command
  shape parked. It is the right ownership direction but the wrong execution
  grain; it adds command/memory overhead instead of collecting the FE gap.
  NC prestart/overlap is also not the next target: the current profile shows
  NC oracle setup as already noise-scale, while NC sumcheck is near its floor.
  Cooperative FE launch grouping was also measured and rejected: single-round,
  all-round, and banded cooperative row-ladder variants cut launches but moved
  the same work into a heavier low-occupancy kernel. Do not retry that family
  as a measured e2e path by only changing launch grouping.
- Direct graph budget gates were rechecked on the current tree. They remain
  byte-identical but slower than the default path (`e2e_graph_two_bench` and
  `e2e_graph_three_bench` are still in the mid-600ms band while default is in
  the high-500ms band). Do not reopen graph-budget work unless paired with a
  new execution grain that removes the Pi_CCS output/proof materialization
  boundary.

Next slice contract — wider fold command stream:
- Keep `neo-reductions` as protocol owner. The CUDA side may change how FE
  row rounds are scheduled, but the engine still validates/adopts the same
  challenges, final sums, and proof logs.
- The next implementation candidate must widen the device-owned boundary, not
  move small work around the existing FE ladder. It is not enough to move
  proof-log copies, replay timing, graph cache keys, fold-plane staging, or FE
  launch grouping around the same row-round body.
- Valid shapes to investigate:
  1. a wider whole-fold command stream where FE row work, transcript state,
     and later proof-log export are scheduled together and measured together;
  2. a different mathematical batching of same-shape FE row rounds that keeps
     Fiat-Shamir order intact while giving each launch enough work.
- Invalid standalone shapes, already measured or ruled out:
  direct coeff-log reduction, FE finalizer fusion, whole-FE replay/graph
  promotion with the current command shape, NC prestart, proof-log D2H
  overlap, next-fold plane staging, single-round cooperative FE, all-round
  cooperative FE, banded cooperative FE, and direct graph-budget replay without
  a new output-materialization boundary.
- Acceptance remains the normal gate: byte-identical parity, residency clean,
  and a repeat-3 `e2e_bench` improvement against the accepted baseline.

Constraints (non-negotiable):
- `neo-reductions` stays the sole protocol owner. The device side owns the
  execution schedule only. Implement by widening the existing seam — one
  whole-phase trace hook returning the full coeff/challenge/absorb log —
  NOT a parallel orchestration framework in the CUDA crate.
- The host replay stays complete in parity mode (every absorb, every
  challenge) and a mismatch stays fatal to that gate. It must not be a
  required step in the fast prover or included in CUDA benchmark time once
  the device transcript can drive the next protocol phase. Hard rule that
  outlives the migration: verifier/decider-side code never consumes
  device-derived challenges without recomputation.

Gates & hardening (includes slice-2 leftovers):
- Keep fast-vs-parity split honest: benchmark runs fast GPU prove without
  host replay; parity gates rerun with full CPU-vs-CUDA replay and byte
  comparison.
- `ccs_prove` / `nifs` assert byte parity through the whole-phase trace.
- Direct NC-trace replay now lives in `gates/ccs.rs`, covering the NC
  prolog and column rounds at the same transcript granularity as FE.
- `RESIDENCY_BUDGETS` now covers the Pi_CCS sumcheck stage so table-sized
  or per-round D2H traffic cannot silently return.

## Parked as standalone work — small leaves

These may still be useful when they fall out of a larger device-owned fold
boundary, but they should not be the next loop's main work:
- `oracle.F` / `oracle.Eval` cleanup.
- `pi_dec.open_children.forms` host prep.
- `pi_rlc.combine_claims` validation leftovers.
- `pi_dec.emit.planes` / `emit.assemble` residuals.

Reason: the current profile shows these as small compared with the Pi_CCS FE
proof-log / command-boundary lever. Chasing them independently risks more
benchmark churn without moving the architecture toward "host enqueues a fold,
GPU owns the protocol work, host downloads proof material."

## Existing structural handoffs to preserve

- Fresh assignment planes uploaded while building device Ajtai commitments are
  retained with their matching host commitments and reused by the next fold's
  ingest when the same fresh claims appear. In steady state, fold ingest
  composes fresh + retained running planes device-to-device instead of
  uploading the fresh witness plane twice.
- `pi_rlc.output` now batches the K-valued `y_ring` and `y_zcol` surfaces
  through one packed device operation (`k_surfaces`) instead of separate
  host-mediated calls. This cleans the output dataflow only; the remaining
  structural RLC work is still the claim/rho authority boundary.
- Π_CCS now carries a recomputable compact `outputs_digest`, and Π_RLC can
  bind that digest directly before sampling `ρ`. This is the intended
  protocol handoff shape: Π_CCS owns the output surface and digest, Π_RLC
  owns rho sampling from that digest. It is not authority; verifiers
  recompute it from the Π_CCS output claims before accepting the proof.
- Do not retry standalone device hashing of full Π_CCS output-claim digest
  preimages as the route to `pi_rlc.combine_claims`. That moves a protocol
  boundary in the right direction but uses the wrong data shape: large
  claim-preimage fanout plus underfilled Poseidon2 hash kernels. Revisit the
  RLC bind only as part of a coarser device-owned fold transcript where that
  compact digest is produced directly from resident GPU Π_CCS outputs.

## Then — slice 4: whole-fold FS chain, graphs, streams

- Extend the device-owned chain from Π_CCS to the full fold: Π_RLC ρ
  sampling already runs through the device sponge
  (`p2_transcript_sample_rlc_rhos`); end state is ONE replay per fold —
  the host enqueues a whole fold and downloads only the proof.
- CUDA graphs / command-stream capture per phase, then per fold
  (`fold.superneo` today: 2181 API calls, 206 syncs; graphs also make the
  140 one-block sponge launches free).
- Streams for genuinely independent work — only after the host gaps are
  gone, else the freed SMs just idle differently: NC digit-table build
  overlapped with FE rounds (no challenge dependency), remaining Π_DEC
  child work in parallel, then cross-fold pipelining of the next fold's
  ingest/commit. Π_RLC K-surface combination already enqueues on a forked
  stream and joins after `Z_mix` / `X`, so do not retry smaller RLC stream
  variants as standalone work.
- Architectural beyond that: pipeline the host F′ witness compile against
  GPU folding of the previous chunk; concurrent chains for aggregate 10x.

## Then — kernel internals (use ncu before changing kernels)

- Register pressure: `nc_col_partials` at 17% theoretical occupancy,
  `fe_round_partials` at 33% before the latest chunk-parallelism pass —
  use ncu counters to steer; do not guess (see the rejected-experiments
  notes in `kernels/ajtai.rs`).
- Tail-round fusion: the last sumcheck rounds are <1 SM of work launched
  as strings of 1-block kernels — fuse into one cooperative kernel.
- Workload ceiling, for honest expectations: mat-vec folding math is
  bandwidth-bound and the FS chain is serial by soundness. The success
  metric is a gap-free GPU timeline approaching the kernel-busy floor
  (gpuprof's floors/levers view), not GEMM-like occupancy.

## How to build / verify

- Build (from `crates/neo-prover-cuda/`):
  `cargo +nightly-2026-04-03 oxide build --features cuda,perf-timers`
- Run gates (from the WORKSPACE ROOT — the binary lands in the root
  `target/`): `./target/release/parity quick`, `./target/release/parity
  transcript`, `./target/release/parity e2e_bench`.
- Phase profile: `python3 scripts/gpuprof/gpuprof.py run e2e_bench --repeat 3
  --assert-residency --json benchmark-results/gpuprof/<ts>-<slug>.json`
  (needs the perf-timers build; errors loudly if stamps are missing).
  Phases come from NVTX ranges (survives async command streams — attribution
  is by enqueue); `idle s/a/h` splits idle causes; TOP LEVERS ranks
  recoverable ms; KERNEL LINT flags spills/occupancy; BOUNDARY SCORECARD
  ranks remaining CPU/GPU crossings by host-owned time, H2D/D2H bytes, joins,
  and launch count; every run appends to `benchmark-results/gpuprof-history.jsonl`.
- Regression gate per iteration:
  `python3 scripts/gpuprof/gpuprof.py check <baseline.json> <candidate.json>`
  (exit 1 on >5% regression; FAIL lines carry structural causes).
- Campaign trajectory: `python3 scripts/gpuprof/gpuprof.py trend e2e_bench
  [--stage <dotted.label>]`.
- Every slice: parity gates first, byte-identical always, never regress,
  `cargo fmt --all`, plain `cargo check --workspace` (cuda off) stays green.

Known traps: kernel-local arrays spill to local memory (use named scalars);
`cargo oxide` takes no `--bin` flag; run the parity binary from the
workspace root, not the crate dir.

## External review feedback (2026-07-05, parallel review session — read-only)

Reviewed the 2026-07-05T00:24Z–05:56Z ledger segment and the current plan.
The discipline is excellent (revert-verified rejections, parked-family
taxonomy, the attribution-drift correction on whole-FE, contract hardening
before boundary moves) — keep all of it. Five specific inputs for the next
iterations, ranked:

1. **Promote the compile/fold pipeline out of slice 4 — it likely beats the
   FE lever outright.** The 2026-07-02 iteration-10 audit found overlapping
   chunk k+1's host F′ compile with fold k was the ONLY item clearing the
   >10% bar, and it has sat unprioritized since. Appends still run
   compile→fold sequentially; at ~30-45ms compile/chunk (re-measure with the
   `[r1cs-compile]` perf-timers first), hiding 3 compiles behind folds
   recovers ~90-135ms online — more than the entire ~107ms FE lever — with
   zero kernel work and zero FS-order risk (independent host reordering;
   parity gates referee). Unverified prereqs from that audit: (a) fold k has
   no dependency on compile k (believed: fold consumes the previous latest +
   placeholder — VERIFY in append_chunk_inner/prepare_next_fold), (b)
   CudaNifsProver/Device/Transcript thread-safety across a scoped thread. If
   compile has since shrunk below ~15ms/chunk, reject it in the ledger with
   that number so it stops haunting the queue.

   Status: audited on the current tree; see `benchmark-results/loop-log.md`.
   This is not the next priority unless the compiler is split into a separate
   precompute stage that no longer needs fold authority.

2. **Before inventing a new FE "mathematical batching": reconcile the FE
   gap first.** The lever row says wall 156.5 / busy ~48 / recoverable ~107,
   but the idle decomposition only classifies ~39ms (sync 0.1 + api 4.2 +
   host 35.1). ~69ms of the flagship stage's gap is unclassified — exactly
   the coverage failure the gpuscope spec's reconciliation invariant exists
   to prevent. One cheap diagnostic: host-side timestamps around the FE
   row-trace enqueue loop in the default path (what does the host do between
   enqueues that costs ~4-5ms/round?). If it's cuda-oxide per-launch host
   cost, that argues one way; if it's engine bookkeeping, another. Also note
   FS seriality bounds this lever: rounds cannot be batched across the
   challenge chain, so if the host loop is already tight, the honest move is
   to declare the FE ladder near its floor and recover e2e time by
   overlapping independent work instead (NC digit tables, item 1).

3. **The whole-FE loss is enumerable churn — spend one bounded slice
   killing it before any permanent park.** The 02:45Z diagnostic numbers are
   owners, not mysteries: launches 384→945, H2D 0→25.2MB, memset
   0.3→561.2MB (unchanged in the 05:56Z recheck, which also broke the new
   0.25MB sumcheck D2H budget). Each has a known fix pattern already used
   elsewhere: persistent phase-workspace buffers (the Y_eval return fix,
   applied to every whole-phase buffer), no re-upload of surfaces that are
   already resident, and no zeroing of buffers whose kernels fully write
   them (the forms/y′ memset fix). Acceptance for the slice: whole-FE memset
   ≈ default, H2D ≈ 0, launches ≤ default + tail count, sumcheck D2H budget
   green — then re-measure. If it still loses after churn parity, park it
   permanently with that clean negative result.

4. **Cooperative tail fusion is NOT in the parked family.** The parked
   cooperative experiments were row-round fusion — big-work regime, where
   the fused body lost on occupancy. The Ajtai tail is the opposite regime:
   ~6 rounds of <1-SM work launched as strings of 1-block kernels (your own
   "kernel internals" section). Fusing the tail into one cooperative kernel
   is launch-bound, not register-bound, so it does not need ncu, and it is
   what the whole-FE path needs to stop adding fine-grained tail kernels.
   Natural companion to item 3.

5. **Add a boundary scorecard so structural slices ratchet numerically.**
   Most overnight entries are honest "structural progress only, no
   performance claim" — but the goal ("host enqueues a fold, downloads only
   the proof") has a countable metric: host-join count per steady fold
   (FE-tail replay join, RLC claim algebra, DEC emit/assemble decode, …) +
   H2D/D2H bytes per fold. gpuprof already measures the pieces; assert
   non-increase like the residency budgets. Then "migration progress" is a
   number that must go down, not a narrative.

CORRECTION (verified 2026-07-05, this session): **ncu is UNBLOCKED and
working.** `/proc/driver/nvidia/params` shows `RmProfilingAdminOnly: 0`,
`/etc/modprobe.d/nvidia-profiling.conf` is in place, and ncu 2025.3.1
successfully collected SpeedOfLight + Occupancy counters from `k_mul_add`
via `parity smoke`. The "kernel internals (blocked on the ncu driver
unlock)" section heading above is stale — that work is now unblocked.
Consequences for the ranking above: (a) the FE-gap reconciliation in item 2
can now include ncu stall/occupancy evidence instead of static-lint guesses;
(b) the cooperative-FE parked family's own exit condition — "redesign that
kernel's arithmetic/memory shape using hardware counters" (05:18Z ledger
entry) — is now satisfiable, so a counter-guided redesign of the fused
row-round body is a legitimate future slice (as a measured mode with the
usual accept/reject gate), distinct from the parked launch-grouping-only
variants; (c) `nc_col_partials` (17% occupancy) and `fe_round_partials`
(33%) can finally be diagnosed rather than estimated. Suggested first ncu
target: `fe_round_partials` and the one-block transcript kernels inside the
default FE window, since that stage holds the top lever.

REVIEWER RESPONSE to the 06:18Z compile/fold audit (2026-07-05): **item 1
is withdrawn — the audit is conclusive on two independent grounds.**
Recursive compile is ~10ms/chunk on the GPU chain (my 30-45ms figure was
stale, pre width-tightening), and compile consumes fold authority
(`prepare_next_fold` runs before `compile_chunk`), so the overlap does not
exist without a compiler split whose payoff cap is now ~30ms total. The
revised ordering is endorsed: (1) bounded whole-FE churn cleanup (item 3
acceptance criteria as written), (2) Ajtai tail fusion inside that slice
(item 4), (3) boundary scorecard (item 5), (4) only then wider fold
scheduling. One refinement of method, not ordering: run the item-2 gap
reconciliation as INSTRUMENTATION inside the churn slice, not as further
candidate edits. The 06:30Z pinned-row-log rejection is what probing the
unclassified ~69ms by trial edit costs — a full build/gate/measure/revert
cycle to learn "too local." One-time host timestamps around the default FE
enqueue loop plus an ncu pass (now unlocked) on the FE-window kernels
locates the 69ms before the next edit, and it sets the whole-FE accept/park
threshold correctly: if the default gap is mostly per-launch host cost that
whole-FE structurally avoids, a churn-clean whole-FE can win big; if it is
engine bookkeeping both paths pay, the ceiling is lower and the final park
decision should say so with that number.

REVIEWER RECOMMENDATION after the deferred-RLC-public-surfaces slice
(2026-07-05, second review pass): **of the two candidates named in the
19:55Z verdict, pick the Π_CCS output-seam widening, not terminal
DEC/proof-output residency.** Grounds:

1. **Magnitude and scaling.** pi_ccs.sumcheck.fe is still ~131ms wall /
   ~104ms recoverable and it is a PER-FOLD cost — it multiplies with chain
   length, so the 8-fold benchmark understates its real-world weight.
   Terminal emit (~19.2ms planes D2H + ~10.2ms assemble) is once-per-chain
   and shrinks in relative terms as chains grow.
2. **Terminal residency has no in-contract win available.** The 09:32Z
   rejection already established the 45MB terminal witness export is
   required under the current benchmark contract; the only exits are a
   symmetric contract change or a GPU decider that consumes the terminal
   witness on device. Both are user-scope decisions / new campaigns, not a
   bounded loop slice. Do not spend iterations there without one of those
   being explicitly approved.
3. **The ledger already converged on the seam — four independent verdicts**
   (12:20Z, 10:42Z, 11:45Z, 17:56Z) all end with: remove the mid-fold host
   Y_eval / full-CeClaim materialization so Π_RLC consumes a device-owned
   Pi_CCS output shell. The 10:55Z RLC-input-authority cleanup was the prep
   (rho sampling and commitment authority no longer need materialized
   CeClaims). The missing piece is `PendingFePhase::download_summary`'s
   mid-fold pull of FE challenges + last coefficients + Y_eval, which
   exists only because the host rebuilds Pi_CCS output claims via host
   Ajtai precompute.
4. **Dual payoff.** The same seam is (a) the top default-path lever
   (row_download ~25.5ms + a large share of the 49 joins) and (b) the
   declared unlock for the parked whole-FE path, whose remaining blockers
   are exactly the summary downloads (0.35MB > 0.25MB sumcheck D2H budget)
   plus the 954-launch shape. Land the seam once, then re-run the whole-FE
   diagnostic for free: if its sumcheck D2H budget goes green and it beats
   the default, promote; if not, park it permanently with that sharper
   number.

Guardrails for the slice: proof bytes and output digests stay canonical;
transcript absorb order unchanged; deferred surfaces get the same
verify-before-proof-export treatment as the RLC y/X/c checks (digests are
compression, not authority — the deferred check must still fail on mutated
data). Do NOT bundle digest-on-GPU into this slice; the 090000Z trap
(p2_hash_fields +97ms) stands — the shell carries non-authoritative
structure and digests are recomputed at egress from authoritative bytes.
Acceptance: byte-identical, residency clean, boundary-scorecard
pi_ccs.sumcheck.fe host-owned ms and join count strictly down (that is the
numeric structural claim), and median vs 596.0 decides ratchet vs
structural-only.

Process note: re-accept a baseline. Two recent strict checks are red purely
for accounting reasons (FE taxonomy children; RLC wait relocated into
pi_dec.open_children.forms). A standing red gate normalizes ignoring the
gate. Accept the current line as baseline with a one-line ledger note
naming the relocated wait, and let the boundary scorecard (joins + bytes
non-increase per boundary) be the guard against cost-shuffling counting as
progress — that is exactly what it was built for at 11:45Z.

## Current fast-contract frontier (2026-07-06)

The fast prover target is now dominated by lifecycle boundaries, not by one
local arithmetic kernel. In the current `e2e_gpu_fast_bench` line, the DEC
child-output arithmetic is resident enough that the next exposed boundary is
`Pi_DEC` proof/output materialization: `DeviceDec::prove` must still return
ordinary host `Children`/`Proof` values because
`NifsProverAdapter::prove` returns `(RunningInstance, NifsProof)`.

Current dependency shape:

```text
Pi_RLC z_mix on device
  -> Pi_DEC split planes on device
     -> child y_ring / y_zcol / commitments on device
        -> host CeClaim materialization
           -> host RunningInstance
           -> host NifsProof.pi_dec.children
           -> audit / recursive proof object
```

The first three nodes are already CUDA-owned. The fourth node is the remaining
hard boundary: host `CeClaim` materialization is not consumed by DEC arithmetic;
it is consumed by the lifecycle proof object and by the concrete
`RunningInstance` stored in `ProofState::Active`. The R1CS F' compiler can skip
native NIFS replay with the CUDA-provided `FPrimeFoldPostSummary`, but the audit
still stores ordinary proof bytes. Therefore a local DEC copy/stream change
cannot remove this boundary. The next real migration must change ownership of
the fold-output object:

```text
Pi_DEC device child surfaces
  -> device-backed fold-output handle
     -> light host shell + post summary for recursive compile
     -> final/parity egress materializes ordinary CeClaims/NifsProof
```

Do not spend another loop on local `pi_dec.emit.download` retiming, pinned-copy
shape changes, or stream moves. Those would only move the same wait to another
bucket while the adapter contract still demands host `CeClaim`s every fold.
The next valid DEC/proof-output slice is a boundary slice:

- introduce a device-backed running/proof-output surface in the CUDA adapter
  path, with a light host shell for lifecycle bookkeeping;
- keep child planes, commitments, y-ring, y-zcol, and public X resident until
  proof egress or parity/debug export;
- preserve verifier semantics: proof export/parity reconstructs ordinary
  CPU-visible `CeClaim`s byte-identically, and verifier/decider code never
  trusts device-derived challenges or digests as authority;
- make the boundary scorecard prove fewer host joins / D2H bytes in the
  repeated fold before counting it as structural progress.

This is the same kind of seam as the Pi_CCS output-shell work: move host
protocol objects out of the repeated online path without changing proof bytes.
If that seam is too large for the current loop, return to the Pi_CCS terminal
output shell. Do not fall back to another 1-20ms local cleanup.

### Proof-carrier constraint for that seam

The adapter alone cannot remove the repeated `Pi_DEC` materialization boundary.
The old lifecycle shape forced ordinary host proof objects immediately:

```text
construction2::StepProof
  -> FoldProof::Recursive(NifsProof)
     -> pi_dec::Proof { children: Vec<CeClaim> }

lifecycle::UncompressedAudit
  -> steps: Vec<StepProof>
  -> decider::Witness { steps, public_batches, final_fold, final_state }
```

`NifsProverAdapter::prove` also returns `(RunningInstance, NifsProof)`, so a
CUDA adapter that keeps only device child surfaces has nowhere canonical to put
the recursive fold proof. The fast e2e gate still compares the recursive
`NifsProof` bytes after every append; terminal claims-only mode only excludes
private terminal witnesses, not per-fold `Pi_DEC` child claims.

Therefore a valid device-backed fold-output implementation must widen the
proof carrier, not just `DeviceDec::prove`:

```text
Pi_DEC device child surfaces
  -> DeviceFoldOutputHandle
     -> RunningInstance shell + FPrimeFoldPostSummary for online compile
     -> deferred StepProof / NifsProof materializer for parity, audit, decider
```

The deferred materializer is the only place allowed to reconstruct ordinary
`CeClaim`s / `NifsProof` from device surfaces. Fast timing may exclude that
materialization only if the benchmark contract explicitly treats it as
parity/audit egress, and the full-audit gate must still prove byte identity.

First structural seam now present: `NifsProverAdapter::prove` returns
`NifsProverOutput`, which owns a `NifsProofCarrier`, instead of a bare
`(RunningInstance, NifsProof)` tuple. `NifsProofCarrier` now has a
`Materialized(NifsProof)` path plus a deferred materializer contract:
`DeferredNifsProofMaterializer::materialize() -> Result<NifsProof, Error>`.
No CUDA path returns a deferred carrier yet, so this does not claim a
performance win. The point is to make the next step explicit:
`NifsProofCarrier::Deferred` is where a future device-backed fold-output
handle must live, while the public `prove_with_adapter` wrapper continues
returning ordinary `(RunningInstance, NifsProof)` for existing CPU callers.
The F' adapter branch now calls the output-returning helper and materializes at
that lifecycle boundary, so the future deferred path has a concrete place to
avoid immediate proof export without changing the CPU wrapper API.

Do not retry these adapter-local variants:

- `DeviceDec` emits only a post-accumulator handle.
- `DeviceDec` skips child claim downloads while still returning ordinary
  `(RunningInstance, NifsProof)`.
- `CudaNifsProver::cache_post_fold_summary` derives authority from an object
  that differs from the proof carrier stored in `StepProof`.

Those variants make the F' post summary, returned `RunningInstance`, stored
`NifsProof`, and next-step Pi_CCS transcript authority disagree. The 2026-07-06
`append 2: Pi_CCS FE initial sum mismatch` rejection is the concrete failure
mode.

Second structural seam now present: `FoldProof::Recursive` stores
`NifsProofCarrier`, so `StepProof` / `UncompressedAudit` can carry either
ordinary proof bytes or a future deferred materializer:

```text
construction2::StepProof
  -> FoldProof::Recursive(NifsProofCarrier)
     -> Materialized(NifsProof)
     -> Deferred(DeferredNifsProofMaterializer)
```

Verifier, decider, R1CS-audit, parity, and test-support paths materialize the
carrier at proof-consumption boundaries. The F' adapter branch can now keep
the carrier in the step proof instead of immediately exporting ordinary proof
bytes. Current CPU/CUDA paths still use `Materialized`, so this is a
behavior-preserving ownership seam, not a performance claim. The next aligned
step is making the CUDA fold output return a real deferred carrier backed by
resident Pi_DEC child/proof surfaces.

Third structural seam now present: post-fold accumulator summary now travels
with `NifsProverOutput` as `NifsPostFoldSummary`. The CUDA adapter attaches the
post-fold accumulator digest plus F' summary to the same output object that
carries the returned `RunningInstance` and `NifsProofCarrier`; F' native step
advance and final-fold advance consume the digest from that output instead of
calling a separate adapter-side `take_post_acc_digest_override` hook. This keeps
the post-fold digest authority aligned with the proof carrier boundary we are
widening. The R1CS F' builder also consumes the same output-carried summary via
an internal lifecycle adapter-output return path, so there is no remaining
adapter summary hook. This is still structural only: current CPU/CUDA paths
materialize ordinary proof bytes, but the lifecycle and compiler now agree on
one output-owned place for post-fold summary authority.

Fourth structural seam now present: `NifsProverOutput` carries the post-fold
running accumulator through `NifsRunningCarrier`, the running-state companion to
`NifsProofCarrier`:

```text
NifsProverOutput
  -> NifsRunningCarrier
     -> Materialized(RunningInstance)
     -> Deferred(DeferredNifsRunningMaterializer)
  -> NifsProofCarrier
     -> Materialized(NifsProof)
     -> Deferred(DeferredNifsProofMaterializer)
  -> NifsPostFoldSummary
```

This closes the last obvious type-level mismatch in the deferred fold-output
boundary: a CUDA backend no longer has to defer proof bytes while still
returning an ordinary host-built `RunningInstance`. Current CPU/CUDA paths still
use `Materialized`, so this is not a timing claim. The next accepted CUDA
implementation must return a real deferred running/proof pair backed by resident
Π_DEC child surfaces, with materialization only at parity/audit/decider egress.

Fifth structural seam now present: F' names the remaining forced
materialization point explicitly. The adapter branch consumes
`NifsProverOutput::into_carriers_with_summary()`, keeps the
`NifsProofCarrier`, and calls `materialize_running_for_proof_state(...)` only
because `ProofState::Active` still stores a concrete `RunningInstance`. That
helper is the next boundary to remove or bypass; a valid GPU-owned lifecycle
path replaces it with a proof-state shape that can carry the device-backed
running carrier, not with another local `Pi_DEC` download shuffle.

Fresh profiler and NCU evidence confirmed this direction: the visible DEC emit
and split kernels are small underfilled helpers, while the repeated online
boundary is lifecycle materialization of child claims/proofs/running state.
Treat the DEC kernels as already past the useful local-tuning point unless a new
`ncu` run shows a real kernel-internal blocker.

REVIEWER COMMENTS on SUPERNEO_CUDA_FLOW_STATE.md (2026-07-06, external
review session — read-only):

Verified: every number in the state table matches the boundary scorecard in
`20260706T172425Z-revert-device-pi-ccs-digest-rlc-bind.json` (oracle
39.4/7.3/22.2ms, 25 joins, 108 launches; FE 115.5/25.9ms, 49 joins, 384
launches; NC 68.9/66.6ms; y′ 58.2/57.7ms). The diagnosis — CPU objects
acting as intermediate protocol modules between GPU stages — is correct and
matches the ledger's repeated convergence. The oracle-first pick is
defensible on ownership grounds (largest classified mid-fold cpu_owned
block; wall mostly non-busy; DAG un-serializes F/Eval/NC/eq builds), and
"no second protocol owner in neo-prover-cuda" is exactly the right
constraint. Four corrections before implementation:

1. **The "How It Should Run" lane diagram is dependency-incorrect under
   the byte-identity axiom — redraw it before anyone implements it.**
   Verified in code: Π_CCS runs ONE sequential Poseidon2 transcript — FE
   row rounds (absorb coeffs, sample challenge, per round), then
   `append_nc_sumcheck_prolog`, then NC column rounds
   (`optimized_engine/phase_trace.rs:252`, `prove.rs:1235`). Therefore:
   (a) Lane 2's NC *rounds* cannot run concurrently with Lane 1's FE
   *rounds*: every NC challenge transitively depends on the full FE round
   chain via Fiat-Shamir. Only NC digit/column table PREP
   (challenge-independent) and NC round-0 coefficient partials may overlap
   FE. (b) Lane 3's y′ cannot start from "resident fold inputs": y′ is the
   Ajtai Y-evaluation at χ(r) where r IS the FE row-challenge vector (the
   07-05 09:38Z slice builds χ(r) from the retained FE challenge log).
   (c) The true DAG is a serial FS spine — FE rounds → NC rounds → tail —
   with only table prep, eq/tensor tables, and commit work legally parallel
   to it. Implementing the diagram as drawn changes challenge derivation →
   different proof bytes → instant parity failure. The correct version of
   this idea is: overlap the *prep* lanes with the FS spine, and overlap
   independent work across the spine's underfilled rounds (rule 4 already
   says this — the diagram contradicts it).

2. **The FE row ("mixed, ~55%") mislabels the problem.** FE cpu_owned is
   2.7ms of 115.5ms wall; the ~87ms gap is neither host-owned nor GPU busy
   — it is the serial round ladder's launch/submission shape (your own NCU
   evidence: fe_round_partials 1 block × 256 threads on a 128-SM GPU, 2.2%
   achieved occupancy). The oracle DAG will NOT collect this; no ownership
   migration will. It ends via round-shape redesign or by overlapping
   non-FS work across the spine. Keep those claims separate or the oracle
   slice will be judged against a ~90ms pool it cannot touch.

3. **State the oracle slice's honest payoff cap: ~30ms e2e (~6-7% of
   451.9ms).** Full success ≈ oracle wall 39.4 → ~8-10ms. Worth the slice,
   and it's the right enabler for FE consuming device buffers — but
   "first module in execution order" must not be read as "biggest lever."

4. **To-do #5 is too strong for the recursive chain.** Per your own
   01:56:30Z and 07-05 06:18Z audits, `compile_chunk` consumes the
   just-produced fold proof/accumulator digest EVERY fold — the recursive
   F′ compiler is a per-fold host consumer on the critical path, not a
   parity/audit/decider boundary. The 16:37Z entry names the residue
   precisely: `ProofState::Active` still stores a concrete
   `RunningInstance`. The fold-output carrier design must therefore export
   a compile-facing surface (proof material + accumulator digest + post
   summary) each fold; add the recursive compiler as a first-class CPU
   node in the boundary map so the carrier work is scoped against it.

Minor: label which benchmark contract each number belongs to now that two
exist (full-audit 451.9ms/4.81x vs claims-only fast 425.6ms/5.02x) —
adjacent ledger entries mix them and a future reader will too. And credit
where due: gpuscope reconciling at max_reconciliation_error_ms=0 is the
instrumentation invariant this plan asked for — the unclassified-gap era
is over, which is exactly why the FE gap can now be named as launch shape
instead of mystery.

NOTE (2026-07-06, review session): per the user's request I corrected
SUPERNEO_CUDA_FLOW_STATE.md in place — the "How It Should Run" section now
draws the serial FS spine explicitly (FE rounds -> Y_eval -> FE tail -> NC
prolog -> NC rounds -> bind), with only challenge-independent prep lanes
parallel to it, plus the chain-level fold->compile->fold pipeline and the
per-fold compile-facing export. The oracle-DAG section gained the missing
FS-order edge. This is a documentation correction, not a scheduling
proposal change — the oracle-first slice stands as planned.

One NEW measurable hypothesis fell out of drawing the true DAG: **output
y_prime (58.2ms wall / 57.7ms busy) may be legally overlappable with the
NC column rounds (68.9ms wall).** Evidence it needs only FE row
challenges: the 07-05 09:38Z slice builds its chi(r) table from the
retained device ROW challenge log, and Pi_CCS output claims evaluate at
the row point. If y_prime's enqueue truly consumes no NC/tail challenge
and no NC-dependent state, moving its mat-vec onto a second stream right
after the FE rows complete could hide up to ~30-50ms behind NC (bounded
by SM contention; nc_col_partials runs at ~16% occupancy, so there is
room). Verify the dependency in code first, then measure with the usual
accept/reject gate. If y_prime turns out to consume NC-finalized state
(digit_rows / eq_beta_m0), record that here and kill the idea cleanly.

KILL RECORD (2026-07-06, review session): **the y_prime/NC overlap
hypothesis above is REFUTED — verified in code, do not pursue.** The stage
timed as `fold.superneo.pi_ccs.output.y_prime` is the `device_ajtai_y_eval`
call (`reduce/ccs/fe.rs`, `ajtai_y_eval` perf block) — the label says
"output" but it executes INSIDE the FE phase: `enqueue_full_fe_phase_body`
runs rows -> Y_eval -> tail, the tail consumes Y_eval, and NC continues
from the FE phase's device transcript afterward
(`phase.rs`, `begin_phase_with_prolog_and_tail_from_device_transcript`).
Output claims reuse the same `DeviceAjtaiYEval` buffer (`output.rs`) — no
second, later, independent y_prime surface exists. So Y_eval is on the FS
spine, not beside it; there is nothing to overlap with NC. The flow-state
doc's diagram and rule 3 have been corrected accordingly (candidate node
removed; Y_eval labeled as tail input reused for output claims).
Credit: the loop's code-path check caught this before a slice was spent.
Standing conclusion unchanged: next targets are the FE row-trace/proof-log
boundary (via the whole-phase route or a contract-level change — the
row-log deferral/pinned/async family stays rejected) and coarser
multi-fold/multi-chain scheduling; dependency-safe in-fold parallelism is
prep lanes, DEC fan-out, and nothing on the spine.

---

REVIEWER VERDICT (2026-07-09) on the multichain8 full-GPU profile and the
DEC-anchored device-carrier proposal: **verified, endorsed, with five
guardrails.**

What I independently verified before endorsing:
- Headline numbers match the profile JSON
  (`20260709T-fullgpu-multichain8.json`): parity line reads "8 independent
  sha256 chains byte-identical (TerminalClaimsOnly); cpu
  aggregate=17966.86ms sequential cuda=2959.60ms (6.07x) parallel
  cuda=1272.14ms (14.12x) overlap=2.33x". Byte-identity held under 8-way
  parallelism — the axiom survives multichain.
- Both code claims are true: `reduce/dec.rs` calls
  `materialize_child_claim_surfaces` unconditionally in the child-open path
  (~line 769) — `DecOutputMode::ResidentOnly` only gates full-witness
  download (`downloads_full_witnesses`, ~line 102); and `adapter.rs`
  destructures the seam as `running_carrier: _,` (~line 572) — the carrier
  we built is currently dead weight.
- Fences-not-bandwidth is supported by the profile's own levers table:
  `pi_dec.emit.download` shows 1008.7ms summed wall vs 9.5ms busy vs
  ~1.1ms transfer floor; `fe.row_download` 235.6ms wall vs 0.0 busy. (Note
  these lever walls SUM across 8 overlapping chains — recoverable_ms is
  not wall-clock; the union math with the 961.5ms active / 282ms idle
  split is the honest wall-clock accounting, and its conclusions check
  out: 30x needs 599ms, idle-elimination alone caps at ~18.7x, so ~38-40%
  active-time reduction is also required.)
- Whole-trace API counts (2,973 cuStreamSynchronize + 448
  cuStreamWaitEvent, 1,135 D2H calls across BOTH the sequential and
  parallel passes) are consistent in magnitude with the cited
  window-scoped 1,256 syncs / 544 D2H.

ENDORSEMENT: the DEC-anchored device carrier is the right next slice. It
is exactly the boundary this file already named ("`ProofState::Active`
still stores a concrete `RunningInstance`" / the ignored
`running_carrier`), now with quantified evidence: DEC child output
6.65MiB, RLC projected X 3.39MiB, CCS y_prime 2.43MiB D2H per the profile,
each dragging a fence chain. The five-step order (device child claims ->
truly-resident ResidentOnly -> shell owns device surfaces + transcript ->
next fold consumes the carrier -> materialize only at egress) is correct.

Guardrails:
1. CONTRACT LABELING. 14.12x is claims-only (TerminalClaimsOnly) with
   PREBUILT chunks — `build_sha256_e2e_fixture` precompiles
   `fixture.chunks`, so the per-fold recursive-compile host consumer
   (~10ms/fold, audited 07-05/06) is outside this window. On the real
   lifecycle each chain re-acquires that serialization (overlappable
   across chains, not free), and the full-audit contract adds 8 x ~44.6MB
   terminal export. Any 30x claim must name the contract it is measured
   on.
2. PER-FOLD COMPILE EXPORT MUST SURVIVE THE CARRIER. The
   `take_post_acc_digest_override` / `take_f_prime_post_summary` seam
   already feeds chunk k+1 compilation without full materialization —
   step 5 ("materialize only at final/parity/audit egress") must not
   regress it. The compile-facing export (acc digest + F' post summary)
   stays per-fold; it is small and host-bound by design.
3. SECURITY (CLAUDE.md discipline). Device-resident surfaces that never
   rematerialize per-fold still get verify-before-proof-export at egress:
   the deferred recomposition checks (y/X/c) must run against
   authoritative device data before any proof leaves the prover, digests
   remain compression not authority, and the transcript stays bit-exact
   Poseidon2. Keep `DecOutputMode::Full` as the parity reference and
   cross-check byte-identical on every slice, as has been the discipline.
4. FS-LEGALITY OF THE PERSISTENT ENGINE. "Batch across chain, round, and
   child dimensions" needs precision: CHAIN batching (8 independent
   transcripts) and CHILD batching are legal and are exactly what fixes
   the 1-10-block launches — e.g. fuse 8 chains' same-round one-block
   Poseidon2 kernels into one 8-block launch. ROUND batching within one
   chain is ILLEGAL on the spine (round i+1 coefficients depend on r_i);
   only challenge-independent prep crosses rounds. This is the same error
   class as the killed y_prime/NC overlap — do not re-derive it at the
   scheduler level.
5. SEQUENCING: carrier slice first, then RE-PROFILE before designing the
   persistent engine. Removing ~1,256 fences will reshape the concurrency
   picture (avg 2.18 concurrent kernels with 8 streams means submission-
   side serialization today — shared allocator/device mutex/host joins);
   the 22-26x estimate is extrapolated from a fence-dominated profile and
   should be re-derived from the post-carrier one.

PROCESS NOTE: 2026-07-09 produced a dozen gpuprof artifact dirs (baseline,
device-fold-output, top-kernel-ncu, async-scratch, fe-tail-workspace,
zero-sync-alloc v1/v2/backtrace/final, fullgpu single+multichain8) but no
loop-log or plan entries since 07-06. Backfill the ledger before starting
the carrier slice — the accept/reject history is the collaboration's
memory, and several of today's attempts (the alloc-sync family) look like
they discovered things worth recording.

---

REVIEWER VERDICT (2026-07-09 evening) on the landed DEC-anchored carrier
slice: **verified and accepted — this is the cleanest slice of the
campaign — with three flags that need action before the next slice.**

Independently verified:
- Final profile parity line: "8 independent sha256 chains byte-identical
  (TerminalClaimsOnly); cpu aggregate=17754.51ms sequential
  cuda=2741.35ms (6.48x) parallel cuda=1203.92ms (14.75x)". Numbers match
  the report. Single-chain full audit 349.65ms/6.29x is also a new best
  on the LATENCY contract (from 449.5ms/4.85x) — the carrier pays on both
  contracts.
- The previously-dead seam is live: `adapter.rs` now calls
  `device_output_from_carrier(running_carrier)` (~:576) instead of
  discarding it. `fold_output.rs` is a well-shaped deep module:
  `DeviceFoldOutput` owns device K-surfaces + host claim shells + parent
  authority + accumulator digest; `materialize_claims` reconstructs full
  `CeClaim`s only at egress, cached.
- Guardrail 3 (verify-before-export) HELD: `verify_reconstruction` still
  runs in-prove (`reduce/dec.rs` ~:280) and the deferred y/X/commitment
  recomposition checks still run in the adapter (~:1065/:1091/:1100).
- The closed-form `reduce_u128` is correct: standard Goldilocks fold
  (2^64 = 2^32-1, 2^96 = -1 mod q); I checked the borrow/carry
  corrections cannot double-wrap and the single conditional subtraction
  canonicalizes. Host parity test covers boundary/noncanonical operands
  (0, 1, q-1, q, u64::MAX). SUGGEST: add explicit `reduce_192` boundary
  triples (max lo/mid/hi) to `tests/goldilocks.rs` — one cheap case.
- Digest-on-CUDA: the standalone park ("p2_hash_fields +97ms",
  "device digest -> rho bind +127.9ms") was legitimately retried here
  because the park was STANDALONE and this landing is bundled with
  cooperative Poseidon + the closed-form reduction, which change the
  economics. Security posture is sound: the device digest is prover-side;
  the verifier recomputes from authoritative claims, so a wrong device
  digest gives an INVALID proof, not an unsound one, and byte-parity pins
  it to the CPU digest. ASK: record in the ledger whether digest-on-CUDA
  was A/B-measured inside this slice or just carried along — the parked
  family needs a closing entry either way.

Three flags:
1. RESIDENCY GATES: the final profile has THREE red gates —
   fold.ingest.running h2d 3.188MB vs 1.0 budget (presumably the packed
   accumulator-plan upload: that is the intended "one upload" boundary,
   so recalibrate the budget rather than eat a standing red),
   fold.commit.fresh d2h 1.759MB vs 1.0 (was 0.688 — MORE THAN DOUBLED,
   explain this one in the ledger), and pi_ccs.sumcheck d2h 1.1MB vs 0.25
   (was 0.944). Standing red gates normalize ignoring gates; recalibrate
   or explain each before the next slice.
2. LEDGER: still no loop-log or plan entries since 07-06, now covering
   two days of slices INCLUDING this accepted one. Backfill is overdue —
   at minimum: this slice's accept entry, the alloc-sync family findings,
   the digest-on-CUDA park closure, and the gate explanations above.
3. COMMIT CHECKPOINT (user decision, flagged to the user): the entire
   `crates/neo-prover-cuda/` crate is UNTRACKED (`?? crates/neo-prover-cuda/`)
   and the neo-fold-clean seams are a large uncommitted diff. Eight days
   of campaign work at 14.75x exists only in the working tree. This is
   the natural checkpoint to commit (DCO sign-off per repo policy).

On 30x: agree with "investigate, don't promise." The listed path (child
commitments + Pi_CCS header/instance authority on device, NC launch
collapse, DEC fan-out concurrency) is consistent with the standing
guardrails. One restatement for the NC collapse: 6,927 NC launches shrink
via CROSS-CHAIN same-round fusion (8 chains' round-i kernels -> one wider
launch) and via challenge-independent prep — never via round fusion
within one chain's transcript. Same FS rule as always; it is the error
class this file has now killed twice.

---

REVIEWER ANALYSIS (2026-07-10) of the "839ms -> 709ms / 25x" three-step
strategy: **[THE BASELINE SECTION BELOW IS WRONG — see the CORRECTION
RECORD at the end of this file. 838.98ms/21.20x IS a real, byte-identical
measured wall time, recorded only in the raw artifact dir
`20260709T-sparse-form-y-eval-artifacts/stdout.txt`, which my artifact
search missed. The three-item endorsements, guardrails, and process asks
below still stand.]**

THE BASELINE DISCREPANCY (superseded — kept for the record):
- No artifact on disk shows 839ms or 21.2x. The best recorded state is
  `20260709T-dec-binary-bitmasks-multichain8.json` (05:18Z): parallel
  cuda **1,092.45ms = 16.32x**, byte-identical, cpu agg 17,830.96ms.
- 839ms is the GPU ACTIVE UNION of that profile's parallel window, not
  its wall time. I recomputed the union independently from
  kernel_enqueue_attribution intervals: ~852ms busy union in the
  parallel pass (methodology delta vs 839 ≈ 1.5%). 17,831/839 = 21.25x —
  that is where the "21.2x" comes from.
- Consequence: the waterfall "839 - 60 - 45 - 25 = 709 -> 25.1x"
  subtracts (mostly active-time) savings from a number that ALREADY
  assumes zero idle. On the honest ledger: wall 1,092 = union 839 +
  idle ~253. If all three items land at face value, union drops to
  ~754ms and the carrier removes some idle (~45ms of fences), leaving
  wall ~= 754 + ~208 = **~960ms ~= 18.5x** — NOT 709ms/25.1x. Reaching
  25x (<=711ms wall) additionally requires eliminating essentially ALL
  remaining host-induced idle in the parallel window. That may well be
  achievable — but it is a fourth work item, not a footnote, and it is
  the same submission-serialization problem flagged at avg concurrency
  2.18. Re-publish the waterfall based on wall time with idle as an
  explicit line.

ON THE THREE ITEMS (all endorsed):
1. Compact sparse forms: 17,055 active / 59,032 dense = 29% occupancy —
   skipping zero blocks is value-exact and parity-safe in principle.
   State in the ledger whether the active-block set is STATIC per matrix
   structure (index list built once per session — clean) or per-fold
   data-dependent (index build must itself be device-side/amortized).
   The overnight ncu artifacts (forms-partition2/4, forms-block-owned,
   forms-low-norm*) show deep forms work already happened — none of it
   is in the ledger.
2. Device claim/proof carrier (DeviceClaimAuthority + resident proof
   log): the natural completion of the landed DEC carrier, upstream
   through Pi_RLC/Pi_CCS. Standing guardrails apply unchanged:
   verify-before-export at egress, per-fold compile summary preserved
   (the diagram keeps "only tiny compile summary -> CPU F' compiler" —
   correct), terminal/parity egress byte-identical via DecOutputMode::Full.
3. Ready-job cross-chain Poseidon scheduling: the design as drawn is
   FS-CORRECT — "chains do not wait until all eight reach the same
   point; only already-ready independent work is combined" is exactly
   the legal cross-chain same-kind fusion, not a barrier. Two
   implementation cautions: (a) the dispatcher is a host-side
   coordination point across 8 chain threads — the submission path is
   already the serialization suspect; measure that it does not add more
   host serialization than it removes GPU underfill; (b) keep the
   batching policy GREEDY (launch immediately with whatever is queued) —
   any wait-for-width window is a soft barrier and adds latency to the
   earliest-ready chain.

PROCESS — now at three strikes, please fix before the next slice:
- The ledger has not been updated since 07-06. There are now ~20
  attempted slices (07-09 day + overnight: poseidon-four-tiles 1275.0,
  device-dec-sticky-status 1167.9, nc-split-clean 1193.4,
  p2-eight-warps 1201.5, digest-plan-carrier 1181.1, toom3-preeval
  1174.7, claim-digest-true-batch8 1295.9 REGRESSION,
  dec-binary-bitmasks 1092.5 BEST) with no accept/reject entries. I can
  no longer tell which attempts are IN the working tree vs reverted —
  the tree state is unlabeled, which breaks auditability of every
  number.
- This message's headline (839/21.2x) is the exact failure mode the
  ledger exists to prevent: an unauditable number. Label metrics as
  wall vs union in every report.
- The crate is still untracked in git; commit checkpoint remains
  recommended (user decision, DCO sign-off).

---

CORRECTION RECORD (2026-07-10, review session): **my baseline audit in
the entry above was WRONG; the loop's rebuttal is correct and I verified
every item of its evidence directly.**

What I verified before conceding:
- `20260709T-sparse-form-y-eval-artifacts/stdout.txt` reads exactly:
  "8 independent sha256 chains byte-identical (TerminalClaimsOnly); cpu
  aggregate=17786.89ms ... parallel cuda=838.98ms (21.20x)". The artifact
  EXISTS — as a raw artifacts dir only (no top-level gpuprof JSON, no
  gpuprof-history.jsonl entry), which is why my search over JSONs +
  history missed it.
- The timer is genuine host wall: `online_start = Instant::now()` after
  the start barrier, blocking on all 8 completion signals, elapsed ->
  `parallel_wall_ms` (`gates/e2e.rs` ~403-409, printed ~718). Not a union,
  not a window metric.
- The sparse profile's GPU active union is 677.94ms — I recomputed it
  from the raw SQLite (CUPTI kernel intervals, parallel pass after the
  2,083ms inter-pass gap): **677.94ms exactly matches the loop's number.**
  So the correct decomposition is wall 838.98 = active 677.94 + gap
  161.04, and the three-item budget (677.94 - 85 active, 161.04 - 45
  gap ~= 709) is coherent as a TARGET ALLOCATION — the loop itself
  correctly labels it that, not a forecast; every slice re-verifies
  against total wall.
- The active-block set is STATIC per matrix structure, resolved:
  `DeviceBarMatrices::upload` (`ring_forms.rs` ~75-97) derives
  `active_blocks` from CSR block offsets once at upload and fingerprints
  the structure. My question is answered; the clean design is already in
  place.

What went wrong in my audit (so it is not repeated): my artifact search
covered top-level `gpuprof/*.json` and `gpuprof-history.jsonl` and
treated them as complete; the sparse run existed only as a raw artifacts
dir — which my own `find -newermt` output had listed and I did not open.
Having failed to find the number, I chose "the number is mislabeled" over
"my search is incomplete" on the strength of a numerical coincidence
(union of the OLDER dec-binary-bitmasks profile ~852ms ~= wall of the
NEWER sparse run 838.98ms, within 1.5%). Two different profiles. A ~1.5%
match is not confirmation. Direct evidence was one file-open away.

Withdrawn: the "~960ms/18.5x honest projection" and the "fourth item =
eliminate essentially all idle" framing — both were computed off the
stale profile. The correct residual: after -60ms active from compact
forms, carrier + Poseidon scheduling must remove ~68ms of the 161ms gap
(~42%); if they fall short, broader submission scheduling becomes the
fourth item. That is the loop's framing and it is right.

Still standing from my entry: all three item endorsements and guardrails
(greedy dispatcher, host-coordination profiling, verify-before-export,
per-fold compile summary), and the process asks — which this incident now
PROVES rather than merely argues: the sparse result was invisible to
audit because it never reached the ledger, history, or a top-level JSON.
The loop's proposed repair is the right minimal fix: one canonical
`gpuprof --json` run of the current tree + one accepted sparse-schedule
ledger entry, then proceed with compact forms.
