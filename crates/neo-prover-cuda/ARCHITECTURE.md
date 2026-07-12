# neo-prover-cuda architecture

## The axiom

**A proof generated on GPU is byte-identical to one generated on CPU.**
Internal parallelization may differ; outputs may not. Every design decision
below derives from this. The parity bin enforces it: each gate asserts
field-identical results against the canonical CPU implementation, and the
`nifs` gates assert full `NifsProof` byte parity (serialized Π_CCS sumcheck,
all claims, running instance, post-prove transcript state) across folds.
That parity validation is a gate, not the fast-path benchmark contract:
reported CUDA proving time should exclude CPU replay/byte-comparison once the
device-owned path no longer needs the host transcript to continue.

Bit-exactness is cheaper than it sounds because field sums are
order-independent: parallel reductions, different multiplication algorithms
(Toom-3 vs schoolbook), and skipped zero terms are all value-equal in exact
arithmetic. What is *not* free is challenge derivation — identical proofs
require identical Fiat-Shamir challenges, which requires identical hashing.

## Ownership map

**Host-owned:**
- Transcript authority. In parity gates, the host Poseidon2 transcript stays
  the canonical comparator. A
  bit-exact device mirror exists (`src/transcript.rs` +
  `kernels/poseidon2.rs`, constants regenerated from `neo-ccs`'s seed and
  pinned by the `transcript` gate + `neo-ccs`'s round-constants test), and
  the FE sumcheck can already run its rounds against it with device-sourced
  challenges (`ccs_fe` gate). Byte-identity is validated by replaying the
  device-logged absorbs into the host transcript in parity mode.
- Claim authority and validation: host transcript replay remains canonical;
  paper-layer validations and the small commitment mixes
  (`ajtai_rlc_mixer` / `ajtai_dec_mixer`) still run on the host. Π_RLC ρ
  sampling and CE surfaces have device paths, then replay/validation pins
  them back to the CPU proof semantics.
- Oracle bookkeeping in `neo-reductions` (challenge tracking, layout state).
- F′ witness compile (upstream of this crate, paid by both chains).

**Device-owned (bulk data, one kernel family each):**

| Phase | Device work | Host round-trip |
|---|---|---|
| Fresh commits | Ajtai ring mat-vec over the PP | commitments down (KB) |
| Π_CCS FE rounds | row-table eval + fold (ping-pong) | coeffs/round on the host-FS path; none on the device-transcript path (one bulk log download after the loop) |
| Π_CCS NC rounds | digit-table eval + fold | coeffs/round, small final state |
| Π_CCS Ajtai tail | forms build + `Y_eval` ring mat-vec | y values down (KB) |
| Π_CCS oracle tables | f-var row tables, carried-ME eval table | device-resident (built from resident witness planes) |
| Π_RLC | ρ sampling, CE surface combines (`X`, `y_ring`, `y_zcol`), witness mix `Σ ρ_i · Z_i` | small proof surfaces down; mixed witness stays resident for Π_DEC |
| Π_DEC | digit split, child `y_ring`/`y_zcol` eval, child commits | resident-only mid-chain (`DecOutputMode`); full download only at chain boundaries |

**Static device state (uploaded once per structure, session-cached by
pointer+shape fingerprint):** Ajtai PP, bar CSR (`DeviceBarMatrices`),
orig CSR (`DeviceRowMatrices`), kernel modules.

**Per-fold shared state:** one witness-planes buffer (`[K+k][cols·D]`,
engine order: fresh then running), uploaded once and lent to the Ajtai
`Y_eval`, the NC digit-table init, and the Π_RLC mix.

## Current state: NIFS adapter, hybrid → device-driven

`CudaNifsProver` (`src/adapter.rs`) implements `NifsProverAdapter` and runs
the full NIFS.P fold — Π_CCS with device FE/NC sumcheck backends threaded
through `pi_ccs::prove_from_parts_with_backends`, device Π_RLC rho sampling,
device CE-surface combines, a device Π_RLC witness mix, then device Π_DEC.
The design started as a deliberate hybrid (small protocol material crosses
PCIe freely, bulk data is resident) and is migrating toward device-driven
orchestration: the device transcript removes the reason for per-round host
round-trips.

**No performance numbers live in this file.** Measurements rot; they belong
to the profiler. For current numbers run
`python3 scripts/gpuprof/gpuprof.py run e2e_bench --repeat 3` (medians +
per-stage levers) and `gpuprof.py trend e2e_bench` (campaign trajectory,
backed by `benchmark-results/gpuprof-history.jsonl`). The roadmap with
expected payoffs is `PLAN-full-gpu.md`.

## Landed capabilities

- **Cross-fold plane residency:** Π_DEC's split planes are retained on
  device under the `cache_output_for_next_step` staging contract and become
  the next fold's running-witness planes (`compose_fold_planes` + the
  `plane_copy` kernel); identity is the child Ajtai commitments (binding).
  `DecOutputMode::ResidentOnly` skips the child download entirely mid-chain.
- **Π_RLC → Π_DEC device chaining:** the mixed witness never round-trips
  (`mix_planes_device` → `DecParentWitness::Device`).
- **Π_RLC CE surfaces on device:** rho sampling, `X`, `y_ring`, `y_zcol`,
  and `ct` derivation now use device-derived surfaces; host validation keeps
  the CPU proof contract canonical.
- **Device digit-table init:** the host digit-table build is skipped when
  the NC backend is active (deferred tables).
- **Resident oracle tables:** the Π_CCS F row tables are built from the
  resident witness planes and stay on device.
- **Device Poseidon2 transcript:** bit-exact sponge mirror with host- and
  device-buffer op streams (`p2_transcript_ops` / `p2_transcript_io_ops`);
  FE sumcheck rounds can absorb coeffs and source challenges entirely on
  device, with a post-loop log replay into the host transcript
  (byte-identity preserved; `transcript` + `ccs_fe` gates).
- **Cache identity:** static uploads key on the Poseidon2 CCS matrix digest;
  the Ajtai PP cache holds its source `Arc` (allocation identity).

## Structural facts that outlive any measurement

- Under host-driven Fiat-Shamir, one host round-trip per sumcheck round is
  the floor (challenges are sequential). The device transcript removes that
  floor; what remains is launch overhead, which command streams/graphs can
  amortize once nothing mid-fold needs a readback.
- CPU replay/byte comparison is a correctness gate, not production work.
  Until the device-owned chain can continue without the host transcript, the
  current hybrid timing still includes some replay barriers; removing those
  barriers is part of the migration, not a benchmark bookkeeping trick.
- The lifecycle e2e pays the host F′ witness compile on both chains; it is
  outside this crate. Overlapping it with GPU folding of the previous chunk
  (pipelining) attacks it without porting it.
- Rejected-experiment notes live next to the kernels they concern (see
  `kernels/ajtai.rs`): E-grouped mat-vec, shared-memory tiling, and the NC
  digit branching fast path all measured slower — do not retry without new
  evidence (e.g. ncu counters).

## Layout

Modules mirror the profiling taxonomy — the code layout *is* the data flow:

- `src/session.rs` — `DeviceSession`: device, kernel modules, Ajtai PP,
  static CSR matrices, scratch, retained child planes. State only.
- `src/ingest.rs` — plane staging: host flatten, fresh H2D, resident d2d
  composition (`fold.ingest.*`).
- `src/commit.rs` — Ajtai commitments (`fold.commit.*`).
- `src/reduce/ccs/{mod,fe,nc}.rs`, `src/reduce/rlc.rs`, `src/reduce/dec.rs`
  — the three SuperNeo reductions (`fold.superneo.pi_*`).
- `src/adapter.rs` — the orchestrator: `prove()` reads as
  ingest → commit → pi_ccs → pi_rlc → pi_dec → accumulate → egress.
- `src/transcript.rs` — device Poseidon2 sponge mirror (state buffer +
  absorb/challenge op streams, host- or device-sourced), seeded from host
  snapshots.
- `src/field.rs` — the host/device word boundary: canonicalizes `u64`
  words before they become field elements.
- `src/ring_forms.rs`, `src/ring_layout.rs`, `src/device.rs` — static
  matrix uploads, layout conversions, device basics.
- `src/kernels/` — `#[cuda_module]` kernel families + launchers
  (`poseidon2.rs` carries the transcript kernels).
- `src/lib.rs` — `perf_timed!` (stderr stamps + NVTX ranges via
  `perf_ranges`, feature `perf-timers`).
- `src/bin/parity/` — the gate runner. Run all:
  `cargo +nightly-2026-04-03 oxide run --features cuda --bin parity`
  (binary lands in the workspace-root `target/`); `quick` skips benches.
- `scripts/gpuprof/` — standalone phase profiler (`run`/`diff`/`check`/
  `trend`): NVTX-attributed per-stage tables, idle-cause split, physics
  floors + levers, kernel lint, residency gate, repeat medians, regression
  exit codes.
