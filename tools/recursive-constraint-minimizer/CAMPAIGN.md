# Constraint-minimization campaign state

One row per completion-bar item. A bar item becomes `met` only after a fresh
evaluation subagent fails to refute it. Evidence paths are relative to the
repository root unless stated.

| Bar | Item | State | Evidence |
|---|---|---|---|
| 1 | cvc5 with finite-field support; three gate tests pass un-ignored | **met** (refutation failed, 2026-08-15 iter 1; re-verified on Linux, iter 13) | cvc5 1.3.4 (Homebrew, cocoa: yes). Gates green live and un-ignored (commit 869f94e98: exactly three `#[ignore]` lines removed, nothing weakened). Independent verifier reran the suite: 38 passed, 0 failed; the three gates execute real solver runs with hard outcome asserts. Linux re-verification (iter 13): official `cvc5-Linux-x86_64-static-gpl` 1.3.4 release binary in `~/.local/bin` (cocoa: yes); both nebula gates green (y_ring Unsat in 1,073 ms; refinement control Inconclusive with pending row 8,665) and the terminal gate green (pending row 56,700, matches README). |
| 2 | Production profile frozen with pinned digests | **met** (refutation failed, 2026-08-15 iter 13) | User decision 2026-08-15: option (a), campaign profile v1 on the minimal shape; re-run classification when the production regime lands. `PROFILE.md` pins source digests, final-plan digest, terminal digests, and geometry for both arms and the terminal relation; non-ignored drift gate `campaign_profile_v1_digests_are_frozen` (13 s) re-derives all pins from fresh audits. Measured: source digests are plan-seed-independent (0xDA vs 0xD9); the final plan digest binds to the 0xDA mirror shape; all pins byte-match the committed Lean mirrors. Rejected regimes stay recorded: λ=125 fails the extension-policy census (114 bits available); λ=114 audit construction alone ran >2 h 06 m. |
| 3 | Seeded-Phi81 sampler equivalence proved | in-progress: Rust side complete, Lean checks await the build | Three production-class conformance fixtures committed (`SeededPhi81ConformanceArtifact.lean`: width-41 uneven multi-chunk, kappa-2, one-rejection seed 0xC3/counter 79,842,272) with a neo-fold-clean drift gate; `tests/SeededPhi81Conformance.lean` native_decides the Lean sampler against them. The bridge mirror (`lean_sampler_mirror.rs`) transcribes the Lean cursor rules over rand_chacha, matches all fixture classes, and matches every one of the frozen profile's 36 production seeded blocks term-for-term (10.9 s; zero replacement draws in production). |
| 4 | Checked bootstrap-recursive assignment committed | **blocked: freeze amendment needed (ask-first)** | Defect NAMED (2026-08-15, 6,142 s full-capture rerun): the second `append_segment_with_constraint_witness_audit` fails the paper's Definition-14 RLC guard — `ΠRLC norm bound violated: count·T·(b-1) = 3·216·1 = 648 must be < b^{k_rho} = 4`; minimum k_rho for count=3 is 10. This is not a code bug: campaign profile v1 (k_rho=2) cannot execute ANY fold, so no honest recursive assignment exists for the frozen shape. The freeze must move to a foldable shape (k_rho >= 10; paper B.2 uses 14) — a bar-2 amendment the user must decide. A k_rho=10 probe (arm sizes, digests, then a 2-segment capture attempt) is running to de-risk that decision. The persisting capture test is ready for the amended profile. |
| 5 | Every census family classified; zero Inconclusive | in-progress: base 6/6 and terminal 8/8 Lean-certified, **both refutation-proof**; recursive pipeline built, awaiting bar 4 | The complete recursive source relation is now a committed Lean authority artifact: string-payload CSR + 527-value table + 36 compact seeded blocks (197 MB payloads, emitted with a full 4.5 M-row replay against the independent recovery, 33 s). Artifact-level `*_of_full_valid` theorems added; the census runner (`recursive_census.rs`) will emit one compact necessity module per family from the shared assignment plus one column override. Base pilot flows end to end (BaseCompactSourceArtifact + BaseCampaignAssignment + BaseCompactStepInitialNecessity), pinned to the committed literal artifact by native_decide equality. y_ring stays the redundancy leg via its scalar certificate. |
| 6 | Removals applied; relations regenerated; fixed point re-solved | unmet | Blocked on 5. |
| 7 | `normalizedFullPlanInclusionMinimalSound` instantiated; ledger zero Open | unmet | One Open entry: `hypernova.recursive_size_closure` (`bridge/src/obligation_ledger.rs`). |
| 8 | Planted redundant + necessary controls flow end to end | in-progress | Necessary leg exercised at real scale: witness -> replay -> generated module -> lake green for six families. Redundant leg: the y_ring compact redundancy module is now emitted and committed (`RecursiveNifsPiRlcVerifyPaddingYRingRedundancy.lean`: 1,120 exact candidate/support rows, native_decide validity against the expanded compact artifact, Artifact-level full-theorem transport); its lake check rides the running build. |
| 9 | Rust, Lean, drift, axioms suites all green | in-progress | Rust suites green (core 18, bridge 39 incl. gates). Lean build not yet run this campaign. |
| 10 | Cost report before/after per relation | unmet | Instrument exists (`emitter_order_constant_affine_run_census_is_exact`). |

## Resource budget (hard, user-set 2026-08-15, updated for the Linux box)

This box has 62 GB of RAM and 16 cores. User directive (2026-08-15, second
environment): use as many CPUs as needed, but never more than 3/4 of total
RAM (~46 GB). A ~200 GB parallel Lean elaboration (~20 workers x ~10 GB on
16 MB generated modules) hard-rebooted the previous machine. This repo's
`lake` has no `-j` flag; throttle Lean worker count with CPU affinity
(`taskset -c 0-2 lake build` gives 3 workers, ~30 GB worst case on the
17-19 MB generated Data modules) and check `free -g` before every heavy
launch. A third-party sglang server holds ~7 GB on this box; respect it.

## Strategy notes

- Redundancy search is healthy: unsat slices stay small (the y_ring production
  family confirms in 290 ms live).
- Necessity search at family granularity on the chained F-prime arms behaves
  like near-full-relation solving: the batched slice reaches solver-hard size
  after one iteration (gb times out at 60 s on ~500-row slices; census #2:
  6/6 inconclusive with the bottleneck moved from iteration count to
  per-query time). If the 300 s solver-mode probe also fails, the plan of
  record is prover-level tampered-witness generation per family (mutate one
  targeted value, recompute honestly downstream, capture the assignment) —
  it produces the counterexample model directly and feeds the same
  fail-closed replay and Lean pipeline.
- Measured 2026-08-15: the full-context one-shot necessity query times out at
  the 300 s cap (gb), and the recursive-arm exact-duplicate sweep (82
  families, 45 min) finds exactly one fully-duplicate family — y_ring. So:
  redundancy batch 1 = y_ring only; necessity moves to witness construction.
  First generic constructor: exclusive-column counterexamples (a column read
  by only one family is mutated to break that family; no other family reads
  it, so the mutated accepted assignment stays valid everywhere else). Full
  replay and the Lean checker remain the authority for every such witness.
- Measured 2026-08-15 (later): the complete terminal artifact renders to a
  324 MB single module — not elaborable under any cap (base: 27 MB, 9.4 min).
  The emitter needs multi-file output (data submodules of chunk defs plus one
  assembly module with the theorems) for the terminal and recursive arms.
  Module-stem naming also keeps underscores (Public_projection); fix with the
  multi-file change.
- Witness census status: base 6/6 (~25 ms each), terminal 8/8 (5 s total).
  Recursive arm blocked on the bar-4 capture defect. Lean emission of the
  base batch is on its fourth design (chunked + hoisted + maxRecDepth 65536,
  validated standalone on the exact failing chunk).
- Measured 2026-08-15 (late): the exclusive-column constructor classifies 6/6
  base families as necessary in ~25 ms each. Lean emission at real scale
  requires chunked list definitions: a monolithic 40k-row literal fails on
  recursion depth regardless of heartbeat budget; 256-element chunks joined
  by List.flatten are the emission contract now. The 4.5 h bootstrap capture
  attempt failed at the second segment append; reason unknown until a rerun
  with full output capture.

## Bar-2 amendment evidence (k_rho=10 foldable shape, measured 2026-08-15)

The frozen v1 shape (k_rho=2) cannot fold (Definition-14 guard). Measured
k_rho=10 minimal shape (identical construction otherwise):

| Relation | k_rho=2 (frozen v1) | k_rho=10 (candidate v2) |
|---|---|---|
| Base arm | 39,949 x 38,626, digest `54bec6fa...` | same geometry, digest `acc3f180...` (coefficients change) |
| Recursive arm | 4,530,315 x 4,480,464, digest `4c0a5164...` | 9,857,455 x 9,759,794, digest `edaf7a51...` |
| Selective fixed point | 1,415,271 x 6,559,326 | 3,216,103 x 11,969,802, plan digest `f87c4841...` |
| Terminal | unchanged (paper-B.2 fixture) | unchanged |

Family counts stay 6/82. Consequences of amending: terminal 8/8
certification survives; base 6/6 re-emits and re-certifies (mechanical:
capture, witnesses, emission, one base-batch Lean rebuild); all compact
recursive artifacts and the y_ring module re-emit through the pipeline
(~2x payload). The two-segment k_rho=10 capture probe is running to prove
foldability end to end; its success plus this table is the decision
package. Base step 0 already passes in 2.9 s at k_rho=10. Encoding
limits measured for the candidate arm: 45,123,011 nnz, still 527
distinct coefficients, max 7,562 terms per row, the same 36 seeded
blocks, zero geometric runs - the string-payload pipeline absorbs the
amended shape without any emitter change (~390 MB of payloads).

## Column-minimization lens (user directive, 2026-08-15)

The user re-anchored the goal on columns and directed that parameter
choices follow the papers' soundness and completeness requirements, not
preference. Consequences now in force:

- The bar-2 amendment is a derivation, not an ask: SuperNeo Definition 14
  requires `(K+k)·T·(b-1) < b^k_rho` with B.2 fixing `b=2, T=216` for
  Goldilocks. The open empirical question is whether the engine's RLC count
  includes the k_rho accumulator limbs (self-consistent minimum k_rho=12
  for one fresh claim) or stays k_rho-independent at 3 (minimum k_rho=10,
  as the engine's own "increase k_rho" fix implies). The running k_rho=10
  two-segment probe discriminates the two laws; the smallest paper-sound
  k_rho wins because the Pi_DEC accumulator carries k_rho full-width limbs
  and is the dominant recurring committed-column driver.
- Nebula Lemma 10 discipline: cost columns are the ACTIVE (nonzero)
  columns, since committing zeros is free in MSM and pay-per-bit Ajtai
  commitments alike. The bar-10 cost report must separate committed-active
  columns from total width.
- Row-family removals shrink columns only when the removed family
  exclusively owns columns. Measured column-ownership census (recursive
  arm, frozen shape): 4,480,464 columns; 4,320,440 (96.4%) exclusively
  owned; 160,024 shared; 0 unused. Top exclusive owners:
  `nifs.pi_ccs.padded_row.binding` 1,532,770 (34%),
  `nifs.pi_rlc.verify.projection_binding.sis_digest` 1,037,600 (23%),
  `nifs.pi_ccs.padded_row.output_digest.sis` 586,400 (13%),
  `fprime.recursive.step.accumulator.output_authority.child_digests`
  454,175 (10%), `nifs.pi_ccs.padded_row.prefix` 184,202 (4%). Four
  families own ~80% of all columns — all fold-verification plumbing
  (padded-row binding traces and Poseidon2/SIS digest projections). These
  are the protocol-side column targets; inclusion-minimal classification
  alone cannot remove them if they prove necessary, so the campaign's
  column deliverable is this ranked ledger plus exact per-family costs.

## Staged solver runbook (cvc5-focused lane, awaiting go)

Zero-compute staging for the user's "focus on cvc5" direction. On "go":
1. Export stage (one `cargo test` run, ~1 min, single process): write each family's
   bounded problem JSON for base (6), terminal (8), and recursive (82) censuses
   to `target/campaign-problems/`. Redundancy-direction slices need no
   accepted assignment, so the recursive arm is NOT blocked by bar 4 here.
2. Solver stage (pure cvc5, one process at a time, one core, <2 GB): for each
   family JSON run the CLI `check --ff-solver gb --timeout-ms 60000`;
   evidence JSON per family under `target/campaign-evidence/`. Worst case
   96 x 60 s ~= 100 min sequential; most queries return in seconds.
3. Certificate stage (in-process, light): derive + validate scalar
   certificates for every unsat; record candidates for the affine-support
   grammar extension where derivation fails.
No Lean builds, no parallel fleets, nothing above single-digit GB.

## Solver sweep result (2026-08-15)

All 88 base+recursive families swept with context-free removal queries
(gb, 60 s): 0 unsat, 26 sat (expected: removal without retained context is
almost always satisfiable), 14 unknown, 48 wall-clock timeouts. Conclusion:
no family is internally self-redundant, and blind whole-family queries are
exhausted. cvc5's remaining role is confirming Rust-proposed candidates with
support context (the y_ring pattern). Evidence: target/campaign-evidence/.

## Iteration log

- 2026-08-15 iteration 14 (compact pipeline): built and committed the full
  string-payload classification pipeline in one sitting — the complete
  recursive source artifact (32 payload modules + assembly with
  expand/coverage/exact-validation theorems), the base pilot (compact
  artifact + shared assignment + one compact necessity module, pinned by
  native_decide equality to the committed literal artifact), the compact
  redundancy renderer and the committed y_ring module, the persisting bar-4
  capture test, and the 82-family recursive census runner. Also added
  Artifact-level `redundant_of_full_valid` /
  `necessary_[normalized_]of_full_valid` to ConstraintMinimization
  (extension only; compiles clean). Operational measurements: complete
  binding-free recursive problem export peaks ~22 GB (the y_ring emission
  gate is therefore `#[ignore]`, run per campaign iteration); the compact
  emission replay covers all 4,530,315 rows in ~33 s. The bar-4 capture
  diagnostic and the full Lean rebuild (invalidated once by the core
  extension, restarted, 1 worker while the capture runs) are both live.

- 2026-08-15 iteration 1: installed cvc5 1.3.4 via Homebrew (CoCoA present, so no
  source build needed); ran the tool's positive and negative controls live;
  ran the three gate tests under `--ignored` (all pass), removed the three
  `#[ignore]` attributes, reran the full bridge suite green. Note the tradeoff:
  the bridge test suite now requires a local cvc5 with finite-field support.
  Next: bar 2 digest printer for the arm profile candidates, then the freeze
  document with a digest drift test.
- 2026-08-15 iteration 2: bar 1 met after independent refutation failed
  (verifier reran everything; commit 869f94e98 confirmed attribute-only).
  Bar 2 hit a real blocker: paper-B.2 λ=125 fails the extension-policy census
  (114 bits available) — recorded as an ask-first regime decision. Added the
  `profile_freeze.rs` measurement printers (`#[ignore]`). Both the λ=114 audit
  build and the two-segment bootstrap-recursive capture exceed the 300 s cap;
  both keep measuring in background for the decision record. Next: bar 8
  planted-control emission through generated Lean modules on a tiny staged
  relation, against the current renderer API.
- 2026-08-15 iteration 3: renderer API re-read — complete-artifact renderers
  emit the coverage theorem themselves and require audit-bound exports, so
  planted controls must ride a real (minimal-profile) audit. Added
  `campaign_probe.rs` with the first live per-family search census over the
  base arm (`analyze_nebula_branch`, gb, 20 s/query, 8 iterations) — running
  in background alongside the two bar-2/bar-4 measurements. The census output
  will drive the first certificate-emission batch and the regime question
  goes to the user once the λ114 numbers land.
- 2026-08-15 iteration 13 (recursive-artifact design): measured the frozen
  recursive source arm: A csc+seeded (9,529,982 explicit nnz + 36 blocks, 0
  geometric runs), B csc 7,516,607, C csc 4,245,757 — 21.3 M explicit terms
  total, far beyond list-literal Lean emission (HANDOFF's "~200k rows" guess
  is falsified; the arm is 4,530,315 rows). Translation-pattern compression
  fails (3.97 M distinct row patterns — row-varying constants), confirming
  README's grammar verdict. New measurements that fix the design: only 527
  distinct coefficient values (a value table + u16 indices), max 1,514 terms
  per row, and a standalone Lean prototype shows a 43 MB string-literal
  payload decodes (base64) and passes native_decide in 45 s / 0.75 GB —
  ~13x lighter than list literals. Plan of record for the complete recursive
  artifact: string-payload CSR + value table + seeded blocks, expanded
  natively in Lean into the existing `Artifact`/`BoundArtifact` types so the
  `*_of_full_bound_valid` theorems apply unchanged; coverage and exactness
  proved once in the assembly module; per-family counterexamples ride one
  shared string-encoded background assignment plus tiny per-column overrides.
- 2026-08-15 iteration 13 (continued): bar 2 marked met after an independent
  no-context refutation attempt failed on all six angles (commit a34a8cce;
  verifier confirmed pins equal the committed Lean mirror digests and traced
  digest selection-independence through `SparseProblemExporter::new` and
  `hash_final_plan`). The verifier's one minor note — family counts not
  asserted — is fixed: the drift gate now pins base 6 and recursive 82 family
  censuses. Also measured the frozen profile's seeded-block geometry for
  bar 3: 36 blocks, all in the recursive arm A matrix (18 x 108 rows with
  kappa=2, 18 x 54 rows with kappa=1), word width 41, 54 seeds total, 2,916
  seeded rows; dense conformance data would be tens of millions of terms, so
  the bar-3 design goes through sampler-semantic validation plus exact-data
  conformance fixtures per code-path class.
- 2026-08-15 iteration 13 (new Linux environment): installed the official
  `cvc5-Linux-x86_64-static-gpl` 1.3.4 release binary (cocoa: yes) — no source
  build needed; all three gate tests re-verified green live. Lean v4.30.0
  toolchain and the full mathlib olean cache fetched. Bar-2 decision received
  (option a): wrote `PROFILE.md` and the `campaign_profile_v1_digests_are_frozen`
  drift gate; measured that source digests are plan-seed-independent while the
  final plan digest binds to the 0xDA mirror shape; measured the recursive
  source arm at 4,530,315 rows x 4,480,464 columns. Bar-4 capture diagnostic
  rerun launched with RUST_BACKTRACE=1 and full output capture (running).
  Operational: complete recursive-arm exports need ~21 GB and were replaced by
  one-row exports for digest pinning (final plan digest and source digest are
  selection-independent).
- 2026-08-15 iteration 12 (wait state): the machine hard-rebooted at ~200 GB
  during the terminal batch build; the user set the hard 64 GB budget now
  recorded above. All campaign processes verified stopped. Every remaining
  lane now waits on the user: (1) bar-2 regime decision a/b/c; (2) bar-4
  capture diagnostic approval (~2 h, single core); (3) approval to rerun the
  terminal batch build throttled to `lake build -j2` (~20 GB peak, 30-45 min);
  (4) bar-3 sampler-equivalence proof work, which also needs Lean builds and
  therefore the same throttled-build approval. No heavy job launches until
  the user answers. Terminal mirrors and drift gate are ready and green;
  only the Lean elaboration of the terminal modules is outstanding.
