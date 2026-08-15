# Constraint-minimization campaign state

One row per completion-bar item. A bar item becomes `met` only after a fresh
evaluation subagent fails to refute it. Evidence paths are relative to the
repository root unless stated.

| Bar | Item | State | Evidence |
|---|---|---|---|
| 1 | cvc5 with finite-field support; three gate tests pass un-ignored | **met** (refutation failed, 2026-08-15 iter 1) | cvc5 1.3.4 (Homebrew, cocoa: yes). Gates green live and un-ignored (commit 869f94e98: exactly three `#[ignore]` lines removed, nothing weakened). Independent verifier reran the suite: 38 passed, 0 failed; the three gates execute real solver runs with hard outcome asserts. |
| 2 | Production profile frozen with pinned digests | **blocked: regime decision, evidence complete** | λ=125 paper B.2: rejected by the extension-policy census (shape provides 114 bits). λ=114 paper B.2: audit construction alone ran >2 h 06 m and was terminated — impractical for iterated campaign use (the minimal shape builds in ~50 s). Options for the user: (a) freeze campaign profile v1 on the minimal shape and re-run classification once the production regime lands (recommended — the regime call is itself still open at the protocol level), (b) λ=114-capped paper B.2 with multi-hour audit cycles, (c) wait for a protocol-side census fix to 125 bits. |
| 3 | Seeded-Phi81 sampler equivalence proved | unmet | README next-work item 1. |
| 4 | Checked bootstrap-recursive assignment committed | unmet, capture defect | The staged capture run failed after ~1.8 h inside the second `append_segment_with_constraint_witness_audit` call (`bridge/tests/nebula.rs:73`); the panic message needs a targeted rerun with full output capture. Until then bar 4 and the recursive-arm witness census are blocked. |
| 5 | Every census family classified; zero Inconclusive | in-progress: base 6/6 and terminal 8/8 Lean-certified, **both refutation-proof**; recursive arm blocked on bar 4 | First certified family: `nifs.pi_rlc.verify.padding.y_ring` (1,120 rows, in-house scalar certificate; live cvc5 confirms Unsat). Lean module emission not yet run. |
| 6 | Removals applied; relations regenerated; fixed point re-solved | unmet | Blocked on 5. |
| 7 | `normalizedFullPlanInclusionMinimalSound` instantiated; ledger zero Open | unmet | One Open entry: `hypernova.recursive_size_closure` (`bridge/src/obligation_ledger.rs`). |
| 8 | Planted redundant + necessary controls flow end to end | in-progress | Necessary leg exercised at real scale: witness -> replay -> generated module -> lake green for six families. Redundant leg (y_ring certificate through Lean) still pending. |
| 9 | Rust, Lean, drift, axioms suites all green | in-progress | Rust suites green (core 18, bridge 39 incl. gates). Lean build not yet run this campaign. |
| 10 | Cost report before/after per relation | unmet | Instrument exists (`emitter_order_constant_affine_run_census_is_exact`). |

## Resource budget (hard, user-set 2026-08-15)

Never use more than 64 GB of memory. A ~200 GB parallel Lean elaboration
(~20 workers x ~10 GB on 16 MB generated modules) hard-rebooted the machine.
Every Lean build of generated modules runs `lake build -j2` (never above
`-j4`), one heavy job at a time, with a `ps` load check before launching.

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
