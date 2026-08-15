# Campaign handoff

Read this first, then CAMPAIGN.md (ledger), COST.md (baseline), README.md
(tool design). Branch `claude/minimizer-campaign` holds all campaign work as
one squashed commit; the working history is in this machine's reflog only.

## Goal (condensed)

Classify every constraint family of the Nightstream Nebula F-prime verifier
(two physical arms + terminal relation) as Redundant or NecessaryForSoundness
with Lean-checked artifact theorems; apply Lean-authorized removals;
regenerate and re-verify the recursive fixed point. Lean is the only
authority; cvc5 and Rust replay are search evidence; fail closed always;
never weaken a theorem, test, validator, or coverage premise. Every bar claim
must survive an adversarial no-context subagent refutation before it counts.

## Verified done (both refutation-tested by fresh subagents)

- cvc5 gates live: Homebrew cvc5 1.3.4 has CoCoA (`--show-config` cocoa: yes).
  Three `installed_cvc5_*` tests run un-ignored.
- Base arm 6/6 and terminal 8/8 families Lean-certified NecessaryForSoundness
  via `necessary_normalized_of_full_[terminal_]bound_valid` over complete
  artifacts (committed mirrors under
  `formal/nightstream-lean/.../Artifacts/MinimizerCampaign/Generated/`).
- Drift gates: `bridge/tests/generated_mirror.rs`, `terminal_mirror.rs`
  (byte-compare mirrors against fresh emission incl. witness re-derivation).
- Axioms guard: `formal/nightstream-lean/tests/Axioms/MinimizerCampaign.lean`
  pins all 24 campaign theorems; tracked set only (Lean 4.30 emits
  per-theorem native_decide certificates, normalized to Lean.trustCompiler).

## Hard-won operational facts

- Lean emission contract (all in `bridge/src/lean_export.rs`): 256-element
  chunk defs; inner lists >256 hoist into their own chunked defs co-located
  immediately before the consuming chunk; splitter cuts only at emitter
  markers (`-- lean-split-safe`); one hoist counter spans sections; generated
  modules set maxHeartbeats 2000000 and maxRecDepth 65536; oversized modules
  split into leaf Data modules + one assembly that imports them. A locality
  rule is mandatory: no Data module may reference a hoistedList it does not
  define (validate before building; a grep suffices).
- Lean memory: one lean worker on a 16 MB generated module needs ~10 GB.
  This machine hard-rebooted at ~200 GB (20 parallel workers). This repo's
  lake has NO -j flag: throttle structurally by batching root targets per
  invocation. Estimate memory before every parallel build, whatever the box.
- Solver: gb is ~10x faster than split on these slices. Context-free
  whole-family queries are exhausted (88-family sweep: 0 unsat; 26 trivially
  sat; rest unknown/timeout at 60 s). cvc5's remaining role: confirm
  Rust-proposed candidates with support context (y_ring pattern, 290 ms).
- Redundancy reality: y_ring (1,120 rows) is the ONLY exact-duplicate family
  among 82 recursive families; scaled/affine duplicates need a
  polynomial-normal-form sweep (not yet written); the scalar certificate
  grammar cannot use the constant-one equation (extend with affine support +
  a Lean checker when needed).
- Necessity reality: solver search behaves like near-full-relation solving
  (one-shot full-context query times out at 300 s). The working constructor
  is exclusive-column witnesses (`bridge/src/witness_search.rs`): base 6/6 in
  ~25 ms each, terminal 8/8. Recursive arm needs an accepted assignment first
  (bar 4).

## Open blockers, exact

1. Bar 2 profile freeze — USER DECISION pending: (a) minimal-shape campaign
   profile v1 (recommended; re-run classification when the production regime
   lands), (b) lambda=114-capped paper B.2 (audit build alone >2 h here), or
   (c) wait for a protocol-side census fix to 125 bits. Paper B.2 lambda=125
   is rejected by the verifier's own extension-policy census (114 bits).
2. Bar 4 capture defect: the two-segment chain
   (`accepted_assignments_cover_and_satisfy_both_physical_source_arms`,
   `bridge/tests/nebula.rs`) panics ~1.8 h into the SECOND
   `append_segment_with_constraint_witness_audit` call. Error surfaces
   through `NebulaFPrimeChainError`
   (`crates/neo-fold-clean/src/frontends/nebula/f_prime/chain.rs:413`, enum
   at :57, ~39 error sites, mostly transparent wrappers). Rerun with
   RUST_BACKTRACE=1 and FULL output capture (no grep truncation) to name it.
   Blocks the recursive-arm witness census and bars 5/6/7.
3. Bar 3: Rust-to-Lean seeded-Phi81 sampler equivalence proof (seeded blocks
   ride into Lean artifacts as seeds+schedule; until proved, seeded-block
   certificates are not fully artifact-checked).
4. Bars 6/7/9/10 follow from the above; COST.md holds the committed baseline
   (base 39,949 rows -> 1,415,271-row fixed point; terminal 58,593 ->
   65,536 x 114,407 Spartan).

## Recommended order on a faster machine

1. Rerun the bar-4 diagnostic (full capture); fix needs owner sign-off if it
   touches protocol crates.
2. Get the regime decision; freeze PROFILE.md with digests + a drift test.
3. Recursive-arm witness census (exclusive-column) + Lean emission via the
   existing split contract; expect ~200k rows -> validate locality, then
   build with a memory-budgeted schedule.
4. Polynomial-normal-form duplicate sweep for scaled duplicates; route new
   unsat candidates through the scalar certifier; extend the grammar (with
   its Lean checker) only when a real candidate demands it.
5. Removal batches (y_ring first) -> regenerate -> re-census -> assembly
   theorems (`normalizedFullPlanInclusionMinimalSound`) -> close the single
   Open obligation (`hypernova.recursive_size_closure`) -> full-bar
   adversarial pass.

## Verification protocol

Never mark your own work done. Per bar item, a fresh no-context subagent gets
only the claim, paths, and commands, and tries to refute it. Two NOT-REFUTED
verdicts are on record (base batch, terminal batch); reproduce that standard.
