# Selected NIFS execution and HyperNova closure

Owner request: 2026-09-11. This is the combined task scope, including the
owner's correction after the app goal was created. Neither part below is
optional. The task ends only when its required claims have closing evidence,
or the applicable project stop rule requires an exact blocker report.

## Required results

1. Resolve the failed logical-assignment mutation check for `piCcsPayload`,
   `runningRoundC0`, and `runningRoundC1`. First identify each allocation,
   actual semantic consumer, correspondence theorem, and rejection evidence.
   Repair a missing binding if found. A change to unused allocation is a
   package change and needs preservation, emission, comparison, and new pins;
   it is not a test-only repair.
2. Run selected-profile PiCCS through the normal production matrix evaluator
   and witness path. Compute its results from actual matrices and witnesses.
   Prepared expected matrix results are comparison data, not prover inputs.
   Match the complete Lean phase result and required rejection cases.
3. Run the actual native NIFS prover and verifier through PiCCS, PiRLC, and
   PiDEC with the same selected inputs. Compare complete phase results,
   canonical proof bytes, transcript states, and final output; run the
   required mutations. Retained prepared-input tests alone do not close this.
4. Close the HyperNova lifecycle, recursive-size, and security Proof and Link
   records in group H. Reuse checked declarations where sufficient. Prove
   accepted predecessor reconstruction, the reverse base case, and history
   composition over the actual selected relation. Keep the actual NIFS
   extraction success and named failure events explicit.
5. Close the selected completeness claims with named theorems connecting
   valid admissible steps to satisfying selected assignments and accepted
   successor terminal proofs. Keep the successor counter range and bounded
   sampler availability explicit. Classify any unconditional perfect-
   completeness statement that the implementation cannot satisfy.
6. Update the local requirements map and evidence only for proved claims and
   checked consumer links. Keep execution evidence distinct from universal
   Rust proofs and from production-backend approval.

## Scope and assumptions

Use the existing Nightstream Goldilocks profile: `b = 2`, `k_rho = 16`,
`B = 65536`, one HyperNova slot containing 16 running claims, one fresh claim,
14 matrices, and Poseidon2 binding. Preserve the existing authority path.

Fixed-public-seed MSIS hardness and the approved parametric classical
Fiat–Shamir boundary remain explicit assumptions. No proof of MSIS hardness
or new general Fiat–Shamir theorem is required. These assumptions cannot
replace missing checks, witness construction, or value correspondence.

Keep use count, query count, and depth symbolic unless a deployment
requirement supplies values. The numerical deployment-security budget is a
separate open claim. State the constant-depth extraction scope and its exact
composition premises; a deterministic history implication is not an
efficient probabilistic extractor theorem. Extractor work uses the declared
mathematical clock.

Architecture work is required only when a demonstrated dependency blocks a
named result above. No general Layout/Export migration, primitive-library
migration, new protocol/profile, Stage 2, or proof backend is part of this task.

## Work and validation

One active implementation obligation names its owner, change, and closing
declaration or test. Independent source review can run in parallel. Keep one
Lean/Rust build queue. **Owner override for this entire session (2026-09-11):
the stop limit is ten rounds per obligation, replacing the repository's
three-round rule.** Attempts already made count toward ten. This is a standing
session instruction, not an exception limited to one declaration.
Reuse still-valid evidence. Do not repeat a failed expensive test to confirm
an unchanged known failure.

Before a Lean checkpoint, run `scripts/validate.sh static`, `build`, and
`axioms` in that order, under the project command caps. Pure refactors retain
package identity. Package changes require the affected exact matrix,
independent assignment, value, mutation, and identity checks before new pins.
Native tests use release mode and the optimized engine.

The first active obligation is the three-block classification. Its result
determines the smallest repair. The current source baseline is
`4fd19b00cf1e052ecbe9bd6923a3463ef186b36d`.

## First work checkpoint

The classification is recorded in `CONFORMANCE_FIXES.md`. All three blocks
are redundant retained copies. Their live consumers use existing proved
source forms. The failed allocation-wide mutation check remains unchanged.

The removal belongs to `RunningTransitionRetainedGeometry`, the existing
canonical assignment schedule, and the existing assignment transport codec.
It removes 1,249,352 logical coordinates. The calculated new logical width is
253,011,231 and its 54-aligned carrier is 253,011,276. These are forecasts,
not new pins. The physical allocation is expected to remain the same.
The smaller Ajtai key is a same-seed prefix; retaining the approved old-key
hardness premise requires the corresponding symbolic zero-extension
reduction. No such reduction or allocation change has been added yet.

The later output-digest recipe mutation also needs its real plan selector:
the current helper names block 27, while the current plan selects block 26.
This check follows the known failure. Repair it with the allocation change;
retain every live proof, transcript, output, and nonempty-block mutation.

Normal PiCCS evaluation first needs a selected package-to-evaluation-cache
connection: `LoadedPackage.ccs_structure_header` returns a matrix-free header,
and `build_superneo_eval_cache` rejects it. The current complete-oracle driver
does not supply that production constructor. Scope the first implementation
change to this exact missing operation before attempting a full native run.

`HyperNovaPredecessor.terminal_implies_predecessorOrCollision` constructs the
exact accepted predecessor from its source openings, or returns the existing
state-hash collision alternative. `terminal_one_implies_baseOrCollision`
recovers the first advice and bottom predecessor without source witnesses.
Independent source review by `/root/nifs_conformance` found no defect in
these statements or their actual-terminal connection.

`HyperNovaInput` passes its fourth focused check. `running_ofClaims` and
`fresh_ofClaims` preserve the exact typed claims in the existing selected
checker input. The earlier third draft and log remain at
`/tmp/nightstream-hypernova-input-draft`. The owner clarified the session-wide
ten-round limit before work resumed. The source-success and history consumers
must still use this conversion; a standalone conversion does not close them.

`HyperNovaSource.sourceReturned_iff_terminalHolds` and its `finishValue`
specialization connect actual returned source values to the exact prior CCS
and CE memberships. They preserve source order and reattach the verifier's
fresh public prefix. The source module passed on round six.

The checkpoint gates passed in order: static boundaries; full library
(3,839 jobs, 4 seconds); full axiom audit (3,929 jobs, 3 seconds). The seven
new public theorems are registered in `tests/AxiomsStage1Security.lean`.
Only `propext`, `Classical.choice`, and `Quot.sound` occur. Logs and source
copies are in `HYPERNOVA_PREDECESSOR_EVIDENCE.zip`. Package data, identity pins,
and Rust code did not change; no new native or emission check was needed for
this proof checkpoint. This is not full-history or native conformance closure.

The next proof consumes these results in an explicit reverse-history
algorithm. Completeness also needs the exact base-advice contract:
`Stage1.Formal.opsAt` places the C/R/D children even at the base step, while
`StepHoldsFor` ignores the base step's unused fresh/proof advice. The encoder
must construct canonical admissible dummy data or prove permitted
normalization of that unused advice. It cannot claim to preserve arbitrary
rejected base proof messages. Actual sampler availability and successor
counter range also remain explicit completeness conditions.

## Deterministic history checkpoint

`HyperNovaHistory.run` consumes the actual returned source values in reverse
order and returns forward-ordered application advice. It preserves unused
results and reports missing entries, source aborts, wrong envelopes and
counter/state mismatches. The base step consumes no source result.

`run_correct` proves exact advice length and final state under accepted
terminal membership, the source-success events at the visited inputs and
absence of the visited state-hash collisions. `HyperNovaInput`,
`HyperNovaSource` and `HyperNovaPredecessor` connect these inputs and values
to the existing NIFS return and terminal predicates. No choice operation
constructs the history. Independent review by `/root/fiat_shamir_model`
found no material defect in that scope.

The focused module passed on attempt five in 2.0 seconds. Static, full build
(3,840 jobs, 4 seconds), and axioms (3,930 jobs, 3 seconds) passed in order.
The axiom set remains `propext`, `Classical.choice`, `Quot.sound`. Evidence is
in `HYPERNOVA_HISTORY_EVIDENCE.zip`. No package data, pins or Rust changed.

The next security obligation is to generate the consumed results through
the existing guarded NIFS experiments and prove the exact event-law link.
Then compose the failure probabilities and declared clocks on that same
law. The deterministic history result is not a probability or runtime bound.

## Source-law and history-probability checkpoint

`VerifierCoinLaw` realizes the existing independent-uniform verifier mean as
a PMF. `SequentialOutputLaw` draws the original context, exact checked prefix
and its actual receipt's suffix. Its event theorem and context marginal are
proved. `HyperNovaSourceLaw` maps that law through the actual stored source
return and proves its decomposition into the unchanged context law and the
same fixed-context kernel. Aborts and captured suffix state are retained.
`NifsClosure.finishValue_probability_and_expected_work` now states its bound
directly on this constructed source-result PMF, with unchanged hypotheses
and declared-work conclusion.

`HyperNovaRealInput.realSuccess_of_terminal` derives the real NIFS event from
accepted recursive terminal membership, no current state-hash collision,
and positive decoded predecessor iteration. It uses the actual decoded
local proof and all current terminal child witnesses. No separate child or
output-correctness premise remains at this boundary.

`HyperNovaHistoryLaw.results` generates source returns under the supplied
state-dependent kernel. Its private structural counter is exactly the
statement iteration. The proved invariant excludes truncation; there is no
extra public depth limit. `accepted_probability_le` bounds initial accepted
mass by complete returned-history mass, the expected sum of visited source
failures from accepted initial openings, and encountered state-hash failure
mass. No independent-call or successful-trace premise is used. Independent
source review confirmed the final initial-acceptance indicator keeps the
original laws and performs no conditioning or renormalization.

The source-failure theorem passed on attempt five and the combined bound on
attempt seven. Earlier well-founded definitions were stopped for slow checks;
structural definitions check in 1.6 seconds. Static, full library (3,846 jobs,
47 seconds), and axioms (3,936 jobs, 46 seconds) passed in order. Only the
allowed axioms occur. Sources, logs and review scope are in
`HYPERNOVA_LAW_EVIDENCE.zip`. Package data, pins, and Rust are unchanged.

`PiCCS.Formal.completePrefix_of_accepted` also passed. It takes actual PiCCS
acceptance and input state binding, and derives the generated phase output
specification from the constructed rows. The existing completeness interface
is retained as a wrapper. The first missing environment fact is loading the
verifier's four expected-context words into their existing source slots while
preserving protocol input readback. Canonical-state framing and selected
whole-package completeness still follow that fact.

Remaining security work: instantiate the source kernel with the selected
NIFS pipeline at every visited input; apply the existing transfer, binding
and moment bounds to those actual visited-context laws; compose expected
source work and declare the reverse walk's own clock. An unconditional FS
model does not supply a point-mass or acceptance-conditioned model for free.
The numerical deployment budget and native conformance work remain open.
