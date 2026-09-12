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
The smaller Ajtai key is a same-seed prefix. The symbolic zero-extension
reduction is now checked in `AjtaiSetupV1.Prefix.extendShortKernel`; it retains
the strict bound and appends zeros after the smaller complete carrier.
The allocation change and its new pins have not been made.

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

## Guarded-call and local witness checkpoint

Checked code cut: `17c0c64a`. `HyperNovaVisitedLaw.visitedDraw_marginal`
and `HyperNovaGuardedSourceLaw.law_eq_guardedDraw` connect each observed
history call to the selected guarded NIFS source law with the same
continuation. The operational history still draws on false-mark paths;
only the reported guarded experiment is masked. The first source-failure
mass is the exact difference between good-active mass and source-success
mass. Acceptance from the mark, the per-visited-law model instantiation,
and declared history work remain to be composed.

`PiCCSProtocolCompleteness.completePrefix` constructs the local PiCCS
witness prefix from actual accepted typed inputs. It loads the existing
verifier-context slots, preserves protocol input readback, derives canonical
state framing, and derives the generated output specification. It does not
yet construct the pilot/hash/application or complete selected package.
`BaseCompleteness.zeroProof_piCcsCheck` supplies the base dummy's actual
PiCCS acceptance without a fresh-opening or sampled-coin premise. Its D
acceptance still needs actual sampler availability and a derived public bound.

The static, library (3,853 jobs, 1 second), and axiom (3,942 jobs, 47 seconds)
gates passed in order. The prior uncached library check took 333 seconds.
Only the allowed axiom set occurs. Evidence and source hashes are retained
in `HYPERNOVA_GUARDED_CONTEXT_EVIDENCE.zip`. Independent review found no
defect in the local PiCCS constructor. Package data, pins, and Rust sources
are unchanged by this checkpoint.

## History probability and native cache checkpoint

Checked code cut: `005e9679`. `HyperNovaVisitedAcceptance` derives acceptance
from the actual history mark. `NifsProviderLaw` fixes the raw calls, tapes
and clocks before choosing any visited law, then proves equality with the
selected supported-provider extension. `HyperNovaVisitedSecurity.history_probability_bound`
composes the first-failure bound over fixed symbolic depth with shared
`g`/`deltaFS`, the exact guarded model instances, actual MSIS reduction masses,
and marked hash events. No source-law or checker-correctness premise remains
at this boundary. Numerical advantage/query bounds remain external.

`HyperNovaHistoryWork` counts every actual source call, including abort and
false-mark paths. Its declared orchestration allowance counts one initial
entry and one processed source return, with expectation at most `D + 1`.
This excludes payload decoding, copying and advice evaluation. The existing
NIFS source clock still needs composition on the unconditional operational
call laws. The allowance alone is not a complete work or machine-time claim.

The canonical base dummy now passes the complete NIFS verifier under actual
sampler availability, with its parent public bound and D checks derived.
The sampler and local R constructors also derive generated outputs from
available executions. The canonical C → R bridge and whole selected-package
assignment construction remain open.

The selected Rust cache now comes from the actual sealed row stream. A
first pass counts storage, exact reservations avoid the measured excessive
buffer growth, and a second pass validates every row and coefficient. The
test executes the actual base witness and logical transport, then matches all
14 × 54 Lean matrix values. It passed in 197.77 seconds of test time,
249.77 seconds including compilation, with 26.90 GiB peak RSS. Normal PiCCS
prover integration and full native C → R → D remain open; the three retained
unused blocks and their failed mutation gate are unchanged.

Ordered static, full library (3,859 jobs, 47 seconds), and full axioms
(3,948 jobs, 47 seconds) passed. Evidence is in
`HYPERNOVA_SECURITY_EVIDENCE.zip` and `NIFS_NATIVE_CACHE_EVIDENCE.zip`.
Rust constructor/rejection tests and formatting passed. Package bytes,
identity pins and the cryptographic assumptions are unchanged.

## Operational source work and C/R checkpoint

Checked code cut: `2b84e0fa45678509a15229fd9cd2673ba4e6c0db`. `HyperNovaSourceWork.expected_work_polynomial_bound`
composes the actual operational source clock over unconditional visited laws,
including abort and false-mark paths. Context/kernel equality and source-return
value agreement are proved. Primitive/storage bounds and finite call moments
remain explicit. The added control allowance excludes payload decoding,
copying and advice evaluation; this is not a machine-runtime claim.

`PiRLCProtocolCompleteness.completePrefix` now constructs canonical C/R
prefixes from actual typed input and sampler availability. It preserves C
rows and derives the production parent and challenges. D, the surrounding
application/state rows, complete selected assignment and honest outer prover
remain open. Normal native PiCCS testing and the unused-allocation repair
are active; their gates have not passed at this cut.

Ordered static, full library (3,861 jobs, 4 seconds), and axiom (3,950 jobs,
3 seconds) gates passed. Evidence: `HYPERNOVA_SOURCE_WORK_EVIDENCE.zip`,
SHA-256 `c367847f1ad9705951ab04a2629549527756f7cd7e066edde4720ffb0536a479`. Package bytes and pins are unchanged.

## Initial envelope and D source checkpoint

Checked cut: `5e7a23ae8f777314cd3b8eaf48a04a821f7d9989`. `HyperNovaInitial.initial_accepted` constructs the
selected initial statement and bottom proof from an initial state of the
fixed public width. Iteration zero, valid counter and equal endpoints are
derived. `PiDECProofInputs` loads the actual D messages and verifier public
digits, proves exact typed readback and preserves every source outside its
existing input interval. It does not assume or prove D acceptance, child
openings or full selected rows. Those remain with the canonical consumer.

Ordered static, full library (3,863 jobs, 3 seconds), and axioms (3,952 jobs,
3 seconds) passed. Evidence: `HYPERNOVA_INITIAL_DEC_INPUTS_EVIDENCE.zip`,
SHA-256 `cd73aad8904691032f084124164f5ceb1f7289486a5fdddf0e52cf509a009d30`. Package pins are unchanged at this cut.

## Recursive step and envelope checkpoint

Checked cut: `23132ebbaa0caced57e1cacbd1998d3952ab2710`. The local C/R/D constructor now takes an actual
valid positive semantic step. Proof/fresh readback, input framing, verifier
context, sampler availability, parent bound and exact D output are derived
from that step and its accepted advice. Pilot/application/transition rows,
complete selected low-norm assignment and the honest next envelope remain
open. Sampler reject/position Boolean facts now follow from their actual
R rows; the selected norm consumer is still separate at this cut.

The accepted envelope's dense field-word bound includes all claim fields
and complete opening domains and is independent of iteration. It does not
claim a Rust wire format or execution bound. Existing checked declarations
also close the model records for fixed application selection, one-based
output pc and 270 logical public words. Their Rust axes remain open.

Ordered static, library and axiom gates passed. Evidence:
`HYPERNOVA_STEP_CONSTRUCTION_EVIDENCE.zip`, SHA-256 `a1579300f77d59dd9eb506a056a0555aea95c2a4f16b421ef266e3814af0853e`.
Actual split/opening/fixed-key commitment primitives also passed their
focused native checks; evidence `NIFS_NATIVE_D_PRIMITIVES_EVIDENCE.zip`,
SHA-256 `5b2033c0008325996892152c2d896915bbe006139c0f61a96d6090d1ed400dcc`. Complete native C/R/D remains open.
Package data, pins and cryptographic assumptions are unchanged at this cut.

## Allocation repair and native execution checkpoint

The allocation repair is committed as `dd38a22f` and merged at
`a62cebc2d3263087985d2e0c5666df8fc5715ba4`, which is on the remote proof
branch. The logical width is now 253011231 and the complete carrier is
253011276. Physical rows and the profile are unchanged. The deterministic
zero-extension reduction retains the previously approved fixed-seed MSIS
instance. Required matrix, assignment, mutation, phase-parity, identity,
loader, build and axiom checks passed; see `NIFS_UNUSED_ALLOCATIONS.md` and
`NIFS_UNUSED_ALLOCATIONS_EVIDENCE.zip`.

Three complete native producer invocations reached the required 300-second
cap without returning a proof. The final timed run completed C in 75.28 s,
R in 8.28 s, and D splitting plus commitments in 64.649 s; D openings had
not finished. C matrix openings took 64.777 s; its Pad work took about
2.955 s. The prepared Pad optimization remains unapplied. The existing
cache reservations and arithmetic are retained.

Approval was requested for one already-built producer invocation with a
420-second cap, followed by the independent Lean and complete saved-result
checks. That request is pending; it is not authorization. All other native
invocations retain the 300-second cap. Complete selected assignment and
honest successor construction continue while this request is pending.

## Complete-witness construction in progress

`StepPhysicalCompleteness.complete` now constructs one Spartan assignment
with the pilot, C/R/D, running-transition and next-preimage rows, exact
returned NIFS output, and prior/next source words. Its focused check passed
in 5.69 seconds (4.8 seconds for the module). It still requires an accepted
local NIFS proof, the stated source well-formedness and public hash link,
and the recursive output agreement. It does not yet prove the complete
selected structural plan or supply the application suffix.

`PaperNonInteractive.Completeness.exists_honest_proof_of_sampler_success`
now constructs causal C messages before the actual sampler response and,
on success, a normal accepted NIFS proof with valid openings for every
returned child. Its first focused check passed in 2.18 seconds. Acceptance
and child validity are conclusions, not premises. This existential theorem
does not claim an efficient executable prover. Its selected accepted-envelope
consumer remains in progress.

`PilotPoseidonCompleteness.rowsZero_of_completed_hashRows` passed its
focused check (56 seconds for the module). The remaining pilot obligation
is to supply those hash rows from the constructed physical prefix. Nine
substantive candidate rounds have been used for this pilot obligation;
moving the remaining connection to another file does not reset the ten-round
session limit. The existing `PermutationCompilerTransport` is the authority
for relocation of the source permutation recipes.

The application suffix and its exact output/public binding, the remaining
physical-to-structural row connections, and the final accepted-next-envelope
theorem remain open. These focused results are not a full ordered
static/build/axiom checkpoint and do not change a requirements status.

`ApplicationWitnessCompleteness.complete` now also passes: its result includes
actual application rows, exact advice readback, output digest equality,
the public projection, and direct application-plan rows. The same check
recompiled `PerApplicationSourceAssignment.completeAssignment_norm_of_physical`,
which derives the complete carrier norm from the actual physical prefix.
`PiDECCompletedAssignment.rowsZero_of_completed` separately closes the direct
PiDEC plan on that copied witness. The selected honest-call consumer
`HyperNovaCompleteness.recursive_nifs_of_sampler_success` passed and derives
its source witnesses from the accepted terminal payload itself.

The pilot's tenth round failed in `HashInvocationRows`: one arithmetic proof
did not expose the local `blocks` alias, and one membership proof supplied a
disjunction after simplification had reduced the goal to `True`. No repair
or retry was made after the limit. A proposed two-part patch is saved at
`/tmp/nightstream-pilot-round11-proposal.patch`; approval was requested for
one additional pilot round and remains pending. The complete pilot connection
is therefore still unchecked. This request does not change any other limit
or grant the separately pending native producer exception.

Independent C and R compiler connections continue. The action-compiler
projection and the existing twelve-child PiCCS list accessor passed. The
PiRLC packet projection now accepts the actual relation width, using the
existing family conformance theorems while retaining the same width-erased
packet definitions; its file check and library build passed. None of these
results replaces the final complete selected-plan theorem or an ordered
checkpoint gate.

`HyperNovaStepData.stepHolds_and_wellFormed` now passes and builds the exact
next semantic state from prior acceptance, an actual accepted local fold,
and the explicit counter condition. `CanonicalPublicOutput.rowsZero` and
`NextPreimageCompleteness.rowsZero_of_completed` pass and close the four
public digest pins and five next-preimage rows on the same canonical
assignment. `PiRLCRetainedCompleteness.rowsZero_of_completed` also passes
and derives the combined product and First54 rows from cumulative physical
rows. These are focused results; C and R permutation/ordinary consumers,
the stopped pilot connection, and the complete structural-plan assembly
remain open. Neither pending approval has been received.

## Compiler consumer checkpoint in progress

The actual C transcript readback now passes in
`PiCCSCompletedReadout.transitionEnv_of_completed` and
`outputValue_of_completed`. `RunningTransitionCompletedAssignment.rowsZero_of_completed`
uses that readback and passes. R product/First54 and ordinary sampler plans
pass from actual cumulative physical rows; the ordinary proof uses the
checked `PiRLCSamplerPoseidonValues.outputValue_of_packets`. The three final
envelope helpers pass in the temporary accepted-next draft. The complete
recursive/base theorems still need the whole selected assignment theorem.

C accounting is explicit: two completed prerequisite contracts (the generic
action-compiler projection and the Layout child-list accessor), followed by
nine C consumer candidates: physical packets 2, ordinary physical projection
5, completed readout 2. The tenth C consumer candidate is reserved for the
complete ordinary/Poseidon/endpoint bundle after source review. A new helper
filename does not start a new budget. R has used six cumulative candidates:
retained plan 1, output values 2, ordinary sampler 3. The pilot remains
stopped at ten, with its separate requested repair round pending.

A separate task saved and pushed WIP commit `28ef2c35` while this work
continued. Its message records an incomplete checkpoint and a failing axiom
check. The later focused results do not convert that commit into a green
checkpoint. The full ordered gates and requirements update remain pending.

The complete C bundle has been placed from the reviewed manifest
`/tmp/nightstream-c-complete-bundle-manifest.json`. C consumer candidate ten
has now started against `PiCCSCompletedAssignment`; this spends the remaining
C candidate and must not be repeated under a helper filename. The target
returns the ordinary, Poseidon and endpoint row conjunction on the same
canonical assignment. R permutation candidate seven is queued separately and
may run only if its C dependencies have passed. `validate.sh build` accepts
only its first target; the two checks require separate capped commands.

## C consumer stopped at round ten

C candidate ten exited with failure after 114.61 seconds. The log is
`/tmp/nightstream-piccs-complete-bundle-build-10.log`. `InvocationInputLaw`
failed in four local proofs; `PiCCSInvocationSlices` failed in getter and
dependent-index conversions; `PiCCSEndpointCompleteness` failed in a
converted-index bound. The complete C theorem was not reached. The compiler
assertions, output-address theorem and existing C readback passed, but these
results do not establish the complete C conclusion.

The C budget is exhausted. No repair or retry is authorized beyond ten.
The proposed repair is `/tmp/nightstream-c-round11-proposal.patch`, with
manifest `/tmp/nightstream-c-round11-proposal.json` and patch SHA-256
`74f5ae68fc4f6e56650c46e877ef540927a16d802766cb22ad284c449f689084`.
It changes only proof bodies at the reported failures. Patch application was
checked without changing the source tree; no Lean validation has run on it.
R remains at six candidates: its next permutation check cannot run because
it would retry the failed C endpoint dependency. The pilot is still stopped
at ten. Its proposed hash-accessor repair also leaves the ordinary and digest
binding row consumers to compose; it does not alone close the whole pilot.
The complete selected assignment and accepted-successor theorems remain open.

A separate task saved and pushed `9b47a570`. Its ordered static and library
gates passed, but its full axiom gate failed in the pilot and C modules named
above. This is an incomplete saved checkpoint. No requirements status or
package identity changed. The separate requests for one extra pilot round
and one 420-second native producer invocation remain unanswered.

## Owner decisions after checkpoint 7c3d33d0

The owner approved one C bundle round eleven using the reviewed patch
`74f5ae68fc4f6e56650c46e877ef540927a16d802766cb22ad284c449f689084`
unchanged, and one pilot bundle round eleven using the saved two-part patch
`e57d6ca264e46c7c6bd5981988f7372fd876697600ae264814074c1e1ef589e1`.
Both patches have been applied. Before either final bundle check, check each
failed declaration in isolation through `scripts/validate.sh file` and fix
the local proof there. Do not add recursion or heartbeat overrides.

The continuing ten-round limit now applies to an isolated declaration, not
an obligation bundle. Prior attempts still count; changing the file name
does not reset them. The two expressly approved final bundle checks are
each single attempts. If either fails, remove its failing draft modules and
necessary draft consumers from audit imports and audit entries, retain their
sources, run static/build/axioms in order, and push only the passing checked
surface. Report the exact excluded claims and errors. This withdrawal is
not proof closure. No further bundle round is authorized without a new
owner decision.

The coordinating task is the sole writer of the closure branch. Every new
commit there must pass static, build and axioms in order. Save unfinished
commits on `wip/<topic>` branches only. Keep the requirements map unchanged
until the checked branch is green, and close records only from their full
evidence. The earlier pending requests above are superseded by these decisions.

The owner did not approve the 420-second native producer run. Preserve C
and R outputs, then stage D split, commitment and opening work under the
300-second command cap and compare saved complete outputs. Do not repeat
the unchanged complete producer that already timed out. After the repaired
checked surface is green, resume R permutation, complete selected assignment,
and accepted-successor composition with explicit sampler and counter
conditions. Unconditional perfect completeness is not claimed for the
fail-closed sampler.

## Approved compiler repair results

Both approved final bundle checks passed. C round eleven took 2.46 seconds
and checked `PiCCSCompletedAssignment.rowsZero_of_completed`: actual C rows
give the ordinary, Poseidon and endpoint row conjunction on the same canonical
assignment. Pilot round eleven took 5.90 seconds and checked
`PilotHashRowsCompleteness.rowsZero_of_spartanRows`: actual cumulative physical
rows supply the pilot hash-chain predicates and its direct Poseidon rows.
This pilot result does not include its ordinary and digest-binding plans.

Every previously failed declaration was checked in isolation first. The C
block getter needed a structural Nat address proof after its approved initial
repair still hit the recursion limit. Downstream checks exposed notation
projections, source aliases, a missing full theorem application, finite-index
conversions and a match reduction. These were repaired and checked locally,
with unchanged intended statements and no new assumptions or option overrides.

Attempt totals for declarations that failed in this repair: C block getter 5;
C retained-source and retained-form proofs 3 each; C slice-values proof 3;
all other failed declarations 2 each. Unchanged predecessor proofs in a
dependency prefix do not start new attempts. The complete C wrapper's row
projection, ordinary proof and final conjunction each passed their first
reached isolated check. Both final bundle allowances are now spent and passed.

The two final consumers are imported by the library root as well as the
existing audit tests. No draft audit entry was removed. Ordered checkpoint
gates passed: static 8.75 seconds, library build 3.79 seconds (3,906 jobs),
axioms 4.79 seconds (3,994 jobs). The axiom set remains `propext`,
`Classical.choice`, `Quot.sound`. Independent source review found no material
issue in the final C/pilot assumptions or source correspondence. Logs and
source hashes are in `COMPILER_REPAIR_EVIDENCE.zip`. Package data, identity
pins and Rust are unchanged. R permutation, pilot ordinary and
digest-binding rows, whole selected assignment and accepted successor remain
open. The native staged execution remains open under the 300-second cap.

## R entry composition stopped at ten

The next R permutation file check reached three new failures. Its generic
S-box converter and initial C-to-R value proof now pass in isolation. The
entry row, input-value and canonical-input proofs also pass separately.
However, their `entry_sboxes` composition failed its tenth conservative
attempt with kernel deep recursion after 41.91 seconds. Exact draft:
`/tmp/PiRLCSamplerPoseidon-entry-isolated-10.lean`, SHA-256
`c12166cf1c636d5861f2b0a7e152b2edec2852030bafba6230bffbdef68cf20e`.
Log: `/tmp/nightstream-rlc-entry-isolated-10.log`.

The ninth attempt's diagnostic specialized the generic theorem to the exact
concrete invocation with temporary proof parameters and passed. That does not
validate the actual composition and those parameters were not added to any
production theorem. The tenth attempt retained the checked row/input proofs,
used an explicit witness-address calculation and removed the local invocation
alias, but still failed. No further entry attempt is authorized. The full R
consumer remains outside production sources and audit roots. This is an open
deterministic compiler link, not a cryptographic assumption.

Independent pilot ordinary/digest completion and native C/R staging continue.
The complete selected-assignment and accepted-successor drafts still depend
on R permutation closure; they cannot replace it with a caller premise.

## Remaining pilot plans completed

`PilotOrdinaryPhysicalCompleteness.rows_of_physical` now derives the ordinary
package rows from the original lowered physical rows and stored hash-output
assertions, preserving the actual auxiliary witness values. Its final file
check passed in 7.66 seconds and its module build in 8.02 seconds. The failed
join attempts used a shadowed membership hypothesis; distinct names fixed the
join without retaining the temporary generic helper.

`PilotCompletedAssignment.rowsZero_of_completed` now proves the ordinary plan
and all eight digest-binding rows on the same canonical completed assignment.
Its three type-name corrections passed isolated checks, its full file passed
in 2.19 seconds and its module build in 7.41 seconds. The hash-output accessor
and two source-support interfaces are audited. Their existing proof bodies
are unchanged. Independent review found no extra source, digest, output or
row premise in either consumer. No package or pin changed.

Together with the checked pilot Poseidon result, all pilot plan components
needed by the aggregate assignment are supplied. The R entry composition
above remains the missing deterministic compiler input to that aggregate.

The pilot checkpoint passed its ordered gates: static 8.76 seconds, library
111.62 seconds (3,908 jobs, including the changed dependencies), and axioms
44.74 seconds (3,996 jobs). Only the allowed axiom set occurs. Sources and logs
are recorded in `PILOT_COMPLETION_EVIDENCE.zip`. The R draft was not imported
or audited and no checked obligation was removed to obtain these results.

## Owner instruction: one branch

The owner now prohibits parallel branches. Continue all work on the existing
`nico/f-prime-constraints-cuda-formal` branch and worktree. Do not create or
use another branch or worktree. The R draft had already been saved and pushed
at `a2e477ef` on `wip/rlc-entry-kernel` before this instruction; preserve that
record without further work there. Temporary source review files are not
checked production proofs. Keep the closure branch green at each commit.

The owner then required all previously diverted work to be merged toward this
same branch. Preserve the stopped R source and failure log as unvalidated
review material outside the production Lean tree, merge that history into
the closure branch, and remove the temporary branch after preservation.

## Actual native C/R checkpoint

The staged producer now calls the same C/R helper as the complete selected
NIFS prover. It builds the actual base witness and selected-key commitment,
uses the production matrix evaluator, and verifies the original running
authority and C/R transcript before saving the parent and witness.

The production command passed in 229.96 seconds: source construction and
commitment 37.30 seconds, C 76.04 seconds, R 8.53 seconds. A separate saved
parent command passed in 134.74 seconds. It reconstructed the original inputs,
replayed both transcript fields, split the saved R witness with the existing
splitter, recomputed all digit commitments, and checked their reconstruction
against the accepted R commitment. All 16 digits were saved; indices 0 through
5 are nonzero. Both commands stayed within the owner's 300-second cap.

The small shared-prefix comparison passed and the changed-witness rejection
test passed. Neither fixture test alone establishes full-profile conformance.
Workspace formatting passed. Ordered Lean gates passed: static 8.68 seconds,
build 0.89 seconds (3,908 jobs), axioms 1.00 second (3,996 jobs). Source hashes,
logs and saved-artifact hashes are in `NATIVE_PARENT_EVIDENCE.zip`.

D openings, complete NIFS assembly, final mutation checks and comparison
with the independent complete Lean result remain open. The package, pins,
protocol, cryptographic assumptions and requirements statuses are unchanged.

## R draft retained on the only work branch

The earlier draft history is merged into `nico/f-prime-constraints-cuda-formal`.
Its exact full source is retained as `R_ENTRY_DRAFT.lean` beside this note,
outside the production Lean tree, with `R_ENTRY_FAILURE.log` and
`R_ENTRY_WIP.md`. Its original source hash is unchanged. This archive is not
a checked theorem and does not close R permutation. All further work uses
the closure branch; the temporary branch is removed after this merge.
