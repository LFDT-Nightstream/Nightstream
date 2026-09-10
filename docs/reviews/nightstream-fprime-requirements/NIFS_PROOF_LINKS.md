# NIFS proof and consumer links

Base source: `0ff1ef1d96d88020c3d7e6673684cf213f0ffd6c`.
Work branch: `nico/nifs-proof-links`.

The task is to finish the remaining NIFS proof obligations and their consuming
links. Existing proved results remain inputs. Closure needs checked evidence
for the actual selected boundary. An assumption, definition, or recorded test
must keep its correct status.

The profile is Goldilocks, `b = 2`, `k_rho = 16`, `B = 65536`, one fresh
source, 16 running sources, 17 PiRLC inputs, 16 PiDEC children, 14 matrices,
28 PiCCS rounds, and Poseidon2 protocol binding.

## Completion evidence

| Requirement | Evidence needed | Status |
|---|---|---|
| `N.binding.prior_authority` | The checked recursive boundary supplies the exact prior preimage used by the NIFS transcript, including all running claims and the selected context. | Local proof and consumer checked |
| `N.binding.context` | Arbitrary accepted opening and checked public input identify the selected context, or a named collision. | Local proof and consumer checked |
| `N.local.actual_step` | The same accepted opening reaches the NIFS consumer with the actual proof, prior claims, and advertised output. | Local proof and consumer checked |
| `N.security.binding` | The actual extraction collision reaches the selected public-seed MSIS assumption with the required norm and execution scope. | Partial; public and Pad checks, coefficient generation and stored arithmetic checked; dense commitment, matrix entry, producer and full runtime contracts open |
| `N.security.fiat_shamir` | The selected Poseidon2 transcript and bounded sampler have a justified security connection under the authorized model. | Open |
| `N.conformance.chain` | One nonzero selected-key input and proof have matching Lean and optimized Rust phase values and final output, with required mutations. | Connected for the retained Lean/optimized execution scope; independent review complete for that scope |
| `N.conformance.executed` | Retained commands, inputs, outcomes, and source identities establish the stated execution scope. | Connected for the recorded local execution scope |
| `N.conformance.owners` | The checked chain consumes the existing semantic, transcript, assignment, and caller owners. | Partial; selected owners independently reviewed; concrete runtime contracts and Fiat–Shamir transfer open |

The existing conditional interactive proofs do not establish Fiat–Shamir
transfer. The fixed-seed MSIS premise is the exact premise recorded in
`PUBLIC_SEED_MSIS_ASSUMPTION.md`; it supplies no numerical success bound.
Concrete execution-correctness gaps cannot become hardness assumptions.

## Checked terminal connection

Code commit: `42a8c110023d63233358840594db7fca077fd001`.

`ActualTerminalSecurity.terminal_implies_nifsOrBaseOrCollision` connects the
exact prior-state public input and advertised running output to the NIFS
consumer. It uses the existing
`ActualContextSecurity.terminal_implies_matchingStepOrCollision` result.
The terminal predicate supplies the opening and public check. There is no
extra canonical-assignment or context-equality premise.

`terminal_implies_parentOrBaseOrCollision` consumes this result for the
actual PiDEC parent opening. `terminal_implies_securityOrCollision` consumes
it for the selected NIFS security outcome, with the existing low-norm
invertibility premise. This is a deterministic theorem with named failure
alternatives. It does not supply a probability bound or a history extractor.

The affected module build passed in 17 seconds. Its explicit Stage 1 security
axiom audit passed in 12 seconds and used only `propext`, `Classical.choice`,
and `Quot.sound`. The boundary gate passed. The dependency build took 390
seconds, using a separate project cache and matching dependency cache.
Every Lean command had the project-required 1,500-second hard cap.

This proof connection changes no circuit rows, transcript schedule, package
identity, or profile. The commands, logs, and source identities are retained
in `NIFS_PROOF_LINKS_EVIDENCE.zip`.
The requirements export and all seven existing export tests passed. The
JavaScript syntax check passed. The local map changes only the three NIFS
records above; other requirement records are preserved.

## Checked binding reduction

Code commit: `93848e5153d19d99cbd78e8becce3f3059fab535`.

The reduction computes a candidate from the two actual weak endpoints. It
uses two scalar differences, three assignment differences, and two scalar
actions. It emits every centered integer coordinate through the existing
charged accessor. The executable output does not select a collision by proof
choice or search the witness space.

`BindingReduction.bindingEvent_implies_success` proves that at least one of
the 17 source coordinates gives a valid short-kernel output when the actual
binding event occurs. The reduction samples a coordinate uniformly in the
existing interactive coin model. `BindingProbability.binding_le_success`
therefore gives

`P(binding event) <= 17 * P(emitted MSIS vector)`.

`BindingProbability.observationSuccess_eq_runPair` connects that probability
to the actual paired-call output. `BindingReduction.runPair_work_eq` retains
both call clocks. `BindingWork.suffixLaw_eq_workLaw` identifies the same
continuation law in the probability and work arguments.

`SupportedExtraction.msis_probability_and_expected_work` connects the checked
prefix to both the source-success bound and the binding reduction's expected
work. Its source loss is

`17 / card(Challenge) + sqrt(17 * P(emitted MSIS vector) + PiCCS test error)`.

After the original context is supplied, the work bound is

`2 * sourceWork + 3 * coordinateWork + carrierWidth * (accessWork + 6) + 12`.

The coefficients come from the actual two source runs and the proved
arithmetic, coordinate-read, centered-lift, uniform-request and return steps.
The work model is the existing interactive coin model. Low-norm invertibility,
call/check/access correctness, primitive bounds and the original global work
moments remain explicit premises. The bound counts abstract operations.

`Poseidon2HashChainV1Setup.productionNifsOutput_is_msis` identifies the emitted
vector with the approved fixed instance: 254260620 integer coordinates,
strict norm 113246208, and the exact `productionAjtaiKey`. The existing setup
identity theorem retains its named Poseidon2 collision alternatives.

The final NIFS and foundation axiom audits each passed in 8 seconds. The
boundary gate and exported-theorem audit coverage check passed. Only
`propext`, `Classical.choice`, and `Quot.sound` occur in the audited proofs.
The foundation dependency build exposed a dependent-type rewrite that reached
the default recursion limit. An explicit record construction fixed that
proof; no resource-limit override was added. All diagnostic logs, including
failed attempts, are retained in `NIFS_BINDING_REDUCTION_EVIDENCE.zip`.

The local map changes only `N.security.binding`, from open to partial. Its
Proof status remains Assumption. No compiler, conformance, or production
approval is inferred from these local proof checks. The site export, all
seven existing export tests, the JavaScript syntax check, and the source
reference check passed. The other 453 requirement nodes and all previous
update records were preserved.

## Preparation now included

Proof commit: `d1785fe1f5dd006df68c49818a8d512f82fd29f4`.

`ContextPreparation.run` executes the original preparation call once, passes
its actual value to the continuation, and retains its work. The preparation
work contract includes private-coin acquisition and fixed-key preprocessing.
`contexts_toReal`, `value_hasSum`, and `summable_iff` connect the exact context
marginal to the original private tapes. They preserve correlation between the
returned context and its preparation cost.

`BindingProbability.prepared_successProbability_eq` uses that same context
law. `BindingWork.prepared_expected_work_polynomial_bound` adds the actual
preparation cost. `SupportedExtraction.msis_probability_and_expected_work`
now consumes both results and the existing checked-prefix proof. The full
bound is

`preparationWork + 2 * sourceWork + 3 * coordinateWork + carrierWidth * (accessWork + 6) + 13`.

The extra step is the preparation-to-continuation dispatch and return. No
uniform bound on individual contexts or calls was added. The NIFS axiom audit
passed in 4 seconds, and the boundary gate passed. The audit permits only
`propext`, `Classical.choice`, and `Quot.sound`.

The final theorem still takes the costed extraction primitives, accessor and
source/parent checker contracts as parameters. The current source has no
selected construction that discharges all of them. In particular, an
arbitrary semantic assignment function cannot be assigned unit access cost.
These are implementation and work-proof obligations. They are not covered by
the public-seed MSIS assumption. `N.security.binding` remains partial.

## Current honest PiCCS check

Tool commit: `a33deedf8f6c0a782ec6c79e03cff6a836a9acd5`.

The existing input-comparison tool now has `optimized-accept` and
`optimized-reject` actions. These use the common optimized verifier and the
complete existing result encoder. The current check used the retained honest
recursive input from `conformance-fixes-evidence.zip`. The selected package
was fetched from Git LFS at the reviewed base and its bytes matched the
retained package. Its source, profile and identity were not substituted with
a small fixture.

| Executed check | Result |
|---|---|
| Current executable Lean on the honest recursive input | Accepted; all 1710813 result bytes match the retained result. The build and run took 290 seconds. |
| Optimized Rust on the same input and proof | Accepted; all 15 complete phase fields match Lean. The run took 19.385 seconds. |
| First-round constant coefficient changed by one | Current Lean and optimized Rust reject. Rust stops at round zero, so a complete rejection trace is not claimed. |
| First output pad coefficient changed by one | The complete result comparison fails at the expected assertion. Exit 134 is the release panic abort, not a timeout. |

The release build took 42.77 seconds. `cargo fmt --all` completed; its stable
toolchain reported the existing nightly-only import-format setting. Each
native validation invocation had the 300-second hard cap. Each Lean command
had the 1500-second hard cap. This work ran no PaperExact action.

The input and proof are concrete and nonzero. The retained source/opening
checks keep their historical scope. This current run checks the PiCCS
verifier prefix; it does not run the full optimized prover or close the
PiRLC, PiDEC, final-output and complete mutation chain. Independent phase
approvals remain separate.

`NIFS_PREPARATION_AND_PICCS_EVIDENCE.zip` retains the exact source, inputs,
outputs, mutations and logs, including failed proof attempts. The map updates
three scoped NIFS records and corrects shifted source lines in two other NIFS
records. Other requirements and their statuses are preserved.

## Current PiCCS-to-PiRLC check

Code commit: `3589e410587edc1e419e355ff5fab3be4580f04f`.

`PiRLCInputCheck` consumes the same honest recursive PiCCS input. It runs the
PiCCS check once, preserves its complete result, and continues with its exact
full transcript state, point and 17 ordered claims. PiCCS rejection stops the
continuation before the sampler. The sampler's existing failure result also
stops it. No probability law is inferred from this execution equality.

`commitments_eq_batch`, `publicInputs_eq_batch` and `evaluations_eq_batch`
identify the actual fields with `PaperStrongInterface.piRlcBatchForProbe`.
`sampled_response` identifies the successful response with
`ProductionKey.piRlcResponse`. The generalized materialized commitment scan
retains its indexed and final-combination proofs. It now accepts the actual
commitments.

The `optimized-rlc` action compares all 15 PiCCS fields and all 11 PiRLC
fields, including all 17 partial combinations. It passes the supplied parent
to the native `pi_rlc::verify` and checks the returned parent and full outgoing
state. The compressed fold digest is not used to restore the transcript.

The honest Lean build and execution took 14 seconds. Its PiCCS prefix is
unchanged. The Rust release builds took 7.23 and 7.14 seconds. The explicit
axiom audit passed in 2 seconds and used only `propext`, `Classical.choice`
and `Quot.sound`. The source boundary gate passed.

All 62 native mutations reject. They cover the parent commitment, public
input, point, pad evaluation, every matrix evaluation, each source commitment,
each transcript lane, malformed evaluation and point lengths, nonzero
padding, and the parent fold digest. Seven serialized mutations also fail
at the expected comparison: the handoff state, rho, each partial-result
family and outgoing state. Each run took about 20.5 seconds and ended at
the expected assertion; the release abort was not a timeout. The existing
injected sampler test passes, including the stream with only 53 accepted
coefficients. Its release build took 46.11 seconds. No PaperExact action ran.

The actual parent commitment, public input and evaluations are nonzero.
Some running children are zero, so the diagnostic flag for every input being
nonzero is false. That flag does not control acceptance. Opening validity,
the complete optimized prover, PiDEC, final output and assignment/matrix
conformance keep their separate obligations.

`NIFS_CCS_RLC_EVIDENCE.zip` retains the source, commands, complete input and
results, mutations, and logs, including the earlier failed proof attempts.
The local map updates the three NIFS conformance records and keeps each
connection partial. This is local execution evidence, not independent phase
approval.

## Rebuilt openings and current optimized prover

Test-tool commit: `3e85884d5bcc2764fd85829a32d0be5e42cc57e0`.

The actual base and recursive fresh openings were rebuilt from the selected
package and retained caller fixtures. Each run checked all 6377559 scalar
matrix rows and recomputed the selected-key commitment. Both commitments
and public projections match the retained PiCCS input. The runs took 42.34
and 43.83 seconds. The first base attempt used a mistyped identity argument
and rejected before writing a cache; that log is retained as a failed
command attempt, not a conformance test.

The current Lean base C/R check passed in 13 seconds and preserves every
retained PiCCS field. The integer fold completed in 20.08 seconds with
254260620 coordinates and maximum norm 62. Its sampler values, complete
state, parent public input, point and all 16 child public inputs match the
retained values. The fixed bound remains 65536. The actual recursive running
prefix rebuilt in 40.04 seconds.

The existing independent fresh-opening test now accepts `family: "ALL"`.
It loads the package and raw carrier once and checks the pad and all 14
matrix families with the existing independent decoder, matrix program and
ring arithmetic. All 810 extension-field coefficients match the actual
recursive PiCCS input. The test passed in 128.84 seconds after a 23.13-second
release test build. Changing the first private carrier coefficient while
preserving its signed-unit bound and public projection fails the pad check
in 22.07 seconds, with the expected test exit 101.

The existing `child-complete-driver` ran the optimized common prover and
verifier on 17 actual native witnesses. It recomputed all 28 round messages
and compared the complete output and transcript. All 662424 emitted input
and proof bytes match the input already checked by current Lean and Rust
C/R execution. The run took 48.38 seconds; the prover reported 19.62 seconds.

This driver uses prepared full ring evaluations through the existing
evaluator interface. The fresh evaluations are independently checked above.
The current distinct running-opening checks, PiRLC witness folding, PiDEC,
final output and complete assignment/matrix conformance remain required.
This evidence does not close the full NIFS prover or grant phase approval.

`NIFS_OPENING_AND_PROVER_EVIDENCE.zip` retains the commands, compact caller
inputs, exact proof bytes, opening metadata, source and logs. The large
reconstructed witness and matrix caches remain in the external working
directory; their byte identities and regeneration commands are recorded.
Every native run used the project-required 300-second hard cap.

## Complete running openings and native PiRLC witness

Code commits: `120425e9` and `24217edb37eb091d2c2a12008f0bb9e10c90ec6f`.

The existing independent child evaluator now shares the selected package and
raw parent across all 15 families. At each point it checks all 16 children,
including every coefficient and the existing changed-word, noncanonical-word,
cancelling-pair and malformed-shape mutations. All 12960 extension-field
coefficients match at the prior point and at the new PiCCS point. The two
runs passed in 133.91 and 132.63 seconds.

The independent selected-key commitment check matches all 19008 coefficients
of the 16 running commitments in 76.61 seconds. The fresh commitment check
matches all 1188 coefficients in 60.04 seconds. The independent raw CCS
evaluator accepts all 6377559 fresh rows in 29.45 seconds. These checks use
the same reconstructed witnesses supplied to the current optimized prover.

`fold-recursive` combines the actual fresh witness and 16 actual running
witnesses with the exact sampled ring actions. Every ring-action basis entry
is checked against the native sampler matrix. The full integer result has
254260620 coordinates, with maximum norm 113. Its public input, point and
sampler state match the checked R result. This preparation took 20.99
seconds. The profile remains `b = 2`, `k_rho = 16`, `B = 65536`.

`child-rlc-driver` now continues the actual optimized C prover through the
native `pi_rlc::prove` and `pi_rlc::verify` owners. It compares the complete
parent and full transcript state with Lean, then compares every private
coefficient with the integer result. The complete run passed in 86.96
seconds. The C input and proof remain byte-identical to the checked input.
Changing the final private coefficient preserves the public input, shape and
norm bound but fails at coordinate `(53, 4708529)`, as required. The expected
release panic has exit 134; it is not a timeout.

The C driver still uses its existing complete-oracle interface. Its prepared
openings now have current independent checks for all 17 sources. The native
R call uses the existing optimized witness mixer. No new crypto assumption,
feature, environment variable or production profile was introduced.

`NIFS_RUNNING_AND_NATIVE_RLC_EVIDENCE.zip` retains exact source, command and
input records, all small results and the validation logs. Large raw caches
remain at the recorded external paths with byte identities and generation
commands. PiDEC, final output and complete matrix/assignment conformance
remain open, as do the separate extraction-cost and Fiat-Shamir obligations.

## Actual PiDEC and final running output

Code commit: `1666d816c321dbbd121d0506af7e1435342bc2b0`.

`PiDECInputCheck` now consumes the typed final values from the same executable
PiRLC trace. It uses the existing paper verifier, strict public bound,
canonical public split, and the selected relation and key. The supplied point
and every child public input must match the verifier's output. Its
`accepted_reduces_knowledge` theorem consumes the existing reduction: valid
openings for these actual children reconstruct an opening for this actual
parent. It does not assume parent opening validity. `child_structure` consumes
the proof that the lazy matrix accessor equals the selected application plan.

The full Lean C/R/D check passed in 15 seconds. Its C/R prefix is unchanged.
It accepts PiDEC and produces the exact supplied 16-child running output.
The earlier runtime constructed the full relation before it was needed and
was stopped. The checked lazy accessor removes that work from public checks;
it does not substitute a smaller relation.

The actual R witness has maximum norm 113. Its 16 signed binary children use
the unchanged Nightstream Goldilocks profile, `b = 2`, `k_rho = 16`,
`B = 65536`. All 19008 child commitment coefficients pass an independent
selected-key check in 88.65 seconds. All 12960 child evaluation coefficients,
including the pad and all 14 matrix families, pass the independent evaluator
and its existing mutation checks in 135.86 seconds.

`child-dec-driver` continues the actual optimized C/R execution through the
native PiDEC split, prover and verifier. Every private digit agrees with the
independent integer witness. Seven digit planes are nonzero; the other nine
are checked through their exact constant-zero representation. All 17 public
result fields agree with Lean. The C proof bytes and R result bytes are
unchanged, and all 446109 final running-output bytes agree. C/R took 88.54
seconds; D took 11.71 seconds. The first comparison used a dense accessor on
a compact matrix and aborted. The corrected indexed comparison passed.

All 55 native D mutations reject. The Lean test rejects 34 public mutations
and 10 encoding errors, the public norm boundary, and a rejected C prefix
that would otherwise supply a D parent. It passed in 20 seconds. The native
prepared-opening path now omits its unused matrix cache; a focused release
regression compares it with the checked cached path and passes.

The final C, R and D axiom audits passed in 8, 2 and 3 seconds. They permit
only `propext`, `Classical.choice` and `Quot.sound`. The source boundary gate
passed. `NIFS_DEC_AND_FINAL_OUTPUT_EVIDENCE.zip` retains source, commands,
complete input/output records, mutation files and logs, including failed
diagnostics. Raw caches keep their recorded external paths and regeneration
commands. Every native test uses the 300-second cap, and every Lean command
uses the 1500-second cap. No PaperExact action ran.

These checks establish the stated selected phase execution. Complete NIFS
caller replay, full expanded-matrix/assignment checks and independent review
remain separate requirements. The three NIFS conformance links stay partial.
The local requirements export, all seven export tests, JavaScript syntax and
affected source references pass. All other 451 requirement nodes are preserved.

## Retained matrix and caller evidence

`NIFS_MATRIX_CALLER_EVIDENCE_LINKS.json` connects four existing passed runs
from `conformance-fixes-evidence.zip` at source `2c63f41d` to the current
selected input. The candidate package and eight shared fixture files have
the same bytes. The matrix interpreters, physical expander, assignment
generator, independent row evaluators, caller checker, arithmetic and
dependency lock are unchanged. The only changes in the compared crates are
the two separate opening-test batch modes. These are retained runs, not new
test executions or independent approval.

The physical check compared every final A/B/C entry. The logical check
compared all 14 matrix families at every active row. The recursive caller
check accepted all 29225729 physical rows, all logical coordinates and
alignment zeros, and all 6377559 logical rows. It bound prior iteration 1
to output iteration 2 and the exact 16 children. Its separate assignment
mutations rejected changed commitment, pad and matrix-evaluation values.
The record retains each original command, elapsed time, result and archive
member identity.

This supplies the stated matrix and raw-caller evidence for the same fresh
assignment used by the current C/R/D run. At this cut, the native
`nifs::verify` interface required a matrix cache. The later complete-NIFS
change below removes this unused public-verifier dependency and retains cache
consistency checks at the callers that own preprocessing. No cache receipt
is used as proof of matrix correctness.

## Verified-cache width correction

Code commit: `35c1318c2726222ef09a950e93a3bf9762b036c9`.

`OptimizedStructureCache::from_verified_artifact` compared the cache's padded
column count with the header's logical count. That rejects a correct cache
when the relation ends inside a ring. The selected relation has 254260583
logical columns and a 254260620-column carrier, with 37 padding columns.
The loader now compares the cache with the padded width and retains the exact
logical shape for later header validation.

The new regression uses `D + 1` logical columns and checks a nonzero
evaluation across both rings. It failed at the old shape check, with exit
101. After the fix, all five cache-artifact tests pass in 0.07 seconds after
a 10.57-second release build. The regression also rejects a later logical
width change within the same padded ring. Existing tamper, size, shape and
round-trip checks pass. Every test command used the 300-second cap, and Rust
formatting completed.

The R-parent proof draft checks the handoff fields, then attempts to identify
the complete parent with the paper R output and consume PiDEC's reconstructed
opening as weak-extraction success. Its combined-output proof remains
unproved. After three build attempts, Lean still reaches its recursion limit
at the public-input combination equality. The project stop-and-report rule
applies. The draft is retained outside the active formal project; it adds no
proved theorem or requirements closure.

`NIFS_CACHE_WIDTH_AND_PARENT_DRAFT.zip` retains the checked Rust source,
regression logs, proof draft and failed proof logs. No recursion override,
Fiat-Shamir law, or new cryptographic assumption was added. The actual selected
cache and complete native NIFS replay remain open.

The next R-parent attempt added a structural width-independence lemma and
made the public equality explicit. Three further checks still failed when
applying that function equality to a public coordinate. The updated draft
and logs are retained in the same archive. The full connection remains
unproved and outside the active formal project. A pointwise width lemma is
the next proof step to check; no recursion limit was raised.

## Checked R-parent and weak-success connection

Code commit: `7c841ac1135851a54231154893aa8a3bf9f9dd27`.

`PiRLCParent.computedParent_eq_combined` proves that the exact materialized
R endpoint passed to D is the paper's combined R claim. It covers the selected
relation, commitment, public input, point, both evaluation families and stage.
`computedParent_outgoing` preserves the complete sampler endpoint.

`checked_children_imply_rlc_success` consumes this equality and the existing
PiDEC knowledge reduction. Its premises are the successful sampler replay,
the returned materialized parent, accepted D messages, and valid openings for
the actual D children. It derives validity of the R-parent opening and supplies
the existing `PaperForkExtraction.Response.Success` witness. Parent opening
validity is a conclusion, not an extra premise.

The same theorem consumes `inputBatch_phi_eq_probe`: the actual batch has
the exact commitment projection used by the existing strong-prefix interface
for every probe. The proof does not assign a probability law to sampler replay
or discharge the remaining extraction work contracts.

A coordinate-level width lemma resolved the earlier recursion failure. It
proves that the five-ring public combination is independent of the private
carrier width. The module built in 3 seconds; the final four-theorem axiom
audit passed in 5 seconds and permits only `propext`, `Classical.choice` and
`Quot.sound`. The source boundary gate passed. No recursion or heartbeat
override was added.

`NIFS_RLC_PARENT_EVIDENCE.zip` retains the checked source, commands and logs.
It supersedes the earlier draft's unproved status. Native protocol behavior
is unchanged, so the retained C/R/D execution checks still have their stated
scope. Full native NIFS replay, concrete extraction contracts, Fiat-Shamir
security and independent review remain open.

Commit `7a1f046772d188c0236471a98774057b0426c486` adds
`computedParent_correct`. For every input and batch, the canonical 17-source
traces return a parent with the exact paper claim and complete sampler state.
The theorem has no successful-return premise. The source traces are nonempty,
and all 14 matrix endpoints have the required size, so this construction
cannot take the incomplete-parent branch. All five theorem audits pass in
4 seconds, and the boundary gate passes. The same evidence archive contains
the updated source and logs, including the first failed totality-proof attempt.

## Complete native NIFS replay

Code commit: `9dcb4e7b9d7b16ac15e7f19c2ab4a4db007708f5`.

The public v1.1 PiCCS verifier does not read a matrix evaluator cache. Its
cached wrapper checked cache consistency, then ran the same public protocol.
The NIFS verifier now takes the caller-selected relation header directly.
Fixed NIFS, F′, finalization and cross-check callers retain cache validation
where they own preprocessing. The selected relation and key remain inputs
chosen by the caller. All C/R/D equations and the prior-parent check remain.
This supersedes the earlier requirement to construct a cache for public NIFS
replay; it does not replace the exact matrix evidence.

The new `child-nifs-driver` runs the actual optimized C prover, native R and D
provers, and the complete `nifs::verify` entry point on the retained nonzero
input. It checks all 16 prior claims against the prior R parent, then compares
every final child, the returned parent cache, the full eight-word transcript
state, and the empty verifier witness list. The final output matches Lean.

The native legacy `fold_digest` on the incoming parent cache is the same
incoming F′ frame digest as on the authoritative prior children. It is not an
extra formal CE coordinate or the previous C transcript endpoint. The prior
commitment, public input, point and evaluations come from the checked prior
C/R result. The complete prior-parent equations are checked; digest agreement
alone is not the acceptance evidence.

All 945983 canonical NIFS proof bytes match a separate encoder that reads raw
Lean fields. That encoder does not call the native proof encoder. The C,
R and D output files retain their previous exact bytes: 662424, 61612 and
446109 bytes. The full verifier rejects 43 mutations covering the prior
parent, all 16 prior child commitments, fresh public input and C/R/D proof
outputs. The same run rejects all 55 native D mutations. The C/R execution
took 88.68 seconds, D took 12.03 seconds, and the full verifier, wire comparison
and NIFS mutations took 0.116 seconds.

The optimized fixed-NIFS, round-trip and cache-substitution suites pass
3, 9 and 1 tests. The final fixed-NIFS test also rejects a same-shape cache
from another relation. Six affected core test targets compile in release
mode. The historical empty-running failure remains recorded; that test was
compiled but not rerun. The Metal test changes only remove the obsolete
argument; no backend was executed. Rust formatting and the source boundary
gate pass. No Lean source changed in this cut.

`NIFS_COMPLETE_REPLAY_EVIDENCE.zip` retains the exact source, common inputs,
complete proof and output bytes, mutation outcomes, commands and logs. It
links the previous matrix, assignment, opening and Lean phase evidence.
Large input buffers retain their recorded external identities and regeneration
commands. The first artifact-log check passed the byte and field comparisons
but used the wrong PiDEC log prefix; its failed diagnostic and corrected check
are both retained.

`N.conformance.executed` is connected only for this recorded execution scope.
The chain and owner links remain partial. Independent phase review is still
required by `FPRIME_STAGE1_GOAL.md`; this local evidence does not grant it.
The local export, seven export tests and JavaScript syntax check pass. All 71
affected source references resolve. One existing reference moved from line
111 to 112 and was corrected. All other 451 nodes and all prior update records
are unchanged.

## Executed PiCCS acceptance and strong probe

Code commit: `bec2bc96f49c4bc0253bbe08efb6ba32d2f3b676`.

`PiCCSInputCheck.execute_accepted_iff` identifies the executed acceptance bit
with `Probe.FixedWidthAccepted` for the exact supplied rounds, full output,
and derived challenges. It consumes the checked fast initial and terminal
equations and the existing fixed-width raw-certificate theorem. The selected
statement and key remain the same.

`PiRLCInputCheck.sampled_fixedWidthAccepted` consumes this equivalence. Every
successful actual C/R handoff now supplies the accepted probe used by the
strong extraction theorem. It does not assume that probe acceptance.
The first draft had record syntax and namespace errors; both are corrected.
The C/R audit passes in 4 seconds, the C audit in 3 seconds, and the boundary
gate passes. Both new theorem audits contain only the permitted axioms.
`NIFS_EXECUTED_PROBE_EVIDENCE.zip` retains the exact source and all proof logs.

This is a deterministic connection for actual checked inputs. The existing
execution functions are unchanged. It supplies no challenge distribution,
ambient-witness checker, primitive cost bound or Fiat–Shamir transfer.
The retained-source check confirms that all previous C/R declarations have
identical bytes. The local export and all seven export tests pass; all 73
affected source references resolve, with the other 451 nodes preserved.

## Complete strong-probe batch and selected public checker

Code commit: `2876011e105da37b60f7d86b3ed33125dcbe0435`.

`PiRLCParent.inputBatch_eq_probe` identifies the entire actual R input with
the batch of the executed C probe. Equality includes all 17 ordered claims,
the selected relation, point, public inputs, commitments and both evaluation
families. `checked_children_imply_rlc_success` now returns the accepted strong
probe and weak-response success for that exact batch. It consumes the checked
C acceptance link and D opening reconstruction. Its remaining witness premise
is validity of the actual final child openings.

`SupportedExtraction.publicCheck` now computes the selected fixed-width public
check from the running fields and the returned probe. It uses the existing
polynomial, coefficient projections and verifier equations. It does not
construct the noncomputable semantic key. `publicCheck_correct` proves its
equivalence to the selected predicate for every probe, including malformed
messages. All five supported extraction results consume this checker. They
no longer take an arbitrary public checker or its correctness as premises.

The actual-batch audit and the NIFS extraction audit each pass in 4 seconds.
The boundary gate passes. The retained failed drafts show one unsimplified
record projection and one attempted computational use of the semantic key;
both are corrected. `NIFS_SELECTED_CHECKER_EVIDENCE.zip` retains the checked
source and logs. The earlier goal turn made progress through committed native
conformance and executed-probe proofs; this turn removes a concrete checker
premise and connects the full weak input.

The witness-operation implementations, ambient witness checker, accessors
and their work bounds remain separate obligations. No new hardness premise,
challenge law, or Fiat–Shamir model was introduced. These two map links remain
partial.
The local export and all seven export tests pass. All 55 affected source
references resolve. The weak-extraction and composition records only refresh
source lines; their proved and connected statuses are unchanged. All other
450 nodes and all earlier update records are preserved.

## External proof references

The online review found [ArkLib](https://github.com/Verified-zkEVM/ArkLib)
and [VCVio](https://github.com/Verified-zkEVM/VCVio). VCVio's
[cost reduction interface](https://github.com/Verified-zkEVM/VCVio/blob/main/VCVio/CryptoFoundations/Asymptotics/ReductionCost.lean)
requires a proof of the concrete cost bound. Its
[stateful Fiat–Shamir bridge](https://github.com/Verified-zkEVM/VCVio/blob/main/VCVio/CryptoFoundations/FiatShamir/Sigma/Stateful/Bridge.lean)
uses a random-oracle interface for a Sigma protocol. These are useful proof
references. They do not supply the missing Nightstream representation-cost
proof or authorize a random-oracle model for this Poseidon2 transcript.
No dependency or new cryptographic assumption was added.

## Stored witnesses, checker, and sampler milestone

Code commit: `5b7168f9c9b0c58b11759cf676d271ea262241a1`.

The stored-witness interfaces from `6e4b08f1` now have an executable
public/ambient checker. `StoredWitnessCheck.check_eq_true_iff` identifies its
result with the existing fixed-width public check and complete ambient
opening predicate. It covers each commitment, public prefix, strict norm,
Pad coefficient, and matrix coefficient. Its MLE traversal does not build a
table with one entry per row.

`StoredWitnessCheckWork.check_value` proves that the charged loop program
returns the same Boolean result. `check_work_le` bounds its returned counter
through the executed loops, including rejected and skipped branches.
`PiCCSStoredWitnessCheck.charged_finish_source_iff` consumes this refinement
at the selected Ajtai key and relation. The remaining primitive correctness
and work fields are explicit. They include the public SumCheck gate, Ajtai
computation, matrix access, public/probe reads, and arithmetic. This is not a
free theorem that these primitives already meet their contracts.

`StoredAssignmentArithmetic` computes array subtraction and weighted
combination. Its selected recomposition first constructs all 16 binary
weights, then materializes the complete parent array. `recompose_value`
proves equality with the existing `PiDEC.Raw.recomposeAssignment`;
`recompose_work_le` includes the weight construction. `Vector.ofFnM` uses a
direct-index, preallocated array loop. The independent delta review checked
that implementation and its exact source hash.

These are value and returned-counter theorems. Full runtime, actual producer
construction, ring-unit inversion, and stored-output projection remain open.
In particular, the older source copier returns `List.Vector` and wraps
successive index reads in closures. Its abstract counter does not establish
all representation and compiler costs. The attempted stored PiDEC-to-PiRLC
consumer reached the project three-round limit at a raw/typed recomposition
rewrite. Its failed source is retained in
`drafts/NIFS_STORED_SUFFIX.lean.txt`, outside the active Lean package. It is
not an audited consumer link.

`FieldPairLaw` proves the exact single-field preimage counts, mixture law,
sharp event-distance bound, and deterministic decoder transport. With
`M=2^32` and `q=M*(M-1)+1`, reducing a uniform field representative modulo
`M` gives `(1-1/q)*Uniform(Fin M) + (1/q)*PointMass(0)`. The exact total
variation is `(M-1)/(M*q)`. This is a finite comparison law; it assigns no
independent-uniform law to Poseidon2 output.

The separate 54-of-64 shortfall-counting attempt also reached the three-round
limit. Its remaining errors concern finite subtype cardinality and proof
elaboration. The source is retained in
`drafts/NIFS_SHORTFALL_BOUND.lean.txt`. The proposed IID-bit bound
`choose(64,11)/65536^11` is not an audited theorem. The field-pair product law,
batch abort probability, and their use under retries remain open.

The [Fiat–Shamir analysis](NIFS_FIAT_SHAMIR_MODEL_DECISION.md) reads the full
2026-03-27 Chiesa–Orrù revision. It records the exact current error formulas
and code interfaces. Additive absorption, permutation timing, initialization,
abort-inclusive codecs, state-restoration extraction, and the transfer to
fixed Poseidon2 still need proof or an explicitly approved external premise
where appropriate. No model was approved. The owner question remains
deferred until the proposed statement is concrete.

The [independent review](NIFS_CONFORMANCE_REVIEW_773f3d0f.md) maps every C/R/D
verifier conjunct to its predicate, circuit and theorem. It confirms the
retained complete Lean/optimized chain and independently reconstructs all
945983 canonical proof bytes. The three historical failed mutation blocks
are duplicate allocations; the actual decoded NIFS consumers read other,
constrained values. The failed diagnostic is preserved. No missing C/R/D
conjunct was found, and no production cleanup was required for that claim.

The broader same-input PaperExact R/D check remains pending. The new public
reference runner passed its release build in 5.15 seconds. It reads the
selected candidate and retained ten-field C/R/D envelope, compares all R11
and D17 fields, and uses no private witness, prover, or proof backend.
PaperExact execution requires explicit approval under `AGENTS.md:117`; that
approval was requested and has not been received. This is separate from the
Fiat–Shamir model decision. The independent review does not declare a phase
Conformance-closed or Production-closed.

The compatible CompPoly tag `v4.30.0-patch1`, commit
`050f0bc7e9780703beb8d178ec533e52bd87d649`, supplies executable extended GCD
with correctness and sufficient-fuel proofs. Its Lean and Mathlib versions
match this package. The review records the exact upstream sources. A local
ring/polynomial representation proof and full work bound remain necessary;
no dependency was added.

The combined `tests.AxiomsNifsClosure` gate passed in four seconds and the
active source-boundary gate passed. All new exported theorems have explicit
axiom checks and use only `propext`, `Classical.choice`, and `Quot.sound`.
The public reference runner passed its release build; it has not been run.
All Lean commands used the required 1500-second cap. The retained public
reference invocation is capped at 300 seconds and still needs approval.
`NIFS_STORED_AND_SAMPLER_EVIDENCE.zip` records the committed sources, passed
and failed logs, both stopped drafts, independent review, and model analysis.
The local map and static exports preserve every other requirement record.

## Stored primitives and scalar failure bounds

Code commit: `aee1232dc47ab98fec64d96237c398976bf3ea3f`.

This update follows the `5b7168f9` milestone above. The source pin, checked
files, and passed and failed logs are retained in
[NIFS_PRIMITIVES_AND_SHORTFALL_EVIDENCE.zip](NIFS_PRIMITIVES_AND_SHORTFALL_EVIDENCE.zip).
The earlier stopped drafts remain historical evidence.

`StoredProbe` now stores Pad and matrix families in separate nested arrays.
Its semantic view preserves the original public coins and raw certificate.
It does not turn a malformed certificate into a valid one. Concrete reads
supply the same Pad and matrix claim values, with four and five counted
operations. The producer still owns the work to create these arrays.

`StoredWitnessCheckPrimitives` implements the dot step, field embedding,
interpolation, equality, and norm checks. The selected public-input reader
uses the actual fresh and running arrays. The selected
`scalar_finish_source_iff` and `scalar_check_work_le` consumers now need only
four primitive contracts: the public gate, commitment check, Pad entry, and
matrix entry. The scalar operations and public/probe reads are proved.
These remain named operation counts, not compiled-runtime bounds.

`CostedWitnessProjection` now uses `List.ofFnM`: a loop over the original
indices, followed by one list reversal. It removes the chain of accessor
functions from the old recursive copier. Its value proof preserves every
fresh private tail and every full running witness. For access bound `a`,
the new projection bound is

```text
freshCount * (privateWidth * (a + 6) + 9)
  + runningCount * (carrierWidth * (a + 6) + 9) + 7.
```

The constants count forward branch/index/cons work, reverse traversal,
initialization, and returns. The displayed polynomial bounds in all six
strong and composed consumers use these counts. The first combined check
found the old constants; the repaired combined check passed. This closes
the direct-copier value/counter link. It does not prove the full producer or
compiled-runtime contract.

The resumed `StoredSuffix.finish_returns_parent` proof also passed. Given
its explicit checker correctness premise, it connects the actual stored
16-child recomposition to the same PiRLC parent opening. The finish counter
includes abort and rejection and is at most the actual checker counter plus
`116 * carrierWidth + 615`. A concrete checker and captured producer law
still have to supply that premise and the actual call work.

The package now pins CompPoly at
`050f0bc7e9780703beb8d178ec533e52bd87d649`. Its Lean and all existing dependency
revisions match this package. `StoredRingInverse` executes normalized
extended GCD and monic reduction modulo `X^54 + X^27 + 1`. It computes the
candidate polynomial once, then copies its 54 coefficients.
`StoredRingInverseCorrect.candidatePolynomial_mul_mod` proves the polynomial
inverse under explicit coprimality. Its degree theorem proves that the copy
does not discard a nonzero coefficient. The RingF-unit-to-coprime and
multiplication links remain open, as does the full inverse work bound.
The RingF polynomial bridge stopped after three failed checks; its source
is retained in `drafts/NIFS_RING_POLYNOMIAL.lean.txt`.

The new execution test checks the candidate with the protocol's own
`ringFMul`, for constant 2, X, and 1 + X. These cases cover normalization,
Phi81 reduction, and a dense cofactor. All 54 product coefficients passed
in each case. This is executed evidence for those cases, not an all-unit
inverse theorem.

`ShortfallBound` proves that the 54-of-64 decoder fails exactly when at least
11 candidates reject. `SamplerShortfall` applies that failure event to the
actual transcript-derived candidate window. The IID-bit bound is now a
checked theorem:

```text
p_scalar <= choose(64, 11) / 65536^11.
```

`FieldShortfall` proves the same upper bound for 32 independent uniform
Goldilocks fields. Its finite bijection swaps the low-32 residue with an
auxiliary uniform lane inside each complete field block. The single final
field value gives two accepted zero candidates, so it can only remove
rejections. The proof preserves dependence inside each field pair and adds
no total-variation penalty to this abort bound. No IID law is assigned to
Poseidon2. Product bias, batch/retry bounds, and their Fiat-Shamir use remain
separate obligations.

The further deterministic 32-field/batch link reached three failed target
checks. The first stopped at the then-unfixed work dependency; the next two
failed in the batch proof, including kernel recursion errors. They took 128
and 122 seconds. The draft is retained in
`drafts/NIFS_SAMPLER_FIELD_LINK.lean.txt`, outside the active package. No
fourth check was run. The finite field abort bound is proved; this additional
actual-field/batch consumer is not an audited link.

The independent review checked the stored interface, concrete primitive
values/counts, direct copier, and all exported work constants. A different
worker checked the polynomial inverse claims. The combined axiom gate for
51 new exports and the affected consumers passed in 297 seconds, including
the dependency rebuild, using only the allowed axioms. Focused primitive,
copy, suffix, and scalar-shortfall jobs took about one to five seconds each.
The inverse execution test passed in seven seconds including its rebuild.
The active source-boundary gate passed. The local map export built all 454
records and its seven tests passed; 451 requirement records were unchanged.
No site was published. The PaperExact execution request remains pending.

## Ring-unit correctness, Pad checks and scalar output laws

Code commit: `dce69b693fc52ff7e7c138198756f9c1400878d6`.

The checked source, reviews, complete passed and failed logs, and two new
stopped drafts are retained in
[NIFS_UNIT_AND_OUTPUT_EVIDENCE.zip](NIFS_UNIT_AND_OUTPUT_EVIDENCE.zip).
This milestone follows `4b4a8ddf`; older archives and drafts are unchanged.

`RingFPolynomial.toPolynomial_ringFMul_mod` proves that the actual
54-coefficient ring product agrees with polynomial multiplication modulo
`X^54 + X^27 + 1`. The coefficient map is injective. The proof makes no
irreducibility assumption about this quotient.
`StoredRingInverseUnit.candidate_eq_unitInverse` then proves correctness
for every unit input of the existing executed candidate. Coprimality is
derived from the unit witness in the proof; the implementation does not
receive an inverse. The original candidate algorithm and its three executed
examples are unchanged. Full inverse work and scalar/storage conversion
work remain open.

`StoredWitnessCheckEntries` now computes the canonical Pad entry, including
native-bar branches, Phi81 kernel sums and Boolean vertex traversal. Its
value and operation-count proofs reach the selected source-return and
work consumers. The selected checker now has three remaining primitive
contracts: public check, dense commitment check and CCS matrix entry.
The Pad result applies to every typed probe and stored witness; it does not
assume an honest transcript or a valid opening.

`SelectivePolynomial` now keeps one table of coefficient/exponent records.
The old semantic monomials are its erasure view. The before/after execution
check compared all 74 ordered coefficients and all 1,036 exponents exactly;
both commands took two seconds. The row-semantics and terminal-consumer
checks passed. This is a storage change, with no new polynomial or relation
identity. It supplies the term reads needed by the unfinished public gate.

The sampler now has both actual-input links and finite output laws:

- `SamplerFieldShortfall.candidateWindow_eq_fieldCandidates` identifies all
  32 ordered field lanes and their 64 low/high candidates. Its selected
  batch theorem equates actual failure with the same field-shortfall event.
- `FieldBatchShortfall.iid_field_batch_shortfall_probability_le` gives
  `a_batch <= 17*u` in the explicit 17-window uniform experiment, where
  `u = choose(64,11)/65536^11`.
- `BitOutputLaw.boundedSample_event_frequency_eq_mixture` proves the exact
  bit-decoder law `(1-a_bits)*UniformSome + a_bits*PointMass(none)`.
- `SamplerOutputLaw.field_output_event_error_le` connects the current scalar
  conversion to the comparison with a uniform successful scalar. For every
  event, its error is at most `32*(M-1)/(M*q) + u`, where `M=2^32` and `q` is
  the Goldilocks prime. It retains abort.

No probability law is assigned to Poseidon2. Uniform successful scalars in
Option have zero abort mass; they are not uniform on the complete Option
type. The total/aborting codec and its interactive adapter remain open.
The model note also retains the exact additive/overwrite equations, the
candidate 12-word C and 36-word R cursor schedules, and the missing joint
prefix/cache, initialization, state-restoration and work proofs. No model,
query budget, retry policy or concrete Poseidon2 transfer was approved.

Two complete attempts stopped under the three-round rule:

| Draft | Final unresolved proof | Retained identity |
|---|---|---|
| `drafts/NIFS_INVERSE_NORMALIZATION_WORK.lean.txt` | The zero-index Array.findIdxRev? scan equation. Full inverse cost remains unproved. | 7,872 bytes; SHA-256 `eb7fe331c7bd0f58ddcdd534891542ff54487c139273d5b11fb1e991b0d82d1b` |
| `drafts/NIFS_STORED_PUBLIC_CHECK.lean.txt` | The 14 port-reader work cases retain free data in unreduced branches before `decide`. The full public gate and its selected integration remain unproved. | 48,887 bytes; SHA-256 `f59264a30d3d069a04b2d18977f6a7099df58ebc0fdabc5a1afaad6a6c715f86` |

Both final files were copied byte for byte outside the active package. No
fourth or narrowed build was run, and no resource setting was raised. The
raw-round foundation had passed earlier; the full public-gate module was
removed after its third failed build. Its passing Pad and term owners stay
active at their own proved scope.

The final combined NIFS audit passed alone in 26 seconds, with 302 complete
records and 39 new exports, using only the allowed axioms. The active
boundary gate passed. The focused output-law and actual-decoder checks took
three seconds each. Independent review found no defect in the new unit,
Pad, term and finite-law claims. The evidence also records a coordinator
queue error: the final focused unit build briefly overlapped the first
field-batch attempt. That timing is not serial performance evidence; the
final combined audit was run alone.

At the 07:20 UTC checkpoint, both external-review files were absent from
this worktree. The original checkout contained the September 4 PiCCS review.
Its public compressed-circuit finding still applies: `nebula/mod.rs`
reexports public F-prime builders that reach the native NIFS circuit and
`padded_row.rs`'s `eval_a.len()+1` check. The selected Lean path keeps Pad and
all 14 matrices separate, but the alternate public route remains a live
authority issue. `N.conformance.owners` stays partial. The overbroad public-path
claim in `CONSTRAINT_TREE.md` was corrected. The old review suite and proof
backends were not run.

The local map update retains all 454 records and changes only the three
relevant NIFS leaves. Its build and seven tests passed. No site was
published. The prepared PaperExact execution remains pending approval.

## Stored public check, batch output and native authority guard

Code commit: `692641958134b46d021639aed088d5574e1c68ce`.

The full public checker's value and work theorems now hold for every typed
stored probe, including arbitrary raw round lists and candidate output values.
`PiCCSStoredPublicCheck.check_value` reaches the existing public predicate;
`check_work_le` bounds the executed named-operation clock by 1,435,806.
The program checks all 28 rounds with width 10, keeps Pad separate from
all 14 genuine matrix families, reads prior claims, checks both points,
and evaluates the retained 74-term table and all 17 norm terms.
`PiCCSStoredWitnessCheck.publicCheck_value` installs this implementation
at the selected key. `scalar_finish_source_iff` and `scalar_check_work_le`
now retain only commitment-check and matrix-entry premises.

`AjtaiSetupV1.Work.coefficient_value` computes the current setup key entry.
Its 27,509-operation bound covers byte reads, all 80 quarter rounds,
shared-array copies, all 16 feed-forward words, first-256-bit packing and
Goldilocks reduction. Fixed-word interpretation retains row and block
range facts; selected dimensions 22 and 4,708,530 satisfy them. This is one
coefficient, not the complete commitment checker.

`StoredCommitment.row_value` now constructs the exact dense Ajtai row from
that generator and the complete stored carrier. The executed direct fold
keeps one key block, witness block, product and accumulator. Its bound is
`223 + blockCount*1,654,418`; it needs no sparse-witness or zero-tail premise.
The row's value and work proofs passed in one second. The selected public
commitment comparison is a separate consumer.

The complete selected-check draft stopped after three full attempts. The
first exposed namespace, fold-bound and index-reduction errors. The next
two were stopped during prolonged elaboration of the selected row path;
no checked selected theorem was produced. The final 14,057-byte, 273-line
file is retained unchanged at `drafts/NIFS_STORED_COMMITMENT_CHECK.lean.txt`
(SHA-256 `4089ad6a02977033253203275054ae5a268c38772677f3e99cd9d53424a9feb3`).
It is outside the active package, has no audit registration and was not
installed in the adapter. No fourth or narrowed check was run. Its source
review is not a validation or approval of its theorems.

`StoredRingArithmetic` materializes every ring result as 54 coefficients.
Its multiplication and addition agree with `ringFMul` and `ringFAdd`, and
its identity agrees with `ringFOne`. The operation bounds are 167,674,
490 and 382 respectively. The imported builder retains its grouped loop
counter. These are proved invocation clocks, not compiled runtime or gas.
The resumed normalization proof also establishes the executed array scan
and copy used by the existing polynomial encoding; full inverse work is
still a separate obligation.

`RingFFrobenius.quotient_pow_card_pow` proves `a^(q^27)=a` in the existing
Phi81 polynomial quotient. `unit_inverse_product` derives the unit inverse
power identity. The proof derives the quotient characteristic, checks
`q^27 mod 81=1`, and uses the coefficient and root maps. It assumes no
irreducibility and uses no pointwise RingF power. This algebra result does
not itself validate an executable inverse or its work.

`StoredRingPowerInverse` supplies a separate executed binary-power candidate.
It proves the returned unit inverse and a bound of 579,844,861 named
operations, including exponent construction. It receives only the input
array. Symbolic wrapper and result lemmas prevent the proof from unfolding
the 1,729-level selected call. Two attempts were stopped after rapid memory
growth; the structural third attempt passed in two seconds without a
resource-setting increase.

The executed comparison matched all 54 inverse and product coefficients
for `2`, `X`, `1+X`, and a dense `(1+X)^53` coefficient vector. All recorded
clocks were 292,431,123, within the proved bound. Timed IO stores force each
complete result before the end timer. The power routine took 10.4--13.2
seconds per input; extended GCD took 15--150 milliseconds. The complete
comparison passed in 50 seconds. An earlier pure-let timing was invalid
and is retained only as functional evidence. This substantial cost
difference is explicit: no existing inverse path was replaced, and the
extractor's final inverse choice and representation link remain open.

The finite batch law and its actual sampler consumers also pass. For 17
independent uniform 32-field windows, every output event differs from a
uniform successful ordered scalar list in Option by at most
`544*(M-1)/(M*q)+17*u`, where `M=2^32` and
`u=choose(64,11)/65536^11`. The stronger separate abort-only bound stays
`17*u`. The actual sampler list and successful final state equal those of
the same ordered field decoder. These deterministic identities give no
independence law for Poseidon2 and no adaptive-query or rewinding law.
The precise remaining transfer obligations stay in the model note.

The new comparison-only scalarwise totalizer also passes. For any supplied
valid fallback scalar, its one-scalar and 17-scalar error bounds are the
same. The actual successful batch's ring list agrees with that comparison.
Failed traces, full verifier acceptance inclusion, inverse-codec fibers
and their work, and the ideal permutation transfer remain open. This
comparison changes no actual sampler behavior and selects no new model.

The normal native NIFS header-bundle entry now rejects with an explicit
unsupported-circuit error. It cannot call the compressed PiCCS composition;
that body and its helper callers are test-only. The regression failed
against the former body, then passed against the guard. It checks the exact
error and unchanged rows, columns, all matrix triplets, witness, encoding
trace, transcript state, cursor and bindings at this entry.

Public Nebula F-prime types remain available. Ordinary profile discovery
fails at the guard; an artifact-restored profile can reach it later during
recursive synthesis. Existing error paths also reach the WASM caller. This
is an intentional rejection of the unsupported circuit, with no replacement
backend or recursive implementation. The enclosing caller can have changed
its own prelude before calling NIFS, so the entry's unchanged-state test is
not an atomicity claim about the complete caller.

The focused Rust regression passed in 19.38 seconds including compilation;
its executed test took under 0.01 seconds. The normal release check for
`neo-fold-clean` and `neo-wasm` passed in 8.27 seconds. `cargo fmt --all`
passed. Four helper methods with only test callers now have `cfg(test)`;
their source review passed. The 08:20 UTC checkpoint read both review paths
in both checkouts. The primary checkout's September 4 finding motivated
this guard. The old complete review suite and proof backends did not run,
and no new broad conformance verdict is claimed.

The initial combined NIFS axiom check passed all 328 registrations,
including 26 new exports, in two seconds, using only the allowed axioms.
The final gate also includes the dense row, power inverse and totalized
comparison exports; its outcome is recorded below. Independent
reviews checked the public gate, selected integration, finite batch laws,
coefficient generator, stored arithmetic and complete audit records. The
coordinator separately checked the quotient proof and native guard source.
Source, reviews and full validation records are retained in
[NIFS_PUBLIC_AND_BATCH_EVIDENCE.zip](NIFS_PUBLIC_AND_BATCH_EVIDENCE.zip).

The final dependency-aware NIFS axiom build passed all 336 records,
including 34 new exports, in four seconds (3,723 jobs). The source-boundary
gate also passed after the stopped draft was removed. A separate worker
verified complete audit coverage and all 20 committed source hashes.

The local map build produced all 454 records and its seven tests passed.
Only the three relevant NIFS records changed; the other 451 records and
the three records' connection statuses are preserved. At 09:20 UTC, both
protected review paths were read again in both checkouts; their contents
were unchanged. No site was published or model approved.

## Stored matrix and field-preimage milestone — `4049d613`

The checked code cut is `4049d6133972475eaa4cd61e27d447da48349392`. Four modules add 25 checked exports.
The dependency-aware NIFS audit passed with 361 complete records, only the
permitted axioms, and a three-second build. Static checks passed. The first
static check found an in-progress, unregistered PiDEC source draft; that draft
was moved out of the active module tree before the successful check.

- `Export/MatrixProgram/SparseWork` constructs and scans actual stored lists.
  It preserves order, zero coefficients and duplicate columns. Add costs
  `10*left.entries.length+13`, scale costs `14*entries.length+13`, and
  coefficient lookup costs at most `10*entries.length+8` named operations.
- `Export/MatrixProgram/RetainedWork` constructs only the requested slot,
  checks the existing complete block geometry, and returns the same form.
  Its returned list has exactly the slot width. The bound is
  `7*width^2+49*width+25`, including the actual Horner tail traversals.
- `Export/MatrixProgram/CoefficientWork` consumes a stored sparse row and
  uses all 54 lanes of the existing Phi81 coefficient expansion. Each live
  logical source lane performs a duplicate-aware scan. The bound is
  `54*(10*entries.length+8+kernelWork+14+8)+3+8`. Its value theorem consumes
  equality to the existing matrix row; row construction remains a caller
  cost. The source-lane guard preserves contributions to queried completion
  columns and does not incorrectly zero every such query.
- `PiRlcSampler/FieldPreimageRectangle` gives computable rank/unrank
  equivalences for reject, one accepted residue, accepted, and unrestricted
  candidate classes. Their sizes are 1, 13107, 65535 and 65536. For low/high
  classes A/B the field preimage count is
  `(2^32-1)*|A|*|B| + [0 in A and 0 in B]`. The extra field is `q-1`.
  Low-class index varies first, then high-class index, then field block.
  Raw alphabet index zero means the centered coefficient -2.

These are named operation clocks, not machine instruction or bit-complexity
proofs. Work counters exclude their own instrumentation. Source review found
and corrected four missing scale operations and the zero-result construction
charges before the final checks. The complete source and failed/successful
logs are retained in [NIFS_MATRIX_AND_PREIMAGE_EVIDENCE.zip](NIFS_MATRIX_AND_PREIMAGE_EVIDENCE.zip).

Two complete criteria stopped under the three-round rule. The resumed dense
commitment checker again did not finish its final value proof; its last
process was stopped after prolonged checking. Diagnostics establish that the
preceding declarations completed in round 2. Its complete resumed draft is
`drafts/NIFS_STORED_COMMITMENT_CHECK_RESUMED.lean.txt`; the historical draft
is unchanged. `TotalizedComparison` round 3 removed the earlier recursion-depth
errors but left two PiDEC decision/output equalities unproved. Its complete
source is `drafts/NIFS_TOTALIZED_COMPARISON.lean.txt`. Both drafts are inactive
and unvalidated, with no consumer or audit registration. No narrower fourth
check was run. The selected checker still requires commitment and matrix
contracts; full actual-verifier acceptance inclusion remains open.

The new matrix primitives have their local consumers above. They do not
supply the selected row generator or a complete selected entry bound. The
rectangle equivalence does not supply complete 32-field decoder fibers, a
random inverse sampler, or its work. Active next work uses typed source-row
generation and separate success/abort fiber counts. The protected reviews
were read at resumption at 15:35 UTC in both checkouts and were unchanged.
No Rust behavior, protocol profile, security assumption, backend or site
publication changed in this milestone.

## Active criteria

Complete the two checker leaves: dense selected-key commitment check and
selected matrix entry. The public SumCheck gate is now proved and installed. Commitment work must include actual key expansion for
arbitrary stored witnesses, and matrix work must include package-row
production and lookup. Stored sparse operations, retained-slot construction
and 54-lane coefficient expansion now have checked value/work proofs. The retained dense-commitment preparation gives a
structural route using one 54-lane key block at a time, with no sparse-witness
premise or full-key table.

Complete the actual stored producer/checker law, inverse and conversion work,
and the full representation/runtime links. The existing inverse value theorem and resumed normalization work are
proved; full inverse work and its chosen executable consumer remain open. Preserve the approved seed and same-key MSIS premise, which supplies
no numerical hardness bound.

Keep the exact replay, matrix/raw-assignment evidence and independent review
at their stated scope. The normal native NIFS entry now rejects the compressed composition.
This scoped guard does not authorize Stage 2, supply a replacement backend
or grant a new complete conformance verdict. The prepared same-input PaperExact R/D comparison
still needs explicit approval before execution and broader per-phase closure.

Complete the mathematical transcript, codec, state-restoration and error/work
connections before requesting a precise Fiat–Shamir model decision. The scalar and independent-batch output comparisons and 17-window abort
bound are proved. The actual-state joint law, adaptive/retry law and exact
Poseidon2 transfer are not proved.

The older `protocol-contract/security-reduction.md` uses a different
transcript, sampler and profile; its numerical limits are not evidence for
this instance. Use the normative paper in the primary checkout. The frozen
Lean corpus remains unused.

Full HyperNova history extraction, Stage 2, proof-backend execution and site
publication remain outside this task. Local diagnostics do not replace
independent review or required owner approval.
