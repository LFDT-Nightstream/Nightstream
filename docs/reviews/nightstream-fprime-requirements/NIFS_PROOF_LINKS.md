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
| `N.security.binding` | The actual extraction collision reaches the selected public-seed MSIS assumption with the required norm and execution scope. | Partial; preparation and reduction work checked |
| `N.security.fiat_shamir` | The selected Poseidon2 transcript and bounded sampler have a justified security connection under the authorized model. | Open |
| `N.conformance.chain` | One nonzero selected-key input and proof have matching Lean and optimized Rust phase values and final output, with required mutations. | Partial; selected C/R/D values and witnesses checked |
| `N.conformance.executed` | Retained commands, inputs, outcomes, and source identities establish the stated execution scope. | Partial; current C/R/D commands and outputs retained |
| `N.conformance.owners` | The checked chain consumes the existing semantic, transcript, assignment, and caller owners. | Partial; selected PiDEC knowledge consumer now checked |

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
assignment used by the current C/R/D run. The full native `nifs::verify`
entry point also requires a selected matrix cache. Its header has no matrix
contents, and no matching verified cache artifact was found in the checkout.
Creating a self-consistent receipt would not establish cache correctness.
That caller boundary remains open.

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

## Active criteria

Discharge the selected extraction primitive, accessor and checker contracts
at their existing owners, with work bounds on their actual representations.
Then apply only the approved same-key MSIS hardness premise. It supplies no
numerical success bound.

Connect the same checked C/R/D proof to the complete NIFS caller, including
the selected matrix-cache and prior-parent authority checks. Consume the
retained exact matrix and raw-assignment results at their stated scope and
complete any remaining gates. Keep local execution evidence separate from
independent phase approval.

Finish the structural R-parent equality and its audited weak-success
consumer. The unproved draft is retained in the cache-width evidence archive.

The lookup for an existing approved Fiat-Shamir model is pending. No new
Poseidon2 idealization, query budget, or security-transfer assumption was
introduced.

The older `protocol-contract/security-reduction.md` uses a different
transcript, sampler, and profile. Its numerical query limits and security
terms are not evidence for this selected NIFS instance.

The primary checkout contains the local normative paper files. The isolated
worktree uses those files for reading. The frozen package instructions are
absent from both checkouts; no frozen proof files are used.

Full HyperNova history extraction, Stage 2, proof-backend execution, and site
publication remain outside this task. Pending independent approvals must not
be replaced by local diagnostic results.
