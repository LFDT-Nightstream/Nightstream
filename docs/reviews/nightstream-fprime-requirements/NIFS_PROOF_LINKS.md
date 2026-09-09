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
| `N.security.binding` | The actual extraction collision reaches the selected public-seed MSIS assumption with the required norm and execution scope. | Partial; reduction after context preparation checked |
| `N.security.fiat_shamir` | The selected Poseidon2 transcript and bounded sampler have a justified security connection under the authorized model. | Open |
| `N.conformance.chain` | One nonzero selected-key input and proof have matching Lean and optimized Rust phase values and final output, with required mutations. | Open |
| `N.conformance.executed` | Retained commands, inputs, outcomes, and source identities establish the stated execution scope. | Open |
| `N.conformance.owners` | The checked chain consumes the existing semantic, transcript, assignment, and caller owners. | Open |

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

## Active criterion

Connect the original context-generation call and its actual preparation
clock to the same reduction. Include preprocessing against the fixed public
setup. The current work theorem starts after that context is supplied. Then
apply only the approved public-seed MSIS premise, with its correct execution
scope. The approved premise supplies no numerical success bound.

The older `protocol-contract/security-reduction.md` uses a different
transcript, sampler, and profile. Its numerical query limits and security
terms are not evidence for this selected NIFS instance.

The primary checkout contains the local normative paper files. The isolated
worktree uses those files for reading. The frozen package instructions are
absent from both checkouts; no frozen proof files are used.

Full HyperNova history extraction, Stage 2, proof-backend execution, and site
publication remain outside this task. Pending independent approvals must not
be replaced by local diagnostic results.
