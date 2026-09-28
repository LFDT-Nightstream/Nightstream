# Formal boundaries for the proposed goal split

Reviewed source: `4fc02857c3aa4207f3290739cc62d794ba86f5f9`.
This is a source audit. It changes no production code and runs no build.
The parent review owns current validation and Rust evidence.

## Scope and assumptions

The audit checks proposal 3 (SuperNeo verifier circuit, then HyperNova lifecycle)
and the Lean part of proposal 4 (full formal F′ package, then production execution).
The existing Stage 1 contract and fixed profile remain the parent contract.
A proposal changes the milestone boundary only; it does not authorize a second
relation, a second package, or changes to the owner files.

The key assumption is that Goal A must be a useful, precise result. This is
necessary: a milestone cannot close if its name still hides an open condition.
A separate repository or independent circuit emitter is not necessary.
The existing accumulator predicate already provides a useful mathematical boundary.

## Main result

Proposal 3 has a real mathematical boundary in the code, but its selected-package
proof is still open. It keeps the main current proof difficulty in Goal A.
It can separate the remaining HyperNova work, but it is not a strong route to a
much faster first completion.

Proposal 4 preserves the current architecture most closely. Much of the formal
infrastructure is present, including the fixed point, domain bound, application
builder, phase assemblers, and terminal predicate. Goal A still needs the complete
arbitrary-assignment soundness and selected-context connection. It does not remove
the main proof difficulty.

The code supports these statements more precisely than a phase-count estimate.
No completion percentage or duration follows from this audit.

## The SuperNeo boundary that already exists

`Lifecycle.Stage1.Accumulator.Holds` means exactly:

```text
Nifs.PaperNonInteractive.verify selectedKey running fresh proof = some output
```

Its `vk` argument is unused: the relation and Ajtai key select the folding
semantics. Context and public-state binding are additional HyperNova obligations.
The theorem `holds_iff_checks` decomposes it into the PiCCS Boolean check,
PiDEC Boolean check over the verifier-computed PiRLC parent, and exact computed
output. [Accumulator.lean:31](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/NightstreamFPrime/Lifecycle/Stage1/Accumulator.lean:31)
[Accumulator.lean:49](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/NightstreamFPrime/Lifecycle/Stage1/Accumulator.lean:49)

The composition theorem `phases_imply_holds_of_wiring` already connects three
exact phase predicates to this accumulator result. Its `PhaseWiring` record
requires:

- the decoded PiCCS proof fields to be the proof fields consumed by the key;
- PiRLC inputs to equal the key's PiCCS outputs;
- the sampler's initial state to equal the PiCCS outgoing transcript state;
- the PiDEC attempt to use the computed parent and the same child messages;
- the returned running instance to equal the computed PiDEC output.

These are explicit value-agreement requirements. They are not supplied by the
existence of phase files. [AccumulatorSemantics.lean:1033](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/NightstreamFPrime/Layout/Stage1/AccumulatorSemantics.lean:1033)
[AccumulatorSemantics.lean:1101](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/NightstreamFPrime/Layout/Stage1/AccumulatorSemantics.lean:1101)

There is an exported `AccumulatorPackage.circuitPackage_implies_accumulatorHolds`
theorem. It takes the older physical `Data.circuitPackage` environment and
PiRLC variable-scope assumptions. It does not state arbitrary-assignment soundness
for the selected per-application direct fixed-point plan.
[AccumulatorPackage.lean:26](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/NightstreamFPrime/Export/Stage1/AccumulatorPackage.lean:26)

Those scope assumptions are ordinary static obligations, not assumed protocol
acceptance: the sampler requires its initial expressions to be below the start
offset, and the combination families require their challenge and input expressions
to be below their start offsets.
[SamplerChain.lean:151](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/NightstreamFPrime/Lifecycle/PiRLC/v1_1/SamplerChain.lean:151)
[CombinationFamily.lean:177](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/NightstreamFPrime/Lifecycle/PiRLC/v1_1/CombinationFamily.lean:177)

The direct accumulator theorem composes the retained plan but still takes
`base`, `groupValue`, `products`, `Encodes`, retained `Semantics`, and the
scope assumptions. This is a reusable proof for agreeing representations. It does
not itself derive agreement from every accepted assignment.
[DirectAccumulatorCommonSemantics.lean:26](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/NightstreamFPrime/Export/Stage1/DirectAccumulatorCommonSemantics.lean:26)

## What is already proved for actual accepted assignments

| Surface | Exact result already present | Limit of that result |
|---|---|---|
| PiCCS | Selected rows and public-input equality imply the exact key's `piCcsCheck = true`. | PiDEC fields remain template fields in this theorem. |
| PiRLC products | Selected rows imply indexed sums of all actual source contributions. Each contribution uses the actual product challenge form and actual PiCCS source form. | The theorem does not identify product challenges with the verifier sampler result. |
| PiDEC | Selected rows and the public boundary imply the exact PiDEC phase predicate in an environment decoded from that assignment. | The parent must still be identified with the key-computed PiRLC parent. |
| Running output | The decoded transition's full output equals the running state in the actual output preimage. | It must still be connected to the key-computed PiDEC output. |
| Application | The selected application rows imply the decoded application step. | This is one component of the full F′ relation. |
| Base step | Selected rows, public-input equality, and iteration zero imply the complete base step under the context read from the assignment. | It does not by itself select the verifier's expected context. |
| State context | The output and prior preimages contain the same context words. | Preservation of a value does not prove it equals the selected verifier context. |

Evidence:
[ActualStep.lean:314](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/NightstreamFPrime/Export/Stage1/ActualStep.lean:314);
[ActualPiRLCValues.lean:35](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/NightstreamFPrime/Export/Stage1/ActualPiRLCValues.lean:35);
[ActualPiRLCValues.lean:240](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/NightstreamFPrime/Export/Stage1/ActualPiRLCValues.lean:240);
[ActualPiDEC.lean:266](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/NightstreamFPrime/Export/Stage1/ActualPiDEC.lean:266);
[ActualRunningTransition.lean:181](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/NightstreamFPrime/Export/Stage1/ActualRunningTransition.lean:181);
[ActualApplicationStep.lean:77](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/NightstreamFPrime/Export/Stage1/ActualApplicationStep.lean:77);
[ActualStep.lean:162](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/NightstreamFPrime/Export/Stage1/ActualStep.lean:162);
[ActualPreimageFraming.lean:142](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/NightstreamFPrime/Export/Stage1/ActualPreimageFraming.lean:142).

The exact current root reduction is
`ActualStep.selectedRowsAndPublic_step_iff_baseOrPiDec`.
For an arbitrary assignment, selected rows, the actual public-input equation,
and a four-word public digest, it proves:

```text
full StepHoldsFor under the decoded context
iff
iteration = 0
or
(piDecCheck selectedKey decodedRunning decodedFresh decodedProof = true
 and selectedKey.output ... = some decodedNextRunning)
```

PiCCS is already discharged in this reduction. Neither of the two recursive
conditions is discharged by the equivalence.
[ActualStep.lean:350](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/NightstreamFPrime/Export/Stage1/ActualStep.lean:350)

This is why the SuperNeo-circuit split retains the current difficult work:
its required result is exactly the missing recursive NIFS part.

## The missing connections, with existing proof inputs

### Actual sampler values must be verifier-derived

`DirectPiRLCSamplerCompletePhaseSemantics.rowsZero_implies_piRlcPhaseHolds`
already proves complete phase semantics from rows, but also takes the separately
supplied source values and their `Encodes` agreement.
[DirectPiRLCSamplerCompletePhaseSemantics.lean:78](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/NightstreamFPrime/Export/Stage1/DirectPiRLCSamplerCompletePhaseSemantics.lean:78)

For an arbitrary assignment, the remaining route must derive the sampler values
and their state from that same assignment and show that the product forms use them.
There must be no caller-supplied challenge equality at the final boundary.

The next reusable semantic edge is
`AccumulatorSemantics.piRlcChallenges_eq_key_of_initialState`. It needs a
valid PiRLC phase and equality of its initial transcript state with the key's
PiCCS outgoing state. It then proves that the key returns exactly those challenges.
[AccumulatorSemantics.lean:461](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/NightstreamFPrime/Layout/Stage1/AccumulatorSemantics.lean:461)

### Actual sums must be the computed PiDEC parent

After challenge agreement, the already-proved product sums must be identified
with the key's 17-source commitment, public-input, separate Eval_K, and separate
14-matrix Eval_A combinations. The PiDEC input point must also agree with the
PiCCS point. The selected PiDEC decoder already has the point-equality result.
[ActualPiDEC.lean:131](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/NightstreamFPrime/Export/Stage1/ActualPiDEC.lean:131)

The existing `keyPiDecAttempt_eq_some_of_wiring` theorem consumes challenge,
parent, and attempt equalities. The existing `piDecCheck_eq_true_of_attempt`
then converts the actual PiDEC phase into the exact verifier check.
[AccumulatorSemantics.lean:499](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/NightstreamFPrime/Layout/Stage1/AccumulatorSemantics.lean:499)
[AccumulatorSemantics.lean:551](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/NightstreamFPrime/Layout/Stage1/AccumulatorSemantics.lean:551)

### Actual child messages and output must be the key's result

The final decoded proof must take PiDEC commitments and evaluations from the
accepted assignment. `ActualStep.withDecodedPiCCS` deliberately fills only the
PiCCS fields; its remaining fields come from a template.
[ActualStep.lean:61](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/NightstreamFPrime/Export/Stage1/ActualStep.lean:61)

The child-message decoding must be connected to `inputAttempt`. The computed
child public-input split and all output fields must then agree with the running
transition's decoded output. The reusable endpoint is
`keyOutput_eq_some_of_attempt`.
[AccumulatorSemantics.lean:587](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/NightstreamFPrime/Layout/Stage1/AccumulatorSemantics.lean:587)

These are concrete proof obligations. This audit does not establish that they all
follow from the current rows. A failed proof attempt may identify a missing
constraint. It would not by itself justify a broad rewrite.

## Why a separate SuperNeo circuit is not already an isolated package

The semantic accumulator result is independent of application execution, but its
current physical input layout is part of F′:

- PiCCS running inputs are read from the pilot prior-state serialization.
- Fresh public inputs reuse pilot hash columns.
- Expected-context words sit at the end of the pilot layout.
- PiRLC reuses PiCCS output values and the outgoing transcript state.
- PiDEC reuses PiRLC output forms.
- Running output is carried in the next-state preimage.
- The final retained width depends on the selected application, and the
  self-derived relation is the complete ordered application plan.

Sources:
[PiCCSInputs.lean:30](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/NightstreamFPrime/Layout/Stage1/PiCCSInputs.lean:30);
[PiCCSInputs.lean:226](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/NightstreamFPrime/Layout/Stage1/PiCCSInputs.lean:226);
[PiRLCInputs.lean:65](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/NightstreamFPrime/Layout/Stage1/PiRLCInputs.lean:65);
[PiRLCInputs.lean:115](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/NightstreamFPrime/Layout/Stage1/PiRLCInputs.lean:115);
[PiDECInputs.lean:119](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/NightstreamFPrime/Layout/Stage1/PiDECInputs.lean:119);
[ActualRunningTransition.lean:181](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/NightstreamFPrime/Export/Stage1/ActualRunningTransition.lean:181);
[PerApplicationFixedPoint.lean:24](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/NightstreamFPrime/Export/Stage1/PerApplicationFixedPoint.lean:24);
[PerApplicationFixedPoint.lean:76](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/NightstreamFPrime/Export/Stage1/PerApplicationFixedPoint.lean:76).

Therefore the smallest implementation of proposal 3 is a theorem boundary inside
the existing package. It can use decoded running/fresh inputs without proving
their complete HyperNova history until Goal B. It should not first remove the
pilot, relocate physical columns, or emit a second production relation.

The current digest-only transcript is a further boundary that Goal A must name.
`ProductionKey.priorDigest` decodes a digest from the fresh public input, and
`publicInputBlocks` ignores the running argument. The key assumes the pilot has
already bound the complete running statement through that digest. The oracle's
initial-state theorem does not prove this binding; it says that the oracle starts
from the supplied prior state.
[ProductionKey.lean:100](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/NightstreamFPrime/Lifecycle/ProductionKey.lean:100)
[ProductionKey.lean:111](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/NightstreamFPrime/Lifecycle/ProductionKey.lean:111)
[ProductionKey.lean:249](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/NightstreamFPrime/Lifecycle/ProductionKey.lean:249).

The selected circuit has a useful actual-assignment result here:
`ActualPiCCSInputs.selectedRowsAndPublic_imply_phaseAndHashes` proves that the
PiCCS running value is the running value in the decoded prior preimage, and the
fresh public input is the encoded hash of that same preimage.
[ActualPiCCSInputs.lean:97](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/NightstreamFPrime/Export/Stage1/ActualPiCCSInputs.lean:97).

A standalone native core can close execution parity without the recursive circuit.
But a secure folding-core claim must keep an authenticated statement boundary:
recompute/check the prior-state digest from authoritative running/state data, or
state this as an explicit caller obligation. Equal caller-supplied digest fields
alone do not establish it. This is relevant to proposal 1 as well as proposal 3.
No change to the approved transcript schedule is implied by this finding.

The distinction must be explicit. A theorem about a folding predicate is a useful
formal milestone. The current architecture's stronger per-phase completion
standard also requires real Rust prover and verifier use. Calling Goal A
production-complete would require those implementation obligations in Goal A.

## Proposal 3: precise split

### Goal A: prove the selected folding verifier circuit

Goal A should establish that the selected matrix rows, with the required
public/constant boundary, imply exact `Accumulator.Holds` for proof and instance
values decoded from that assignment.

The sampler equality, exact PiRLC parent, actual PiDEC message decoding, and
computed output equality all remain in A. Layout preservation, coverage, the
selected profile/relation/Ajtai authority, and relevant soundness assumptions
also remain in A. If A claims executable or conformance closure, its complete
nonzero Lean–Rust value, matrix, assignment, mutation, and review evidence remains
in A too.

A can defer the application step, HyperNova base/recursive dispatch,
public-state history, selected-context binding in recursive preimages, outer
terminal use, and final production lifecycle integration. These are B's work.
Where the existing circuit already supplies their rows, A can leave them in place.
A secure SuperNeo claim must still state how its digest authenticates the input
statement. It cannot defer that obligation and then claim unconditional security.

### Standing

The semantic boundary and the conditional phase-composition proof already exist.
The actual-assignment PiCCS, PiDEC, product-sum, and running-state proofs provide
substantial inputs. The selected folding-circuit root is still missing.

This is a clear ownership split, but it does not avoid the current sampler and
cross-phase proof work. It is not the preferred split if the main purpose is a
much earlier completed result.

## Proposal 4: formal F′ package, then production execution

### Formal pieces already present

| Piece | Code evidence | What the code establishes |
|---|---|---|
| Concrete F′ relation | [Relation.lean:98](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/NightstreamFPrime/Lifecycle/Relation.lean:98) | `StepHoldsFor` is the concrete fixed augmented transition. |
| Phase assemblers | [Completeness.lean:279](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/NightstreamFPrime/Lifecycle/PiCCS/v1_1/Completeness.lean:279), [Completeness.lean:499](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/NightstreamFPrime/Lifecycle/PiRLC/v1_1/Completeness.lean:499), [Completeness.lean:266](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/NightstreamFPrime/Lifecycle/PiDEC/v1_1/Completeness.lean:266) | Each phase has a FormalCircuit with soundness/completeness and its semantic interface. |
| Stage 1 assembler | [Formal.lean:158](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/NightstreamFPrime/Lifecycle/Stage1/Formal.lean:158), [Formal.lean:329](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/NightstreamFPrime/Lifecycle/Stage1/Formal.lean:329) | Eight ordered child calls; rows imply their specifications under their assumptions. |
| Root completion | [AssemblerApplicationCompleteness.lean:202](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/NightstreamFPrime/Layout/Stage1/AssemblerApplicationCompleteness.lean:202) | Canonical layout supplies Stage 1's proof-only completion record. |
| Physical preservation | [PreservationClosure.lean:253](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/NightstreamFPrime/Layout/Stage1/PreservationClosure.lean:253) | Physical rows imply the compact child specifications. Full-step bridge still takes a representation record. |
| Self-derived relation | [PerApplicationFixedPoint.lean:90](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/NightstreamFPrime/Export/Stage1/PerApplicationFixedPoint.lean:90) | Rebuilding with the derived matrices gives exactly the same plan. |
| Domain closure | [Poseidon2HashChainV1Package.lean:172](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/NightstreamFPrime/Export/Stage1/Poseidon2HashChainV1Package.lean:172) | The selected logical rows and carrier fit the common 2^28 domain. |
| Matrix-program identity | [Poseidon2HashChainV1Package.lean:190](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/NightstreamFPrime/Export/Stage1/Poseidon2HashChainV1Package.lean:190) | The emitted matrix program is exact for the selected structural plan and source rows. |
| Canonical package and context | [PerApplicationCanonicalPackage.lean:123](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/NightstreamFPrime/Export/Stage1/PerApplicationCanonicalPackage.lean:123), [PerApplicationCanonicalPackage.lean:533](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/NightstreamFPrime/Export/Stage1/PerApplicationCanonicalPackage.lean:533) | The package, relation, application, and setup have one Lean-owned construction. |
| Terminal semantics | [PerApplicationTerminal.lean:35](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/NightstreamFPrime/Export/Stage1/PerApplicationTerminal.lean:35) | The selected package/key/context instantiate the outer terminal predicate. |

The selected source proves 6,377,559 logical active rows, logical width
254,260,583, carrier width 254,260,620, 29,225,729 physical rows, and 29,344,425
physical columns. These are exact source facts, not conformance status.
[Poseidon2HashChainV1Package.lean:140](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/NightstreamFPrime/Export/Stage1/Poseidon2HashChainV1Package.lean:140)
[Poseidon2HashChainV1Setup.lean:29](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/NightstreamFPrime/Export/Stage1/Poseidon2HashChainV1Setup.lean:29)

### What the existing closure theorem does not finish

`Poseidon2HashChainV1Closure.rowsZero_implies_stepHoldsFor` takes
`RawValues` and rows on `(bound raw).assignment`.
The raw packet contains source values, product-group values, and first-54 products.
Its assignment function chooses a canonical block encoding.
[Poseidon2HashChainV1Closure.lean:35](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/NightstreamFPrime/Export/Stage1/Poseidon2HashChainV1Closure.lean:35)
[PerApplicationCanonicalAssignment.lean:59](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/NightstreamFPrime/Export/Stage1/PerApplicationCanonicalAssignment.lean:59)

`bound` overwrites the four expected-context words with the selected digest
before constructing the assignment. The final theorem thus cannot substitute for
the arbitrary-assignment theorem merely because its header says accepted rows.
[PerApplicationVerifierBoundAssignment.lean:50](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/NightstreamFPrime/Export/Stage1/PerApplicationVerifierBoundAssignment.lean:50)

Goal A must complete the actual-assignment root described above and close the
selected-context connection. The actual context-preservation theorem proves that
prior/output words agree; it does not prove they equal the verifier's selected
digest. That connection belongs at the actual acceptance boundary, with the
permitted named collision/binding alternative where required.

### Security and terminal status

The existing security theorem takes a valid full step and the explicit
`LowNormInvertibility` boundary, then returns the step plus a base or recursive
`SecurityOutcome`.
[PerApplicationSecurity.lean:395](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/NightstreamFPrime/Export/Stage1/PerApplicationSecurity.lean:395)

`SecurityOutcome` is a deterministic disjunction: extracted knowledge, mixing
failure, sumcheck failure, parent-opening binding failure, PiRLC forking failure,
or missing valid PiDEC child openings. It is not a theorem that bounds each
failure's probability.
[PaperSecurityComposition.lean:466](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/NightstreamFPrime/Spec/Folding/Nifs/PaperSecurityComposition.lean:466)

The owner goal permits remaining SuperNeo reductions as an explicit assumption.
This audit does not add a requirement to formalize an adversary/probability model.
The unresolved issue for a complete formal package is the exact acceptance-to-step
connection and the intended security composition, including terminal use.

Terminal semantics are further along than the phrase “terminal phase missing”
suggests. The base terminal form checks iteration zero and equal initial/current
state. The recursive form checks the prior public link, all 16 running CE
openings, and the fresh CCS opening. It performs no extra NIFS fold.
[Terminal.lean:45](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/NightstreamFPrime/Lifecycle/Stage1/Terminal.lean:45)
[Terminal.lean:75](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/NightstreamFPrime/Lifecycle/Stage1/Terminal.lean:75)
[Terminal.lean:108](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/NightstreamFPrime/Lifecycle/Stage1/Terminal.lean:108)

The package terminal metadata covers the complete existing relation and adds no
rows or columns. The existing terminal theorems are exact definition-equivalence
results; they do not establish production verifier execution.
[TerminalPackage.lean:20](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/NightstreamFPrime/Export/Stage1/TerminalPackage.lean:20)
[PerApplicationTerminal.lean:60](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/NightstreamFPrime/Export/Stage1/PerApplicationTerminal.lean:60)

Two older listed debts are already resolved in source: PiDEC has a constructive
decision procedure, and its operational split rejects an out-of-bound parent.
Do not count those as remaining new implementation work.
[PaperAlgebra.lean:519](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/NightstreamFPrime/Lifecycle/PaperAlgebra.lean:519)
[PaperVerifier.lean:83](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/NightstreamFPrime/Spec/Folding/PiDEC/PaperVerifier.lean:83)

### Precise split and standing

Goal A retains the full actual-assignment proof, selected-context connection,
complete intended deterministic/security composition, canonical layout/package
proofs, and selected terminal semantics. Existing fixed-point and domain theorems
can be reused while the relation stays unchanged.

Goal B owns the exact Rust implementation links and complete production path,
including loading the validated package as the only relation and the separately
approved backend's final proof/verification execution. A formal milestone can
close before B, but it cannot be called conformance-closed or production-closed.

The current owner goal requires phase conformance in order. Deferring all
conformance until Goal B would change that work order; a written split needs the
owner to make this choice explicitly. This review itself does not change it.

Proposal 4 is the least disruptive split. It separates the final production
obligation, but the key proof work stays in A. It offers a clean formal milestone,
not evidence of a short route to that milestone.

## Rejected extra work

- Extracting a new SuperNeo package is not required to express or prove the existing accumulator boundary.
- Re-proving the unchanged fixed point or domain bound is not required to assess either proposal.
- A complete new probabilistic-security framework is not required by the owner's permitted explicit-assumption boundary.
- Adding a terminal folding circuit is not justified: the current outer terminal verifier reuses the relation and performs no extra fold.
