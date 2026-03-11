# ProtocolRelations

## Purpose

- **What it is**: The CCS and CE relation predicates on protocol-target context. Defines `ccsRelation` (protocol target holds), `ceRelation` (CCS plus accepted SumCheck transcript), and `ceRelaxedRelation` (CCS only). Builds `sumcheckInstanceOfContext` from context and `SumCheckTransitionWitness` carrying round-consistency facts.
- **Key property**: `ceRelation ctx ↔ ccsRelation ctx ∧ ∃ tr, SumCheckAccepted (sumcheckInstanceOfContext ctx) tr`; and `ceRelation ctx → ceRelaxedRelation ctx`.
- **Protocol role**: PiCCS, PiRLC, PiDEC, and FoldingProtocol depend on these relation predicates for the Section 7 folding reductions (Π_CCS, Π_RLC, Π_DEC).

## Target Formulas

- `ccsRelation(ctx) ↔ protocolTargetProp(ctx)` (CCS = protocol target)
- `ceRelation(ctx) ↔ ccsRelation(ctx) ∧ ∃ tr, SumCheckAccepted inst tr` where `inst = sumcheckInstanceOfContext ctx`
- `ceRelaxedRelation(ctx) ↔ ccsRelation(ctx)`
- `sumcheckFullFieldDenominatorAlignment(ctx) ↔ ctx.cset.size = Goldilocks.q`
- `GoldilocksFullFieldLundBoundary.ofCsetCardinality(hCard)` packages the active Goldilocks/full-field Lund setup boundary from `hCard : ctx.cset.size = Goldilocks.q`
- `ProtocolSection71Objects(ctx)` packages one shared Definition-14 `GlobalParams` package, one shared norm bound, and one coherent CCS/CE tuple pair for the current theorem instance
- `ProtocolSection71Realization(ctx)` packages a realized Section 7.1 Definition-14 global-parameter package, one shared norm bound, and the induced CCS/CE presentation of the compact protocol relations
- `ProtocolSection71Provider(ctx)` packages one concrete Section 7.1 theorem instance: a shared Definition-14 realization together with concrete proof-system `CCS.Holds` / `CE.Holds`
- `ProtocolSection71Specialization(ctx, hInst)` packages the compact-context specialization theorems for one generic proof-system `Section71Instance`
- `ProtocolSection71Setup(ctx)` packages one generic proof-system `Section71Instance` together with its compact-context specialization as one theorem-native Section 7.1 setup owner
- `ProtocolSection71TheoremInstance(ctx)` packages one fully paper-faithful Section 7.1 theorem instance specialized to the compact protocol context
- `ccsRelation_of_section71Specialization`: proof-system `Section71Instance` + compact specialization → `ccsRelation ctx`
- `ceRelation_of_section71Specialization`: proof-system `Section71Instance` + compact specialization → `ceRelation ctx`
- `ProtocolSection71Setup.ccsRelation`: theorem-native Section 7.1 setup → `ccsRelation ctx`
- `ProtocolSection71Setup.ceRelation`: theorem-native Section 7.1 setup → `ceRelation ctx`
- `ccsRelation_of_section71TheoremInstance`: one paper-faithful Section 7.1 theorem instance → `ccsRelation ctx`
- `ceRelation_of_section71TheoremInstance`: one paper-faithful Section 7.1 theorem instance → `ceRelation ctx`
- `ccsRelation_of_protocolTargetData`: one explicit protocol-side Section 7.5 target-data owner → `ccsRelation ctx`
- `ceRelation_of_protocolTargetData`: one explicit protocol-side Section 7.5 target-data owner + accepted transition witness → `ceRelation ctx`
- `ceRelation_of_ccsRelation`: `ccsRelation ctx → SumCheckTransitionWitness ctx → ceRelation ctx`
- `ceRelation_of_ccsRelation_claimTrue`: `ccsRelation ctx → SumCheckClaimTrue inst → ceRelation ctx`
- `ccsRelation_of_paperCarrierDiff`: active paper-facing route data → `ccsRelation ctx`
- `ccsRelation_of_basisKernelAssumption`: finite basis-kernel Thm-3 witness + active paper-facing route data → `ccsRelation ctx`
- `ccsRelation_of_basisKernelCheck`: executable finite basis-kernel checker + active paper-facing route data → `ccsRelation ctx`
- `ccsRelation_of_native_paperCarrierDiff`: active native paper-facing route data → `ccsRelation ctx`
- `ceRelation_of_paperCarrierDiff`: active paper-facing route data + witness → `ceRelation ctx`
- `ceRelation_of_basisKernelAssumption`: finite basis-kernel Thm-3 witness + active paper-facing route data + witness → `ceRelation ctx`
- `ceRelation_of_basisKernelCheck`: executable finite basis-kernel checker + active paper-facing route data + witness → `ceRelation ctx`
- `ceRelation_of_native_paperCarrierDiff`: active native paper-facing route data + witness → `ceRelation ctx`
- `ceRelation_of_claimTrue`: `ProtocolRelationsAssumptions ctx → SumCheckClaimTrue inst → ceRelation ctx`
- `ceRelation_of_native_claimTrue`: `ProtocolRelationsNativeAssumptions ctx → SumCheckClaimTrue inst → ceRelation ctx`
- `ceClaimTrue_of_ce`: `ceRelation ctx → SumCheckClaimTrue inst`
- `ceClaimTrue_of_native_ce`: `ceRelation ctx → SumCheckClaimTrue inst`
- `ceRelaxedRelation_of_ce`: `ceRelation ctx → ceRelaxedRelation ctx`

## Paper Anchors

Source: ./formal/superneo-lean/SuperNeo.pdf.md

- Definition 12 (Norm-bounded CCS), Section 7.1, lines 457-459.
- Definition 13 (Norm-bounded CCS Evaluation Relation), Section 7.1, lines 461-465.
- Section 7.1 (Relations), lines 449-465.

## Module Mapping

| Lean file | Paper section |
|---|---|
| `SuperNeo/ProtocolRelations.lean` | Section 7.1, Definitions 12–13 |

## Contract Surface

| Group | Lean symbol | Kind | Role | Guarantee |
|---|---|---|---|---|
| Instance | `sumcheckInstanceOfContext` | def | Definitional | Builds SumCheckInstance from ctx |
| Instance | `sumcheckFullFieldDenominatorAlignment` | def | Theorem-Target | The protocol SumCheck instance satisfies the full-field Lund denominator requirement |
| Boundary | `GoldilocksFullFieldLundBoundary` | structure | Boundary | Named setup-side boundary for replaying the active Goldilocks/full-field Lund endpoint |
| Witness | `SumCheckTransitionWitness` | structure | Definitional | transcript, accepted, initialRound, roundSumStep |
| Relations | `ccsRelation` | def | Definitional | protocolTargetProp ctx |
| Relations | `ceRelation` | def | Definitional | ccsRelation ∧ ∃ tr, SumCheckAccepted |
| Relations | `ceRelaxedRelation` | def | Definitional | ccsRelation ctx |
| Boundary | `ProtocolSection71Objects` | structure | Boundary | Owns one shared Definition-14 `GlobalParams` package and one coherent CCS/CE tuple pair for the current theorem instance |
| Boundary | `ProtocolSection71Realization` | structure | Boundary | Specializes the compact protocol relations into explicit Section 7.1 proof-system objects built from one shared Definition-14 global-parameter package |
| Boundary | `ProtocolSection71Provider` | structure | Boundary | Owns one concrete Section 7.1 theorem instance: a shared Definition-14 realization plus CCS/CE membership proofs |
| Boundary | `ProtocolSection71Specialization` | structure | Boundary | Specializes one generic proof-system `Section71Instance` to the compact protocol relations |
| Boundary | `ProtocolSection71Setup` | structure | Boundary | Owns one generic proof-system `Section71Instance` together with its compact specialization as one theorem-native Section 7.1 setup object |
| Boundary | `ProtocolSection71TheoremInstance` | structure | Theorem-Target | Owns one fully paper-faithful specialized Section 7.1 theorem instance as a single object |
| Assumptions | `ProtocolRelationsAssumptions` | structure | Boundary | Bundles target only |
| Assumptions | `ProtocolRelationsNativeAssumptions` | structure | Boundary | Bundles native target only |
| Constructors | `ProtocolRelationsAssumptions.ofPaperCarrierDiff`, `ProtocolRelationsNativeAssumptions.ofPaperCarrierDiff` | def | Theorem-Target | Canonical relations bundles from the paper-facing `paperCarrier`-difference route on the active Goldilocks path |
| Constructors | `ProtocolRelationsAssumptions.ofLowNormAtLeastFive`, `ProtocolRelationsNativeAssumptions.ofLowNormAtLeastFive` | def | Theorem-Target | Canonical relations bundles from a stronger strict low-norm invertibility theorem with threshold at least `5` |
| Theorems | `ccsRelation_of_assumptions` | theorem | Theorem-Target | Assumptions → ccsRelation |
| Theorems | `ccsRelation_of_native_assumptions` | theorem | Theorem-Target | Native assumptions → ccsRelation |
| Theorems | `ccsRelation_of_protocolTargetProp` | theorem | Theorem-Target | `protocolTargetProp → ccsRelation` |
| Theorems | `ccsRelation_of_protocolTargetData` | theorem | Theorem-Target | Protocol-side Section 7.5 target data → `ccsRelation` |
| Theorems | `ccsRelation_of_paperCarrierDiff` | theorem | Theorem-Target | Active paper-facing route data → ccsRelation |
| Theorems | `ccsRelation_of_basisKernelAssumption` | theorem | Theorem-Target | Finite basis-kernel Thm-3 witness + active paper-facing route data → ccsRelation |
| Theorems | `ccsRelation_of_basisKernelCheck` | theorem | Theorem-Target | Executable finite basis-kernel checker + active paper-facing route data → ccsRelation |
| Theorems | `ccsRelation_of_native_paperCarrierDiff` | theorem | Theorem-Target | Active native paper-facing route data → ccsRelation |
| Theorems | `ccsRelation_iff_protocolTargetProp` | theorem | Theorem-Target | `ccsRelation ↔ protocolTargetProp` |
| Theorems | `ceRelation_iff` | theorem | Theorem-Target | `ceRelation ↔ ccsRelation ∧ ∃ tr, accepted` |
| Theorems | `ceRelaxedRelation_iff` | theorem | Theorem-Target | `ceRelaxedRelation ↔ ccsRelation` |
| Theorems | `ceRelation_of_ccsRelation` | theorem | Theorem-Target | CCS + witness → ceRelation |
| Theorems | `ceRelation_of_protocolTargetData` | theorem | Theorem-Target | Protocol-side Section 7.5 target data + witness → `ceRelation` |
| Theorems | `ceRelation_of_ccsRelation_claimTrue` | theorem | Theorem-Target | CCS + claimTrue → ceRelation |
| Theorems | `ceRelation_of_assumptions` | theorem | Theorem-Target | Assumptions + witness → ceRelation |
| Theorems | `ceRelation_of_native_assumptions` | theorem | Theorem-Target | Native assumptions + witness → ceRelation |
| Theorems | `ceRelation_of_paperCarrierDiff` | theorem | Theorem-Target | Active paper-facing route data + witness → ceRelation |
| Theorems | `ceRelation_of_basisKernelAssumption` | theorem | Theorem-Target | Finite basis-kernel Thm-3 witness + active paper-facing route data + witness → ceRelation |
| Theorems | `ceRelation_of_basisKernelCheck` | theorem | Theorem-Target | Executable finite basis-kernel checker + active paper-facing route data + witness → ceRelation |
| Theorems | `ceRelation_of_native_paperCarrierDiff` | theorem | Theorem-Target | Active native paper-facing route data + witness → ceRelation |
| Theorems | `ceRelation_of_claimTrue` | theorem | Theorem-Target | Assumptions + claimTrue → ceRelation |
| Theorems | `ceRelation_of_native_claimTrue` | theorem | Theorem-Target | Native assumptions + claimTrue → ceRelation |
| Theorems | `ceClaimTrue_of_ce` | theorem | Theorem-Target | ceRelation → claimTrue |
| Theorems | `ceClaimTrue_of_native_ce` | theorem | Theorem-Target | ceRelation → claimTrue |
| Theorems | `ceRelaxedRelation_of_ce` | theorem | Theorem-Target | ceRelation → ceRelaxedRelation |
| Theorems | `ProtocolSection71Realization.challengeSet_eq_cset` | theorem | Theorem-Target | Recover the compact challenge-set from the realized Definition-14 package |
| Theorems | `ProtocolSection71Realization.sharedCommitment_eq` | theorem | Theorem-Target | Recover that the realized CCS and CE statements share one commitment |
| Theorems | `ProtocolSection71Realization.sharedPublicInput_eq` | theorem | Theorem-Target | Recover that the realized CCS and CE statements share one public input |
| Theorems | `ProtocolSection71Realization.ceAssignment_eq_fullVector` | theorem | Theorem-Target | Recover that the CE witness assignment is the CCS full vector `[x, w]` |
| Theorems | `ProtocolSection71Realization.ccsHolds_of_relation` | theorem | Theorem-Target | Compact `ccsRelation` → realized proof-system CCS membership |
| Theorems | `ProtocolSection71Realization.relation_of_ccsHolds` | theorem | Theorem-Target | Realized proof-system CCS membership → compact `ccsRelation` |
| Theorems | `ProtocolSection71Realization.ceHolds_of_relation` | theorem | Theorem-Target | Compact `ceRelation` → realized proof-system CE membership |
| Theorems | `ProtocolSection71Realization.relation_of_ceHolds` | theorem | Theorem-Target | Realized proof-system CE membership → compact `ceRelation` |
| Constructors | `ProtocolSection71Specialization.realization` | def | Theorem-Target | Convert one proof-system `Section71Instance` plus compact specialization into the realized Section 7.1 boundary |
| Constructors | `ProtocolSection71Setup.realization` | def | Theorem-Target | Convert one theorem-native Section 7.1 setup into the realized Section 7.1 boundary |
| Constructors | `ProtocolSection71TheoremInstance.realization` | def | Theorem-Target | Convert one paper-faithful Section 7.1 theorem instance into the realized Section 7.1 boundary |
| Constructors | `ProtocolSection71TheoremInstance.setup` | def | Theorem-Target | Recover the packaged theorem-native Section 7.1 setup from one paper-faithful theorem instance |
| Constructors | `ProtocolSection71TheoremInstance.provider` | def | Theorem-Target | Recover the concrete Section 7.1 provider from one paper-faithful theorem instance |
| Constructors | `ProtocolSection71Provider.ofSection71CE` | def | Theorem-Target | Build the canonical Section 7.1 provider from one realized proof-system CE instance |
| Constructors | `ProtocolSection71Provider.ofSpecialization` | def | Theorem-Target | Build the compact-context Section 7.1 provider from one proof-system `Section71Instance` plus compact specialization |
| Constructors | `ProtocolSection71Provider.ofSetup` | def | Theorem-Target | Build the compact-context Section 7.1 provider from one theorem-native Section 7.1 setup |
| Theorems | `ProtocolSection71Provider.sharedCommitment_eq` | theorem | Theorem-Target | Recover that the provider CCS and CE statements share one commitment |
| Theorems | `ProtocolSection71Provider.sharedPublicInput_eq` | theorem | Theorem-Target | Recover that the provider CCS and CE statements share one public input |
| Theorems | `ProtocolSection71Provider.ceAssignment_eq_fullVector` | theorem | Theorem-Target | Recover that the provider CE witness assignment is the CCS full vector `[x, w]` |
| Theorems | `ccsRelation_of_section71Provider` | theorem | Theorem-Target | One concrete Section 7.1 provider bundle → compact `ccsRelation` |
| Theorems | `ceRelation_of_section71Provider` | theorem | Theorem-Target | One concrete Section 7.1 provider bundle → compact `ceRelation` |
| Theorems | `ccsRelation_of_section71Specialization` | theorem | Theorem-Target | One proof-system `Section71Instance` plus compact specialization → compact `ccsRelation` |
| Theorems | `ceRelation_of_section71Specialization` | theorem | Theorem-Target | One proof-system `Section71Instance` plus compact specialization → compact `ceRelation` |
| Theorems | `ProtocolSection71Setup.ccsRelation` | theorem | Theorem-Target | One theorem-native Section 7.1 setup → compact `ccsRelation` |
| Theorems | `ProtocolSection71Setup.ceRelation` | theorem | Theorem-Target | One theorem-native Section 7.1 setup → compact `ceRelation` |
| Theorems | `ProtocolSection71TheoremInstance.challengeSet_eq_cset` | theorem | Theorem-Target | Recover the compact challenge-set from one paper-faithful Section 7.1 theorem instance |
| Theorems | `ProtocolSection71TheoremInstance.sharedCommitment_eq` | theorem | Theorem-Target | Recover that one paper-faithful Section 7.1 theorem instance shares a commitment across CCS/CE |
| Theorems | `ProtocolSection71TheoremInstance.sharedPublicInput_eq` | theorem | Theorem-Target | Recover that one paper-faithful Section 7.1 theorem instance shares a public input across CCS/CE |
| Theorems | `ProtocolSection71TheoremInstance.ceAssignment_eq_fullVector` | theorem | Theorem-Target | Recover that the CE witness assignment is the CCS full vector `[x, w]` for one paper-faithful Section 7.1 theorem instance |
| Theorems | `ccsRelation_of_section71TheoremInstance` | theorem | Theorem-Target | One paper-faithful Section 7.1 theorem instance → compact `ccsRelation` |
| Theorems | `ceRelation_of_section71TheoremInstance` | theorem | Theorem-Target | One paper-faithful Section 7.1 theorem instance → compact `ceRelation` |
| Theorems | `sumcheckFullFieldDenominatorAlignment_iff` | theorem | Theorem-Target | `sumcheckFullFieldDenominatorAlignment ctx ↔ ctx.cset.size = Goldilocks.q` |
| Constructors | `GoldilocksFullFieldLundBoundary.ofCsetCardinality` | def | Theorem-Target | Builds the named Goldilocks/Lund setup boundary from `ctx.cset.size = Goldilocks.q` |
| Theorems | `GoldilocksFullFieldLundBoundary.csetCardinality_eq` | theorem | Theorem-Target | Recover `ctx.cset.size = Goldilocks.q` from the named setup boundary |
| Witness | `SumCheckTransitionWitness.accepted_exists` | theorem | Theorem-Target | Witness → ∃ tr, accepted |

## Proof Obligations

- The theorem-native relation surfaces are `ccsRelation`, `ceRelation`, and `ceRelaxedRelation`.
- `ProtocolSection71Objects` is the explicit owner of the paper-facing Definition-14 data for one theorem instance: one shared `GlobalParams` package, one shared norm bound, and one coherent CCS/CE tuple pair.
- `ProtocolSection71Realization` is the explicit compatibility boundary between these compact relations and the proof-system Section 7.1 objects in `ProofSystem.ConstraintSystem.CCS`.
- `ProtocolSection71Specialization` is the theorem-native compact-context bridge from one generic proof-system `Section71Instance` to the compact relation predicates.
- `ProtocolSection71Setup` is the smallest explicit upstream owner once one generic proof-system `Section71Instance` and its compact specialization theorems are available together.
- `ProtocolSection71TheoremInstance` is the single-object paper-faithful owner once the specialized Section 7.1 theorem instance itself is available upstream.
- The object/realization boundary must carry one shared Definition-14 `GlobalParams` package, recover the compact challenge-set from that package, and enforce that CCS and CE share the same commitment, public input, and witness vector rather than storing unrelated tuples.
- `ProtocolSection71Provider` is the smallest concrete owner for one specialized Section 7.1 theorem instance once a realized CE witness is available; it is a convenience owner, not the only theorem-native entrypoint.
- Direct relation constructors exist from `protocolTargetProp`, from CCS/claim-truth/witness bridges, and from the active paper-facing protocol data.
- Direct relation constructors also exist from the single theorem-native Section 7.5 owner `ProtocolTargetData`, without routing back through a compatibility bundle.
- `ProtocolRelationsAssumptions` and `ProtocolRelationsNativeAssumptions` are compatibility bundles carrying the same upstream protocol-target closure when a bundled surface is still more convenient.
- Claim-true and witness bridges factor through the relation predicates themselves rather than duplicating separate relation-specific assumption bundles.
- Canonical constructors exist from an already-built protocol-target bundle, from the paper-facing `paperCarrier` difference route for `ctx.invDelta`, and from the stronger strict low-norm invertibility route.

## Assumption Ledger

- `ProtocolRelationsAssumptions` bundles `ProtocolTargetAssumptions`.
- `ProtocolRelationsNativeAssumptions` bundles `ProtocolTargetNativeAssumptions`.
- `ProtocolTargetData` is the explicit theorem-native owner for the Section 7.5 target inputs used by the direct compact relation constructors.
- `ProtocolSection71Objects` is the paper-facing Definition-14 data boundary for one theorem instance.
- `ProtocolSection71Realization` is the specialization boundary from compact protocol data to explicit Section 7.1 proof-system objects built from one Definition-14 package.
- `ProtocolSection71Specialization` is the smallest explicit compact-context bridge for one generic proof-system `Section71Instance`.
- `ProtocolSection71Setup` is the smallest explicit theorem-native owner combining one generic proof-system `Section71Instance` with its compact specialization.
- `ProtocolSection71TheoremInstance` is the smallest explicit owner once the specialized Section 7.1 theorem instance and its concrete proof-system membership proofs are carried together as one object.
- `ProtocolSection71Provider` is the smallest explicit owner of one concrete specialized Section 7.1 proof-system theorem instance.
- No separate SumCheck boundary bundle is introduced in this module; SumCheck data enters through `SumCheckTransitionWitness` and the relation bridges.

## Dependency and Consumer Map

Upstream dependencies:
- `SuperNeo/ProtocolTarget.lean`: imports `protocolTargetProp`, `ProtocolTargetAssumptions`, `ProtocolTargetNativeAssumptions`, `ProtocolTargetContext`.
- `SuperNeo/SumCheck.lean`: imports `SumCheckInstance`, `SumCheckTranscript`, `SumCheckAccepted`, `SumCheckClaimTrue`, `sumcheckSoundness_constructive`, `sumcheckCompleteness_constructive`.

Downstream consumers:
- `SuperNeo/PiCCS.lean`: uses `ceRelation`, `ceRelation_of_ccsRelation`, `ceClaimTrue_of_ce`, `SumCheckTransitionWitness`, `sumcheckInstanceOfContext`.
- `SuperNeo/PiRLC.lean`: uses `ceRelaxedRelation`, `ceRelaxedRelation_of_ce`, `piCCSStrongStatement`.
- `SuperNeo/PiDEC.lean`: uses `ceRelaxedRelation`, `piRLCWeakStatement`.
- `SuperNeo/FoldingProtocol.lean`: imports ProtocolRelations for folding relation predicates.
- `SuperNeo/ProtocolReduction.lean`: imports ProtocolRelations.

## Design Notes

Assumption bundling is used only as a compatibility layer for carrying upstream protocol-target closure into the relation constructors. The canonical theorem-facing targets remain the relation predicates and their direct witness/claim-true / active-route bridges.

## Quality Expectations

Relation definitions must match paper CCS/CE semantics. Soundness/completeness bridges (`ceRelation_of_ccsRelation_claimTrue`, `ceClaimTrue_of_ce`) must be proved.

## Acceptance Criteria

- `lake build` succeeds.
- No `sorry`.
- All relation theorems proved.

## Out of Scope

- Generic standalone SumCheck redesign beyond the now proof-complete Definition-6 surface used by this module.
