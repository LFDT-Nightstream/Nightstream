# InteractiveReductions

## Purpose

- **What it is**: The composition layer for the reduction pipeline Π_RLC ∘ Π_CCS and Π_DEC ∘ Π_RLC ∘ Π_CCS. It retains `InteractiveReductionAssumptions` as a legacy compatibility bundle, but the theorem-native surfaces are the direct composition theorems from compact Section 7.1 / Section 7.5 owners.
- **Key property**: `strongCompositionStatement` (Π_RLC ∘ Π_CCS is strong) and `weakCompositionStatement` (Π_DEC ∘ Π_RLC ∘ Π_CCS is weak) are proved directly from theorem-native Section 7.1 / Section 7.5 owners, with the old assumption bundle remaining only as a convenience wrapper.
- **Protocol role**: ProtocolTheorem uses composition statements. This is the composition capstone for all three reduction steps (CCS → RLC → DEC).

## Target Formulas

- `strongCompositionStatement ctx ↔ piDECKnowledgeStatement ctx`
- `weakCompositionStatement ctx ↔ ceRelaxedRelation ctx ∧ SumCheckClaimTrue (sumcheckInstanceOfContext ctx)`
- `InteractiveReductionAssumptions ctx → strongCompositionStatement ctx`
- `InteractiveReductionAssumptions ctx → weakCompositionStatement ctx`
- `ProtocolTargetData ctx → SumCheckTransitionWitness ctx → strongCompositionStatement ctx`
- `ProtocolTargetData ctx → SumCheckTransitionWitness ctx → weakCompositionStatement ctx`
- `ProtocolSection71Realization ctx + CE.Holds → strongCompositionStatement ctx`
- `ProtocolSection71Realization ctx + CE.Holds → weakCompositionStatement ctx`
- `ProtocolSection71Provider ctx → strongCompositionStatement ctx`
- `ProtocolSection71Provider ctx → weakCompositionStatement ctx`
- `ProtocolSection71Specialization ctx hInst + Section71Instance hInst → strongCompositionStatement ctx`
- `ProtocolSection71Specialization ctx hInst + Section71Instance hInst → weakCompositionStatement ctx`
- `ProtocolSection71Setup ctx → strongCompositionStatement ctx`
- `ProtocolSection71Setup ctx → weakCompositionStatement ctx`
- `ProtocolSection71TheoremInstance ctx → strongCompositionStatement ctx`
- `ProtocolSection71TheoremInstance ctx → weakCompositionStatement ctx`
- `ProtocolSection71Context → strongCompositionStatement target`
- `ProtocolSection71Context → weakCompositionStatement target`
- `ProtocolSection71Data ctx → strongCompositionStatement ctx`
- `ProtocolSection71Data ctx → weakCompositionStatement ctx`
- `InteractiveReductionAssumptions.ofProtocolTargetData : ProtocolTargetData ctx → SumCheckTransitionWitness ctx → InteractiveReductionAssumptions ctx`
- `InteractiveReductionAssumptions ctx + (∀ n, 0 ≤ eps n) → SoundnessFailureAdvantageBound(sumcheckInstanceOfContext ctx, witnessTranscript, eps)`

## Paper Anchors

Source: ./formal/superneo-lean/SuperNeo.pdf.md

- Theorem 6 (Strong-Weak Composition), Section 6, lines 438-447.
- Definition 9 (Weak Interactive Reductions), lines 404-416.
- Definition 10 (Strong Interactive Reductions), lines 418-436.

## Module Mapping

| Lean file | Paper section |
|---|---|
| `SuperNeo/InteractiveReductions.lean` | Theorem 6, Definitions 9–10 |

## Contract Surface

| Group | Lean symbol | Kind | Role | Guarantee |
|---|---|---|---|---|
| Assumptions | `InteractiveReductionAssumptions` | structure | Boundary | Bundles protocol-target assumptions + SumCheck transition witness |
| Constructor | `InteractiveReductionAssumptions.ofProtocolRelations` | def | Theorem-Target | Compatibility constructor from protocol-relations assumptions by extracting the target component |
| Constructor | `InteractiveReductionAssumptions.ofProtocolTargetData` | def | Theorem-Target | Compatibility constructor from one explicit protocol-side Section 7.5 target-data owner + witness |
| Constructor | `InteractiveReductionAssumptions.ofPaperCarrierDiff` | def | Theorem-Target | Canonical constructor from thm3 + arithmetic + active `paperCarrier`-difference route + witness, assembled directly at protocol-target level |
| Constructor | `InteractiveReductionAssumptions.ofBasisKernelAssumption` | def | Theorem-Target | Canonical constructor from finite basis-kernel Thm-3 witness + arithmetic + active `paperCarrier`-difference route + witness |
| Constructor | `InteractiveReductionAssumptions.ofBasisKernelCheck` | def | Theorem-Target | Canonical constructor from executable finite basis-kernel checker + arithmetic + active `paperCarrier`-difference route + witness |
| Constructor | `InteractiveReductionAssumptions.ofNativePaperCarrierDiff` | def | Theorem-Target | Canonical constructor from the active native-bar `paperCarrier`-difference route + witness, discharging generic Thm 3 from `thm3CoreAssumption_native` |
| Constructor | `InteractiveReductionAssumptions.ofLowNormAtLeastFive` | def | Theorem-Target | Canonical constructor from thm3 + arithmetic + stronger strict low-norm invertibility theorem with threshold at least `5` + witness |
| Constructor | `InteractiveReductionNativeAssumptions.ofPaperCarrierDiff` | def | Theorem-Target | Native canonical constructor from the same route + witness |
| Constructor | `InteractiveReductionNativeAssumptions.ofLowNormAtLeastFive` | def | Theorem-Target | Native canonical constructor from the stronger strict low-norm route + witness |
| Statements | `strongCompositionStatement` | def | Definitional | Π_RLC ∘ Π_CCS strong |
| Statements | `weakCompositionStatement` | def | Definitional | Π_DEC ∘ Π_RLC ∘ Π_CCS weak |
| Theorems | `strongComposition_of_assumptions` | theorem | Theorem-Target | Assumptions → strong |
| Theorems | `weakComposition_of_assumptions` | theorem | Theorem-Target | Assumptions → weak |
| Theorems | `strongComposition_of_section71_ce` | theorem | Theorem-Target | Realized proof-system CE membership → strong composition |
| Theorems | `weakComposition_of_section71_ce` | theorem | Theorem-Target | Realized proof-system CE membership → weak composition |
| Theorems | `strongComposition_of_section71Provider` | theorem | Theorem-Target | One concrete Section 7.1 provider bundle → strong composition |
| Theorems | `weakComposition_of_section71Provider` | theorem | Theorem-Target | One concrete Section 7.1 provider bundle → weak composition |
| Theorems | `strongComposition_of_section71Specialization` | theorem | Theorem-Target | One proof-system `Section71Instance` plus compact specialization → strong composition |
| Theorems | `weakComposition_of_section71Specialization` | theorem | Theorem-Target | One proof-system `Section71Instance` plus compact specialization → weak composition |
| Theorems | `strongComposition_of_section71Setup` | theorem | Theorem-Target | One theorem-native Section 7.1 setup → strong composition |
| Theorems | `weakComposition_of_section71Setup` | theorem | Theorem-Target | One theorem-native Section 7.1 setup → weak composition |
| Theorems | `strongComposition_of_section71TheoremInstance` | theorem | Theorem-Target | One paper-faithful Section 7.1 theorem instance → strong composition |
| Theorems | `weakComposition_of_section71TheoremInstance` | theorem | Theorem-Target | One paper-faithful Section 7.1 theorem instance → weak composition |
| Theorems | `strongComposition_of_section71Context` | theorem | Theorem-Target | One theorem-native Section 7.1 context object → strong composition |
| Theorems | `weakComposition_of_section71Context` | theorem | Theorem-Target | One theorem-native Section 7.1 context object → weak composition |
| Theorems | `strongComposition_of_section71Data` | theorem | Theorem-Target | One protocol-side Section 7.1 Definition-14 data package → strong composition |
| Theorems | `weakComposition_of_section71Data` | theorem | Theorem-Target | One protocol-side Section 7.1 Definition-14 data package → weak composition |
| Theorems | `strongComposition_of_protocolTargetData` | theorem | Theorem-Target | One protocol-side Section 7.5 target-data owner + witness → strong composition |
| Theorems | `weakComposition_of_protocolTargetData` | theorem | Theorem-Target | One protocol-side Section 7.5 target-data owner + witness → weak composition |
| Theorems | `sumcheckFailureAdvantageBound_of_assumptions` | theorem | Theorem-Target | Witness-level SumCheck failure-advantage bound from reduction assumptions |
| Theorems | `sumcheckFailureAdvantageBound_of_native_assumptions` | theorem | Theorem-Target | Native-path witness-level SumCheck failure-advantage bound |

## Proof Obligations and Closure Plan

- `InteractiveReductionAssumptions ctx` is retained as a compatibility bundle carrying a protocol-target instantiation and an accepted SumCheck transition witness.
- The theorem-native composition surfaces are the direct theorems from `ProtocolTargetData` + witness and from the Section 7.1 theorem-native owners.
- Realized Section 7.1 CE objects, one proof-system `Section71Instance` plus compact specialization, the packaged theorem-native `ProtocolSection71Setup`, and the smaller `ProtocolSection71Provider` bundle must all feed the same compact composition statements directly.
- `ProtocolSection71TheoremInstance` is the single-object paper-faithful Section 7 owner for those same composition statements once a specialized theorem instance is available upstream.
- `ProtocolSection71Context` packages that specialized theorem instance together with its compact target context as one direct broad-protocol owner.
- `ProtocolSection71Data` is the explicit protocol-side Definition-14 owner that canonically builds both of those narrower owners.
- `ProtocolTargetData` is the explicit protocol-side Section 7.5 owner feeding the direct composition theorems and the compatibility constructor.
- Canonical constructors must exist both from the compatibility protocol-relations bundle and directly from the narrower protocol-target paper-facing routes, including the finite basis-kernel Theorem-3 providers.

## Assumption Ledger

- `InteractiveReductionAssumptions`: boundary assumption bundling protocol-target assumptions and a transition witness.
- `InteractiveReductionNativeAssumptions`: native compatibility bundle carrying bar equality, protocol-target instantiation, and a transition witness.
- `ProtocolTargetData`: theorem-native Section 7.5 owner for the direct composition route.

## Dependency and Consumer Map

Upstream dependencies:
- `SuperNeo/PiDEC.lean`: imports `PiDECAssumptions`, `piDECKnowledgeStatement`, `ceRelaxedRelation`, `SumCheckClaimTrue`, `sumcheckInstanceOfContext`, `piDEC_of_assumptions`.
- `SuperNeo/SumCheck.lean`: constructive SumCheck truth is used directly in witness-level failure-advantage bounds.

Downstream consumers:
- `SuperNeo/ProtocolTheorem.lean`: uses composition statements for the full protocol reduction.

## Implementation Plan

1. Define the strong/weak composition statements in the compact protocol vocabulary.
2. Prove both composition theorems from `InteractiveReductionAssumptions`.
3. Provide canonical constructors from protocol-relations bundles and direct paper-facing/native paper-facing protocol data.

## Quality Expectations

Composition statements must match Theorem 6 (Strong-Weak Composition). Strong/weak definitions must align with Definitions 9 and 10.

## Acceptance Criteria

- `lake build` succeeds.
- No `sorry`.

## Out of Scope

- Concrete deployment/setup instantiation beyond the theorem-native Section 7.1 and Section 7.5 owners consumed here.
- Proof of the underlying cryptographic assumptions themselves.
