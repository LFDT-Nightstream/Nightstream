# ProtocolRelations

## Purpose

- **What it is**: The CCS and CE relation predicates on protocol-target context. Defines `ccsRelation` (protocol target holds), `ceRelation` (CCS plus accepted SumCheck transcript), and `ceRelaxedRelation` (CCS only). Builds `sumcheckInstanceOfContext` from context and `SumCheckTransitionWitness` carrying round-consistency facts. Owns the theorem-native Section 7.1 owner `ProtocolSection71TheoremInstance`.
- **Key property**: `ceRelation ctx ↔ ccsRelation ctx ∧ ∃ tr, SumCheckAccepted (sumcheckInstanceOfContext ctx) tr`; and `ceRelation ctx → ceRelaxedRelation ctx`.
- **Protocol role**: PiCCS, PiRLC, PiDEC, and FoldingProtocol depend on these relation predicates for the Section 7 folding reductions (Π_CCS, Π_RLC, Π_DEC).

## Target Formulas

- `ccsRelation(ctx) ↔ protocolTargetProp(ctx)` (CCS = protocol target)
- `ceRelation(ctx) ↔ ccsRelation(ctx) ∧ ∃ tr, SumCheckAccepted inst tr` where `inst = sumcheckInstanceOfContext ctx`
- `ceRelaxedRelation(ctx) ↔ ccsRelation(ctx)`
- `sumcheckFullFieldDenominatorAlignment(ctx) ↔ ctx.cset.size = Goldilocks.q`
- `GoldilocksFullFieldLundBoundary.ofCsetCardinality(hCard)` packages the active Goldilocks/full-field Lund setup boundary from `hCard : ctx.cset.size = Goldilocks.q`
- `ProtocolSection71TheoremInstance(ctx)` packages one paper-faithful Section 7.1 theorem instance specialized to the compact protocol context: shared Definition-14 `GlobalParams`, one norm bound, coherent CCS/CE statement-witness pairs, sharing facts, concrete `CCS.Holds` / `CE.Holds`, and two-way bridges to the compact relation predicates
- `ProtocolSection71TheoremInstance.ccsRelation`: one theorem-native Section 7.1 instance → `ccsRelation ctx`
- `ProtocolSection71TheoremInstance.ceRelation`: one theorem-native Section 7.1 instance → `ceRelation ctx`
- `ccsRelation_of_protocolTargetProp`: `protocolTargetProp ctx → ccsRelation ctx`
- `ceRelation_of_ccsRelation`: `ccsRelation ctx → SumCheckTransitionWitness ctx → ceRelation ctx`
- `ceRelation_of_ccsRelation_claimTrue`: `ccsRelation ctx → SumCheckClaimTrue inst → ceRelation ctx`
- `ceClaimTrue_of_ce`: `ceRelation ctx → SumCheckClaimTrue inst`
- `ceRelaxedRelation_of_ce`: `ceRelation ctx → ceRelaxedRelation ctx`

## Paper Anchors

Source: ./formal/superneo-lean/SuperNeo.pdf.md

- Definition 12 (Norm-bounded CCS), Section 7.1, lines 457-459.
- Definition 13 (Norm-bounded CCS Evaluation Relation), Section 7.1, lines 461-465.
- Section 7.1 (Relations), lines 449-465.

## Module Mapping

| Lean file | Paper section |
|---|---|
| `SuperNeo/FoldingProtocol/ProtocolRelations.lean` | Section 7.1, Definitions 12–13 |

## Contract Surface

| Group | Lean symbol | Kind | Role | Guarantee |
|---|---|---|---|---|
| Instance | `sumcheckInstanceOfContext` | def | Definitional | SumCheck instance induced by one protocol context |
| Alignment | `sumcheckFullFieldDenominatorAlignment` | def | Definitional | Challenge-set size equals the Goldilocks field size |
| Boundary | `GoldilocksFullFieldLundBoundary` | structure | Boundary | Active Goldilocks/full-field Lund denominator setup |
| Witness | `SumCheckTransitionWitness` | structure | Boundary | Accepted transcript + round-consistency facts |
| Relation | `ccsRelation` | def | Definitional | Protocol target proposition |
| Relation | `ceRelation` | def | Definitional | CCS plus accepted SumCheck transcript |
| Relation | `ceRelaxedRelation` | def | Definitional | CCS only |
| Owner | `ProtocolSection71TheoremInstance` | structure | Boundary | Definition-14 data + relation bridges |
| Theorem | `ProtocolSection71TheoremInstance.ccsRelation` | theorem | Theorem-Target | Owner → compact CCS relation |
| Theorem | `ProtocolSection71TheoremInstance.ceRelation` | theorem | Theorem-Target | Owner → compact CE relation |
| Theorem | `ccsRelation_of_protocolTargetProp` | theorem | Theorem-Target | Protocol target → CCS relation |
| Theorem | `ccsRelation_iff_protocolTargetProp` | theorem | Theorem-Target | CCS relation unfolding |
| Theorem | `ceRelation_iff` / `ceRelaxedRelation_iff` | theorem | Theorem-Target | Relation unfoldings |
| Theorem | `ceRelation_of_ccsRelation` | theorem | Theorem-Target | CCS + witness → CE |
| Theorem | `ceRelation_of_ccsRelation_claimTrue` | theorem | Theorem-Target | CCS + claim truth → CE |
| Theorem | `ceClaimTrue_of_ce` | theorem | Theorem-Target | CE → SumCheck claim truth |
| Theorem | `ceRelaxedRelation_of_ce` | theorem | Theorem-Target | CE → relaxed CE |

## Proof Obligations and Closure Plan

- The relation predicates are definitional over `protocolTargetProp` and the SumCheck acceptance surface.
- `ceClaimTrue_of_ce` is discharged by constructive SumCheck soundness; `ceRelation_of_ccsRelation_claimTrue` by constructive completeness.
- `ProtocolSection71TheoremInstance` closes the Definition-14 ↔ compact-relation bridge by carrying both directions as fields and discharging the relation theorems from its `Holds` proofs.

## Assumption Ledger

- `SumCheckTransitionWitness` is the only carried witness boundary; SumCheck truth is derived from it, never assumed separately.
- `ProtocolSection71TheoremInstance` is a theorem-native owner: all of its fields are concrete data or proved facts supplied by the instantiating site.

## Dependency and Consumer Map

Upstream dependencies:
- `SuperNeo/FoldingProtocol/ProtocolTarget.lean`: imports `protocolTargetProp`, `ProtocolTargetAssumptions`, `ProtocolTargetContext`.
- `SuperNeo/Primitives/SumCheck.lean`: imports `SumCheckInstance`, `SumCheckTranscript`, `SumCheckAccepted`, `SumCheckClaimTrue`, `sumcheckSoundness_constructive`, `sumcheckCompleteness_constructive`.
- `SuperNeo/ProofSystem/ConstraintSystem`: imports the paper-facing Section 7.1 CCS/CE objects.

Downstream consumers:
- `SuperNeo/FoldingProtocol/PiCCS.lean`: uses `ceRelation`, `ceRelation_of_ccsRelation`, `ceClaimTrue_of_ce`, `SumCheckTransitionWitness`, `sumcheckInstanceOfContext`.
- `SuperNeo/FoldingProtocol/PiRLC.lean`: uses `ceRelaxedRelation_of_ce`, `ceClaimTrue_of_ce`, `ceRelation_of_ccsRelation`.
- `SuperNeo/FoldingProtocol/PiDEC.lean`: uses `ceRelaxedRelation`.
- `SuperNeo/FoldingProtocol/ProtocolSection71Context.lean`: wraps `ProtocolSection71TheoremInstance` with its target context as the single-object owner.
- `SuperNeo/FoldingProtocol.lean`: imports ProtocolRelations for folding relation predicates.

## Design Notes

The relation predicates and their direct witness/claim-truth bridges are the
canonical theorem-facing targets. Definition-14 evidence enters through one
owner, `ProtocolSection71TheoremInstance`; evidence-specific assumption
packaging lives upstream with `ProtocolTargetAssumptions`.

## Quality Expectations

- No `sorry`.
- Relation predicates stay definitional; bridges stay theorem-native.

## Acceptance Criteria

- `lake build` succeeds.
- `ceRelation_iff`, `ceRelaxedRelation_iff`, and the witness/claim-truth bridges are proved.
- `ProtocolSection71TheoremInstance.ccsRelation` / `.ceRelation` are proved from the carried `Holds` fields.

## Out of Scope

- The strong/weak reduction statements themselves (PiCCS/PiRLC/PiDEC).
- Final protocol composition (ProtocolTheorem).
