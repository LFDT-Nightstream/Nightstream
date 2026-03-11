# ProtocolSection71Data Spec

## Purpose

- **What it is**: An explicit protocol-side owner for one Section 7.1 theorem instance specialized to a compact `ProtocolTargetContext`.
- **Key property**: The paper's Definition-14 ingredients remain visible as fields: challenge set, commitment map `L`, input projector `L_in`, structure `s`, one norm bound, one coherent CCS/CE tuple pair, and their membership/specialization proofs.
- **Protocol role**: This is the smallest protocol-side source from which the repo can canonically build both `ProtocolSection71TheoremInstance` and `ProtocolSection71Context`.

## Target Formulas

- `ProtocolSection71Data.params : GlobalParams h.Commitment`
- `ProtocolSection71Data.challengeSet_eq_cset : h.challengeSet = ctx.cset`
- `ProtocolSection71Data.sharedCommitment_eq : h.ccsStatement.commitment = h.ceStatement.commitment`
- `ProtocolSection71Data.sharedPublicInput_eq : h.ccsStatement.publicInput = h.ceStatement.publicInput`
- `ProtocolSection71Data.ceAssignment_eq_fullVector : h.ceWitness.assignment = CCS.fullVector h.ccsStatement h.ccsWitness`
- `ProtocolSection71Data.theoremInstance : ProtocolSection71TheoremInstance ctx`
- `ProtocolSection71Data.context : ProtocolSection71Context`
- `ProtocolSection71Data.ccsRelation : ccsRelation ctx`
- `ProtocolSection71Data.ceRelation : ceRelation ctx`
- `ccsRelation_of_section71Data : ProtocolSection71Data ctx → ccsRelation ctx`
- `ceRelation_of_section71Data : ProtocolSection71Data ctx → ceRelation ctx`

## Paper Anchors

- Source: `./formal/superneo-lean/SuperNeo.pdf.md`
- Definition 11 (Structure), lines 449-455
- Definition 12 (Norm-bounded CCS), lines 457-459
- Definition 13 (Norm-bounded CCS Evaluation Relation), lines 461-465
- Definition 14 (Global Reduction Parameters), lines 467-475

## Module Mapping

- Implementation: `SuperNeo.ProtocolSection71Data`
- Interface: `SuperNeo.ProtocolSection71DataInterface`

## Contract Surface

| Group | Lean surface | Guarantee | Role |
|---|---|---|---|
| Context | `ProtocolSection71Data` | Owns one compact target's paper-facing Definition-14 data and one coherent CCS/CE theorem instance | Theorem-Target |
| Projection | `ProtocolSection71Data.params` | Recover the shared Definition-14 parameter package | Theorem-Target |
| Projection | `ProtocolSection71Data.theoremInstance` | Build the compact specialized Section 7.1 theorem instance | Theorem-Target |
| Projection | `ProtocolSection71Data.context` | Build the single-object compact Section 7.1 context owner | Theorem-Target |
| Projection | `ProtocolSection71Data.ccsRelation` | Recover compact CCS relation from the packaged theorem instance | Theorem-Target |
| Projection | `ProtocolSection71Data.ceRelation` | Recover compact CE relation from the packaged theorem instance | Theorem-Target |
| Projection | `ProtocolSection71Data.challengeSet_eq_cset` | Recover the compact challenge-set equality | Theorem-Target |
| Projection | `ProtocolSection71Data.sharedCommitment_eq` | Recover the shared commitment across CCS/CE | Theorem-Target |
| Projection | `ProtocolSection71Data.sharedPublicInput_eq` | Recover the shared public input across CCS/CE | Theorem-Target |
| Projection | `ProtocolSection71Data.ceAssignment_eq_fullVector` | Recover that CE uses the CCS full vector `[x,w]` | Theorem-Target |
| Theorem | `ccsRelation_of_section71Data` | One protocol-side Section 7.1 data owner implies compact `ccsRelation` | Theorem-Target |
| Theorem | `ceRelation_of_section71Data` | One protocol-side Section 7.1 data owner implies compact `ceRelation` | Theorem-Target |

## Proof Obligations

- `params` must reconstruct the exact shared Definition-14 parameter package from the explicit paper-facing fields.
- `theoremInstance` must build `ProtocolSection71TheoremInstance` without adding any extra assumptions.
- The compact relation theorems must be immediate consequences of the constructed theorem instance.

## Assumption Ledger

- This module introduces no new theorem-level assumptions.
- The carried specialization and membership proofs are explicit fields of the protocol-side data object.
- Concrete construction of the data object from a specific protocol setup remains an upstream instantiation task.

## Dependency and Consumer Map

- Upstream dependencies:
  - `SuperNeo/ProtocolSection71Context.lean`
- Downstream consumers:
  - `SuperNeo/PiCCS.lean`
  - `SuperNeo/PiRLC.lean`
  - `SuperNeo/PiDEC.lean`
  - `SuperNeo/InteractiveReductions.lean`
  - `SuperNeo/ProofSystem/Folding/*`

## Quality Expectations

- Keep the component thin and paper-facing.
- Keep the Definition-14 ingredients explicit instead of collapsing them back into opaque bundles.
- Do not duplicate proof content already owned by the constructed theorem instance.

## Acceptance Criteria

1. `lake build` succeeds.
2. `lake exe check` succeeds.
3. No `sorry`.
