# ProtocolSection71Context Spec

## Purpose

- **What it is**: A single theorem-native owner for one compact Section 7.1 protocol instance: a `ProtocolTargetContext` paired with one specialized paper-faithful `ProtocolSection71TheoremInstance`.
- **Key property**: Downstream Section 7 reductions can consume one object instead of threading `ctx` and a separate specialized theorem instance in parallel.
- **Protocol role**: This is the smallest explicit upstream owner once the actual Definition-14 package and its specialization back to the compact protocol context have been constructed.

## Target Formulas

- `ProtocolSection71Context.ccsRelation : ccsRelation h.target`
- `ProtocolSection71Context.ceRelation : ceRelation h.target`
- `ProtocolSection71Context.realization : ProtocolSection71Realization h.target`
- `ProtocolSection71Context.setup : ProtocolSection71Setup h.target`
- `ProtocolSection71Context.provider : ProtocolSection71Provider h.target`
- `ProtocolSection71Context.challengeSet_eq_cset : h.theoremInstance.params.challengeSet = h.target.cset`
- `ProtocolSection71Context.sharedCommitment_eq : h.theoremInstance.ccsStatement.commitment = h.theoremInstance.ceStatement.commitment`
- `ProtocolSection71Context.sharedPublicInput_eq : h.theoremInstance.ccsStatement.publicInput = h.theoremInstance.ceStatement.publicInput`
- `ProtocolSection71Context.ceAssignment_eq_fullVector : h.theoremInstance.ceWitness.assignment = CCS.fullVector h.theoremInstance.ccsStatement h.theoremInstance.ccsWitness`
- `ccsRelation_of_section71Context : ProtocolSection71Context → ccsRelation target`
- `ceRelation_of_section71Context : ProtocolSection71Context → ceRelation target`

## Paper Anchors

- Source: `./formal/superneo-lean/SuperNeo.pdf.md`
- Definition 11 (Structure), lines 449-455
- Definition 12 (Norm-bounded CCS), lines 457-459
- Definition 13 (Norm-bounded CCS Evaluation Relation), lines 461-465
- Definition 14 (Global Reduction Parameters), lines 467-475

## Module Mapping

- Implementation: `SuperNeo.ProtocolSection71Context`
- Interface: `SuperNeo.ProtocolSection71ContextInterface`

## Contract Surface

| Group | Lean surface | Guarantee | Role |
|---|---|---|---|
| Context | `ProtocolSection71Context` | Owns one compact target and one specialized paper-faithful Section 7.1 theorem instance | Theorem-Target |
| Projection | `ProtocolSection71Context.ccsRelation` | Recover compact CCS relation from the packaged theorem instance | Theorem-Target |
| Projection | `ProtocolSection71Context.ceRelation` | Recover compact CE relation from the packaged theorem instance | Theorem-Target |
| Projection | `ProtocolSection71Context.realization` | Recover the explicit Definition-14 realization boundary | Theorem-Target |
| Projection | `ProtocolSection71Context.setup` | Recover the packaged generic Section 7.1 setup bundle | Theorem-Target |
| Projection | `ProtocolSection71Context.provider` | Recover the concrete specialized Section 7.1 provider | Theorem-Target |
| Projection | `ProtocolSection71Context.challengeSet_eq_cset` | Recover the compact challenge-set equality | Theorem-Target |
| Projection | `ProtocolSection71Context.sharedCommitment_eq` | Recover the shared commitment across CCS/CE | Theorem-Target |
| Projection | `ProtocolSection71Context.sharedPublicInput_eq` | Recover the shared public input across CCS/CE | Theorem-Target |
| Projection | `ProtocolSection71Context.ceAssignment_eq_fullVector` | Recover that CE uses the CCS full vector `[x,w]` | Theorem-Target |
| Theorem | `ccsRelation_of_section71Context` | One packaged Section 7.1 context implies compact `ccsRelation` | Theorem-Target |
| Theorem | `ceRelation_of_section71Context` | One packaged Section 7.1 context implies compact `ceRelation` | Theorem-Target |

## Proof Obligations

- The context object must be a pure packaging layer; it introduces no new assumptions beyond the carried `ProtocolSection71TheoremInstance`.
- The projections back to realized/setup/provider/compact relation surfaces must be definitional or immediate theorem wrappers over the carried theorem instance.

## Assumption Ledger

- This module introduces no new theorem-level assumptions.
- Construction of `ProtocolSection71Context` remains an upstream task: the repo still needs a canonical source of the specialized `ProtocolSection71TheoremInstance`.

## Dependency and Consumer Map

- Upstream dependencies:
  - `SuperNeo/ProtocolRelations.lean`
- Downstream consumers:
  - `SuperNeo/PiCCS.lean`
  - `SuperNeo/PiRLC.lean`
  - `SuperNeo/PiDEC.lean`
  - `SuperNeo/InteractiveReductions.lean`
  - `SuperNeo/ProofSystem/Folding/*`

## Quality Expectations

- Keep the component thin and stable.
- Do not duplicate theorem content already owned by `ProtocolSection71TheoremInstance`.

## Acceptance Criteria

1. `lake build` succeeds.
2. `lake exe check` succeeds.
3. No `sorry`.
