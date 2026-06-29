# ProtocolSection71Context Spec

## Purpose

- **What it is**: A single theorem-native owner for one compact Section 7.1 protocol instance: a `ProtocolTargetContext` paired with one specialized paper-faithful `ProtocolSection71TheoremInstance`.
- **Key property**: Downstream consumers can take one object instead of threading `ctx` and a separate specialized theorem instance in parallel.
- **Protocol role**: This is the smallest explicit upstream owner once the actual Definition-14 package and its specialization back to the compact protocol context have been constructed.

## Target Formulas

- `ProtocolSection71Context.ccsRelation : ccsRelation h.target`
- `ProtocolSection71Context.ceRelation : ceRelation h.target`

## Paper Anchors

- Source: `./formal/superneo-lean/SuperNeo.pdf.md`
- Definition 11 (Structure), lines 449-455
- Definition 12 (Norm-bounded CCS), lines 457-459
- Definition 13 (Norm-bounded CCS Evaluation Relation), lines 461-465
- Definition 14 (Global Reduction Parameters), lines 467-475

## Module Mapping

- Implementation: `SuperNeo.FoldingProtocol.ProtocolSection71Context`
- Interface: `SuperNeo.FoldingProtocol.ProtocolSection71ContextInterface`

## Contract Surface

| Group | Lean surface | Guarantee | Role |
|---|---|---|---|
| Context | `ProtocolSection71Context` | Owns one compact target and one specialized paper-faithful Section 7.1 theorem instance | Theorem-Target |
| Projection | `ProtocolSection71Context.ccsRelation` | Recover compact CCS relation from the packaged theorem instance | Theorem-Target |
| Projection | `ProtocolSection71Context.ceRelation` | Recover compact CE relation from the packaged theorem instance | Theorem-Target |

Definition-14 share facts (challenge-set equality, shared commitment, shared
public input, full-vector assignment) are read through the carried
`theoremInstance` field, which owns those theorem surfaces.

## Proof Obligations

- The context object must be a pure packaging layer; it introduces no new assumptions beyond the carried `ProtocolSection71TheoremInstance`.
- The relation projections must be immediate theorem wrappers over the carried theorem instance.

## Assumption Ledger

- This module introduces no new theorem-level assumptions.
- Construction of `ProtocolSection71Context` remains an upstream task: the repo still needs a canonical source of the specialized `ProtocolSection71TheoremInstance`.

## Dependency and Consumer Map

- Upstream dependencies:
  - `SuperNeo/FoldingProtocol/ProtocolRelations.lean`
- Downstream consumers:
  - `formal/direct-ccs-fprime-lean`: carries `ProtocolSection71Context` producers in its stage-computation packages and consumes `.target` / `.ceRelation`.

## Quality Expectations

- Keep the component thin and stable.
- Do not duplicate theorem content already owned by `ProtocolSection71TheoremInstance`.

## Acceptance Criteria

1. `lake build` succeeds.
2. `lake exe check` succeeds.
3. No `sorry`.
