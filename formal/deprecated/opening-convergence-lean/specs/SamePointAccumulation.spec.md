# SamePointAccumulation Specification

## Purpose

Prove that Phase 2a (same-object same-point identity collapse) preserves
the evaluation relation and provenance. Claims sharing the same opened
object, unified point r*, and payload (guaranteed by
SameObjectPayloadUniqueness from Phase 1) are collapsed into a single
ReducedEvalClaim without loss of semantic content.

## Target Formulas

### Theorem 7: Phase2IdentityCollapse

Given a Phase2Group where all payloads are identical (from
SameObjectPayloadUniqueness):

**Semantic equivalence:**
```
(forall i: f(r*) = v*_i)  <->  f(r*) = v*
```

where v* = v*_0 (any representative, since all are identical).

**Provenance preservation:**
- `source_claim_ids` are in canonical bucket order
- `reduced_claim_digest` is deterministic

### Theorem 8: SingletonPassthrough

If a Phase 2 group has exactly one claim, the output ReducedEvalClaim is
identical to the input (modulo type wrapper).

The five singleton families (Stage2RegisterReads, Stage2RegisterWrites,
Stage2RamEvents, Stage2TwistLinks, Stage3Continuity) use this path in v1.

## Explicit Type Signatures

| Lean symbol | Type | Lives in |
|---|---|---|
| `Phase2Group` | Structure | Groups claims by (opened_object, point) |
| `Phase2Group.openedObject` | `OpenedObjectId` | Shared opened object |
| `Phase2Group.point` | `Fin ell -> K` | Shared evaluation point r* |
| `Phase2Group.payloads` | `Fin groupSize -> FamilyEvalPayload K` | All payloads in group |
| `Phase2Group.hIdentical` | Proof | All payloads are equal |
| `ReducedEvalClaim` | Structure | Output: collapsed claim |
| `sourceIds` | `Fin groupSize -> Nat` | Canonical ordered source IDs |

## Phase 2 Semantics

Phase 2 operates purely on the output of Phase 1. Its input is a list of
unified claims at r*, grouped by (opened_object, point). Within each
group, SameObjectPayloadUniqueness guarantees all payloads are equal.

The collapse is therefore trivially sound: picking any representative
payload preserves the evaluation relation exactly.

The nontrivial contribution of this module is:
1. Formalizing that "trivially sound" as a machine-checked theorem
2. Proving provenance is preserved (source IDs, deterministic digest)
3. Handling the singleton case explicitly for the five non-batched families

## Paper Anchors

- SuperNeo Section 6: Deduplication after batch reduction
- Nightstream architecture: Phase 2a identity collapse

## Module Mapping

| Existing module | Import | What it provides |
|---|---|---|
| `OpeningConvergence.Basic` | Core types | FamilyEvalPayload, ReducedEvalClaim |
| `OpeningConvergence.BatchEvalReductionInterface` | SameObjectPayloadUniqueness | hIdentical justification |

## Contract Surface

| Lean symbol | Kind | Role | Guarantee |
|---|---|---|---|
| `phase2IdentityCollapse` | Theorem | P0 | Collapsed claim preserves evaluation relation |
| `phase2IdentityCollapseProvenance` | Theorem | P1 | Source IDs preserved in canonical order |
| `singletonPassthrough` | Theorem | P0 | Singleton group produces identical output |
| `Phase2Group` | Structure | Input | Groups with identical payloads |
