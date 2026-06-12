# PiRLC

## Purpose

- **What it is**: The weak interactive-reduction step Π_RLC. Defines `piRLCWeakStatement` as the conjunction of `ceRelaxedRelation ctx` and `SumCheckClaimTrue (sumcheckInstanceOfContext ctx)`.
- **Key property**: `piRLCWeak_of_ce` gives the theorem-native entrypoint from `ceRelation ctx`; `piRLCWeak_of_assumptions` derives the same statement from the upstream protocol-target assumption bundle plus one accepted transition witness. The weak statement relaxes CE to ceRelaxed (CCS only).
- **Protocol role**: PiDEC depends on `piRLCWeakStatement` for the weak→knowledge composition. Section 7.4 (Π_RLC) performs random linear combination of CE claims.

## Target Formulas

- `piRLCWeakStatement(ctx) ↔ ceRelaxedRelation(ctx) ∧ SumCheckClaimTrue(sumcheckInstanceOfContext ctx)`
- `piRLCWeak_of_ce`: `ceRelation ctx → piRLCWeakStatement ctx`
- `piRLCWeak_of_assumptions`: `ProtocolTargetAssumptions ctx → SumCheckTransitionWitness ctx → piRLCWeakStatement ctx`
- Weak reduction (Lemma 4): Π_RLC : CE^{K+k} → CE(B) is weak for φ projecting commitments.

## Paper Anchors

Source: ./formal/superneo-lean/SuperNeo.pdf.md

- Section 7.4 (Π_RLC), lines 550-571.
- Lemma 4 (Π_RLC is weak), lines 569-570.

## Module Mapping

| Lean file | Paper section |
|---|---|
| `SuperNeo/FoldingProtocol/PiRLC.lean` | Section 7.4, Lemma 4 |

## Contract Surface

| Group | Lean symbol | Kind | Role | Guarantee |
|---|---|---|---|---|
| Statement | `piRLCWeakStatement` | def | Definitional | ceRelaxedRelation ∧ SumCheckClaimTrue |
| Theorem | `piRLCWeak_of_ce` | theorem | Theorem-Target | CE relation → weak statement |
| Theorem | `piRLCWeak_of_assumptions` | theorem | Theorem-Target | Protocol-target assumptions + witness → weak statement |

Richer upstream evidence is converted upstream (Section 7.1 owners yield
`ceRelation`; evidence-specific constructors live with
`ProtocolTargetAssumptions`); Π_RLC exposes only the two canonical routes.

## Proof Obligations and Closure Plan

- `piRLCWeak_of_ce` derives both conjuncts from one CE witness.
- `piRLCWeak_of_assumptions` derives `ccsRelation` from the assumption bundle and lifts it to `ceRelation` through the accepted transition witness.

## Assumption Ledger

- SumCheck truth is discharged from the accepted transition witness rather than introduced as a separate local boundary here.

## Dependency and Consumer Map

Upstream dependencies:
- `SuperNeo/FoldingProtocol/PiCCS.lean`: precedes Π_RLC in the Section 7 composition chain.
- `SuperNeo/FoldingProtocol/ProtocolRelations.lean`: uses `ceRelaxedRelation_of_ce`, `ceClaimTrue_of_ce`, `ceRelation_of_ccsRelation`.
- `SuperNeo/FoldingProtocol/ProtocolTarget.lean`: uses `ProtocolTargetAssumptions`, `protocolTargetProp_of_assumptions`.

Downstream consumers:
- `SuperNeo/FoldingProtocol/PiDEC.lean`: uses `piRLCWeakStatement`, `piRLCWeak_of_ce`, `piRLCWeak_of_assumptions`.
- `formal/direct-ccs-fprime-lean`: consumes `piRLCWeakStatement` and `piRLCWeak_of_ce` through `PiRLCInterface`.

## Implementation Plan

1. Define `piRLCWeakStatement` as `ceRelaxedRelation ctx ∧ SumCheckClaimTrue (sumcheckInstanceOfContext ctx)`.
2. Factor the theorem-native proof through `ceRelation`.
3. Derive the assumptions route from the same compact relation proof.

## Quality Expectations

`piRLCWeakStatement` must match Lemma 4: relaxed CE relation plus sum-check claim truth.

## Acceptance Criteria

- `lake build` succeeds.
- No `sorry`.
- `piRLCWeak_of_ce` proved.
- `piRLCWeak_of_assumptions` proved.

## Out of Scope

- Full protocol execution (ProofSystem layer).
- Probabilistic weak-reduction proof (Lemma 4 proof deferred to appendix).
