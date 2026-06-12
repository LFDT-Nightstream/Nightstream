# PiCCS

## Purpose

- **What it is**: The strong interactive-reduction step Π_CCS. Defines `piCCSStrongStatement` as the conjunction of `ceRelation ctx` and `SumCheckClaimTrue (sumcheckInstanceOfContext ctx)`.
- **Key property**: `piCCSStrong_of_ce` gives the theorem-native entrypoint from `ceRelation ctx`; `piCCSStrong_of_assumptions` derives the same statement from the upstream protocol-target assumption bundle plus one accepted transition witness.
- **Protocol role**: PiRLC depends on `piCCSStrongStatement` for the strong→weak composition (Theorem 6). Section 7.3 (Π_CCS) reduces CCS instances to CE instances via sum-check.

## Target Formulas

- `piCCSStrongStatement(ctx) ↔ ceRelation(ctx) ∧ SumCheckClaimTrue(sumcheckInstanceOfContext ctx)`
- `piCCSStrong_of_ce`: `ceRelation ctx → piCCSStrongStatement ctx`
- `piCCSStrong_of_assumptions`: `ProtocolTargetAssumptions ctx → SumCheckTransitionWitness ctx → piCCSStrongStatement ctx`
- Strong reduction (Lemma 3): Π_CCS : CCS^K × CE^k → CE^{K+k} is strong for φ projecting commitments.

## Paper Anchors

Source: ./formal/superneo-lean/SuperNeo.pdf.md

- Section 7.3 (Π_CCS), lines 481-548.
- Lemma 3 (Π_CCS is strong), lines 545-546.

## Module Mapping

| Lean file | Paper section |
|---|---|
| `SuperNeo/FoldingProtocol/PiCCS.lean` | Section 7.3, Lemma 3 |

## Contract Surface

| Group | Lean symbol | Kind | Role | Guarantee |
|---|---|---|---|---|
| Statement | `piCCSStrongStatement` | def | Definitional | ceRelation ∧ SumCheckClaimTrue |
| Theorem | `piCCSStrong_of_ce` | theorem | Theorem-Target | CE relation → strong statement |
| Theorem | `piCCSStrong_of_assumptions` | theorem | Theorem-Target | Protocol-target assumptions + witness → strong statement |

Richer upstream evidence (a Section 7.1 theorem instance, paper-carrier
difference data, basis-kernel Theorem-3 witnesses) is converted upstream:
owners yield `ceRelation` via `ProtocolSection71TheoremInstance.ceRelation` /
`ProtocolSection71Context.ceRelation`, and evidence-specific constructors live
with `ProtocolTargetAssumptions`. Π_CCS itself exposes only the two canonical
routes above.

## Proof Obligations and Closure Plan

- `piCCSStrong_of_ce` is the compact theorem-native entrypoint.
- `piCCSStrong_of_assumptions` derives `ccsRelation` from the assumption bundle and lifts it to `ceRelation` through the accepted transition witness.

## Assumption Ledger

- SumCheck truth is discharged from the accepted transition witness rather than introduced as a separate local boundary here.

## Dependency and Consumer Map

Upstream dependencies:
- `SuperNeo/FoldingProtocol/ProtocolRelations.lean`: uses `ceRelation`, `ccsRelation`, `SumCheckTransitionWitness`, `sumcheckInstanceOfContext`, `ceRelation_of_ccsRelation`, `ceClaimTrue_of_ce`.
- `SuperNeo/FoldingProtocol/ProtocolTarget.lean`: uses `ProtocolTargetAssumptions`, `protocolTargetProp_of_assumptions`.

Downstream consumers:
- `SuperNeo/FoldingProtocol/PiRLC.lean`: uses `piCCSStrongStatement`.
- `formal/direct-ccs-fprime-lean`: consumes `piCCSStrongStatement` and `piCCSStrong_of_ce` through `PiCCSInterface`.

## Implementation Plan

1. Define `piCCSStrongStatement` as `ceRelation ctx ∧ SumCheckClaimTrue (sumcheckInstanceOfContext ctx)`.
2. Factor the theorem-native proof through `ceRelation`.
3. Derive the assumptions route from the same compact relation proof.

## Quality Expectations

`piCCSStrongStatement` must match Lemma 3: CE relation plus sum-check claim truth. Derivation must thread assumptions correctly.

## Acceptance Criteria

- `lake build` succeeds.
- No `sorry`.
- `piCCSStrong_of_ce` proved.
- `piCCSStrong_of_assumptions` proved.

## Out of Scope

- Full protocol execution (ProofSystem layer).
- Probabilistic strong-reduction proof (Lemma 3 proof deferred to appendix).
