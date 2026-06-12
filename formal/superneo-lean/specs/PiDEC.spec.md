# PiDEC

## Purpose

- **What it is**: The decomposition reduction step Π_DEC. Defines `piDECKnowledgeStatement` as the existence of `deltaInv` with `mulRq ctx.invDelta deltaInv = oneRq`, plus `ceRelaxedRelation ctx` and `SumCheckClaimTrue`.
- **Key property**: `piDEC_of_weak` gives the theorem-native entrypoint from the weak `Π_RLC` statement; `piDEC_of_ce` bridges directly from the compact relation layer; `piDEC_of_assumptions` derives the same statement from the upstream protocol-target assumption bundle plus one accepted transition witness. Invertibility is extracted directly from `protocolTargetProp` (via `ceRelaxedRelation`), not from a separate low-norm boundary input.
- **Protocol role**: ProtocolTheorem and FoldingProtocol depend on `piDECKnowledgeStatement` for the knowledge-soundness chain. Section 7.5 (Π_DEC) reduces norm from B to b via decomposition.

## Target Formulas

- `piDECKnowledgeStatement(ctx) ↔ ∃ deltaInv, mulRq ctx.invDelta deltaInv = oneRq ∧ ceRelaxedRelation(ctx) ∧ SumCheckClaimTrue(sumcheckInstanceOfContext ctx)`
- `piDEC_of_weak`: `piRLCWeakStatement ctx → piDECKnowledgeStatement ctx`
- `piDEC_of_ce`: `ceRelation ctx → piDECKnowledgeStatement ctx`
- `piDEC_of_assumptions`: `ProtocolTargetAssumptions ctx → SumCheckTransitionWitness ctx → piDECKnowledgeStatement ctx`
- Theorem 7: Π_DEC : CE(B) → CE(b)^k is a reduction of knowledge.

## Paper Anchors

Source: ./formal/superneo-lean/SuperNeo.pdf.md

- Section 7.5 (Π_DEC), lines 585-593.
- Theorem 7 (Π_DEC is reduction of knowledge), lines 594-596.

## Module Mapping

| Lean file | Paper section |
|---|---|
| `SuperNeo/FoldingProtocol/PiDEC.lean` | Section 7.5, Theorem 7 |

## Contract Surface

| Group | Lean symbol | Kind | Role | Guarantee |
|---|---|---|---|---|
| Statement | `piDECKnowledgeStatement` | def | Definitional | ∃ deltaInv, inverse ∧ ceRelaxed ∧ claimTrue |
| Theorem | `piDEC_of_weak` | theorem | Theorem-Target | Weak statement → knowledge statement |
| Theorem | `piDEC_of_ce` | theorem | Theorem-Target | CE relation → knowledge statement |
| Theorem | `piDEC_of_assumptions` | theorem | Theorem-Target | Protocol-target assumptions + witness → knowledge statement |

Richer upstream evidence is converted upstream (Section 7.1 owners yield
`ceRelation`; evidence-specific constructors live with
`ProtocolTargetAssumptions`); Π_DEC exposes only the canonical routes above.

## Proof Obligations and Closure Plan

- `piDEC_of_weak` is the compact theorem-native entrypoint; it extracts invertibility from `protocolTargetProp` carried inside the relaxed relation.
- `piDEC_of_ce` factors through `piRLCWeak_of_ce`.
- `piDEC_of_assumptions` factors through `piRLCWeak_of_assumptions`.

## Assumption Ledger

- No extra invertibility boundary is threaded at `PiDEC` level; invertibility is already required upstream in `ProtocolTargetAssumptions`.
- No separate SumCheck bundle is introduced locally here.

## Dependency and Consumer Map

Upstream dependencies:
- `SuperNeo/FoldingProtocol/PiRLC.lean`: uses `piRLCWeakStatement`, `piRLCWeak_of_ce`, `piRLCWeak_of_assumptions`.
- `SuperNeo/FoldingProtocol/ProtocolTarget.lean`: `protocolTargetProp` carries `invertibleRq ctx.invDelta`.

Downstream consumers:
- `SuperNeo/SecurityModel/InteractiveReductions.lean`: `strongComposition_of_assumptions` is `piDEC_of_assumptions` under the composed assumption registry.
- `formal/direct-ccs-fprime-lean`: consumes `piDECKnowledgeStatement` and `piDEC_of_ce` through `PiDECInterface`.

## Implementation Plan

1. Define `piDECKnowledgeStatement` as inverse existence together with `ceRelaxedRelation` and `SumCheckClaimTrue`.
2. Factor the theorem-native proof through the weak `Π_RLC` statement.
3. Derive the CE and assumptions routes from the same compact proof.

## Quality Expectations

`piDECKnowledgeStatement` must match Theorem 7: inverse existence plus relaxed CE and sum-check claim. Derivation must use invertibility assumption correctly.

## Acceptance Criteria

- `lake build` succeeds.
- No `sorry`.
- `piDEC_of_assumptions` proved.

## Out of Scope

- Proof of Theorem 7 (deferred to appendix).
- Concrete invertibility bound instantiation.
