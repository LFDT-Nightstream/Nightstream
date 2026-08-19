# InteractiveReductions

## Purpose

- **What it is**: The composition layer for the reduction pipeline Π_RLC ∘ Π_CCS and Π_DEC ∘ Π_RLC ∘ Π_CCS. `InteractiveReductionAssumptions` is the composed assumption registry: one protocol-target assumption bundle plus one accepted SumCheck transition witness.
- **Key property**: `strongCompositionStatement` (Π_RLC ∘ Π_CCS is strong) and `weakCompositionStatement` (Π_DEC ∘ Π_RLC ∘ Π_CCS is weak) are proved from `InteractiveReductionAssumptions` by composing the Π_CCS/Π_RLC/Π_DEC theorems.
- **Protocol role**: ProtocolTheorem embeds `InteractiveReductionAssumptions` in its final assumption registry and consumes the composition statements. This is the composition capstone for all three reduction steps (CCS → RLC → DEC).

## Target Formulas

- `strongCompositionStatement ctx ↔ piDECKnowledgeStatement ctx`
- `weakCompositionStatement ctx ↔ ceRelaxedRelation ctx ∧ SumCheckClaimTrue (sumcheckInstanceOfContext ctx)`
- `InteractiveReductionAssumptions ctx → strongCompositionStatement ctx`
- `InteractiveReductionAssumptions ctx → weakCompositionStatement ctx`
- `InteractiveReductionAssumptions ctx + (∀ n, 0 ≤ eps n) → SoundnessFailureAdvantageBound(sumcheckInstanceOfContext ctx, witnessTranscript, eps)`

## Paper Anchors

Source: ./formal/superneo-lean/SuperNeo.pdf.md

- Theorem 6 (Strong-Weak Composition), Section 6, lines 438-447.
- Definition 9 (Weak Interactive Reductions), lines 404-416.
- Definition 10 (Strong Interactive Reductions), lines 418-436.

## Module Mapping

| Lean file | Paper section |
|---|---|
| `SuperNeo/SecurityModel/InteractiveReductions.lean` | Theorem 6, Definitions 9–10 |

## Contract Surface

| Group | Lean symbol | Kind | Role | Guarantee |
|---|---|---|---|---|
| Assumptions | `InteractiveReductionAssumptions` | structure | Boundary | Bundles `ProtocolTargetAssumptions` + SumCheck transition witness |
| Statements | `strongCompositionStatement` | def | Definitional | Π_RLC ∘ Π_CCS strong |
| Statements | `weakCompositionStatement` | def | Definitional | Π_DEC ∘ Π_RLC ∘ Π_CCS weak |
| Theorems | `strongComposition_of_assumptions` | theorem | Theorem-Target | Assumptions → strong |
| Theorems | `weakComposition_of_assumptions` | theorem | Theorem-Target | Assumptions → weak |
| Theorems | `sumcheckFailureAdvantageBound_of_assumptions` | theorem | Theorem-Target | Witness-level SumCheck failure-advantage bound from reduction assumptions |

Evidence-specific packaging (paper-carrier difference, basis-kernel Theorem-3
witnesses, native bar) is composed at construction sites from
`ProtocolTargetAssumptions` constructors plus a witness; this module exposes
only the composed registry and the canonical composition theorems.

## Proof Obligations and Closure Plan

- `strongComposition_of_assumptions` composes `piDEC_of_assumptions` over the carried bundle and witness.
- `weakComposition_of_assumptions` projects the weak statement from the strong one.
- `sumcheckFailureAdvantageBound_of_assumptions` discharges the failure event constructively from SumCheck soundness over the carried witness transcript.

## Assumption Ledger

- `InteractiveReductionAssumptions`: boundary assumption bundling protocol-target assumptions and a transition witness; both components remain explicit fields.

## Dependency and Consumer Map

Upstream dependencies:
- `SuperNeo/FoldingProtocol/PiDEC.lean`: imports `piDECKnowledgeStatement`, `ceRelaxedRelation`, `SumCheckClaimTrue`, `sumcheckInstanceOfContext`, `piDEC_of_assumptions`.
- `SuperNeo/SumCheck.lean`: constructive SumCheck truth is used directly in witness-level failure-advantage bounds.

Downstream consumers:
- `SuperNeo/FoldingProtocol/ProtocolTheorem.lean`: embeds `InteractiveReductionAssumptions` in `FinalTheoremAssumptions` and uses the composition statements and the witness-level SumCheck advantage bound.

## Implementation Plan

1. Define the strong/weak composition statements in the compact protocol vocabulary.
2. Prove both composition theorems from `InteractiveReductionAssumptions`.
3. Prove the witness-level SumCheck failure-advantage bound from the carried transcript.

## Quality Expectations

Composition statements must match Theorem 6 (Strong-Weak Composition). Strong/weak definitions must align with Definitions 9 and 10.

## Acceptance Criteria

- `lake build` succeeds.
- No `sorry`.

## Out of Scope

- Concrete deployment/setup instantiation.
- Proof of the underlying cryptographic assumptions themselves.
