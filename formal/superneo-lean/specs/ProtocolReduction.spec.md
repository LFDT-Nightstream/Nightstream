# ProtocolReduction Spec

## Purpose

- **What it is**: Protocol skeleton composition from Section 7 reduction steps. Defines `p10ForClaim` and `arithmeticBundleForClaim` as claim-level surfaces, then derives `CEValid` from props, Thm3 assumption, or checks.
- **Key property**: `p10ForClaim ∧ arithmeticBundleForClaim ∧ witness/norm → CEValid`; `thm3CoreAssumption ∧ arithmeticBundleForClaim ∧ witness/norm → CEValid`; check-driven assumptions imply CEValid via `protocolMathTargetWithThm3Prop_of_checks`.
- **Protocol role**: Composes protocol reduction steps; derived from Section 7 reduction steps (Π_CCS, Π_RLC, Π_DEC) and ProtocolRelations.

## Target Formulas (Paper → Lean)

- `p10ForClaim ctx claim ↔ p10CoreProp ctx.bar claim.a claim.b`
- `arithmeticBundleForClaim ctx claim ↔ arithmeticBundleProp ... (claim fields) ...`
- `superneoMathProtocolSkeleton_of_props`: `p10ForClaim ∧ arithmeticBundleForClaim ∧ ClaimShapeValid ∧ witness.z = claim.z ∧ norm < ceNormBound → CEValid`

- `superneoMathProtocolSkeleton_of_thm3_assumption`: `thm3CoreAssumption ∧ arithmeticBundleForClaim ∧ ... → CEValid`
- `superneoMathProtocolSkeleton_of_checks`: check-driven (P10, P6, P12, P14, module, P17, P18, P19) ∧ witness/norm → CEValid
- `smoke_checks_imply_props`: checks → `p10ForClaim ∧ arithmeticBundleForClaim`
- `smoke_props_imply_check_subset`: props → check subset (P10, P6, P12, module, P17, P18, P19)
- `smoke_protocolMathTarget_compose`: props ∧ witness/norm → CEValid

## Paper Anchors

- Source: `./formal/superneo-lean/SuperNeo.pdf.md`
- Anchors:
  - Section 7 (Neo's folding scheme for CCS), lines 447–596: Relations, reduction steps (Π_CCS, Π_RLC, Π_DEC)
  - Section 7.2–7.5, lines 467–596: Folding scheme via interactive reductions

## Module Mapping

- Implementation: `SuperNeo.ProtocolReduction`
- Interface: `SuperNeo.ProtocolReductionInterface`

## Contract Surface

| Contract group | Lean surface | Preconditions | Guarantee | Role | Used by |
|---|---|---|---|---|---|
| Claim surfaces | `p10ForClaim`, `arithmeticBundleForClaim` | None | Claim-level P10 and arithmetic bundle | Definitional | — |
| Skeleton from props | `superneoMathProtocolSkeleton_of_props` | ClaimShapeValid, IsDBarMatrix, IsDVec, p10ForClaim, arithmeticBundleForClaim, witness, norm | CEValid | Theorem-Target | — |
| Skeleton from Thm3 | `superneoMathProtocolSkeleton_of_thm3_assumption` | ClaimShapeValid, IsDBarMatrix, IsDVec, thm3CoreAssumption, arithmeticBundleForClaim, witness, norm | CEValid | Theorem-Target | — |
| Skeleton from checks | `superneoMathProtocolSkeleton_of_checks` | ClaimShapeValid, IsDBarMatrix, IsDVec, check-driven (P10, P6, P12, P14, module, P17, P18, P19), witness, norm | CEValid | Theorem-Target | — |
| Smoke: checks → props | `smoke_checks_imply_props` | Check-driven assumptions | `p10ForClaim ∧ arithmeticBundleForClaim` | Theorem-Target | — |
| Smoke: props → check subset | `smoke_props_imply_check_subset` | ClaimShapeValid, p10ForClaim ∧ arithmeticBundleForClaim | P10, P6, P12, module, P17, P18, P19 checks | Theorem-Target | — |
| Smoke: compose | `smoke_protocolMathTarget_compose` | ClaimShapeValid, IsDBarMatrix, IsDVec, props, witness, norm | CEValid | Theorem-Target | — |

## Proof Obligations and Closure Plan

All obligations closed. `superneoMathProtocolSkeleton_of_props` uses `protocolMathTargetProp_of_arithmeticBundle` and `protocolMathTargetProp_to_CEValid`. `superneoMathProtocolSkeleton_of_thm3_assumption` derives P10 from `p10Core_of_assumption` then applies props skeleton. `superneoMathProtocolSkeleton_of_checks` uses `protocolMathTargetWithThm3Prop_of_checks` and `protocolMathTargetWithThm3Prop_to_CEValid`. Smoke theorems are compile-only sanity checks.

## Assumption Ledger

No open boundary assumptions in this module.

## Dependency and Consumer Map

- Upstream dependencies:
  - `SuperNeo/ProtocolRelations.lean`: imports `ProtocolCtx`, `CEClaim`, `CEWitness`, `CEValid`, `ClaimShapeValid`, `protocolMathTargetProp_to_CEValid`, `protocolMathTargetWithThm3Prop_to_CEValid`, `arithmeticBundleProp_of_checks`, `arithmeticBundleProp_props_imply_check_subset`, `arithmeticBundleProp_props_imply_module_checks`, `p10CoreCheck_sound`, `p10CoreCheck_complete`
- Downstream consumers:
  - `SuperNeo/FoldingProtocol.lean`: uses ProtocolReduction for protocol composition

## Implementation Plan

1. `p10ForClaim` and `arithmeticBundleForClaim` defined as claim-level wrappers.
2. `superneoMathProtocolSkeleton_of_props` composes `protocolMathTargetProp_of_arithmeticBundle` with `protocolMathTargetProp_to_CEValid`.
3. `superneoMathProtocolSkeleton_of_thm3_assumption` derives P10 from Thm3 then applies props skeleton.
4. `superneoMathProtocolSkeleton_of_checks` uses `protocolMathTargetWithThm3Prop_of_checks` and `protocolMathTargetWithThm3Prop_to_CEValid`.
5. Smoke theorems prove check/prop bridges and composition.

## Quality Expectations

- No `sorry` in any theorem.
- All declarations proved natively.

## Acceptance Criteria

1. `lake build` succeeds.
2. `lake exe check` succeeds.
3. All surfaces exported through the interface.

## Out of Scope

- Concrete instantiation of ProtocolCtx or CEClaim; those belong to protocol setup.
- `protocolMathTargetProp_to_CEValid` and `protocolMathTargetWithThm3Prop_to_CEValid` are in ProtocolRelations.
