# MatrixTransform Spec

## Purpose

- **What it is**: Theorem-4 matrix-vector product transform layer. Defines `matrixVecDirect` (direct Mz) and `matrixVecCtBar` (ring-level computation: block-wise `ct(mulRqPhi(bar(row_j), z_j))` summed over d-sized blocks), the executable check `matrixTransformIdentity`, and the universal contract `matrixTransformAssumption`.
- **Key property**: `Mz = ct(M̄z)` — for shape-compatible `m`, `z`, the direct field-level matrix-vector product equals the ring-level product under block-wise left-bar transform and constant-term extraction. Proved from Theorem 3 via block decomposition (Appendix D.1).
- **Protocol role**: EvalLink depends on `matrixTransformAssumption`. ProtocolTarget uses matrix-transform for the evaluation homomorphism (Theorem 5) chain.

## Target Formulas (Paper → Lean)

- Paper formula: Theorem 4 (Matrix-Vector Product Transform): `Mz = ct(M̄z)`
  - Appendix D.1 proof shape: `ct(⟨M̄_i, z⟩) = Σ_j ct(M̄_{i,j} · z_j) = Σ_j ⟨M_{i,j}, z_j⟩ = ⟨M_i, z⟩`
- Lean mapping:
  - `ctBarDot bar a b` : `ct(mulRqPhi(superneoBarBlock bar a, b))` — the Theorem-3 kernel
  - `extractBlock v j` : j-th d-sized block of v
  - `ringBlockDot bar row z` : `Σ_j ctBarDot bar (extractBlock row j) (extractBlock z j)` — ring-level dot product
  - `matrixVecCtBar bar m z` : `m.map (fun row => ringBlockDot bar row z)` — ring-level matrix-vector product
  - `matrixVecDirect m z` : direct Mz (row-wise field dot product)
  - `matrixTransformAssumption bar m` : `∀ z, MatrixRowsCompatible m z → matrixVecDirect m z = matrixVecCtBar bar m z`
- Target statement: `matrixTransformAssumption` derived from the Theorem-3 kernel through block decomposition, with theorem-native entrypoints from `thm3CoreAssumption`, the finite basis criterion, and the finite checker.

## Paper Anchors

- Source: `./formal/superneo-lean/SuperNeo.pdf.md`
- Anchors:
  - Theorem 4 (Matrix-Vector Product Transform), Section 5, lines 384-386: `Mz = ct(M̄z)`
  - Appendix D.1 (Proof of Theorem 4), lines 782-788: block decomposition proof

## Module Mapping

- Implementation: `SuperNeo.MatrixTransform`
- Interface: `SuperNeo.MatrixTransformInterface`

## Contract Surface

| Contract group | Lean surface | Preconditions | Guarantee | Role | Used by |
|---|---|---|---|---|---|
| Dot product | `dotVec a b` | None (size guard) | `dotVec a b = innerProduct a b` | Definitional | — |
| Direct product | `matrixVecDirect m z` | None | Row-wise `directBlockDot row z` where `directBlockDot = Σ_j innerProduct(row_j,z_j)` over d-sized blocks | Definitional | — |
| Theorem-3 kernel | `ctBarDot bar a b` | None | `ct(mulRqPhi(superneoBarBlock bar a, b))` | Definitional | — |
| Block extraction | `extractBlock v j` | None | j-th d-sized block via `v.extract (j*d) ((j+1)*d)` | Definitional | — |
| Ring dot product | `ringBlockDot bar row z` | None | `Σ_j ctBarDot bar (extractBlock row j) (extractBlock z j)` | Definitional | — |
| Bar-lifted product | `matrixVecCtBar bar m z` | None | Row-wise `ringBlockDot bar row z` | Definitional | `EvalHom.lean`, `Checks.lean` |
| Dot/inner equivalence | `dotVec_eq_innerProduct` | None | `dotVec a b = innerProduct a b` | Theorem-Target | — |
| Row compatibility | `MatrixRowsCompatible m z` | None | `z.size % d = 0 ∧ ∀ i, (m[i]).size = z.size` | Definitional | — |
| Check surface | `matrixTransformIdentity bar m z` | None | `true ↔ MatrixRowsCompatible m z ∧ matrixVecDirect m z = matrixVecCtBar bar m z` | Theorem-Target | — |
| Theorem-facing boundary | `matrixTransformAssumption bar m` | None | `∀ z, MatrixRowsCompatible m z → matrixVecDirect m z = matrixVecCtBar bar m z` | Boundary | `EvalLink.lean` |
| Check-facing boundary | `matrixTransformCheckAssumption bar m` | None | `∀ z, MatrixRowsCompatible m z → matrixTransformIdentity bar m z = true` | Theorem-Target | — |
| Thm3 closure | `matrixTransformAssumption_of_thm3CoreAssumption` | `thm3CoreAssumption bar` | `matrixTransformAssumption bar m` | Theorem-Target | — |
| P10 closure | `matrixTransformAssumption_of_p10` | `thm3CoreAssumption bar` | `matrixTransformAssumption bar m` | Theorem-Target | — |
| Basis-kernel closure | `matrixTransformAssumption_of_basisKernelAssumption` | `thm3BasisKernelAssumption bar` | `matrixTransformAssumption bar m` | Theorem-Target | — |
| Basis-kernel checker closure | `matrixTransformAssumption_of_basisKernelCheck` | `thm3BasisKernelCheck bar = true` | `matrixTransformAssumption bar m` | Theorem-Target | — |
| P10+P11 compatibility | `matrixTransformAssumption_of_p10_p11` | `thm3CoreAssumption bar`, `barLiftLinearityAssumption bar` | `matrixTransformAssumption bar m` (via P10 constructor) | Theorem-Target | — |
| Check/prop bridges | `matrixTransformIdentity_sound`, `_complete`, `_iff_prop`, `_of_assumption`, `_of_checkAssumption`, `_iff_checkAssumption` | None | Theorem ↔ check equivalence | Theorem-Target | — |

## Assumption Ledger

- Upstream theorem source: `Thm3Core.lean`.
  Theorem 4 accepts the canonical Theorem-3 surface `thm3CoreAssumption bar`, and also admits the theorem-native finite basis-kernel witnesses `thm3BasisKernelAssumption bar` and `thm3BasisKernelCheck bar = true`.

## Dependency and Consumer Map

- Upstream dependencies:
  - `SuperNeo/Thm3Core.lean`: uses `thm3CoreAssumption`, `thm3BasisKernelAssumption`, `thm3BasisKernelCheck`, and `innerProduct` for Theorem-3-based derivation.
  - `SuperNeo/Ring.lean` (transitive): uses `ct`, `mulRqPhi`, `superneoBarBlock` for ring-level computation.
  - `SuperNeo/BarLift.lean`: only for compatibility constructor `matrixTransformAssumption_of_p10_p11`; core closure path is P10-only.
- Downstream consumers:
  - `SuperNeo/EvalLink.lean`: depends on `matrixTransformAssumption` for evaluation link.
  - `SuperNeo/EvalHom.lean`: uses `matrixVecCtBar` in eval-hom proposition.
  - `SuperNeo/Checks.lean`: uses `matrixVecCtBar` in check functions.
  - `SuperNeo/ArithmeticBundle.lean`: uses `matrixTransformAssumption` in theorem stack.

## Implementation Plan

1. `ctBarDot`, `extractBlock`, `ringBlockDot` defined as ring-level block operations.
2. `matrixVecCtBar` redefined using `ringBlockDot` (paper-faithful).
3. `matrixTransformEq_of_thm3CoreAssumption` is proved by block-wise reduction and per-block use of `thm3CoreAssumption`.
4. Finite basis-kernel constructors are obtained by composing the Theorem-3 basis-kernel bridges with the main Theorem-4 derivation.
5. Sound/complete and check/prop bridges are proved via `decide` reasoning.

## Quality Expectations

- All theorem STATEMENTS are paper-faithful.

## Acceptance Criteria

1. `lake build` succeeds.
2. `matrixTransformAssumption_of_thm3CoreAssumption`, `matrixTransformAssumption_of_basisKernelAssumption`, `matrixTransformAssumption_of_basisKernelCheck`, and all bridges are exported through the interface.

## Out of Scope

- Classifying additional concrete bar designs beyond the theorem-native Theorem-3 surfaces exported by `Thm3Core.lean`.
