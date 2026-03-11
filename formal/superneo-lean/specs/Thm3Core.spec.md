# Thm3Core Spec

## Purpose

- **What it is**: The Theorem-3 inner-product transform module. It defines `innerProduct` (dot product with size guard) and the theorem-facing boundary `thm3CoreAssumption` stating that `ct(mulRqPhi(bar(a), b)) = ⟨a, b⟩` for d-sized blocks — the constant term of the cyclotomic ring product of the left bar-transformed block with the right block equals the field inner product. Also provides the P10 compatibility surface (`p10CoreProp`, `p10CoreCheck`) for executable checks.
- **Key property**: `∀ a b, a.size = d → b.size = d → ct(mulRqPhi(bar(a), b)) = innerProduct a b`.
- **Protocol role**: MatrixTransform uses `thm3CoreAssumption` to derive Theorem 4 (matrix-vector product transform). The chain feeds into EvalLink and EvalHom.

## Target Formulas (Paper → Lean)

- Paper formula in current executable convention: `ct(ā·b) = ⟨a, b⟩` (Theorem 3 boundary surface)
- Lean mapping:
  - `innerProduct a b` : dot product with size guard
  - `thm3CoreAssumption bar` : `∀ a b, a.size = d → b.size = d → ct (mulRqPhi (superneoBarBlock bar a) b) = innerProduct a b`
  - `p10CoreProp bar a b` : `a.size = d ∧ b.size = d ∧ ct (mulRqPhi (superneoBarBlock bar a) b) = innerProduct a b`
  - `p10CoreCheck bar a b = true ↔ p10CoreProp bar a b`
- Target statement:
  - generic boundary surface: `thm3CoreAssumption bar`
  - native constructive closure: `thm3CoreAssumption_native : thm3CoreAssumption nativeBarMatrix`

## Paper Anchors

- Source: `./formal/superneo-lean/SuperNeo.pdf.md`
- Anchors:
  - Theorem 3 (Inner Product Transform), Section 5, lines 368-372.

## Module Mapping

- Implementation: `SuperNeo.Thm3Core`
- Interface: `SuperNeo.Thm3CoreInterface`

## Contract Surface

| Contract group | Lean surface | Preconditions | Guarantee | Role | Used by |
|---|---|---|---|---|---|
| Inner product | `innerProduct a b` | None (size guard returns 0 if mismatch) | Dot product `Σ_i a[i]·b[i]` when `a.size = b.size` | Definitional | `MatrixTransform.lean` |
| Theorem-3 boundary | `thm3CoreAssumption bar` | None | `∀ a b, a.size = d → b.size = d → ct (mulRqPhi (superneoBarBlock bar a) b) = innerProduct a b` | Boundary | `MatrixTransform.lean` |
| Finite basis criterion | `thm3BasisKernelAssumption bar` | None | `∀ i j : Fin d, ct(mulRqPhi(superneoBarBlock bar (basisVec i), basisVec j)) = (if i=j then 1 else 0)` | Theorem-Target | generic closure |
| Finite checker | `thm3BasisKernelCheck bar` | None | `thm3BasisKernelCheck bar = true ↔ thm3BasisKernelAssumption bar` | Theorem-Target | generic closure |
| Reference-name alias | `thm3CoreAssumption_ref36_64 bar` | None | Compatibility alias to canonical `thm3CoreAssumption` naming | Boundary | theorem-native bridging |
| Shape predicates | `IsDVec`, `IsDBarMatrix` | None | `a.size = d`, `bar.size = d ∧ bar.all (fun row => row.size == d) = true` | Definitional | P10 wrappers |
| P10 proposition | `p10CoreProp bar a b` | None | `a.size = d ∧ b.size = d ∧ ct(mulRqPhi(bar(a), b)) = innerProduct a b` | Definitional | — |
| P10 check | `p10CoreCheck bar a b` | None | `p10CoreCheck bar a b = true ↔ p10CoreProp bar a b` | Theorem-Target | — |
| Sound/complete | `p10CoreCheck_sound`, `p10CoreCheck_complete` | Check true / Prop holds | Bidirectional bridge | Theorem-Target | — |
| From preconditions | `p10Core_of_preconditions`, `p10Core_of_preconditions_props`, `p10Core_of_assumption` | Shape + check / Thm3 | `p10CoreProp bar a b` | Theorem-Target | — |

## Proof Obligations and Closure Plan

Closed now:
- P10 sound/complete and `p10Core_of_*` theorems.
- `p10Core_of_assumption` derives P10 from `thm3CoreAssumption` + vector shapes.
- `nativeBarMatrix` is defined constructively and matched to generated artifacts.
- Native kernel closure is constructive: `thm3CoreAssumption_native`.
- Generic closure reduction is constructive:
  - `thm3CoreAssumption_of_basisKernelAssumption`
  - `thm3BasisKernelAssumption_of_thm3CoreAssumption`
  - `thm3CoreAssumption_iff_basisKernelAssumption`
  - `thm3CoreAssumption_of_basisKernelCheck`
  - `thm3CoreAssumption_iff_basisKernelCheck`

Remaining for broader generalization (not required for native paper-faithful proof-complete):
- Classify all `bar` that satisfy the finite basis criterion (optional strengthening). The generic boundary itself is already reduced to and equivalent with the finite basis characterization/checker.

## Assumption Ledger

- Open boundary: `thm3CoreAssumption` — requires a bar transform satisfying `ct(mulRqPhi(bar(a), b)) = ⟨a, b⟩`.
  Closure status:
  - native instance `nativeBarMatrix`: closed via `thm3CoreAssumption_native`.
  - generic `bar`: reduced to finite basis-kernel criterion (`thm3CoreAssumption_iff_basisKernelAssumption`).
- `thm3CoreAssumption_ref36_64` is now only a compatibility alias (not a trusted axiom surface).

## Dependency and Consumer Map

- Upstream dependencies:
  - `SuperNeo/BarLift.lean`: imports `superneoBarBlock` for the bar-transform kernel.
  - `SuperNeo/Ring.lean` (transitive): uses `ct`, `mulRqPhi` for ring-level operations.
- Downstream consumers:
  - `SuperNeo/MatrixTransform.lean`: uses `thm3CoreAssumption` and `innerProduct` to derive `matrixTransformAssumption` (Theorem 4).

## Implementation Plan

1. `innerProduct` defined with size guard; returns 0 on mismatch.
2. `thm3CoreAssumption` stated as theorem boundary: `ct(mulRqPhi(bar(a), b)) = innerProduct a b` for d-sized blocks.
3. P10 check/prop bridges proved via `decide_eq_true` / `decide_eq_true_eq`.
4. Generic Thm3 boundary reduced to finite basis criterion and finite boolean checker (`thm3BasisKernelCheck`).
4. `p10Core_of_assumption` derives from `thm3CoreAssumption` and shape predicates.

## Quality Expectations

- No `sorry` in any theorem.
- Native closure theorem remains constructive and axiom-free at module level.

## Acceptance Criteria

1. `lake build` succeeds.
2. `thm3CoreAssumption_native` and all P10 theorems exported through the interface.

## Out of Scope

- Closed-form classification of all valid `bar` matrices beyond the finite basis criterion/checker equivalence.
