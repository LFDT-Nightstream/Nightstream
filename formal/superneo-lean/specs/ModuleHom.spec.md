# ModuleHom Spec

## Purpose

- **What it is**: The module-hom layer formalizes R_F-module homomorphisms (Definition 15 implicit). It defines `VecModuleHom` and `ScalarModuleHom` structures, one-point linearity obligations (`vecModulePropPair`, `scalarModulePropPair`), executable checks (`vecModuleCheckPair`, `scalarModuleCheckPair`, `preservesAddVec`, `preservesScaleVec`, etc.), and universal theorem/check-facing contracts (`vecModuleAssumption`, `scalarModuleAssumption`, `vecModuleCheckAssumption`, `scalarModuleCheckAssumption`).
- **Key property**: `map(x+y) = map(x)+map(y)` and `map(s•x) = s•map(x)` for vector maps; `map(x+y) = map(x)+map(y)` and `map(s•x) = s*map(x)` for scalar maps — i.e. additivity and scalar-linearity.
- **Protocol role**: EvalHom uses module-hom assumptions to derive the evaluation homomorphism (Theorem 5). ArithmeticBundle checks module-hom properties.

## Target Formulas (Paper → Lean)

- Paper formula(s):
  - Vector hom: `L(x+y) = L(x) + L(y)` and `L(s·x) = s·L(x)`
  - Scalar hom: `L_in(x+y) = L_in(x) + L_in(y)` and `L_in(s·x) = s·L_in(x)`
- Lean mapping:
  - `vecModulePropPair h s x y` : `h.map (vecAdd x y) = vecAdd (h.map x) (h.map y) ∧ h.map (vecScale s x) = vecScale s (h.map x)`
  - `scalarModulePropPair h s x y` : `h.map (vecAdd x y) = h.map x + h.map y ∧ h.map (vecScale s x) = s * h.map x`
  - `vecModuleAssumption h` : `(∀ x y, h.map (vecAdd x y) = vecAdd (h.map x) (h.map y)) ∧ (∀ s x, h.map (vecScale s x) = vecScale s (h.map x))`
  - `scalarModuleAssumption h` : `(∀ x y, h.map (vecAdd x y) = h.map x + h.map y) ∧ (∀ s x, h.map (vecScale s x) = s * h.map x)`
  - `vecModuleCheckPair h s x y = true ↔ vecModulePropPair h s x y` (and scalar analogue)
- Target statement: All sound/complete and assumption bridges proved.

## Paper Anchors

- Source: `./formal/superneo-lean/SuperNeo.pdf.md`
- Anchors:
  - Theorem 5, Section 5, lines 390-400: R_F-module homomorphisms L and L_in
  - Definition 15 (implicit): module homomorphism structure

## Module Mapping

- Implementation: `SuperNeo.ModuleHom`
- Interface: `SuperNeo.ModuleHomInterface`

## Contract Surface

| Contract group | Lean surface | Preconditions | Guarantee | Role | Used by |
|---|---|---|---|---|---|
| Structures | `VecModuleHom`, `ScalarModuleHom` | None | Map signatures `Array F → Array F` / `Array F → F` | Definitional | `EvalHom.lean` |
| One-point prop | `vecModulePropPair`, `scalarModulePropPair` | None | Additivity ∧ scaling at one point | Definitional | — |
| One-point check | `vecModuleCheckPair`, `scalarModuleCheckPair` | None | `checkPair = true ↔ propPair` | Theorem-Target | `ArithmeticBundle.lean` |
| Compatibility checks | `preservesAddVec`, `preservesScaleVec`, `preservesAddScalar`, `preservesScaleScalar` | Size equality for add | Executable linearity checks | Theorem-Target | — |
| Sound/complete | `preservesAdd*_sound/complete`, `preservesScale*_sound/complete` | Check true / Prop holds | Bidirectional bridge | Theorem-Target | — |
| Prop/check bridges | `vecModulePropPair_of_checkPair`, `vecModuleCheckPair_of_propPair`, `vecModuleCheckPair_iff_propPair`, scalar analogues | — | One-point ↔ check | Theorem-Target | — |
| Universal theorem | `vecModuleAssumption`, `scalarModuleAssumption` | None | `∀` linearity | Definitional | `EvalHom.lean` |
| Universal check | `vecModuleCheckAssumption`, `scalarModuleCheckAssumption` | None | `∀ s x y, checkPair = true` | Definitional | — |
| Assumption bridges | `vecModuleAssumption_of_checkAssumption`, `vecModuleCheckAssumption_of_assumption`, `vecModuleAssumption_iff_checkAssumption`, scalar analogues | — | Theorem ↔ check universal | Theorem-Target | — |

## Proof Obligations and Closure Plan

All obligations closed. Sound/complete theorems proved via `decide_eq_true` / `decide_eq_true_eq`. Universal assumption bridges proved by instantiating one-point pairs (e.g. `s := 0`, `y := x` for scaling).

## Assumption Ledger

No open boundary assumptions in this module.

## Dependency and Consumer Map

- Upstream dependencies:
  - `SuperNeo/Ring.lean`: imports `vecAdd`, `vecScale` for linearity definitions.
- Downstream consumers:
  - `SuperNeo/EvalHom.lean`: uses `vecModuleAssumption` and `scalarModuleAssumption` to derive `evalHomAssumption` (Theorem 5).
  - `SuperNeo/ArithmeticBundle.lean`: imports and checks module-hom properties.

## Implementation Plan

1. `VecModuleHom` / `ScalarModuleHom` defined as structures with `map` field.
2. Prop/check pairs defined; decidable instances for `decide`.
3. Sound/complete proved via `decide_eq_true_eq` / `decide_eq_true`.
4. Universal assumption bridges: check → theorem via `vecModulePropPair_of_checkPair`; theorem → check via `vecModuleCheckPair_of_propPair`; iff surfaces make the module theorem-facing enough for downstream use.

## Quality Expectations

- No `sorry` in any theorem.
- Module-hom is the single semantic surface for R_F-linear maps; check/prop duality is explicit.

## Acceptance Criteria

1. `lake build` succeeds.
2. `lake exe check` succeeds.
3. All module-hom theorems exported through the interface.

## Out of Scope

- Full evaluation-hom instantiation (belongs to EvalHom).
- Algebraic module theory (associativity, etc.).
