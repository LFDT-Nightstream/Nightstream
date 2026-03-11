# ArithmeticBundle

## Purpose

- **What it is**: A proposition-level bundle composing arithmetic obligations from Sections 4–5: decomposition (P6), matrix transform (P12), evaluation homomorphism (P14), module linearity (P15), invertibility (P16), sampling (P17), polynomial (P18), and interpolation (P19). Defines `arithmeticBundleProp` as the conjunction of all these.
- **Key property**: `arithmeticBundleProp_of_checks`: executable checks (splitRoundTrip, matrixTransformIdentity, evalHom2, preservesAddVec, etc.) imply `arithmeticBundleProp`. Conversely, `arithmeticBundleProp_props_imply_check_subset` and `arithmeticBundleProp_props_imply_module_checks` give check-level obligations from the proposition.
- **Protocol role**: ProtocolTarget and ArithmeticObligations depend on `arithmeticBundleProp` for the Section 7 protocol composition context. Bundles arithmetic side-conditions for folding reductions.

## Target Formulas

- `arithmeticBundleProp ↔ arithmeticDecompProp ∧ MatrixRowsCompatible ∧ matrixVecDirect = matrixVecCtBar ∧ arithmeticEvalHomProp ∧ arithmeticVecModuleProp ∧ arithmeticScalarModuleProp ∧ invertibilityPreconditionsProp ∧ arithmeticSamplingProp ∧ arithmeticPolyProp ∧ arithmeticInterpProp`
- `splitRoundTrip zDecomp b k = true → arithmeticDecompProp zDecomp b k`
- `matrixTransformIdentity bar m z = true → MatrixRowsCompatible m z ∧ matrixVecDirect m z = matrixVecCtBar bar m z`
- `arithmeticBundleProp_of_checks`: check preconditions → arithmeticBundleProp
- `arithmeticBundleProp_props_imply_check_subset`: arithmeticBundleProp → splitRoundTrip = true ∧ matrixTransformIdentity = true ∧ …

## Paper Anchors

Source: ./formal/superneo-lean/SuperNeo.pdf.md

- Section 4 (Preliminaries): decomposition, matrix transform, evaluation homomorphism.
- Section 5 (Embedding products): Theorem 4 (Mz = ct(M̄z)), Theorem 5 (Evaluation homomorphism).
- Section 7 (Folding scheme): arithmetic obligations composed for protocol context, lines 449–596.

## Module Mapping

| Lean file | Paper section |
|---|---|
| `SuperNeo/ArithmeticBundle.lean` | Sections 4–5, 7 (composition) |

## Contract Surface

| Group | Lean symbol | Kind | Role | Guarantee |
|---|---|---|---|---|
| Props | `arithmeticEvalHomProp` | def | Definitional | evalHom2Prop |
| Props | `arithmeticVecModuleProp` | def | Definitional | add/scale linearity for VecModuleHom |
| Props | `arithmeticScalarModuleProp` | def | Definitional | add/scale linearity for ScalarModuleHom |
| Props | `arithmeticSamplingProp` | def | Definitional | samplingExpansionProp |
| Props | `arithmeticPolyProp` | def | Definitional | eqLiftAllBoolean ∧ setSize ≠ 0 ∧ totalDegree ≤ setSize |
| Props | `arithmeticDecompProp` | def | Definitional | splitBalancedRoundTripProp |
| Props | `arithmeticInterpProp` | def | Definitional | interpolationProp |
| Bundle | `arithmeticBundleProp` | def | Definitional | Conjunction of all above |
| Theorems | `arithmeticDecompProp_iff_splitRoundTrip_true` | theorem | Theorem-Target | Prop ↔ check |
| Theorems | `arithmeticBundleProp_of_props` | theorem | Theorem-Target | Component props → bundle |
| Theorems | `arithmeticBundleProp_of_theorem_stack` | theorem | Theorem-Target | P10 + thm3 + module assumptions → bundle |
| Theorems | `arithmeticBundleProp_checks_imply_props` | theorem | Theorem-Target | Checks → bundle |
| Theorems | `arithmeticBundleProp_props_imply_check_subset` | theorem | Theorem-Target | Bundle → check subset |
| Theorems | `arithmeticBundleProp_props_imply_module_checks` | theorem | Theorem-Target | Bundle + size guard → module checks |
| Theorems | `arithmeticBundleProp_of_checks` | theorem | Theorem-Target | Checks → bundle (convenience) |

## Proof Obligations and Closure Plan

All obligations closed. Proposition/check bridges proved. Theorem-native constructor (`arithmeticBundleProp_of_theorem_stack`) derives from thm3CoreAssumption and module assumptions.

## Assumption Ledger

No open boundary assumptions in this module. The theorem-native path uses `thm3CoreAssumption`, `vecModuleAssumption`, and `scalarModuleAssumption` from upstream; those have closure plans in their respective modules.

## Dependency and Consumer Map

Upstream dependencies:
- `SuperNeo/MatrixTransform.lean`: uses `matrixVecDirect`, `matrixVecCtBar`, `MatrixRowsCompatible`, `matrixTransformIdentity`.
- `SuperNeo/EvalHom.lean`: uses `evalHom2Prop`, `evalHomAssumption`.
- `SuperNeo/ModuleHom.lean`: uses `VecModuleHom`, `ScalarModuleHom`, `vecModuleAssumption`, `scalarModuleAssumption`.
- `SuperNeo/InvertibilityAxioms.lean`: uses `invertibilityPreconditionsProp`.
- `SuperNeo/SamplingSet.lean`: uses `samplingExpansionProp`.
- `SuperNeo/PolyLemmas.lean`: uses `eqLiftAllBoolean`, `schwartzZippelBoundLeOne`.
- `SuperNeo/Decomp.lean`: uses `splitRoundTrip`, `splitBalancedRoundTripProp`.
- `SuperNeo/Interp.lean`: uses `interpolationProp`, `interpolationCase`.

Downstream consumers:
- `SuperNeo/ProtocolTarget.lean`: depends on arithmetic bundle for protocol-target context.
- `SuperNeo/ArithmeticObligations.lean`: uses arithmetic bundle for obligation composition.

## Implementation Plan

Current scope complete. All constructors and bridges proved. Theorem stack path threads P10 through matrix transform and eval hom.

## Quality Expectations

Bundle must be a true conjunction of all paper obligations. Check/prop bridges must be sound and complete for their respective subsets.

## Acceptance Criteria

- `lake build` succeeds.
- No `sorry`.
- All bridge theorems proved.

## Out of Scope

- Concrete parameter instantiation (Goldilocks, etc.).
- Proof of individual upstream assumptions (Thm3, bar-lift, module hom).
