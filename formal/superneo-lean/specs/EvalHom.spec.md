# EvalHom Spec

## Purpose

- **What it is**: The evaluation-hom layer formalizes Theorem 5 (Evaluation Homomorphism). It defines `evalBarMzAt` as multilinear evaluation (`mleInnerProductForm`) over the `y`-space, `evalHom2Prop` as the actual linearity equation, `evalHom2` as executable Theorem 5 check, and theorem/check-facing boundaries `evalHomAssumption` and `evalHomCheckAssumption`.
- **Key property**: Under compatible preconditions, evaluation preserves linear combinations:
  `eval(ρ₁·y₁ + ρ₂·y₂) = ρ₁·eval(y₁) + ρ₂·eval(y₂)`,
  instantiated with `y₁ := matrixVecCtBar bar m z₁`, `y₂ := matrixVecCtBar bar m z₂`.
- **Protocol role**: ArithmeticBundle checks `evalHomAssumption`. ProtocolMathTarget depends on closed eval-hom for the folding protocol.

## Target Formulas (Paper → Lean)

- Paper formula: Theorem 5 (Evaluation Homomorphism) — evaluation at challenge point `r` is an R_F-module homomorphism.
- Lean mapping:
  - `evalBarMzAt bar m z r` : `mleInnerProductForm z r`
  - `evalHom2Prop bar m z1 z2 r ρ1 ρ2` :
    `z1.size = z2.size ∧ MatrixRowsCompatible m z1 ∧`
    `evalBarMzAt bar m (linComb2Vec ρ1 ρ2 (matrixVecCtBar bar m z1) (matrixVecCtBar bar m z2)) r`
    `= ρ1 * evalBarMzAt bar m (matrixVecCtBar bar m z1) r +`
      `ρ2 * evalBarMzAt bar m (matrixVecCtBar bar m z2) r`
  - `evalHom2 bar m z1 z2 r ρ1 ρ2` : executable check
  - `evalHomAssumption bar m r ρ1 ρ2` : `∀ z1 z2, z1.size = z2.size → MatrixRowsCompatible m z1 → evalHom2Prop bar m z1 z2 r ρ1 ρ2`
  - `evalHomCheckAssumption bar m r ρ1 ρ2` : `∀ z1 z2, z1.size = z2.size → MatrixRowsCompatible m z1 → evalHom2 bar m z1 z2 r ρ1 ρ2 = true`
- Target statement: `evalHom2 bar m z1 z2 r ρ1 ρ2 = true ↔ evalHom2Prop bar m z1 z2 r ρ1 ρ2` (sound/complete/iff_prop proved).

## Paper Anchors

- Source: `./formal/superneo-lean/SuperNeo.pdf.md`
- Anchors:
  - Theorem 5 (Evaluation Homomorphism), Section 5, lines 390-400

## Module Mapping

- Implementation: `SuperNeo.EvalHom`
- Interface: `SuperNeo.EvalHomInterface`

## Contract Surface

| Contract group | Lean surface | Preconditions | Guarantee | Role | Used by |
|---|---|---|---|---|---|
| Evaluator | `evalBarMzAt` | None | Multilinear evaluation (`mleInnerProductForm`) on input `z` at challenge `r` | Theorem-Target | `ArithmeticBundle.lean` |
| Eval-hom prop | `evalHom2Prop`, `evalHom2` | `z1.size = z2.size`, row compatibility | Theorem-5 linearity equation over `matrixVecCtBar` images | Theorem-Target | `ArithmeticBundle.lean` |
| Theorem-facing boundary | `evalHomAssumption bar m r ρ1 ρ2` | None | `∀ z1 z2, ... → evalHom2Prop ...` | Theorem-Target | `ProtocolMathTarget` |
| Check-facing boundary | `evalHomCheckAssumption bar m r ρ1 ρ2` | None | `∀ z1 z2, ... → evalHom2 = true` | Theorem-Target | — |
| Constructive closure | `evalHomAssumption_constructive` | None | `evalHomAssumption bar m r ρ1 ρ2` (derived from MLE linearity; theorem-native) | Theorem-Target | — |
| Compatibility alias | `evalHomAssumption_native` | None | Backward-compatible alias to constructive closure | Theorem-Target | — |
| Sound/complete | `evalHom2_sound`, `evalHom2_complete`, `evalHom2_iff_prop` | Check true / Prop holds | Bidirectional bridge | Theorem-Target | — |
| From eval-link + module | `evalHomAssumption_of_evalLink_and_moduleAssumptions` | `evalLinkAssumption`, `vecModuleAssumption`, `scalarModuleAssumption` | `evalHomAssumption` | Theorem-Target | — |
| From Thm3 + module | `evalHomAssumption_of_thm3_and_moduleAssumptions` | `thm3CoreAssumption`, module assumptions | `evalHomAssumption` | Theorem-Target | — |
| From P10 + module | `evalHomAssumption_of_p10_and_moduleAssumptions` | `thm3CoreAssumption`, module assumptions | `evalHomAssumption` | Theorem-Target | — |
| From P10+P11 + module | `evalHomAssumption_of_p10_p11_and_moduleAssumptions` | `thm3CoreAssumption`, `barLiftLinearityAssumption`, module assumptions | `evalHomAssumption` (compatibility path) | Theorem-Target | — |

## Proof Obligations and Closure Plan

All obligations closed. `evalHomAssumption_constructive` proves the Theorem-5 linearity equation via:
1. `mleInnerProductLinearityAssumption_holds`,
2. helper lemmas linking `linComb2Vec` to `MLE.linComb`,
3. derived scaling lemma for `mleInnerProductForm`.
Constructors from eval-link, Thm3, P10, and P10+P11 chain through `evalHomAssumption_of_evalLink_and_moduleAssumptions`.

## Assumption Ledger

No open boundary assumptions in this module.

## Dependency and Consumer Map

- Upstream dependencies:
  - `SuperNeo/EvalLink.lean`: imports `evalLinkAssumption` for eval-hom constructor.
  - `SuperNeo/ModuleHom.lean`: imports `vecModuleAssumption`, `scalarModuleAssumption` for eval-hom constructor.
- Downstream consumers:
  - `SuperNeo/ArithmeticBundle.lean`: uses `evalHomAssumption` for checks.
  - `SuperNeo/ProtocolMathTarget.lean`: depends on closed eval-hom for the folding protocol.

## Implementation Plan

1. Define `evalBarMzAt` as `mleInnerProductForm`.
2. Define `evalHom2Prop` / `evalHom2` with explicit Theorem-5 linearity equation.
3. Prove helper lemmas:
   - zero/scale behavior of `mleInnerProductForm`,
   - `linComb2Vec`-to-`linComb` bridge.
4. Prove `evalHom2_sound`, `evalHom2_complete`, `evalHom2_iff_prop`.
5. Prove `evalHomAssumption_constructive`, then bridge constructors (`eval-link`, `P10`, `P11`).

## Quality Expectations

- No `sorry` in any theorem.
- Eval-hom is the single semantic surface for Theorem 5; check/prop duality is explicit.

## Acceptance Criteria

1. `lake build` succeeds.
2. `lake exe check` succeeds.
3. All eval-hom theorems exported through the interface.

## Out of Scope

- Protocol-level composition (belongs to ProtocolMathTarget).
