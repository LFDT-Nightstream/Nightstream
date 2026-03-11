# Checks — Sanity-Check Harness

## Purpose

- **What it is**: The executable deterministic cross-check harness used by `Main.lean`'s `lake exe check`. Contains `checkSuperCase`, `checkRingCase`, `checkNormCase`, `checkSplitCase`, `checkEqCase`, `checkMleCase`, `checkEmbeddingVecCase`, `checkEmbeddingMatrixCase`, `checkBarLiftVecCase`, `checkBarLiftMatrixCase`, `checkMatrixTransformCase`, `checkEvalLinkCase`, `checkEvalHomCase`, `checkSamplingCase`, `checkEqLiftCase`, `checkInterpCase`, and aggregators `checkSuperNeoCases`, `checkRingMulCases`, etc.
- **Key property**: For each module with golden vectors, `check*Cases = true` iff the module's executable semantics match the expected outputs from `SuperNeo.Generated.Vectors`. Formally: `∀ c ∈ superneoCases, checkSuperCase bar c → ct(mulRqPhi(superneoBarBlock bar a) b) = expectedCt ∧ dot a b = expectedDot`.
- **Protocol role**: Compile-time validation that all module sanity checks pass. Validates Definition 1 (Fields) and later definitions (coefficient maps, norm, decomposition, MLE, embedding, bar lift, matrix transform, eval link, eval hom) against golden vectors.

## Target Formulas

- `checkSuperNeoCases = true ↔ ∀ c ∈ superneoCases, ct(mulRqPhi(superneoBarBlock bar a) b) = expectedCt ∧ dot a b = expectedDot ∧ ct(...) = dot(...)`
- `checkRingMulCases = true ↔ ∀ c ∈ ringMulCases, mulRqPhi a b = expected`
- `checkNormCases = true ↔ ∀ c ∈ normCases, normInfCoeffs a = expectedNorm`
- `checkSplitCases = true ↔ ∀ c ∈ splitCases, splitBalancedVec input base k = expectedDigits ∧ recomposeSplitDigits gotDigits base = input`
- `checkMleCases = true ↔ ∀ c ∈ mleCases, mleByInnerProduct v r = mleByFoldingExec v r = expectedInner`
- `checkEmbeddingVecCases = true ↔ ∀ c ∈ embeddingVecCases, embedVec input = expectedBlocks ∧ unembedVec gotBlocks = input`
- `checkMatrixTransformCases = true ↔ ∀ c ∈ matrixTransformCases, matrixVecDirect m z = matrixVecCtBar bar m z` (up to expected values)
- `checkEvalLinkCases = true ↔ ∀ c ∈ evalLinkCases, matrixVecCtBar bar m z = expectedY ∧ ct gotY = expectedCtY`
- `checkEvalHomCases = true ↔ ∀ c ∈ evalHomCases, vecAdd (vecScale rho1 gotY1) (vecScale rho2 gotY2) = matrixVecCtBar bar m (linComb2Vec rho1 rho2 z1 z2)`

## Paper Anchors

- **Source**: `./formal/superneo-lean/SuperNeo.pdf.md`
- Definition 1 (Fields, Rings, and Dimensions), lines 275–282: sanity checks validate Definition 1 and later definitions.
- Definitions 2–8, 11–14: checks validate coefficient maps, norm, decomposition, MLE, embedding, bar lift, matrix transform, eval link, eval hom.

## Module Mapping

| Lean module | Paper section |
|-------------|----------------|
| `SuperNeo.Checks` | Infrastructure; no direct paper definition. Validates Definitions 1–14 via golden vectors. |

## Contract Surface

| Group | Lean symbol | Kind | Role | Guarantee |
|-------|-------------|------|--------|-----------|
| SuperNeo | `checkSuperNeoCases` | def | Definitional | `superneoCases.all (checkSuperCase bar)` |
| Ring | `checkRingMulCases` | def | Definitional | `ringMulCases.all checkRingCase` |
| Norm | `checkNormCases` | def | Definitional | `normCases.all checkNormCase` |
| Decomp | `checkSplitCases` | def | Definitional | `splitCases.all checkSplitCase` |
| EqPoly | `checkEqCases` | def | Definitional | `eqCases.all checkEqCase` |
| MLE | `checkMleCases` | def | Definitional | `mleCases.all checkMleCase` |
| Embedding | `checkEmbeddingVecCases`, `checkEmbeddingMatrixCases` | def | Definitional | Embedding round-trip and expected blocks |
| BarLift | `checkBarLiftVecCases`, `checkBarLiftMatrixCases` | def | Definitional | Bar lift linearity and expected outputs |
| MatrixTransform | `checkMatrixTransformCases` | def | Definitional | `matrixVecDirect m z = matrixVecCtBar bar m z` |
| EvalLink | `checkEvalLinkCases` | def | Definitional | Eval link identity and expected Y |
| EvalHom | `checkEvalHomCases` | def | Definitional | Eval homomorphism `ρ₁·Y₁ + ρ₂·Y₂ = bar·M·(ρ₁·z₁ + ρ₂·z₂)` |
| CoeffMaps | `checkCoeffMapCases` | def | Definitional | Round-trip sanity for coeff maps |
| Parameters | `checkParameterCases` | def | Definitional | `goldilocksShapeSanity ∧ Parameters.Goldilocks.sanity` |

## Proof Obligations and Closure Plan

All obligations closed. Each `check*Cases` is a `Bool`-valued executable; `lake exe check` runs them and asserts `true`. No theorem-level proof obligations in this module.

## Assumption Ledger

No open boundary assumptions in this module.

## Dependency and Consumer Map

- **Dependencies**: imports `Ring`, `CoeffMaps`, `Norm`, `Decomp`, `EqPoly`, `MLE`, `Embedding`, `BarLift`, `MatrixTransform`, `EvalLink`, `EvalHom`, `ModuleHom`, `InvertibilityAxioms`, `SamplingSet`, `PolyLemmas`, `Dimensions`, `Parameters`, `Interp`, `Generated.Vectors`.
- **Consumers**:
  - `Main.lean`: uses `checkSuperNeoCases`, `checkRingMulCases`, etc. via `lake exe check` to validate all module sanity.
  - `SuperNeo.ChecksInterface`: imports this module and re-exports curated `check*Cases` symbols.

## Implementation Plan

Keep harness minimal; add new `check*Cases` only when a new module with golden vectors is introduced. All checks must be deterministic and executable.

## Quality Expectations

Every `check*Cases` must pass on the current golden vectors. Golden vectors in `Generated.Vectors` must be consistent with paper definitions.

## Acceptance Criteria

- `lake exe check` succeeds with `all_checks = true`.
- Spec contains explicit paper anchors.
- Each check corresponds to a paper definition or module.

## Out of Scope

- Theorem-level proofs (checks are executable only).
- Non-deterministic or probabilistic checks.
