# MLE

## Purpose

- **What it is**: Multilinear extension (MLE) evaluators for functions defined on the Boolean hypercube `{0,1}^ℓ`, providing three equivalent computation paths: guarded executable (`mleEval`), inner-product sum form (`mleInnerProductForm`), and iterative folding (`mleByFolding`), plus chi-weight and dot-product forms.
- **Key property**: `mleEval f r = mleInnerProductForm f r` when `f.size = 2^|r|` (identity theorem), `mleInnerProductForm f r = mleViaChiDot f r` (chi-dot equivalence), and `mleByInnerProduct = mleByFolding` (folding equivalence). MLE is linear in the table input: `mleEval(f + δ·g, r) = mleEval(f, r) + δ · mleEval(g, r)`.
- **Protocol role**: MLE evaluation is the workhorse of the sum-check protocol (Definition 6). The folding scheme (Section 7.3) reduces claims about MLE values via the sum-check-to-evaluation reduction. Linearity of MLE in the table argument enables the folding combination `z' = ρ₁·z₁ + ρ₂·z₂`.

## Target Formulas

- `ṽ(r) = Σ_{j∈{0,1}^ℓ} eq(r, j) · v_j` (Definition, line 273)
- `eq(x,y) = Π_i (x_i · y_i + (1-x_i)(1-y_i))` (eq polynomial)
- `χ_r(j) = Π_i (r_i · b_i + (1-r_i)(1-b_i))` where `b = bits(j)` (chi weight)
- `mleEval(f+δ·g, r) = mleEval(f, r) + δ · mleEval(g, r)` (linearity)

## Paper Anchors

Source: ./formal/superneo-lean/SuperNeo.pdf.md

- Section 4, line 273: `ṽ(X) = Σ_j eq(X,j) · v_j` — MLE definition.
- Definition 6, Section 4, lines 352-355: sum-check uses MLE claims.
- Section 7.3 (Π_CCS), lines 440-470: MLE evaluation in folding.

## Module Mapping

| Lean file | Paper section |
|---|---|
| `SuperNeo/MLE.lean` | MLE definition (Section 4), sum-check linkage (Definition 6) |

## Contract Surface

| Group | Lean symbol | Kind | Role | Guarantee |
|---|---|---|---|---|
| Predicates | `IsBit`, `IsBitVec` | def | Definitional | Boolean-cube membership |
| Encoding | `bitsToFieldArray` | def | Definitional | Index-mask to bit-vector |
| Evaluators | `mleEval` | def | Definitional | Guarded executable evaluator |
| Evaluators | `mleInnerProductForm` | def | Definitional | Unguarded sum form |
| Evaluators | `mleByInnerProduct` | def | Definitional | Inner-product route |
| Evaluators | `mleByFolding`, `mleByFoldingExec` | def | Definitional | Iterative folding route |
| Executable check | `mleIdentity` | def | Definitional | executable inner-vs-folding identity check |
| Chi weights | `chiWeight`, `chi` | def | Definitional | Basis-weight selector |
| Chi dot | `mleViaChiDot`, `dot` | def | Definitional | Chi/dot evaluation route |
| Table ops | `linComb` | def | Definitional | `f + δ·g` pointwise |
| Compatibility | `rHat` | def | Definitional | Compatibility vector |
| Identity | `mleEval_eq_innerProductForm_of_size` | theorem | Theorem-Target | Exec = sum when `|f| = 2^|r|` |
| Executable check | `mleIdentity_sound` | theorem | Theorem-Target | `mleIdentity = true → size guard ∧ inner = folding` |
| Executable check | `mleIdentity_complete` | theorem | Theorem-Target | proposition implies `mleIdentity = true` |
| Executable check | `mleIdentity_eq_true_iff` | theorem | Theorem-Target | Bool↔Prop closure for executable identity |
| Folding equiv | `mleByInnerProduct_eq_mleByFolding_of_size` | theorem | Theorem-Target | Inner = folding |
| Chi equiv | `mleInnerProductForm_eq_mleViaChiDot_of_size` | theorem | Theorem-Target | Sum = chi/dot |
| Linearity | `mleEval_linComb_of_assumptions` | theorem | Theorem-Target | Table-linear under packages |
| Size | `rHat_size`, `chi_size`, `linComb_size` | theorem | Theorem-Target | Output sizes |
| Packages | `mleIdentityAssumption` | def | Definitional | Identity package target |
| Packages | `mleIdentityAssumption_holds` | theorem | Theorem-Target | Package closed |
| Packages | `mleChiIdentityAssumption` | def | Definitional | Chi package target |
| Packages | `mleChiIdentityAssumption_holds` | theorem | Theorem-Target | Package closed |
| Packages | `mleInnerProductLinearityAssumption` | def | Definitional | Linearity package target |
| Packages | `mleInnerProductLinearityAssumption_holds` | theorem | Theorem-Target | Package closed |
| Packages | `mleEvalLinearityAssumption` | def | Definitional | Eval linearity package |
| Packages | `mleEvalLinearityAssumption_holds` | theorem | Theorem-Target | Package closed |
| Bridge | `eqPolyDeltaOnBitsAssumption` | def | Definitional | EqPoly delta on bits |
| Bridge | `eqPolyDeltaOnBitsAssumption_holds_of_eqPolyAssumption` | theorem | Theorem-Target | Closed from EqPoly |

## Proof Obligations and Closure Plan

All package targets are closed (all `*_holds` theorems proved). No open obligations.

## Assumption Ledger

No open boundary assumptions. The `eqPolyDeltaOnBitsAssumption` is bridged from `EqPoly.eqPolyAssumption` (already proved in `EqPoly.lean`).

## Dependency and Consumer Map

Upstream dependencies:
- `SuperNeo/EqPoly.lean`: `eqPoly`, `eqPolyAssumption` for MLE weights.

Downstream consumers:
- `SuperNeo/SumCheck.lean`: uses MLE evaluators for sum-check claims.
- `SuperNeo/EvalLink.lean`: uses MLE identity for evaluation linkage.
- `SuperNeo/ProtocolRelations.lean`: uses `mleEval` for CCS relation checks.

## Implementation Plan

No further work required; module is proof-complete for its scope.

## Quality Expectations

All three evaluation routes must agree under appropriate size guards. Linearity must be explicitly proved (not assumed). Package targets provide a stable API for downstream consumers without exposing proof internals.

## Acceptance Criteria

- `lake build` succeeds.
- No `sorry`.
- All 4 `*_holds` package closures proved.

## Out of Scope

- Multivariate polynomial ring formalization (MLE is treated as a function, not a ring element).
- Efficient NTT-based MLE evaluation.
- Concrete numeric MLE evaluations (sanity checks live in `Checks.lean`).
