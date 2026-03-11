# EqPoly Spec

## Purpose
- **What it is**: The equality polynomial `eq(x,y) = Π_{i=1}^ℓ (x_i·y_i + (1-x_i)·(1-y_i))` is the Kronecker selector for the Boolean hypercube `{0,1}^ℓ`.
- **Key property**: For bit-vectors `x, y` of equal length: `eqPoly x y = if x = y then 1 else 0`.
- **Protocol role**: `MLE.lean` uses `eq` as the Lagrange basis for multilinear extensions (`ṽ(r) = Σ_{j∈{0,1}^ℓ} eq(r,j)·v_j`). In Π_CCS (Section 7.3, lines 481-548), `eq(X, α)` and `eq(X, r)` serve as point-selectors in the sum-check helper polynomials `Q` and `Eval`.
- **Scope**: Current formalization covers the base-field (`F` = Goldilocks) eq-polynomial. Extension-field (`K`) generalization is deferred until consumers require it directly.

## Target Formulas (Paper -> Lean)
- Paper formula(s):
  - `eq(x,y) = Π_{i=1}^ℓ (x_i·y_i + (1-x_i)·(1-y_i))`
  - `ṽ(r) = Σ_{j∈{0,1}^ℓ} eq(r, j)·v_j` (MLE via eq basis)
  - For `x, y ∈ {0,1}^ℓ`: `eq(x,y) = δ_{x,y}` (Kronecker delta)
- Lean mapping:
  - `eq(x,y)` -> `SuperNeo.eqPoly : Array F → Array F → F`
  - single-coordinate factor `x_i·y_i + (1-x_i)·(1-y_i)` -> `SuperNeo.eqTerm : F → F → F`
  - `x ∈ {0,1}` -> `SuperNeo.IsBit : F → Prop`
  - `x ∈ {0,1}^ℓ` -> `SuperNeo.IsBitVec : Array F → Prop`
  - Kronecker-delta theorem -> `SuperNeo.eqPoly_eq_delta_of_isBitVec`
- Target statement:
  - `eqPoly x y = if x = y then 1 else 0` when `x.size = y.size`, `IsBitVec x`, `IsBitVec y`

## Paper Anchors
- Source: `./formal/superneo-lean/SuperNeo.pdf.md`
- Anchors:
  - Definition of `eq`, Section 4 (Preliminaries, *Polynomials* paragraph), line 274
  - Lemma 6 (`eq(X,Z)·Q(X)` zero-sum characterization), line 737
  - Section 7.3 (Π_CCS helper polynomials `Q`, `Eval` using `eq`), lines 481-548

## Module Mapping
- Implementation: `SuperNeo.EqPoly`
- Interface: `SuperNeo.EqPolyInterface`

## Contract Surface
| Contract group | Lean surface (interface) | Preconditions | Guarantee | Role | Used by |
|---|---|---|---|---|---|
| Bit predicates | `IsBit`, `IsBitVec` | None | `IsBit x ↔ x = 0 ∨ x = 1`; `IsBitVec v ↔ ∀ i, IsBit v[i]` | Theorem-Target | `MLE.lean`, `PolyLemmas.lean` |
| Core eq definitions | `eqTerm`, `eqPoly`, `bitsToFArray` | `eqPoly` returns 0 when sizes differ | `eqTerm x y = x*y + (1-x)*(1-y)`; `eqPoly x y = Π_i eqTerm x[i] y[i]` | Theorem-Target | `MLE.lean`, `Checks.lean` |
| Size-mismatch theorem | `eqPoly_eq_zero_of_size_ne` | `x.size ≠ y.size` | `eqPoly x y = 0` | Theorem-Target | `MLE.lean` |
| Bit-level selector | `eqTerm_eq_delta_of_isBit` | `IsBit x`, `IsBit y` | `eqTerm x y = if x = y then 1 else 0` | Theorem-Target | internal (proof chain) |
| Boolean-cube selector | `eqPoly_eq_delta_of_isBitVec` | `x.size = y.size`, `IsBitVec x`, `IsBitVec y` | `eqPoly x y = if x = y then 1 else 0` | Theorem-Target | `MLE.lean`, `PolyLemmas.lean` |
| Package wrapper (closed) | `eqPolyAssumption`, `eqPolyAssumption_holds` | Same as selector | Bundled `Prop` carrier for downstream assumption-threading; discharged by `eqPolyAssumption_holds` | Theorem-Target | `MLE.lean`, `ArithmeticObligations.lean` |

## Proof Obligations and Closure Plan
All obligations closed. Every theorem in the contract surface is fully proved; no boundary assumptions remain.

## Assumption Ledger
No open boundary assumptions in this module.

## Dependency and Consumer Map
- Upstream dependencies:
  - `SuperNeo/Field.lean` (field type `F` and arithmetic)
- Downstream consumers:
  - `SuperNeo/MLE.lean`: uses `eqPoly`, `bitsToFArray`, `eqPolyAssumption` to define and close the MLE sum-form identity `mleEval f r = Σ_i f[i] * eqPoly(bits(i), r)`.
  - `SuperNeo/PolyLemmas.lean`: uses `eqPoly_eq_delta_of_isBitVec` for polynomial simplification lemmas over the Boolean cube.
  - `SuperNeo/ArithmeticObligations.lean`: uses `eqPolyAssumption_holds` to close the MLE delta bridge package.
  - `SuperNeo/Checks.lean`: uses `eqPoly`, `bitsToFArray` for executable sanity checks.

## Implementation Plan (How to Achieve)
1. `eqTerm_eq_delta_of_isBit` proved by case-splitting on `IsBit` disjunctions and `decide`.
2. `eqPoly_eq_delta_of_isBitVec` proved by induction on the foldl product: all-one case when `x = y` (every `eqTerm` is 1), exists-zero case when `x ≠ y` (some `eqTerm` is 0, product collapses).
3. `eqPolyAssumption_holds` trivially wraps the selector theorem for consumers expecting a bundled `Prop`.

## Quality Expectations
- Selector theorem must be stated with the exact Kronecker-delta formula, not prose.
- Contract rows separate definitions, structural lemmas, and the main selector theorem by mathematical role.
- `eqPolyAssumption` is documented as a closed compatibility wrapper, not an open boundary.

## Acceptance Criteria
1. `eqPoly_eq_delta_of_isBitVec` exported through the interface with full hypotheses.
2. `eqPolyAssumption_holds` closes the assumption package with no remaining axioms.
3. MLE consumers can derive delta behavior through closed EqPoly surfaces without axioms.
4. `lake build` succeeds.
5. `lake exe check` succeeds with `proof_import_wall=true` and `all_checks=true`.

## Out of Scope
- Extension-field (`K`) EqPoly theorem layer (no current consumer requires it).
- Lemma-6-style `eq·Q` zero-sum characterization (belongs to a future SumCheck or PolyLemmas module).

## Cleanup Notes
- Gradually replace consumer-side `eqPolyAssumption`-threaded usage with direct `eqPoly_eq_delta_of_isBitVec` calls where it improves clarity.
