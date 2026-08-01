# Polynomial

## Purpose
- **What it is**: Sparse multivariate polynomial `SparsePoly<F>` in `t` indeterminates with `Term<F>` entries, used as the CCS constraint polynomial `f` in Definition 11.
- **Key invariant**: `eval(x)` computes `sum_k coeff_k * prod_j x_j^{exp_{k,j}}`; `arity()` equals the number of indeterminates `t`; `insert_var_at_front` and `append_zero_vars` preserve evaluation semantics.
- **Protocol role**: The polynomial `f` in `CcsStructure` determines how matrix outputs are combined -- e.g., R1CS uses `f(X_0, X_1, X_2) = X_0 * X_1 - X_2` (element-wise Hadamard constraint).

## Target Formulas (Paper -> Rust)

| Paper notation | Paper reference | Rust identifier | Notes |
|---|---|---|---|
| `f in F[X_1, ..., X_t]` | Def 11, line 449 | `SparsePoly<F>` | Sparse representation |
| Term `c * X_1^{e_1} * ... * X_t^{e_t}` | Def 11 | `Term<F>` | `coeff` + `exps: Vec<u32>` |
| `f(v_1, ..., v_t)` evaluation | Def 12 | `SparsePoly::eval(x)` | Row-wise in CCS check |
| Degree `deg(f)` | Def 11 | `SparsePoly::max_degree()` | Max sum of exponents |

## Paper Anchors

Source: ./docs/superneo-paper/

- Definition 11 (CCS Structure), Section 7.1, lines 449-455: polynomial `f` is part of the CCS structure.
- Note: `f` is the "encoded image family" in the Lean formalization.

## Lean Cross-Reference

| Lean spec | Lean module | Relationship |
|---|---|---|
| (none) | (none) | Polynomial representation is an implementation detail not formalized in Lean |

## Contract Surface

| Group | Rust symbol | Kind | Role | Guarantee |
|---|---|---|---|---|
| Type | `SparsePoly<F>` | struct | Core | Sparse polynomial in `t` variables |
| Type | `Term<F>` | struct | Core | Single term: `coeff * prod x_j^{exp_j}` |
| Constructor | `SparsePoly::new(t, terms)` | fn | Core | Create with given arity and terms |
| Query | `SparsePoly::arity()` | fn | Core | Number of indeterminates |
| Query | `SparsePoly::terms()` | fn | Core | Slice of terms |
| Query | `SparsePoly::max_degree()` | fn | Core | Highest sum of exponents in any term |
| Eval | `SparsePoly::eval(x)` | fn | Core | Evaluate at point `x in F^t` (panics if lengths mismatch) |
| Eval | `SparsePoly::eval_in_ext(x)` | fn | Core | Evaluate with inputs in extension field K |
| Transform | `SparsePoly::insert_var_at_front()` | fn | Helper | Prepend dummy variable at index 0 |
| Transform | `SparsePoly::append_zero_vars(extra)` | fn | Helper | Append dummy variables |

## Invariant Obligations

| Invariant | Verification method | Lean theorem counterpart |
|---|---|---|
| Evaluation: `f(x) = sum coeff_k * prod x_j^{exp_{k,j}}` | Unit test | (none) |
| Arity matching: `eval` panics on wrong-length input | Unit test (should_panic) | (none) |
| `insert_var_at_front`: eval with `x[0]=0` equals original eval | Unit test | (none) |
| `append_zero_vars`: eval with trailing zeros equals original eval | Unit test | (none) |
| `max_degree` returns highest exponent sum across terms | Unit test | (none) |

## Assumption Ledger

| Assumption | Source | Justification |
|---|---|---|
| Exponent vectors have length `t` (arity) | Constructor | Not validated at construction; caller responsibility |

## Dependency and Consumer Map

Upstream dependencies:
- `p3-field`: `Field` trait

Downstream consumers:
- `neo-ccs::relations`: `CcsStructure` stores `SparsePoly<F>` as its polynomial `f`
- `neo-ccs::r1cs`: constructs `f = X_0*X_1 - X_2` for R1CS embedding
- `neo-ccs::utils`: `direct_sum*` combines polynomials

## Lean Oracle Conformance

| Test file | Oracle family | What it checks |
|---|---|---|
| (none) | (none) | Polynomial evaluation is an implementation detail; verified via unit tests |

## Quality Expectations

- No `unsafe` (enforced crate-wide)
- `eval` must panic (not silently return wrong result) on arity mismatch
- Terms with zero coefficients are allowed but have no semantic effect

## Acceptance Criteria

- `cargo test -p neo-ccs --release` succeeds
- All polynomial invariant tests pass

## Out of Scope

- Dense polynomial representation
- Polynomial factoring or GCD
- Symbolic simplification
