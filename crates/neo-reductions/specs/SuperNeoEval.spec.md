# SuperNeoEval

## Purpose

- **What it is**: SuperNeo transformed-matrix evaluators that compute multilinear extension dot products using bar-transformed matrices and ring constant-term products, enabling Theorem 4's matrix-vector product identity.
- **Key invariant**: `superneo_row_dot_transformed_matrix(bar(M), row, z) = sum_t ct(bar(a_t) * z_t)` over D-coefficient blocks, which equals `(M * z)[row]` when `z` satisfies `MatrixRowsCompatible` (Theorem 4).
- **Protocol role**: Provides the evaluation kernel used by `OptimizedOracle` to compute Q polynomial evaluations via the SuperNeo transform instead of direct matrix-vector products. Enables incremental adoption without protocol rewiring.

## Target Formulas (Paper -> Rust)

| Paper notation | Paper reference | Rust identifier | Notes |
|---|---|---|---|
| `ct(bar(a) * z)` | Theorem 3, line 375 | `ct(&a_ring.mul(&Rq(z_re)))` | Core ring product + constant-term extraction |
| `(M z)[i] = sum_t ct(cf_inv(bar(m_{i,t})) * cf_inv(z_t))` | Theorem 4, line 384 | `superneo_row_dot_transformed_matrix(mat_bar, row, z)` | Row dot product via transformed matrix |
| `bar(M)` | Def 8 (Lifting) | `CcsStructure::transform_matrices_superneo()` (in neo-ccs) | Pre-computed bar transform of CCS matrices |
| `SuperneoMatrixCache` | (internal) | `SuperneoMatrixCache` | Cached sparse block structure per matrix |
| `SuperneoLinearForm` | (internal) | `SuperneoLinearForm` | Sparse (col, value) form for a single row's evaluation |
| `eval_vec_k(z)` | (internal) | `SuperneoLinearForm::eval_vec_k(z)` | Evaluate linear form against witness vector |
| `eval_packed_digits_k(z)` | (internal) | `SuperneoLinearForm::eval_packed_digits_k(z)` | Evaluate returning all D ring digits |

## Paper Anchors

Source: ./docs/superneo-paper/

- Theorem 3 (Inner Product Transform), Section 5, lines 372-378: `ct(bar(a) * b) = <a, b>`.
- Theorem 4 (Matrix-Vector Product Transform), Section 5, lines 384-386: `(Mz)[i] = sum_t ct(cf_inv(bar(m_t)) * cf_inv(z_t))`.
- Theorem 5 (Evaluation Homomorphism), Section 5, lines 390-400: linearity of evaluation under ring operations.
- Remark 2 (Matrix-vector Product Evaluation), Section 5, lines 388-389.
- Definition 8 (Lifting / bar transform), Section 5, line 370.

## Lean Cross-Reference

| Lean spec | Lean module | Relationship |
|---|---|---|
| `MatrixTransform.spec.md` | `SuperNeo/MatrixTransform.lean` | `matrixVecDirect` vs `matrixVecCtBar`: Theorem 4 identity |
| `EvalHom.spec.md` | `SuperNeo/EvalHom.lean` | `evalHomAssumption`: Theorem 5 linearity |
| `EvalLink.spec.md` | `SuperNeo/EvalLink.lean` | `evalLinkIdentity`: Remark 2 evaluation identity |
| `Thm3Core.spec.md` | `SuperNeo/Thm3Core.lean` | `thm3CoreAssumption`: inner-product transform |
| `BarLift.spec.md` | `SuperNeo/BarLift.lean` | `bar` transform definition |

## Contract Surface

| Group | Rust symbol | Kind | Role | Guarantee |
|---|---|---|---|---|
| Row eval | `superneo_row_dot_transformed_matrix(mat_bar, row, z)` | fn | Core | Returns `sum_t ct(bar(a_t) * z_t)` for row `row` |
| Cache | `SuperneoMatrixCache` | struct | Core | Pre-built sparse block structure for a matrix |
| Cache | `SuperneoMatrixCache::build(mat_bar, s_m)` | fn | Core | Build cache from bar-transformed matrix |
| Linear form | `SuperneoLinearForm` | struct | Core | Sparse representation of one row's evaluation |
| Linear form | `SuperneoLinearForm::eval_vec_k(z)` | fn | Core | Evaluate against full witness vector |
| Linear form | `SuperneoLinearForm::eval_packed_digits_k(z)` | fn | Core | Evaluate returning `[K; D]` ring digits |
| Block | `RowBlock` | struct | Helper | Per-block `(blk_idx, bar_coeffs, orig_coeffs)` |

## Invariant Obligations

| Invariant | Verification method | Lean theorem counterpart |
|---|---|---|
| `superneo_row_dot` matches direct matrix-vector product for compatible inputs | Unit test | `matrixTransformIdentity` (Theorem 4) |
| `SuperneoMatrixCache` produces same results as direct `superneo_row_dot` | Unit test | (none) |
| `SuperneoLinearForm::eval_vec_k` matches `superneo_row_dot` | Unit test | (none) |
| `eval_packed_digits_k` returns all D digits correctly | Unit test | (none) |
| Block structure handles zero rows (sparse) correctly | Unit test | (none) |
| Out-of-bounds row returns `K::ZERO` | Unit test | (none) |

## Assumption Ledger

| Assumption | Source | Justification |
|---|---|---|
| Input matrix `mat_bar` is the bar-transform of the original matrix | Caller (neo-ccs `transform_matrices_superneo`) | Transformation applied at structure creation |
| Witness `z` satisfies `MatrixRowsCompatible` for Theorem 4 to hold | Protocol guarantee | Packed SuperNeo witness layout |
| Ring multiplication `Rq::mul` is correct | neo-math | Verified by Ring.spec.md |

## Dependency and Consumer Map

Upstream dependencies:
- `neo-math`: `Rq`, `ct`, `superneo_bar_block`, `D`, `F`, `K`, `KExtensions`
- `neo-ccs`: `CcsMatrix`, `CcsStructure`

Downstream consumers:
- `neo-reductions::engines::optimized_engine::oracle`: uses `SuperneoMatrixCache` in `OptimizedOracle`
- `neo-reductions::engines::optimized_engine::verify`: uses cached evaluation for terminal identity

## Lean Oracle Conformance

| Test file | Oracle family | What it checks |
|---|---|---|
| `crates/neo-ccs/spec-tests/lean_oracles.rs` | `matrix_eval_v1` | bar-transform ct, eval-link, eval-hom (covers the same math) |

## Quality Expectations

- `superneo_row_dot_transformed_matrix` handles partial blocks (when `z.len()` is not a multiple of D)
- `SuperneoMatrixCache` skips all-zero blocks for performance
- No `unsafe` code

## Acceptance Criteria

- `cargo test -p neo-reductions --release` succeeds
- Spec-derived tests in `spec-tests/superneo_eval.rs` pass
- Existing `superneo_eval_equivalence` and `superneo_eval_perf` tests pass
- `cargo clippy -p neo-reductions --all-targets --release -- -D warnings` clean

## Out of Scope

- Bar transform implementation (in neo-math)
- CCS structure transformation (in neo-ccs)
- Protocol-level Q polynomial (see PiCCS.spec.md / Engines.spec.md)
