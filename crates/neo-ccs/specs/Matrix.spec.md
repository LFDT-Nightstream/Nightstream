# Matrix

## Purpose
- **What it is**: Dense row-major matrix `Mat<T>`, compressed sparse row `CsrMatrix`, compressed sparse column `CscMat<F>`, the `CcsMatrix<F>` enum (Identity sentinel | CSC), and per-matrix sparse cache `SparseCache` -- the matrix representations backing CCS structures.
- **Key invariant**: `Mat::from_row_major(n, m, data)` stores `data[i*m + j]` as entry `(i,j)`; `CscMat::from_dense_row_major(M)` produces the same matrix-vector products as the dense form; the identity sentinel `CcsMatrix::Identity{n}` acts as `I_n` without storing `n` diagonal entries.
- **Protocol role**: CCS matrices `{M_j}` are stored in these types. The folding and reduction engines use `add_mul_into` / `add_mul_transpose_into` for efficient sparse `M*x` and `M^T*x` operations. The identity hint provides fast-path optimization for `M_0 = I_n`.

## Target Formulas (Paper -> Rust)

| Paper notation | Paper reference | Rust identifier | Notes |
|---|---|---|---|
| `M_j in F^{n x m}` | Def 11, line 449 | `Mat<F>` (dense) or `CcsMatrix<F>` (sparse/identity) | CCS matrices |
| `M * z` | Def 12, line 457 | `add_mul_into(x, y, n_eff)` | `y += M * x` |
| `M^T * r` | Def 13, line 461 | `add_mul_transpose_into(x, y, n_eff)` | `y += M^T * x` |
| `I_n` (identity) | (implicit) | `CcsMatrix::Identity{n}` | Sentinel without storage |

## Paper Anchors

Source: ./docs/superneo-paper/

- Definition 11 (CCS Structure), Section 7.1, lines 449-455: matrices `{M_j}` are part of the CCS structure.

## Lean Cross-Reference

| Lean spec | Lean module | Relationship |
|---|---|---|
| (none) | (none) | Matrix representation is an implementation detail not formalized in Lean |

## Contract Surface

| Group | Rust symbol | Kind | Role | Guarantee |
|---|---|---|---|---|
| Dense | `Mat<T>` | struct | Core | Row-major `data[i*cols + j]` |
| Dense | `Mat::from_row_major(rows, cols, data)` | fn | Core | Constructor |
| Dense | `Mat::zero(rows, cols, zero)` | fn | Helper | All-zero matrix |
| Dense | `Mat::identity(n)` | fn | Core | `I_n` with `identity_hint = true` |
| Dense | `Mat::is_identity()` | fn | Core | Check via hint or full validation |
| Dense | `Mat::is_identity_hint()` | fn | Helper | Returns hint flag only |
| Dense | `Mat::is_column_selector()` | fn | Helper | Each column has exactly one 1 |
| Dense | `Mat::rows()`, `Mat::cols()` | fn | Core | Dimension accessors |
| Dense | `Mat::as_slice()`, `Mat::as_mut_slice()` | fn | Core | Raw data access (mut clears identity_hint) |
| Dense | `Mat::row(i)`, `Mat::row_mut(i)` | fn | Core | Row slice access (mut clears identity_hint) |
| Dense | `Mat::set(row, col, val)` | fn | Core | Element set (clears identity_hint) |
| Dense | `Mat::append_zero_rows(k, zero)` | fn | Helper | Extend with zero rows (clears identity_hint) |
| CSR | `CsrMatrix` | struct | Core | True compressed sparse row (non-zeros only) |
| CSR | `CsrMatrix::from_dense(dense)` | fn | Core | Dense-to-CSR conversion |
| CSR | `CsrMatrix::spmv_transpose(r_pairs)` | fn | Core | O(nnz) sparse M^T*v multiply |
| CSR | `CsrMatrix::row_nz(row)`, `CsrMatrix::nnz()` | fn | Helper | Sparsity introspection |
| View | `MatRef<'a, T>` | struct | Helper | Borrowed row-major view |
| CSC | `CscMat<F>` | struct | Core | Compressed sparse column format |
| CSC | `CscMat::from_triplets(triplets, nrows, ncols)` | fn | Core | From (row, col, val); dedup by summing |
| CSC | `CscMat::from_dense_row_major(mat)` | fn | Core | Parallel dense-to-CSC conversion |
| CSC | `CscMat::add_mul_into(x, y, n_eff)` | fn | Core | `y += A * x` |
| CSC | `CscMat::add_mul_transpose_into(x, y, n_eff)` | fn | Core | `y += A^T * x` |
| Enum | `CcsMatrix<F>` | enum | Core | `Identity{n}` or `Csc(CscMat)` |
| Enum | `CcsMatrix::is_identity()` | fn | Core | True for identity sentinel |
| Enum | `CcsMatrix::rows()`, `CcsMatrix::cols()` | fn | Core | Dimension accessors |
| Cache | `SparseCache<F>` | struct | Helper | Per-matrix CSC cache (None = identity) |

## Invariant Obligations

| Invariant | Verification method | Lean theorem counterpart |
|---|---|---|
| Row-major layout: `M[i,j] == data[i*cols + j]` | Unit test | (none) |
| Identity construction and detection: `Mat::identity(n).is_identity() == true` | Unit test | (none) |
| Identity hint cleared on mutation: `set()`, `row_mut()`, `as_mut_slice()` clear hint | Unit test | (none) |
| CSC round-trip: dense -> CSC -> mul matches dense mul | Unit test | (none) |
| CSC from_triplets: duplicates summed, zeros skipped | Unit test | (none) |
| CSC transpose mul: `add_mul_transpose_into` matches `M^T * x` | Unit test | (none) |
| `CcsMatrix::Identity` sentinel acts as `I_n` | Unit test | (none) |
| `is_column_selector` correctly identifies selector matrices | Unit test | (none) |

## Assumption Ledger

| Assumption | Source | Justification |
|---|---|---|
| `identity_hint` is excluded from equality and serde | Design choice | Optimization-only flag; correctness verified by full check when hint absent |
| CSC format sorts by column | `from_triplets` impl | Standard CSC convention |

## Dependency and Consumer Map

Upstream dependencies:
- `neo-math`: `F` (Goldilocks) type alias
- `p3-field`: `PrimeCharacteristicRing` trait
- `p3-matrix`: `DenseMatrix` conversion

Downstream consumers:
- `neo-ccs::relations`: CCS structures store matrices as `Vec<CcsMatrix<F>>`
- `neo-ajtai`: uses `Mat<F>` for witness decomposition
- `neo-fold`: sparse matrix-vector products in folding verifier

## Lean Oracle Conformance

| Test file | Oracle family | What it checks |
|---|---|---|
| (none) | (none) | Matrix operations are implementation details; verified via unit tests |

## Quality Expectations

- No `unsafe` (enforced crate-wide)
- `identity_hint` is not serialized (serde skip) -- it is a fast-path optimization only
- All mutable accessors must clear `identity_hint` to maintain correctness
- `CscMat::from_triplets` must handle duplicate entries by summing (not replacing)

## Acceptance Criteria

- `cargo test -p neo-ccs --release` succeeds
- All matrix invariant tests pass
- Identity hint behavior is correctly validated

## Out of Scope

- GPU-accelerated matrix operations
- Blocked/tiled matrix formats
- Parallel sparse matrix construction optimizations beyond `from_dense_row_major`
