# Relations

## Purpose
- **What it is**: Core CCS/CE relation predicates (Definitions 11-14) implementing `f(M_1 z, ..., M_t z) = 0` (CCS row-wise check) and `y_j = Z M_j^T r^⊗` (CE evaluation consistency), plus R1CS embedding, tensor products, direct-sum composition, and the `SModuleHomomorphism` commitment trait.
- **Key invariant**: `check_ccs_rowwise_zero(s, x, w)` returns `Ok(())` iff `f((M_1 z)[i], ..., (M_t z)[i]) == 0` for all rows `i`; `check_ce_consistency` returns `Ok(())` iff all CE equalities hold (commitment binding, public input projection, ring-digit outputs, core-term alignment).
- **Protocol role**: Defines the statement/witness structure consumed by every reduction in the folding pipeline (Pi_CCS, Pi_RLC, Pi_DEC). The `CcsStructure` carries the constraint matrices and polynomial; claims/witnesses pair commitments with openings.

## Target Formulas (Paper -> Rust)

| Paper notation | Paper reference | Rust identifier | Notes |
|---|---|---|---|
| `s = ({M_j}, f, n, m)` | Def 11, line 449 | `CcsStructure<F>` | Structure: matrices + polynomial + dimensions |
| `f(M_1 z, ..., M_t z) = 0` (row-wise) | Def 12, line 457 | `check_ccs_rowwise_zero(s, x, w)` | CCS satisfiability check |
| `f(M z) = e * u` (relaxed) | Def 12 (relaxed) | `check_ccs_rowwise_relaxed(s, x, w, u, e)` | Relaxed CCS for folding |
| `c = L(Z)` | Def 12, line 458 | `check_ccs_claim_opening(l, inst, wit)` | Commitment binding |
| `y_j = Z M_j^T r^⊗` | Def 13, line 461 | `check_ce_consistency(params, s, l, inst, wit)` | CE evaluation relation |
| `(c, x)` CCS claim | Def 12 | `CcsClaim<C, F>` | Statement type |
| `w` CCS witness | Def 12 | `CcsWitness<F>` | Witness type |
| `(c, X, r, {y_j}, ct)` CE claim | Def 13 | `CeClaim<C, F, K>` | CE statement type |
| `Z` CE witness | Def 13 | `CeWitness<F>` | CE witness type |
| `r^⊗` tensor product | Def 6 (MLE) | `tensor_point(r)` | chi_r expansion, length `2^ℓ` |
| `A ∘ B = C` R1CS | Standard | `r1cs_to_ccs(a, b, c)` | R1CS-to-CCS embedding via `f = X_0·X_1 - X_2` |
| Block-diagonal CCS sum | (internal) | `direct_sum(ccs1, ccs2)` | Stacks constraints independently |
| Secure mixed CCS sum | (internal) | `direct_sum_transcript_mixed(ccs1, ccs2, digest)` | Cancellation-resistant: `f_1 + beta·f_2` |
| `L: F^{d×m} -> C` commitment | Def 4, Def 18 | `SModuleHomomorphism<F,C>` trait | Minimal commitment interface |

## Paper Anchors

Source: ./docs/superneo-paper/

- Definition 11 (CCS Structure), Section 7.1, lines 449-455: `s = ({M_j}_{j in [t]}, f, n, m)`.
- Definition 12 (Norm-bounded CCS relation), Section 7.1, lines 457-459: `c = L(z)`, `||z||_inf < b`, `f(M z) = 0`.
- Definition 13 (Norm-bounded CE evaluation relation), Section 7.1, lines 461-465: `y_j = Z M_j^T r^⊗`.
- Definition 14 (Global reduction parameters), Section 7.1, lines 467-475: `L`, `L_x`, challenge set, structure.
- Section 4.1 (CCS consistency equalities), lines 298-301.

## Lean Cross-Reference

| Lean spec | Lean module | Relationship |
|---|---|---|
| `ProofSystem/ConstraintSystem/CCS.spec.md` | `SuperNeo/ProofSystem/ConstraintSystem/CCS.lean` | Defines `CCSStructure`, `CCS.Holds`, `CE.Holds` matching Defs 11-13 |
| `ProtocolRelations.spec.md` | `SuperNeo/ProtocolRelations.lean` | Protocol-specific CCS/CE relation predicates; `ceRelation <-> ccsRelation` bridge |
| `MLE.spec.md` | `SuperNeo/MLE.lean` | `mle_tensor` corresponds to `tensor_point` |

## Contract Surface

| Group | Rust symbol | Kind | Role | Guarantee |
|---|---|---|---|---|
| Structure | `CcsStructure<F>` | struct | Core | Carries matrices `{M_j}`, polynomial `f`, dimensions `n x m` |
| Structure | `CcsStructure::new(matrices, f)` | fn | Core | Validates shapes, converts to CSC |
| Structure | `CcsStructure::new_sparse(matrices, f)` | fn | Core | From pre-built CcsMatrix enums |
| Structure | `CcsStructure::t()` | fn | Core | Number of matrices (arity of `f`) |
| Structure | `CcsStructure::max_degree()` | fn | Core | Max degree of CCS polynomial |
| Structure | `CcsStructure::ensure_identity_first()` | fn | Core | Insert I_n at M_0 if absent |
| Structure | `CcsStructure::ensure_identity_first_owned()` | fn | Core | Owned variant (avoids clone if already identity) |
| Structure | `CcsStructure::assert_m0_is_identity_for_nc()` | fn | Core | Strict M_0 = I_n validation for Ajtai/NC pipeline |
| Structure | `CcsStructure::transform_matrices_superneo()` | fn | Core | SuperNeo bar transform M -> bar(M) row-wise |
| Claims | `CcsClaim<C, F>` | struct | Core | CCS statement: `(c, x, m_in)` |
| Claims | `CcsWitness<F>` | struct | Core | CCS witness: `(w, Z)` |
| Claims | `CeClaim<C, F, K>` | struct | Core | CE statement: `(c, X, r, y_ring, ct, ...)` |
| Claims | `CeWitness<F>` | struct | Core | CE witness: `Z in F^{d x m}` |
| Checks | `check_ccs_rowwise_zero(s, x, w)` | fn | Core | Verify `f(Mz) = 0` row-wise |
| Checks | `check_ccs_rowwise_relaxed(s, x, w, u, e)` | fn | Core | Verify `f(Mz) = e*u` row-wise |
| Checks | `check_ccs_claim_opening(l, inst, wit)` | fn | Core | Verify `c = L(Z)` |
| Checks | `check_ce_consistency(params, s, l, inst, wit)` | fn | Core | Full CE membership check |
| Utils | `tensor_point(r)` | fn | Core | `chi_r = ⊗(r_i, 1-r_i)` |
| Utils | `validate_power_of_two(n)` | fn | Helper | `n != 0 && (n & (n-1)) == 0` |
| Utils | `mat_vec_mul_ff(m, n_rows, n_cols, v)` | fn | Helper | F-matrix * F-vector |
| Utils | `mat_vec_mul_fk(m, n_rows, n_cols, v)` | fn | Helper | F-matrix * K-vector (with F->K embedding) |
| Utils | `direct_sum(ccs1, ccs2)` | fn | Core | Block-diagonal CCS composition |
| Utils | `direct_sum_mixed(ccs1, ccs2, beta)` | fn | Core | Mixed CCS composition `f_1 + beta·f_2` |
| Utils | `direct_sum_transcript_mixed(ccs1, ccs2, digest)` | fn | Core | Cancellation-resistant: beta from transcript |
| R1CS | `r1cs_to_ccs(a, b, c)` | fn | Core | R1CS -> CCS via `f = X_0·X_1 - X_2` |
| Trait | `SModuleHomomorphism<F, C>` | trait | Core | Commitment interface: `commit`, `project_x` |
| Errors | `CcsError` | enum | Core | Dimension, length, power-of-two, row-fail, relation-fail |
| Errors | `RelationError` | enum | Core | Structure validation errors |
| Errors | `DimMismatch` | struct | Core | Expected vs got dimensions |

## Invariant Obligations

| Invariant | Verification method | Lean theorem counterpart |
|---|---|---|
| CCS rowwise zero: `f(M_j z) = 0` for valid `(x, w)` | Unit test | `CCS.Holds` |
| CCS rowwise relaxed: `f(M_j z) = e*u` for correct slack | Unit test | (none) |
| CCS claim opening: `c = L(Z)` round-trip | Unit test | (none -- requires neo-ajtai) |
| CE consistency: `y_j = Z M_j^T r^⊗` | Unit test | `CE.Holds` |
| Structure validation: mismatched matrix shapes rejected | Unit test | (none) |
| Structure validation: wrong polynomial arity rejected | Unit test | (none) |
| R1CS embedding: `r1cs_to_ccs` produces CCS satisfying `A*z . B*z = C*z` | Unit test | (none) |
| `tensor_point`: entries sum to 1 for any input | Unit test + `lean_oracles` (removed lane) | `mle_tensor` |
| `tensor_point`: Lean oracle conformance (`mle_tensor_v1.json`) | `lean_oracles` (removed lane) | Direct |
| `direct_sum`: block-diagonal preserves independent satisfaction | Unit test | (none) |
| `direct_sum_transcript_mixed`: beta != 0 and beta != 1 | Unit test | (none) |
| `ensure_identity_first`: inserts I_n when M_0 != I | Unit test | (none) |
| `assert_m0_is_identity_for_nc`: rejects non-identity M_0 | Unit test | (none) |
| `validate_power_of_two`: correct for powers and non-powers | Unit test | (none) |

## Assumption Ledger

| Assumption | Source | Justification |
|---|---|---|
| Matrices `M_j` fit in memory as CSC | Runtime | Large CCS structures may require specialized allocation |
| `SModuleHomomorphism` implementor (neo-ajtai) is correct | neo-ajtai spec | Verified by AjtaiCommit.spec.md |
| Goldilocks field arithmetic is correct | neo-math spec | Verified by Goldilocks.spec.md |
| `direct_sum_transcript_mixed` digest has sufficient entropy | Poseidon2 | Standard hash assumption |

## Dependency and Consumer Map

Upstream dependencies:
- `neo-math`: `Fq`, `Rq`, `D`, `ETA`, `cf`, `cf_inv`, `superneo_bar_block`
- `neo-params`: `NeoParams` (concrete protocol parameters)
- `p3-field`: `Field`, `PrimeCharacteristicRing` traits
- `p3-goldilocks`: Goldilocks field type

Downstream consumers:
- `neo-ajtai`: uses `Mat`, `SModuleHomomorphism`
- `neo-fold-clean`: uses `CcsStructure`, `CcsClaim`, `CcsWitness`, `CeClaim`
- `neo-reductions`: uses `CcsStructure`, `CeClaim` for Pi_CCS/Pi_RLC/Pi_DEC

## Lean Oracle Conformance

| Test file | Oracle family | What it checks |
|---|---|---|
| `spec-tests/lean_oracles.rs` | `mle_tensor_v1` | `tensor_point` matches Lean MLE tensor expansion |
| `spec-tests/lean_oracles.rs` | `matrix_eval_v1` | Matrix-vector product, bar-transform ct, eval-link, eval-hom linearity |

## Quality Expectations

- No `unsafe` (enforced crate-wide via `#![forbid(unsafe_code)]`)
- All public items must have doc comments (`#![deny(missing_docs)]`)
- `check_ccs_rowwise_zero` and `check_ccs_rowwise_relaxed` must not leak witness values through error messages
- `direct_sum_transcript_mixed` is the recommended production API (cancellation-resistant)

## Acceptance Criteria

- `cargo test -p neo-ccs --release` succeeds (runs both `tests/` and `spec-tests/`)
- All `lean_oracles` (removed lane) families (`mle_tensor_v1`, `matrix_eval_v1`) pass
- Spec-derived tests in `spec-tests/relations.rs` pass
- `cargo clippy -p neo-ccs --all-targets --release -- -D warnings` clean

## Out of Scope

- Sumcheck protocol (belongs to neo-reductions layer)
- Folding verifier/session packaging (belongs to neo-fold-clean)
- Specific circuit designs (application-layer concern)
- Witness generation from program traces (frontend/application concern)
