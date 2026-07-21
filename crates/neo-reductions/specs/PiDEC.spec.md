# PiDEC

## Purpose

- **What it is**: Decomposition reduction step Pi_DEC that splits a parent CE claim into `k` child CE claims via balanced base-b digit decomposition, reducing witness norms for the next folding iteration.
- **Key invariant**: The paper verifier computes `split_b(parent.X, k, b)`, requires exact equality with the ordered child `X` family, and checks radix recomposition of only the public commitments and evaluation tuples. Each child witness digit matrix `Z_i` has entries in the balanced range `[-(b-1), +(b-1)]`. Cached constant terms, advice, delayed points, digests, padding, and `y_zcol` are implementation/lifecycle sidecars and require separate derivation or ownership proofs.
- **Protocol role**: Third (final) step in the folding composition `Pi_DEC o Pi_RLC o Pi_CCS` (Theorem 1, Theorem 7). Produces `k` child claims with bounded norms that become the ME inputs for the next folding step.

## Target Formulas (Paper -> Rust)

| Paper notation | Paper reference | Rust identifier | Notes |
|---|---|---|---|
| `Pi_DEC` | Section 7.5, line 585 | `api::dec_children_with_commit`, `api::verify_dec_public` | Decomposition reduction of knowledge |
| `Z = Sigma b^i * Z_i` | Section 7.5, line 587 | `split_b_matrix_k(Z, k, b)` | Balanced base-b decomposition |
| `||Z_i||_inf < b` | Section 7.5, line 588 | Signed digit range guarantee | Norm reduction |
| `X_child_i = split_b(X_parent)_i` | Section 7.5, line 589 | `verify_dec_public` computes and checks this | Verifier-determined public input decomposition |
| `y_parent_j = Sigma b^i * y_child_{i,j}` | Section 7.5, line 590 | `verify_dec_public` checks this | Ring-digit output decomposition |
| `c_parent = Sigma b^i * c_child_i` (S-action) | Section 7.5 | `combine_b_pows` closure | Commitment decomposition via S-action |
| `b` decomposition base | Appendix B.2 | `params.b` | Concrete parameter from NeoParams |
| `k` decomposition depth | Appendix B.2 | `params.k_rho` | Exact number of ordered digit layers |

## Paper Anchors

Source: ./formal/superneo-lean/SuperNeo.pdf.md

- Section 7.5 (Pi_DEC decomposition reduction), lines 585-593.
- Theorem 7 (Pi_DEC is a reduction of knowledge), lines 594-596.
- Theorem 1 composition, lines 438-447: Pi_DEC is the final step.
- Section 4 (balanced base-b decomposition `split_b`), line 294.

## Lean Cross-Reference

| Lean spec | Lean module | Relationship |
|---|---|---|
| `PiDEC.spec.md` | `SuperNeo/PiDEC.lean` | `piDECKnowledgeStatement` = exists `deltaInv`, invertibility + `ceRelaxedRelation` + `SumCheckClaimTrue` |
| `ProofSystem/Folding/PiDEC.spec.md` | `SuperNeo/ProofSystem/Folding/PiDEC.lean` | Proof-system wrapper; `final_of_assumption` |
| `Decomp.spec.md` | `SuperNeo/Decomp.lean` | `split_b` round-trip and digit bound formalization |

## Contract Surface

| Group | Rust symbol | Kind | Role | Guarantee |
|---|---|---|---|---|
| DEC (prover) | `api::dec_children_with_commit(mode, s, params, parent, Z_split, ell_d, child_commitments, combine_b_pows)` | fn | Core | Returns `(children, ok_y, ok_X, ok_c)` |
| DEC (cached) | `api::dec_children_with_commit_cached(...)` | fn | Core | Same as above but reuses caller-provided `SparseCache` |
| DEC (verifier) | `api::verify_dec_public(s, params, parent, children, combine_b_pows, ell_d)` | fn | Core | Requires `params.k_rho` children and returns `true` iff their public `X` equals the verifier-computed split and all remaining decompositions are valid |
| Splitting | `split_b_matrix_k(Z, k, b)` | fn | Core | Returns `Vec<Mat<F>>` of `k` digit matrices |
| Splitting | `split_b_matrix_k_with_nonzero_flags(Z, k, b)` | fn | Core | Also returns per-digit nonzero flags |

## Invariant Obligations

| Invariant | Verification method | Lean theorem counterpart |
|---|---|---|
| DEC round-trip: `Sigma b^i * Z_i = Z` entry-wise | Unit test | `splitBalancedRoundTripProp_holds` |
| Digit bound: all entries of `Z_i` lie in `[-(b-1), +(b-1)]` | Unit test | `splitBase2DigitsWithinBound` |
| Canonical X decomposition: `X_child_i = split_b(X_parent)_i` | Unit test | `PiDEC.PaperVerifier.OutputAccepted.childPublicInput_eq` |
| Parent X representability in exactly `params.k_rho` digits | Unit test | Rust verifier boundary; the typed Lean split is total and uses an exact fallback outside the `CE(B)` norm domain |
| y decomposition: `y_parent_j = Sigma b^i * y_child_{i,j}` | Unit test | `PiDEC.PaperVerifier.OutputAccepted.toRecompositionAccepted` |
| ct consistency with the semantic evaluation carrier | Implementation-sidecar test | `PiDecStrictProductionCompiler.Accepted.legacy.ct`; paper ownership is derived/open |
| Commitment decomposition: `c_parent = combine_b_pows(children, b)` | Unit test | `PiDecStrictProductionCompiler.PaperBridge.commitmentEquation` |
| Optional advice recomposition | Implementation-sidecar test | Excluded from the active exact-paper/no-advice profile |
| Overflow rejection: entries exceeding `b^k` range produce error | Unit test | (none) |
| Prover-verifier agreement: `dec_children_with_commit` output passes `verify_dec_public` | Unit test | (none) |
| Delayed NC channel | Split-NC/lifecycle tests | `s_col` and `y_zcol` are deliberately outside paper PiDEC; `y_zcol` is checked through the delayed block-lane boundary |

## Assumption Ledger

| Assumption | Source | Justification |
|---|---|---|
| Witness values representable in `k` balanced base-b digits | RLC norm bound | `||Z||_inf < b^k / 2` ensured by RLC sampling |
| `combine_b_pows` correctly implements `Sigma b^i * c_i` via S-action | Caller (neo-fold) | Uses `neo_ajtai::b_power_combine` |
| Parent claim `ct` is consistent with `y_ring` | Validated at boundary | `validate_ct_constant_term` |

## Dependency and Consumer Map

Upstream dependencies:
- `neo-math`: `F`, `K`, `D`, `balanced::to_balanced_i128`
- `neo-ajtai`: `Commitment` type
- `neo-ccs`: `CcsStructure`, `CeClaim`, `Mat`
- `neo-params`: `NeoParams` (b, k)

Downstream consumers:
- `neo-fold::shard`: calls DEC after RLC to produce child claims for next folding step
- `neo-fold::shard::verifier`: calls `verify_dec_public` to verify decomposition

## Lean Oracle Conformance

| Test file | Oracle family | What it checks |
|---|---|---|
| `dec_public_paper_exact_x.rs` | Independent paper boundary | Exact child arity, verifier-computed public split, overflow rejection, and nonbinary canonical examples |
| `f_prime_pi_dec_source_lean_artifact` | Bounded Rust/Lean source artifact | Exact `kappa = 4` strict source rows refine the independently defined reduced compiler; this does not cover final selective rows |

## Quality Expectations

- `split_b_matrix_k` returns `Err` on overflow (not panic)
- `verify_dec_public` logs specific failure point (X, y, ct, or c mismatch)
- `s_col` consistency is checked as an implementation sidecar; delayed
  `y_zcol` authority is owned by the separate block-lane lifecycle boundary

## Acceptance Criteria

- `cargo test -p neo-reductions --release` succeeds
- Spec-derived tests in `spec-tests/pi_dec.rs` pass
- Existing `rlc_dec_k_gt1`, `dec_reduction_y_zcol`, `dec_public_extra_openings` tests pass
- `cargo clippy -p neo-reductions --all-targets --release -- -D warnings` clean

## Out of Scope

- RLC step -- see PiRLC.spec.md
- Commitment scheme S-action implementation -- see neo-ajtai specs
- Shard-level orchestration of RLC -> DEC -> next Pi_CCS -- belongs to neo-fold
