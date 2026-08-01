# PiRLC

## Purpose

- **What it is**: Random Linear Combination (RLC) reduction step Pi_RLC that combines multiple CE claims into a single parent CE claim using rotation-matrix challenges `rho_i`, preserving the CE evaluation relation under S-module homomorphism.
- **Key invariant**: `parent = Sigma rho_i * input_i` component-wise (X, y, ct, commitment), and the combined witness `Z_mix = Sigma rho_i * Z_i` satisfies the CE relation for the parent claim.
- **Protocol role**: Second step in the folding composition `Pi_DEC o Pi_RLC o Pi_CCS` (Theorem 1). Takes multiple CE claims produced by Pi_CCS and collapses them into one, reducing the number of obligations before decomposition.

## Target Formulas (Paper -> Rust)

| Paper notation | Paper reference | Rust identifier | Notes |
|---|---|---|---|
| `Pi_RLC` | Section 7.4, line 549 | `api::rlc_with_commit`, `api::rlc_public` | Weak interactive reduction |
| `rho_i in S` (rotation matrices) | Section 7.4, line 551 | `RotRho` (typed), `rot_rhos_to_mats` | S-module challenge elements |
| `Z_mix = Sigma rho_i * Z_i` | Section 7.4, line 553 | `rlc_with_commit` return value (witness) | Combined witness |
| `X_out = Sigma rho_i * X_i` | Section 7.4, line 554 | `rlc_public` computes this | Public input combination |
| `y_out_j = Sigma rho_i * y_{i,j}` | Section 7.4, line 555 | `rlc_public` computes this | Ring-digit output combination |
| `c_out = Sigma rho_i * c_i` (S-action) | Section 7.4 | `mix_commits` closure | Commitment combination via S-module homomorphism |
| `RotRing` | (internal) | `RotRing` | Ring metadata for rotation matrix norm bound |

## Paper Anchors

Source: ./docs/superneo-paper/

- Section 7.4 (Pi_RLC weak interactive reduction), lines 549-583.
- Lemma 4 (Pi_RLC is a weak interactive reduction), lines 582-583.
- Theorem 1 composition, lines 438-447: Pi_RLC appears in the middle of the pipeline.

## Lean Cross-Reference

| Lean spec | Lean module | Relationship |
|---|---|---|
| `PiRLC.spec.md` | `SuperNeo/PiRLC.lean` | `piRLCWeakStatement` = `ceRelaxedRelation` AND `SumCheckClaimTrue` |
| `ProofSystem/Folding/PiRLC.spec.md` | `SuperNeo/ProofSystem/Folding/PiRLC.lean` | Proof-system wrapper; `soundness_relations` |
| `InteractiveReductions.spec.md` | `SuperNeo/InteractiveReductions.lean` | `weakCompositionStatement` uses Pi_RLC |

## Contract Surface

| Group | Rust symbol | Kind | Role | Guarantee |
|---|---|---|---|---|
| RLC (prover) | `api::rlc_with_commit(mode, s, params, rhos, me_inputs, Zs, ell_d, mix_commits)` | fn | Core | Returns `(parent_claim, Z_mix)` |
| RLC (verifier) | `api::rlc_public(s, params, rhos, inputs, mix_rhos_commits, ell_d)` | fn | Core | Recomputes parent without witnesses |
| Sampling | `sample_rot_rhos_n(ell, norm_sq_bound)` | fn | Core | Sample `ell` rotation matrices with squared-norm bound |
| Sampling | `sample_rot_rhos_n_typed(ring)` | fn | Core | Type-safe sampling with `RotRing` metadata |
| Types | `RotRho` | struct | Core | Typed, validated rotation-matrix challenge |
| Types | `RotRing` | struct | Core | Ring metadata for rotation sampling |
| Conversion | `rot_rhos_from_mats(rhos)` | fn | Helper | Convert `Mat<F>` to `RotRho` |
| Conversion | `rot_rhos_to_mats(rhos)` | fn | Helper | Convert `RotRho` to `Mat<F>` |

## Invariant Obligations

| Invariant | Verification method | Lean theorem counterpart |
|---|---|---|
| RLC linearity: `X_out = Sigma rho_i * X_i` | Unit test | `piRLCWeak_of_ce` |
| RLC linearity: `y_out_j = Sigma rho_i * y_{i,j}` | Unit test | `piRLCWeak_of_ce` |
| RLC linearity: `ct_out = ct(y_out)` consistency | Unit test | (none) |
| Prover-verifier agreement: `rlc_with_commit` and `rlc_public` produce same claim (sans witness) | Unit test | (none) |
| `RotRho` norm bound: sampled rotation matrices satisfy `||rho||^2 <= bound` | Unit test | (none) |
| NC channel mixing: `y_zcol_out = Sigma rho_i * y_zcol_i` when present | Unit test | (none) |
| aux_openings mixing: `aux_out = Sigma rho_i * aux_i` (scalar projection) | Unit test | (none) |
| Input validation: empty inputs rejected | Unit test | (none) |
| Input validation: `|rhos| != |inputs|` rejected | Unit test | (none) |

## Assumption Ledger

| Assumption | Source | Justification |
|---|---|---|
| Rotation-matrix norm bound prevents blowup | Paper Section 7.4 | `||rho||^2 <= norm_sq_bound` ensures MSIS security margin |
| S-module homomorphism is commitment-preserving | neo-ajtai | `L(rho * Z) = rho * L(Z)` by S-module structure |
| All ME inputs share the same `r` vector | Caller guarantee | Validated at API boundary |

## Dependency and Consumer Map

Upstream dependencies:
- `neo-math`: `F`, `K`, `D`, ring arithmetic
- `neo-ajtai`: `Commitment` type
- `neo-ccs`: `CcsStructure`, `CeClaim`, `Mat`
- `neo-params`: `NeoParams`
- `neo-reductions::common`: `left_mul_acc`, `ct_from_y_ring_for_ccs_m`

Downstream consumers:
- `neo-fold::shard`: calls `rlc_with_commit` after Pi_CCS, before Pi_DEC
- `neo-fold::shard::verifier`: calls `rlc_public` to verify RLC step

## Lean Oracle Conformance

| Test file | Oracle family | What it checks |
|---|---|---|
| (none yet) | (none) | RLC is algebraically simple; verified via roundtrip tests |

## Quality Expectations

- Input validation before computation (fail-fast)
- `RotRho` enforces norm bounds at construction time
- NC channel handled consistently when present or absent

## Acceptance Criteria

- `cargo test -p neo-reductions --release` succeeds
- Spec-derived tests in `spec-tests/pi_rlc.rs` pass
- Existing `rlc_public_y_zcol` and `rlc_dec_k_gt1` tests pass
- `cargo clippy -p neo-reductions --all-targets --release -- -D warnings` clean

## Out of Scope

- DEC decomposition step -- see PiDEC.spec.md
- Commitment scheme internals -- see neo-ajtai specs
- Shard-level orchestration -- belongs to neo-fold
