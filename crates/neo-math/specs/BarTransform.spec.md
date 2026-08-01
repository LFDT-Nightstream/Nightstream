# BarTransform

## Purpose

- **What it is**: The SuperNeo inner-product transform `bar(·)` (Theorem 3) and its block-wise lifting to vectors and matrices (Definition 8). The transform satisfies `ct(bar(a) · b) = <a, b>` for `d`-sized blocks, enabling field-native sum-check over ring-committed witnesses.
- **Key invariant**: The bar-transform matrix `M` is the inverse of the constant-term Gram matrix `G[i,j] = ct(X^i · X^j mod Phi_81)`. This ensures `ct(cf_inv(bar(a)) · cf_inv(b)) = <a, b>`.
- **Protocol role**: The bar transform is the mathematical core that enables SuperNeo's field-native arithmetic (desideratum D3). It allows CCS matrix-vector products `Mz` to be computed as `ct(M_bar · z_ring)` (Theorem 4, line 384), which the sum-check then reduces to point evaluations.

## Target Formulas (Paper -> Rust)

| Paper notation | Paper reference | Rust identifier | Notes |
|---|---|---|---|
| `bar(·) : F^d → F^d` | Thm 3, Section 5, line 368 | `superneo_bar_block(v: [Fq; D]) -> [Fq; D]` | Linear transform on one block |
| `ct(bar(a) · bar(b)) = <a, b>` | Thm 3, line 370 | Verified by `lean_oracles` (removed lane) `ring_ct_v1` | Inner-product identity |
| `bar(v)` for `v ∈ F^{n_F}` | Def 8, Section 5, line 378 | `superneo_bar_vec(v: &[Fq]) -> Vec<Fq>` | Block-wise lift; `len` must be multiple of `D` |
| `Mz = ct(M_bar · z)` | Thm 4, Section 5, line 384 | (composition of `superneo_bar_vec` + ring mul + `ct`) | Matrix-vector product transform |

## Paper Anchors

Source: ./docs/superneo-paper/

- Theorem 3 (Inner Product Transform), Section 5, lines 368-372: `ct(bar(a) · bar(b)) = <a, b>`.
- Definition 8 (Lifting the Transform), Section 5, lines 376-382: block-wise vector/matrix lift.
- Theorem 4 (Matrix-Vector Product Transform), Section 5, lines 384-386: `Mz = ct(M_bar · z)`.
- Remark 1 (Efficiency and Sparsity Preservation), line 382: for trinomial cyclotomics, the transform involves permutations and additions only.

## Lean Cross-Reference

| Lean spec | Lean module | Relationship |
|---|---|---|
| `specs/Thm3Core.spec.md` | `SuperNeo/Thm3Core.lean` | `thm3CoreAssumption`: `ct(mulRqPhi(bar(a),b)) = innerProduct a b`; closed constructively for native bar matrix |
| `specs/BarLift.spec.md` | `SuperNeo/BarLift.lean` | Block-wise lifting, linearity (`barLiftVector_add`, `barLiftVector_scale`) |
| `specs/MatrixTransform.spec.md` | `SuperNeo/MatrixTransform.lean` | `matrixTransformAssumption`: `Mz = ct(M_bar · z)` from Thm 3 |
| `specs/EvalHom.spec.md` | `SuperNeo/EvalHom.lean` | Evaluation homomorphism (Thm 5) depends on bar transform linearity |
| `specs/Embedding.spec.md` | `SuperNeo/Embedding.lean` | `embedVec`/`unembedVec` define the coefficient embedding that bar operates on |

## Contract Surface

| Group | Rust symbol | Kind | Role | Guarantee |
|---|---|---|---|---|
| Block transform | `superneo_bar_block(v)` | fn | Core | Applies `M · v` where `M = G^{-1}` (Gram inverse); output satisfies Theorem 3 |
| Vector transform | `superneo_bar_vec(v)` | fn | Core | Block-wise `superneo_bar_block` over chunks of `D`; panics if `len % D != 0` |
| Matrix access | `superneo_bar_matrix()` | fn | Core | Returns `&'static [[Fq; D]; D]`; lazily computed and cached |

## Invariant Obligations

| Invariant | Verification method | Lean theorem counterpart |
|---|---|---|
| `ct(cf_inv(bar(a)) * cf_inv(b)) = <a, b>` | `lean_oracles` (removed lane) (`ring_ct_v1`): `expected_ct_bar_dot = expected_dot` | `thm3CoreAssumption_native` |
| Bar matrix `M` satisfies `M^T · G = I` | Runtime assertion in `build_superneo_bar_matrix` | `superneoBarBlock_eq_id` |
| `superneo_bar_vec` is linear: `bar(a + b) = bar(a) + bar(b)` | Property test | `barLiftVector_add` |
| `superneo_bar_vec` is linear: `bar(s · a) = s · bar(a)` | Property test | `barLiftVector_scale` |
| Block chunking matches embedding: blocks are `D`-sized | `lean_oracles` (removed lane) (`embedding_bar_v1`) | `embedVec_map_superneoBarBlock_eq` |

## Assumption Ledger

| Assumption | Source | Justification |
|---|---|---|
| Gram matrix `G` is invertible for `Phi_81` | Runtime assertion | Verified at first invocation; would panic if false |

## Dependency and Consumer Map

Upstream dependencies:
- `neo-math::ring`: `Rq`, `ct`, `cf_inv` for Gram matrix construction
- `neo-math::field`: `Fq` arithmetic

Downstream consumers:
- `neo-ccs`: CCS matrix embedding applies `superneo_bar_vec` to matrix rows
- `neo-reductions`: Π_CCS oracle polynomial construction uses bar-transformed matrices
- `neo-ajtai`: commitment verification uses bar transform for evaluation checks

## Lean Oracle Conformance

All spec-derived tests (lean oracles + invariant obligations) live in `spec-tests/`.

| Test file | Oracle family | What it checks |
|---|---|---|
| `crates/neo-math/spec-tests/lean_oracles.rs` | `ring_ct_v1` | `ct(bar(a)·b) = <a,b>` via `expected_ct_bar_dot == expected_dot` |
| `crates/neo-math/spec-tests/lean_oracles.rs` | `embedding_bar_v1` | Block decomposition and `superneo_bar_vec` output match Lean vectors |

## Quality Expectations

- Bar matrix is computed once and cached (`OnceLock`); no repeated inversions
- `superneo_bar_block` must not allocate (operates on fixed `[Fq; D]`)
- `superneo_bar_vec` panics on misaligned input rather than silently truncating

## Acceptance Criteria

- `cargo test -p neo-math --release` succeeds (runs both `tests/` and `spec-tests/`)
- `lean_oracles` (removed lane) `ring_ct_v1` and `embedding_bar_v1` pass
- Spec-derived tests in `spec-tests/bar_transform.rs` pass
- Bar matrix sanity check (`M^T G = I`) passes on first invocation

## Out of Scope

- The Galois/conjugation automorphism trick (used in the paper's proof of Theorem 3, not in the implementation)
- Sparse bar transform optimizations (Remark 1 notes permutation-only for trinomials; current impl uses dense matrix-vector multiply)
- Bar transform for non-`Phi_81` cyclotomics
