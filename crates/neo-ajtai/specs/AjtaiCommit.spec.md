# AjtaiCommit

## Purpose

- **What it is**: The Ajtai lattice-based commitment scheme `L: F_q^{d*m} -> C` via `c = M*z` where `M in R_q^{kappa x m}` is the public parameter matrix.
- **Key invariant**: `verify_open(pp, commit(pp, Z), Z) == true` and S-homomorphism: `s_mul(rho, commit(pp, Z))` is consistent with the S-action on the witness.
- **Protocol role**: Commitment is the foundation for all protocol layers — every folded instance carries Ajtai commitments that bind the witness.

## Target Formulas (Paper -> Rust)

| Paper notation | Paper reference | Rust identifier | Notes |
|---|---|---|---|
| `Setup(kappa, m) -> M` | Def 18, line 753 | `setup(rng, d, kappa, m) -> AjtaiResult<PP<RqEl>>` | Uniform random matrix `M in R_q^{kappa x m}` |
| `Commit(pp, z) -> M*z` | Def 18, line 755 | `try_commit(pp, Z) -> AjtaiResult<Commitment>` | Column-major matrix-vector product |
| `c = M*z` | Def 18, line 756 | `commit(pp, Z) -> Commitment` | Panic wrapper over `try_commit` |
| `verify(pp, c, z) := (c == M*z)` | (implicit in Def 18) | `verify_open(pp, c, Z) -> bool` | Direct recomputation check |
| `rho * c` (S-action on commitment) | Thm 2, line 319 | `s_mul(rho, c) -> Commitment` | Left multiplication by ring element |

## Paper Anchors

Source: ./docs/superneo-paper/

- Definition 4 (Ring Commitment Scheme), Section 4, lines 298-301.
- Definition 18 (Ajtai Commitment), lines 753-756: `Setup(kappa,m) -> M; Commit(pp, z) -> Mz`.
- Theorem 2 (Properties), lines 319-321: homomorphic, B-binding, (B,C)-relaxed binding.

## Lean Cross-Reference

| Lean spec | Lean module | Relationship |
|---|---|---|
| `ProofSystem/Lattice.spec.md` | `SuperNeo/Lattice.lean` | Defines `opensTo`, `AjtaiBindingAssumption`, `AjtaiRelaxedBindingAssumption` |
| `ProofSystem/LatticePaper.spec.md` | `SuperNeo/LatticePaper.lean` | Concrete parameters, Theorem 2 properties |

## Contract Surface

| Group | Rust symbol | Kind | Role | Guarantee |
|---|---|---|---|---|
| Setup | `setup(rng, d, kappa, m) -> AjtaiResult<PP<RqEl>>` | fn | Core | Uniform random matrix |
| Setup | `setup_par(rng, d, kappa, m) -> AjtaiResult<PP<RqEl>>` | fn | Core | Parallel deterministic setup |
| Types | `PP<RqEl>` | struct | Core | Public parameters (kappa, m, d, m_rows) |
| Types | `Commitment` | struct | Core | `F_q^{d x kappa}` column-major flat matrix |
| Types | `Commitment::zeros(d, kappa)` | fn | Helper | Zero commitment |
| Types | `Commitment::col(c)` | fn | Helper | Column slice access |
| Types | `Commitment::col_mut(c)` | fn | Helper | Mutable column slice |
| Types | `Commitment::add_inplace(rhs)` | fn | Helper | In-place addition |
| Commit | `try_commit(pp, Z) -> AjtaiResult<Commitment>` | fn | Core | Column-major commit with dimension checking |
| Commit | `commit(pp, Z) -> Commitment` | fn | Core | Panic wrapper |
| Commit | `try_commit_row_major(pp, Z) -> AjtaiResult<Commitment>` | fn | Core | Row-major `Mat` input |
| Commit | `commit_row_major(pp, Z) -> Commitment` | fn | Core | Panic wrapper |
| Commit | `commit_masked_ct(pp, Z) -> Commitment` | fn | Core | Constant-time masked accumulation |
| Commit | `commit_precomp_ct(pp, Z) -> Commitment` | fn | Core | Constant-time precomputed rotations |
| Verify | `verify_open(pp, c, Z) -> bool` | fn | Core | Direct recomputation check (`#[must_use]`) |
| Verify | `verify_split_open(pp, c, b, c_is, Z_is) -> bool` | fn | Core | Split opening verification |
| S-Hom | `s_mul(rho, c) -> Commitment` | fn | Core | Left multiplication by ring element |
| S-Hom | `s_lincomb(rhos, cs) -> AjtaiResult<Commitment>` | fn | Core | Linear combination |
| Internal | `rot_step(cur, next)` | fn | Helper | Rotation step for `Phi_81` (feature-gated: testing) |
| Internal | `precompute_rot_columns(a, cols)` | fn | Helper | Precompute `d` rotation columns (doc-hidden) |
| Internal | `sample_uniform_rq(rng)` | fn | Helper | Uniform ring element sampling (doc-hidden) |
| Internal | `seeded_pp_chunk_seeds(master_seed, kappa, m)` | fn | Helper | Chunk seed derivation (doc-hidden) |
| Internal | `commit_row_major_seeded(seed, d, kappa, m, Z)` | fn | Helper | Seeded PP commit (doc-hidden) |

## Invariant Obligations

| Invariant | Verification method | Lean theorem counterpart |
|---|---|---|
| `verify_open(pp, commit(pp, Z), Z) == true` | Unit test | `opensTo` |
| `verify_open(pp, c, Z') == false` for `Z' != Z` | Unit test | (probabilistic — binding) |
| `s_mul(rho, c) == commit(pp, rho*Z)` (S-homomorphism) | Unit test | (none) |
| `setup_par` determinism: same seed -> same PP | Unit test | (none) |
| `commit_masked_ct(pp, Z) == commit(pp, Z)` | Unit test | (none) |
| `commit_precomp_ct(pp, Z) == commit(pp, Z)` | Unit test | (none) |
| Seeded PP commit matches materialized PP commit | Unit test | (none) |

## Assumption Ledger

| Assumption | Source | Justification |
|---|---|---|
| MSIS hardness for `kappa=18`, `B=2^14` | Paper Appendix B.2 | Lattice estimator gives ~129 bits |
| PRG seed produces uniform-looking matrix | `rand_chacha` | Standard cryptographic assumption |
| Goldilocks field order `q = 2^64 - 2^32 + 1` is prime | neo-math spec | Proved in `Goldilocks.spec.md` |

## Dependency and Consumer Map

Upstream dependencies:
- `neo-math`: `Fq`, `Rq`, `D`, `ETA`, `cf`, `cf_inv`, `SAction`
- `neo-ccs`: `Mat`
- `neo-params`: concrete protocol parameters

Downstream consumers:
- `neo-fold-clean`: root-lane and opening commitments
- `neo-reductions`: `Pi_DEC`

## Lean Oracle Conformance

All spec-derived tests (lean oracles + invariant obligations) live in `spec-tests/`.

| Test file | Oracle family | What it checks |
|---|---|---|
| (none) | (none) | Ajtai commitment correctness is verified via unit tests; no lean oracle vectors are generated for this module |

## Quality Expectations

- No `unsafe` (enforced crate-wide)
- `commit_masked_ct` and `commit_precomp_ct` must be constant-time (no branching on witness values)
- `verify_open` is annotated `#[must_use]` — callers must consume the boolean result
- Internal helpers (`rot_step`, `precompute_rot_columns`, `sample_uniform_rq`) are not part of the contract surface

## Acceptance Criteria

- `cargo test -p neo-ajtai --release` succeeds
- All invariant obligations have spec-tests
- `verify_open(pp, commit(pp, Z), Z)` returns `true` for random witnesses
- S-homomorphism property holds for random `rho` and `Z`

## Out of Scope

- Ring inversion
- NTT-based fast multiplication
- Key generation ceremony
- Multi-party setup
