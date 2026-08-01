# Ring

## Purpose

- **What it is**: The cyclotomic ring `R_q = F_q[X] / (Phi_81)` where `Phi_81(X) = X^54 + X^27 + 1`, with coefficient maps `cf`/`cf_inv`, constant-term extraction `ct`, and schoolbook multiplication.
- **Key invariant**: Ring multiplication is correct modulo `Phi_81`; `cf` and `cf_inv` are inverse bijections; `ct` extracts the degree-0 coefficient.
- **Protocol role**: Ring elements are the algebraic objects committed via Ajtai (`c = A·z` for `z ∈ R_q^n`). The coefficient embedding (Definition 7) maps field vectors `z ∈ F^{d·n}` into ring vectors `z ∈ R_q^n` by chunking into `d=54`-sized blocks.

## Target Formulas (Paper -> Rust)

| Paper notation | Paper reference | Rust identifier | Notes |
|---|---|---|---|
| `R_F = F[X]/(Phi(X))` | Def 1, Section 4, line 280 | `Rq` | `Phi_81(X) = X^54 + X^27 + 1`, degree `d = 54` |
| `d = deg(Phi_eta)` | Def 1, line 280 | `D = 54` | Cyclotomic degree |
| `eta` | Def 1, line 280 | `ETA = 81` | Cyclotomic index |
| `cf(a) ∈ F^d` | Def 2, Section 4, line 286 | `cf(a: Rq) -> [Fq; D]` | Coefficient vector |
| `cf^{-1}(v)` | Def 2, line 286 | `cf_inv(v: [Fq; D]) -> Rq` | Inverse embedding |
| `ct(a) ∈ F` | Def 2, line 288 | `ct(a: &Rq) -> Fq` | Constant term (degree-0 coeff) |
| `‖a‖_∞` | Def 3, Section 4, line 292 | `inf_norm(a: &Rq) -> u128` | Max centered coefficient |
| `bar(·)` transform | Thm 3, Section 5, line 368 | `superneo_bar_block(v)` | Inner-product transform on `d`-block |
| `bar(v)` vector lift | Def 8, Section 5, line 378 | `superneo_bar_vec(v)` | Block-wise bar transform |

## Paper Anchors

Source: ./docs/superneo-paper/

- Definition 1 (Fields, Rings, Dimensions), Section 4, lines 278-282: `R_F := F[X]/(Phi(X))`, `d = deg(Phi)`.
- Definition 2 (Coefficient maps), Section 4, lines 286-288: `cf`, `ct`.
- Definition 3 (Norm), Section 4, lines 290-292: `‖a‖_∞` over centered representatives.
- Appendix B.2, lines 709-713: `eta = 81`, `Phi = X^54 + X^27 + 1`, `d = 54`.

## Lean Cross-Reference

| Lean spec | Lean module | Relationship |
|---|---|---|
| `specs/Goldilocks.spec.md` | `SuperNeo/Goldilocks.lean` | Defines the base field `q` underlying ring coefficients |
| `specs/Parameters.spec.md` | `SuperNeo/Parameters.lean` | Concrete `d`, `eta`, `b`, `k`, `B` values |
| `specs/Embedding.spec.md` | `SuperNeo/Embedding.lean` | `embedElem`/`unembedElem` are the Lean equivalent of `cf_inv`/`cf` |
| `specs/Thm3Core.spec.md` | `SuperNeo/Thm3Core.lean` | `ct(mulRqPhi(bar(a),b)) = <a,b>` uses ring multiplication |

## Contract Surface

| Group | Rust symbol | Kind | Role | Guarantee |
|---|---|---|---|---|
| Constants | `ETA` | const | Core | `81` — cyclotomic index |
| Constants | `D` | const | Core | `54` — degree of `Phi_81`, ring dimension |
| Ring type | `Rq` | struct | Core | `[Fq; D]` coefficient representation |
| Ring ops | `Rq::zero()` | fn | Core | Additive identity |
| Ring ops | `Rq::one()` | fn | Core | Multiplicative identity (`1 + 0·X + ...`) |
| Ring ops | `Rq::add(&self, &rhs)` | fn | Core | Coefficient-wise addition |
| Ring ops | `Rq::sub(&self, &rhs)` | fn | Core | Coefficient-wise subtraction |
| Ring ops | `Rq::mul(&self, &rhs)` | fn | Core | Schoolbook mul mod `Phi_81`; constant-time |
| Ring ops | `Rq::mul_by_monomial(j)` | fn | Helper | Fast `a(X) · X^j mod Phi_81` |
| Coeff maps | `cf(a)` | fn | Core | Coefficient vector extraction `R_q → F_q^d` |
| Coeff maps | `cf_inv(v)` | fn | Core | Inverse embedding `F_q^d → R_q` |
| Coeff maps | `ct(a)` | fn | Core | Constant-term extraction `R_q → F_q` |
| Norm | `inf_norm(a)` | fn | Core | `‖a‖_∞` over centered representatives |
| Bar transform | `superneo_bar_block(v)` | fn | Core | Theorem-3 inner-product transform on one `d`-block |
| Bar transform | `superneo_bar_vec(v)` | fn | Core | Block-wise bar transform over `F_q^{d·n}` |
| Bar transform | `superneo_bar_matrix()` | fn | Core | Lazily-computed `d×d` bar transform matrix `M` |
| Internal | `rot_apply_vec(a, v)` | fn | Helper | S-action `cf(a · cf_inv(v))` |
| Constructors | `Rq::from_field_coeffs(v)` | fn | Helper | Ring element from coeff slice |
| Constructors | `Rq::from_field_scalar(s)` | fn | Helper | Embed scalar as constant polynomial |
| Random | `Rq::random_small(rng, bound)` | fn | Helper | Random element with `‖·‖_∞ ≤ bound` |
| Random | `Rq::random_uniform(rng)` | fn | Helper | Uniform random ring element |

## Invariant Obligations

| Invariant | Verification method | Lean theorem counterpart |
|---|---|---|
| `cf(cf_inv(v)) = v` for all `v ∈ F_q^d` | `lean_oracles` (removed lane) (`coeff_maps_v1`) | `unembedElem_embedElem` |
| `cf_inv(cf(a)) = a` for all `a ∈ R_q` | `lean_oracles` (removed lane) (`coeff_maps_v1`) | `embedElem_unembedElem` |
| `ct(a) = cf(a)[0]` | Unit test | `ct` definition in `CoeffMaps` |
| Ring mul is correct: `a·b mod Phi_81` | `lean_oracles` (removed lane) (`ring_ct_v1`) | Ring multiplication in Lean |
| `Phi_81` reduction: `X^54 ≡ -X^27 - 1` | Unit test | Cyclotomic reduction identity |
| `ct(bar(a) · b) = <a, b>` (Theorem 3) | `lean_oracles` (removed lane) (`ring_ct_v1`) | `thm3CoreAssumption_native` |
| Bar matrix is inverse of ct-Gram matrix | Internal assertion in `build_superneo_bar_matrix` | `superneoBarBlock_eq_id` |
| `inf_norm` uses centered representatives | `lean_oracles` (removed lane) (`invertibility_v1`) | Norm definition in `Norm.lean` |

## Assumption Ledger

| Assumption | Source | Justification |
|---|---|---|
| `Phi_81(X) = X^54 + X^27 + 1` is the 81st cyclotomic polynomial | Number theory | Standard; Phi_81 factors as (X^81-1)/((X^27-1)(X^3-1)/(X-1)·(X^9-1)/(X^3-1)) |
| Bar matrix is well-defined (ct-Gram matrix is invertible) | Runtime assertion | Checked at first use; panics if fails |

## Dependency and Consumer Map

Upstream dependencies:
- `neo-math::field`: `Fq` type and `GOLDILOCKS_MODULUS`
- `p3-field`: `Field`, `PrimeField64` traits

Downstream consumers:
- `neo-math::s_action`: `SAction` wraps ring multiplication
- `neo-ajtai`: commitment `c = A·z` uses `Rq` vectors and `cf`/`cf_inv`
- `neo-reductions`: Π_CCS/Π_RLC/Π_DEC operate on ring-level commitments
- `neo-ccs`: CCS matrix embedding uses `superneo_bar_vec`

## Lean Oracle Conformance

All spec-derived tests (lean oracles + invariant obligations) live in `spec-tests/`.

| Test file | Oracle family | What it checks |
|---|---|---|
| `crates/neo-math/spec-tests/lean_oracles.rs` | `ring_ct_v1` | Ring mul, `ct(bar(a)·b) = <a,b>`, dot product |
| `crates/neo-math/spec-tests/lean_oracles.rs` | `coeff_maps_v1` | `cf`/`cf_inv` round-trip, `ct` extraction |
| `crates/neo-math/spec-tests/lean_oracles.rs` | `embedding_bar_v1` | `superneo_bar_block`, `superneo_bar_vec`, block chunking |

## Quality Expectations

- No `unsafe` (enforced crate-wide)
- `Rq::mul` must be constant-time (no branching on coefficient values)
- `reduce_mod_phi_81` is `pub(crate)` only — not part of the contract surface
- Bar matrix computation is lazy (computed once, cached in `OnceLock`)

## Acceptance Criteria

- `cargo test -p neo-math --release` succeeds (runs both `tests/` and `spec-tests/`)
- All three `lean_oracles` (removed lane) families (`ring_ct_v1`, `coeff_maps_v1`, `embedding_bar_v1`) pass
- Spec-derived tests in `spec-tests/ring.rs` pass
- `superneo_bar_matrix()` internal sanity check (`M^T G = I`) does not panic

## Out of Scope

- NTT-based fast multiplication (explicitly avoided per Neo design)
- Power-of-two cyclotomic polynomials (`X^d + 1`); only `Phi_81` is supported
- Ring inversion (belongs to `InvertibilityAxioms` / neo-ajtai layer)
- Sparse ring multiplication optimizations beyond `mul_sparse_bits`
