# Goldilocks

## Purpose

- **What it is**: The Goldilocks base field `F_q` with `q = 2^64 - 2^32 + 1` and the quadratic extension `K = F_{q^2}` used throughout SuperNeo for sum-check soundness.
- **Key invariant**: `q` is prime; `K` provides at least 128-bit soundness (`|K| ≈ 2^128`); conjugation satisfies `conj(a + bu) = a - bu`.
- **Protocol role**: Every arithmetic operation in the folding scheme operates over `F_q`. The extension `K` is the domain for sum-check challenges (Definition 6) and evaluation points `r ∈ K^{log m}` (Definitions 12-13).

## Target Formulas (Paper -> Rust)

| Paper notation | Paper reference | Rust identifier | Notes |
|---|---|---|---|
| `F` (base field of prime order `q`) | Def 1, Section 4, line 278 | `Fq` (type alias for `p3_goldilocks::Goldilocks`) | `q = 2^64 - 2^32 + 1` |
| `K` (extension `F_{q^2}`) | Def 1, Section 4, line 278 | `K` (type alias for `BinomialExtensionField<Fq, 2>`) | Degree-2 extension, `|K| ≈ 2^128` |
| `q = 2^64 - 2^32 + 1` | Appendix B.2, line 709 | `GOLDILOCKS_MODULUS` | `18446744069414584321` |

## Paper Anchors

Source: ./formal/superneo-lean/SuperNeo.pdf.md

- Definition 1 (Fields, Rings, Dimensions), Section 4, lines 275-282: defines `F`, `K`, and the requirement `1/|K| = negl(λ)`.
- Appendix B.2 (Goldilocks parameters), lines 709-727: `q = 2^64 - 2^32 + 1`, `K = F_{q^2}`.

## Lean Cross-Reference

| Lean spec | Lean module | Relationship |
|---|---|---|
| `specs/Goldilocks.spec.md` | `SuperNeo/Goldilocks.lean` | Defines `q = 18446744069414584321` and `halfQ`; all theorems closed via `native_decide`/`omega` |
| `specs/GoldilocksPrime.spec.md` | `SuperNeo/GoldilocksPrime.lean` | Proves `Nat.Prime q` via Lucas primality with witness `a = 7` |
| `specs/Parameters.spec.md` | `SuperNeo/Parameters.lean` | `extDegreeK = 2` for the extension field degree |

## Contract Surface

| Group | Rust symbol | Kind | Role | Guarantee |
|---|---|---|---|---|
| Base field | `Fq` | type | Core | Alias for `p3_goldilocks::Goldilocks`; prime field with `q = 2^64 - 2^32 + 1` |
| Extension field | `K` | type | Core | `BinomialExtensionField<Fq, 2>`; quadratic extension `F_{q^2}` |
| Extension trait | `KExtensions` | trait | Core | Conjugation, inversion, coefficient access for `K` |
| Extension trait | `KExtensions::conj` | fn | Core | `(a + bu) ↦ (a - bu)`; self-inverse |
| Extension trait | `KExtensions::inv` | fn | Core | Multiplicative inverse; panics on zero |
| Extension trait | `KExtensions::as_coeffs` | fn | Core | Extract `[real, imag]` components |
| Extension trait | `KExtensions::from_coeffs` | fn | Core | Construct `K` from `[real, imag]` |
| Extension trait | `KExtensions::scale_base` | fn | Helper | Multiply by base-field scalar (faster than full extension mul) |
| Constructor | `from_complex(real, imag)` | fn | Core | Shorthand for `K::from_coeffs([real, imag])` |
| Internal | `GOLDILOCKS_MODULUS` | const | Helper | `18446744069414584321u128`; used by norm computations |

## Invariant Obligations

| Invariant | Verification method | Lean theorem counterpart |
|---|---|---|
| `q = 18446744069414584321` | Compile-time const | `Goldilocks.q` |
| `q` is prime | Lucas primality test (witness `a = 7`, factors of `q-1`) | `Goldilocks.q_prime` |
| `conj(conj(x)) = x` for all `x ∈ K` | Property test | (follows from `conjugate` definition) |
| `x * inv(x) = 1` for all nonzero `x ∈ K` | Property test | (follows from `Field::inverse`) |
| `from_coeffs(as_coeffs(x)) = x` round-trip | Property test | (follows from constructor) |
| Extension irreducibility: `u^2 - 7` has no root in `F_q` | Euler's criterion (`7^((q-1)/2) = -1 mod q`) | (implicit in `BinomialExtensionField`) |

## Assumption Ledger

| Assumption | Source | Justification |
|---|---|---|
| `p3_goldilocks` correctly implements Goldilocks arithmetic | `p3-goldilocks` crate | Widely used, audited by Plonky3 team |
| `BinomialExtensionField<Fq, 2>` correctly implements `F_{q^2}` | `p3-field` crate | Extension polynomial `u^2 - 7` is irreducible over Goldilocks |

## Dependency and Consumer Map

Upstream dependencies:
- `p3-goldilocks`: provides `Goldilocks` field type
- `p3-field`: provides `BinomialExtensionField`, `Field`, `PrimeField64` traits

Downstream consumers:
- `neo-math::ring`: `Rq` coefficients are `Fq` elements
- `neo-math::norms`: norm computations use `GOLDILOCKS_MODULUS`
- `neo-math::s_action`: S-action on `K`-vectors uses `KExtensions`
- Every downstream crate uses `Fq` and `K`

## Lean Oracle Conformance

All spec-derived tests (lean oracles + invariant obligations) live in `spec-tests/`.

| Test file | Oracle family | What it checks |
|---|---|---|
| `crates/neo-math/spec-tests/lean_oracles.rs` | `coeff_maps_v1` | Round-trip `cf(cf_inv(v)) = v` exercises `Fq` arithmetic |
| `crates/neo-math/spec-tests/lean_oracles.rs` | `ring_ct_v1` | Ring multiplication exercises `Fq` field ops |

## Quality Expectations

- No `unsafe` in this module (enforced by `#![forbid(unsafe_code)]` at crate root)
- Type aliases must reference stable `p3` types
- `KExtensions` trait must be the single extension point for `K` operations

## Acceptance Criteria

- `cargo test -p neo-math --release` succeeds (runs both `tests/` and `spec-tests/`)
- `lean_oracles` (removed lane) conformance tests pass (exercises `Fq` arithmetic indirectly)
- Spec-derived tests in `spec-tests/goldilocks.rs` pass
- All `KExtensions` methods are consistent with their mathematical definitions

## Out of Scope

- Alternative base fields (Mersenne-61, Almost Goldilocks from Appendix B.1/B.3)
- Higher-degree extensions beyond `K = F_{q^2}`
- NTT/FFT over Goldilocks (explicitly avoided per Neo design)
