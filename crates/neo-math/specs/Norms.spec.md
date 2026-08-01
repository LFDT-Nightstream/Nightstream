# Norms

## Purpose

- **What it is**: Centered-representative norm and norm-bound predicates for field elements, ring elements, and coefficient vectors.
- **Key invariant**: For `a ∈ F_q`, the centered representative is `min(a, q - a)`, giving `‖a‖_∞ ∈ [0, (q-1)/2]`. The NC bound check `within_nc_bound(v, b)` returns true iff `‖v‖_∞ < b`.
- **Protocol role**: Norm bounds are the security backbone of SuperNeo. CCS(b,L) and CE(b,L) require `‖z‖_∞ < b` (Definitions 12-13). The sum-check `NC(X)` polynomial encodes norm constraints (Section 7.3, line 509). Π_DEC decomposes high-norm witnesses into low-norm digits (Section 7.5).

## Target Formulas (Paper -> Rust)

| Paper notation | Paper reference | Rust identifier | Notes |
|---|---|---|---|
| `‖a‖_∞` for `a ∈ F` | Def 3, Section 4, line 292 | `to_balanced_i128(v)` | Centered representative in `[-(q-1)/2, (q-1)/2]` |
| `‖a‖_∞ < b` | Def 12-13, Section 7.1 | `within_nc_bound(v, b)` | NC bound check |
| `‖a‖_∞` for `a ∈ R_F` | Def 3, Section 4, line 292 | `inf_norm(a: &Rq)` (in `ring.rs`) | Max over coefficient centered reps |

## Paper Anchors

Source: ./docs/superneo-paper/

- Definition 3 (Norm), Section 4, lines 290-292: `‖a‖_∞` for field elements via centered representative; for ring elements via max over coefficients.
- Definition 12 (CCS), Section 7.1, line 459: `‖z‖_∞ < b` norm bound in CCS relation.
- Definition 13 (CE), Section 7.1, line 465: `‖z‖_∞ < b` norm bound in CE relation.

## Lean Cross-Reference

| Lean spec | Lean module | Relationship |
|---|---|---|
| `specs/Decomp.spec.md` | `SuperNeo/Decomp.lean` | Digit bounds `‖d_j‖_∞ < b` use the same norm definition |
| `specs/InvertibilityAxioms.spec.md` | `SuperNeo/InvertibilityAxioms.lean` | `strictInvertibilityWindowProp B a` uses `0 < ‖a‖_∞ < B` |
| `specs/SamplingSet.spec.md` | `SuperNeo/SamplingSet.lean` | `samplingNormBoundProp` uses `‖·‖_∞ ≤ B` |

## Contract Surface

| Group | Rust symbol | Kind | Role | Guarantee |
|---|---|---|---|---|
| Centering | `to_balanced_i128(v)` | fn | Core | Returns centered representative in `[-(q-1)/2, (q-1)/2]` for any `v ∈ F_q` |
| Bound check | `within_nc_bound(v, b)` | fn | Core | Returns `true` iff `|to_balanced_i128(v)| < b` (i.e., `v ∈ {-(b-1), ..., (b-1)}`) |

## Invariant Obligations

| Invariant | Verification method | Lean theorem counterpart |
|---|---|---|
| `to_balanced_i128(0) = 0` | Unit test | (follows from definition) |
| `to_balanced_i128(v)` is in `[-(q-1)/2, (q-1)/2]` | Property test | Norm definition in `Norm.lean` |
| `within_nc_bound(v, 2)` iff `v ∈ {-1, 0, 1}` | Unit test | `digitsWithinBaseProp` with `b = 2` |
| `within_nc_bound(v, b)` returns false for `b < 2` | Unit test | (guard condition) |
| Norm of ring element equals max norm of coefficients | `lean_oracles` (removed lane) (`invertibility_v1`) | Ring norm = max coefficient norm |

## Assumption Ledger

No external assumptions. All computations are pure arithmetic over `u128`.

## Dependency and Consumer Map

Upstream dependencies:
- `p3-field`: `PrimeField64`, `PrimeCharacteristicRing` traits

Downstream consumers:
- `neo-math::ring`: `inf_norm` uses `GOLDILOCKS_MODULUS` and centered-rep logic
- `neo-ajtai::decomp`: digit-bound checks use `within_nc_bound`
- `neo-reductions`: NC polynomial construction uses norm predicates
- `neo-math/spec-tests/lean_oracles.rs`: `centered_norm` helper uses `to_balanced_i128`

## Lean Oracle Conformance

All spec-derived tests (lean oracles + invariant obligations) live in `spec-tests/`.

| Test file | Oracle family | What it checks |
|---|---|---|
| `crates/neo-math/spec-tests/lean_oracles.rs` | `invertibility_v1` | `expected_norm` matches `centered_norm`; `within_nc_bound` matches `expected_weak_window` |

## Quality Expectations

- No `unsafe`
- `to_balanced_i128` must be generic over `PrimeField64`
- No allocations in norm computation

## Acceptance Criteria

- `cargo test -p neo-math --release` succeeds (runs both `tests/` and `spec-tests/`)
- `lean_oracles` (removed lane) `invertibility_v1` tests pass (norm values and window checks match Lean)
- Spec-derived tests in `spec-tests/norms.rs` pass

## Out of Scope

- L2 norms (SuperNeo uses only `l_∞`)
- Norm-growth tracking across folding steps (belongs in `neo-reductions`)
- Invertibility predicates (belong in `InvertibilityAxioms` / neo-ajtai)
