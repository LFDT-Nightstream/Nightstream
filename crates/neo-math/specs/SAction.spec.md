# SAction

## Purpose

- **What it is**: The S-action `rot(a)` — left multiplication by a ring element `a ∈ R_q`, lifted to vectors in `F_q^d`, matrices, and extension-field vectors in `K^d`. Composition satisfies `rot(a) ∘ rot(b) = rot(a·b)`.
- **Key invariant**: `apply_vec(a, v) = cf(a · cf_inv(v))` — the S-action is definitionally equivalent to ring multiplication followed by coefficient extraction.
- **Protocol role**: S-actions implement the random linear combination step in Π_RLC (Section 7.4): folding `z* = Σ ρ_i · z_i` where `ρ_i ∈ C ⊂ R_q` are sampling-set challenges. The S-action extends to `K`-vectors for folding ME evaluation claims `y_j ∈ R_K` (Definition 13).

## Target Formulas (Paper -> Rust)

| Paper notation | Paper reference | Rust identifier | Notes |
|---|---|---|---|
| `ρ · z` for `ρ ∈ R_F`, `z ∈ R_F^n` | Π_RLC, Section 7.4, line 577 | `SAction::apply_vec(&self, v)` | Ring-scalar action on coefficient vector |
| `rot(a)` matrix | (implicit in ring multiplication) | `SAction::to_matrix(&self)` | Full `d×d` matrix: column j = `cf(a · X^j)` |
| `ρ · y` for `y ∈ R_K` | Π_RLC, Section 7.4, line 573 | `SAction::apply_k_vec(&self, y)` | Extension via real/imag decomposition |

## Paper Anchors

Source: ./formal/superneo-lean/SuperNeo.pdf.md

- Section 7.4 (Π_RLC), lines 571-579: random linear combination `z ← Σ ρ_i · z_i` with `ρ_i ∈ C`.
- Definition 15 (Module Homomorphism), line 741: `R_F`-module homomorphism properties.
- Theorem 5 (Evaluation Homomorphism), lines 390-398: linearity of evaluation under ring-scalar combination.

## Lean Cross-Reference

| Lean spec | Lean module | Relationship |
|---|---|---|
| `specs/EvalHom.spec.md` | `SuperNeo/EvalHom.lean` | Theorem-5 linearity `eval(ρ₁·z₁ + ρ₂·z₂, r) = ρ₁·eval(z₁,r) + ρ₂·eval(z₂,r)` — the S-action is the `ρ·z` operation |
| `specs/SamplingSet.spec.md` | `SuperNeo/SamplingSet.lean` | Challenge elements `ρ ∈ C` are the ring scalars that drive S-actions |

## Contract Surface

| Group | Rust symbol | Kind | Role | Guarantee |
|---|---|---|---|---|
| Constructor | `SAction::from_ring(a: Rq)` | fn | Core | Wraps ring element as S-action |
| Constructor | `SAction::scalar(f: Fq)` | fn | Helper | Embeds base-field scalar as constant-polynomial S-action |
| Matrix | `SAction::to_matrix(&self)` | fn | Helper | Full `d×d` rotation matrix; column j = `cf(a · X^j mod Phi_81)` |
| Fq action | `SAction::apply_vec(&self, v)` | fn | Core | `cf(a · cf_inv(v))`; action on `F_q^d` |
| Composition | `SAction::compose(&self, other)` | fn | Core | `rot(a·b)` — ring multiplication of underlying elements |
| K action | `SAction::apply_k_vec(&self, y)` | fn | Core | S-action on `K`-vector via real/imag decomposition; rejects nonzero padding beyond `D` |

## Invariant Obligations

| Invariant | Verification method | Lean theorem counterpart |
|---|---|---|
| `apply_vec(a, v) = cf(a · cf_inv(v))` | Unit test (definitional) | (follows from construction) |
| `compose(a, b).apply_vec(v) = apply_vec(a, apply_vec(b, v))` | Property test | (ring associativity) |
| `scalar(f).apply_vec(v) = f * v` (coefficient-wise) | Unit test | (constant polynomial multiplication) |
| `to_matrix` column j equals `cf(a · X^j mod Phi_81)` | Unit test | (definitional) |
| K-action: `apply_k_vec` rejects nonzero elements at index ≥ D | Unit test | (security: prevents dimension-mismatch attacks) |
| K-action linearity: `apply_k_vec(a, u + v) = apply_k_vec(a, u) + apply_k_vec(a, v)` | Property test | (follows from Fq-linearity applied to real/imag) |

## Assumption Ledger

No external assumptions. All operations reduce to ring multiplication in `Rq`.

## Dependency and Consumer Map

Upstream dependencies:
- `neo-math::ring`: `Rq`, `cf`, `cf_inv`
- `neo-math::field`: `Fq`, `K`, `from_complex`, `KExtensions`

Downstream consumers:
- `neo-ajtai::s_module`: uses `SAction` for S-homomorphic commitment operations
- `neo-reductions`: Π_RLC uses S-action for random linear combination of witnesses
- `neo-fold`: shard folding uses S-action for ME claim aggregation

## Lean Oracle Conformance

All spec-derived tests (lean oracles + invariant obligations) live in `spec-tests/`.
No dedicated `lean_oracles` (removed lane) test family for S-action. Coverage is indirect through
ring multiplication oracles (`ring_ct_v1`) which exercise the same `Rq::mul` path.

## Quality Expectations

- No `unsafe`
- `apply_k_vec` must validate padding (security-critical: nonzero tail elements beyond `D` are rejected with `SActionError`)
- `apply_vec` must not allocate (operates on fixed `[Fq; D]`)

## Acceptance Criteria

- `cargo test -p neo-math --release` succeeds (runs both `tests/` and `spec-tests/`)
- Spec-derived tests in `spec-tests/s_action.rs` pass
- K-vector padding rejection works correctly for vectors longer than `D`
- Composition is consistent with ring multiplication

## Out of Scope

- S-action decomposition (splitting `rot(a)` into sub-actions; belongs in neo-ajtai)
- Challenge sampling from `C` (belongs in neo-transcript / neo-reductions)
- S-action on arbitrary-length field vectors (only `D`-sized blocks are supported)
