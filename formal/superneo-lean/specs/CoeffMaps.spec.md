# CoeffMaps

## Purpose

- **What it is**: The coefficient maps `cf : R_q → F_q^d` and `cf⁻¹ : F_q^d → R_q` that convert between the polynomial-ring representation and the field-vector representation, proved to be mutual inverses.
- **Key property**: `cfInv (cf a) = a` and `cf (cfInv v) = v` — perfect round-trip.
- **Protocol role**: `cf` and `cfInv` mediate between ring-algebraic operations (`mulRq`, norm bounds) and field-vector operations (embedding, MLE evaluation). The embedding `embed` (Definition 7) operates on `cf`-images, and the result is lifted back via `cfInv`.

## Target Formulas

- `cf(a) = a` (identity, since `Coeffs = Array F`)
- `cf⁻¹(v) = v` (identity)
- `cf⁻¹(cf(a)) = a` (round-trip)
- `cf(cf⁻¹(v)) = v` (round-trip)

## Paper Anchors

Source: ./formal/superneo-lean/SuperNeo.pdf.md

- Definition 2 (Coefficient maps), Section 4, lines 284-288: `cf : R_q → F_q^d` and `cf⁻¹ : F_q^d → R_q`.

## Module Mapping

| Lean file | Paper section |
|---|---|
| `SuperNeo/Primitives/CoeffMaps.lean` | Definition 2 |

## Contract Surface

| Group | Lean symbol | Kind | Role | Guarantee |
|---|---|---|---|---|
| Maps | `cf` | def | Definitional | Identity on `Array F` |
| Maps | `cfInv` | def | Definitional | Identity on `Array F` |
| Round-trip | `cfInv_cf` | theorem | Theorem-Target | `cfInv (cf a) = a` |
| Round-trip | `cf_cfInv` | theorem | Theorem-Target | `cf (cfInv v) = v` |
| Size | `cf_size` | theorem | Theorem-Target | `(cf a).size = a.size` |
| Size | `cfInv_size` | theorem | Theorem-Target | `(cfInv v).size = v.size` |
| Compatibility | `ct_cf` | theorem | Theorem-Target | `ct (cf a) = ct a` |
| Compatibility | `ct_cfInv` | theorem | Theorem-Target | `ct (cfInv v) = ct v` |
| Shape | `hasRingDegreeShape_cf_iff` | theorem | Theorem-Target | Shape preserved by `cf` |
| Shape | `hasRingDegreeShape_cfInv_iff` | theorem | Theorem-Target | Shape iff `v.size = D` |
| Shape | `ringMulShapeProp_cf_iff` | theorem | Theorem-Target | Multiplication shape preserved |
| Bundle | `coeffMapRoundTripProp` | def | Definitional | Conjunction of both round-trips |
| Bundle | `coeffMapRoundTrip_theorem` | theorem | Theorem-Target | `coeffMapRoundTripProp v` for all `v` |
| Bundle | `coeffMapRoundTrip_complete` | theorem | Theorem-Target | `coeffMapRoundTripProp v → coeffMapRoundTrip v = true` |
| Bundle | `coeffMapRoundTrip_eq_true_iff` | theorem | Theorem-Target | executable round-trip iff proposition |
| Shape | `ringMulShapeProp_cfInv_iff` | theorem | Theorem-Target | multiplication shape preserved through `cfInv` |

## Proof Obligations and Closure Plan

All obligations closed. The remaining useful closure is the full Bool↔Prop round-trip surface around `coeffMapRoundTrip`.

## Assumption Ledger

No open boundary assumptions in this module.

## Dependency and Consumer Map

Upstream dependencies:
- `SuperNeo/Primitives/Ring.lean`: `Coeffs`, `ct`, `hasRingDegreeShape`, `mulRq`.

Downstream consumers:
- `SuperNeo/EmbeddingTheory/Embedding.lean`: uses `cf`/`cfInv` to convert between ring and field representations for `embedElem`.
- `SuperNeo/EmbeddingTheory/Thm3Core.lean`: uses `cfInv_cf` for the core embedding theorem.

## Implementation Plan

No further work required; module is proof-complete.

## Quality Expectations

All round-trip proofs should be `rfl`-reducible to confirm the representation is genuinely shared (no data copying).

## Acceptance Criteria

- `lake build` succeeds.
- No `sorry`.
- Both round-trip theorems discharge by `rfl`.

## Out of Scope

- NTT coefficient maps.
- `cf` extended to matrices/vectors of ring elements.
