# PayloadSemantics Specification

## Purpose

Prove that the object carried in `FamilyEvalPayload` is exactly the object
the Ajtai PCS opens, and that UNPACK recovers the correct field-element view
for CCS verification.

This is the highest-priority module because if the payload representation is
wrong, all later proofs are correct about the wrong statement.

## Target Formulas

### Theorem 1: UnpackLinearity

For all evaluation points `r ∈ K^ell`, for all committed row tables `T`:

```
UNPACK(MLE_eval(packed_columns(T), r)) = MLE_eval(field_columns(T), r)
```

Hypothesis: positions `full_width..D*m` in each committed row are zero
(PaddingInvariant, enforced by `encode_vector_for_full_width`).

Proof reduces to: `UNPACK ∘ PACK = id` on the `full_width`-dimensional
subspace, plus the padding invariant.

### Theorem 2: PayloadPcsConsistency

For committed packed column `j` with ring-valued MLE `F_j : {0,1}^ell → R_F`:

```
PackedColumnEval_encode(MLE_eval_ring(F_j, r)) = payload.column_evals[j]
```

The R_K element has a unique coefficient representation in `K[X]/(X^D + 1)`.
The nontrivial step: the ring homomorphism `R_F → R_K` induced by `Fq → K`
commutes with MLE evaluation (because both are K-linear after lifting).

## Explicit Type Signatures

| Lean symbol | Type | Lives in |
|---|---|---|
| `packed_evals` | `Fin m → PackedColumnEval K` | `(R_K)^m` in coefficient view |
| `field_evals` | `Fin full_width → K` | `K^full_width` |
| `PACK` | `(Fin full_width → K) → (Fin m → PackedColumnEval K)` | linear map |
| `UNPACK` | `(Fin m → PackedColumnEval K) → (Fin full_width → K)` | linear map |
| `mleEval` | `(Fin (2^ell) → K) → (Fin ell → K) → K` | MLE evaluation |
| `coeffPolynomials` | `Fin AJTAI_D → (Fin (2^ell) → K)` | D coefficient polys |

## R_K vs K Boundary

- **R_K objects**: `PackedColumnEval.coeffs` (D=54 K coefficients representing one R_K element)
- **K objects**: individual coefficients, MLE evaluation of coefficient polynomials, scalarized values
- **Linearization boundary**: none in this module — that happens in Module 2

## Paper Anchors

- SuperNeo Section 4: Ajtai commitment scheme over `R_F`
- SuperNeo Definition 3: CE evaluation relation
- Jolt Section 4.2: packed-column MLE evaluation

## Module Mapping

| Existing module | Import | What it provides |
|---|---|---|
| `SuperNeo.MLE` | `mleEval`, linearity proofs | MLE evaluation |
| `SuperNeo.EqPoly` | `eqPoly`, Boolean-cube properties | Kronecker delta |
| `SuperNeo.Field` | `F`, field axioms | Base field |

## Contract Surface

| Lean symbol | Kind | Role | Guarantee |
|---|---|---|---|
| `unpackLinearity` | Theorem | P0 | UNPACK ∘ MLE_eval(packed) = MLE_eval(field) |
| `payloadPcsConsistency` | Theorem | P0 | Coefficient extraction commutes with MLE eval through ring embedding |
| `PaddingInvariant` | Hypothesis | Explicit | Positions beyond full_width are zero |
| `pack` / `unpack` | Definition | Foundation | PACK/UNPACK linear maps over K |
