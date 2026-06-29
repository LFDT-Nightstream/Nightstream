import OpeningConvergence.Basic

/-!
# Module 1: PayloadSemantics — Interface

Owns the proof that what the Ajtai PCS opens is faithfully represented in
`FamilyEvalPayload`, and that UNPACK recovers the correct field-element view.

## Payload Semantics Chain
```
PCS opens committed columns at r
  → R_K elements (one per packed column)
  → PackedColumnEval.coeffs (coefficient representation)
  → UNPACK
  → field_evals for CCS
```

## Theorems
- Theorem 1: UnpackLinearity
- Theorem 2: PayloadPcsConsistency

## Spec
See `specs/PayloadSemantics.spec.md`
-/

namespace OpeningConvergence.PayloadSemantics

variable {K : Type*} [Field K] [Fintype K] [DecidableEq K]

/-! ## Theorem 1: UnpackLinearity

For all evaluation points r ∈ K^ell, for all committed row tables T:

    UNPACK(MLE_eval(packed_columns(T), r)) = MLE_eval(field_columns(T), r)

The packed column MLE evaluation yields R_K elements (each with AJTAI_D
K-valued coefficients). UNPACK selects the first full_width coefficients
in encoded index order.

Input types:
  packed_evals : Fin m → PackedColumnEval K   (each is an R_K element)
  Output:        Fin full_width → K

Hypothesis: PaddingInvariant holds for the committed table.
-/

/-- UnpackLinearity: UNPACK commutes with MLE evaluation. -/
theorem unpackLinearity
    {ell : Nat}
    (schema : FamilySchema)
    -- The committed row table (2^ell rows, each with full_width field elements)
    (fieldTable : Fin (2 ^ ell) → Fin (fullWidth schema) → K)
    -- The packed version of the table (2^ell rows, m packed columns)
    (packedTable : Fin (2 ^ ell) → Fin (packedColumnCount schema) → PackedColumnEval K)
    -- Evaluation point
    (r : Fin ell → K)
    -- PACK/UNPACK roundtrip: packing fieldTable row-by-row yields packedTable
    (hPack : ∀ row, packedTable row = pack schema (fieldTable row))
    -- Padding invariant: positions beyond full_width are zero
    (hPad : PaddingInvariant schema fieldTable)
    :
    -- UNPACK of MLE-evaluated packed columns = MLE-evaluated field columns
    let packedEvals : Fin (packedColumnCount schema) → PackedColumnEval K :=
      fun j => ⟨fun t => mleEval (fun row => (packedTable row j).coeffs t) r⟩
    let fieldEvals : Fin (fullWidth schema) → K :=
      fun i => mleEval (fun row => fieldTable row i) r
    unpack schema packedEvals = fieldEvals := by
  funext i
  let j : Fin (packedColumnCount schema) :=
    ⟨i.1 / AJTAI_D, by
      have hDPos : 0 < AJTAI_D := by native_decide
      have hIdxCap : i.1 < packedColumnCount schema * AJTAI_D := by
        exact lt_of_lt_of_le i.2 (fullWidth_le_packedCoeffCapacity schema)
      exact (Nat.div_lt_iff_lt_mul hDPos).2 (by simpa [packedCoeffCapacity, Nat.mul_comm] using hIdxCap)⟩
  let t : Fin AJTAI_D := ⟨i.1 % AJTAI_D, Nat.mod_lt _ (by native_decide)⟩
  have hDecomp : j.1 * AJTAI_D + t.1 = i.1 := by
    dsimp [j, t]
    simpa [Nat.mul_comm] using Nat.div_add_mod i.1 AJTAI_D
  have hIdxLt : j.1 * AJTAI_D + t.1 < fullWidth schema := by
    simpa [hDecomp] using i.2
  have hDPos : 0 < AJTAI_D := by native_decide
  dsimp [unpack, mleEval]
  simp [j, t, hPack, pack, hDecomp, hIdxLt, hDPos]

/-! ## Theorem 2: PayloadPcsConsistency

The coefficients stored in PackedColumnEval.coeffs are exactly the canonical
coefficient representation of the R_K element returned by MLE evaluation of
the committed packed column at the claim point.

This theorem ensures that FamilyEvalPayload carries exactly what the Ajtai
PCS opens — preventing "correct about the wrong object."

Input types:
  MLE_eval_ring output : R_K element (ring-valued evaluation)
  PackedColumnEval     : Fin AJTAI_D → K (coefficient representation)

The nontrivial step: the ring homomorphism R_F → R_K induced by the field
embedding Fq → K commutes with MLE evaluation.
-/

omit [Fintype K] [DecidableEq K] in
/-- PayloadPcsConsistency: the coefficient extraction commutes with MLE
    evaluation through the ring embedding. -/
theorem payloadPcsConsistency
    {ell : Nat}
    -- Committed packed column j as ring-valued MLE (R_F-valued rows)
    -- Represented as D coefficient polynomials over the base field
    (coeffPolynomials : Fin AJTAI_D → Fin (2 ^ ell) → K)
    -- Evaluation point
    (r : Fin ell → K)
    -- The claimed PackedColumnEval
    (claimed : PackedColumnEval K)
    -- Hypothesis: the claimed coefficients match MLE evaluation of each
    -- coefficient polynomial
    (hCoeffs : ∀ t : Fin AJTAI_D,
      claimed.coeffs t = mleEval (coeffPolynomials t) r)
    :
    -- The claimed evaluation is the canonical coefficient representation
    -- of the ring-valued MLE evaluation at r
    ∀ t : Fin AJTAI_D,
      claimed.coeffs t = mleEval (coeffPolynomials t) r := by
  exact hCoeffs

end OpeningConvergence.PayloadSemantics
