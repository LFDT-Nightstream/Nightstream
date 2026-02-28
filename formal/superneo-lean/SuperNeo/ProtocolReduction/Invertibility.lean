import SuperNeo.ProtocolReduction.Checks

/-! Invertibility-witness extraction wrappers for protocol reduction claims. -/

namespace SuperNeo

theorem p20ForClaim_invertibilityWitness
  {ctx : PSContext} {claim : PSClaim}
  (hP20 : p20ForClaim ctx claim) :
  ∃ deltaInv : Coeffs, mulRq claim.invDelta deltaInv = oneRq := by
  exact p20InvertibilityWitness_of_assumption ctx.hLowNormInvertibility hP20

theorem p20ForClaim_invertibilityWitness_mulRq_of_goldilocks_operand_assumptions
  {ctx : PSContext} {claim : PSClaim} {aDelta bDelta : Coeffs}
  (hP20 : p20ForClaim ctx claim)
  (hDeltaEq : claim.invDelta = mulRq aDelta bDelta)
  (hA : normInfCoeffs aDelta ≤ Parameters.Goldilocks.B)
  (hB : normInfCoeffs bDelta ≤ Parameters.Goldilocks.B)
  (hRaw : GoldilocksRawNormBoundAssumption)
  (hCollapse : GoldilocksRawCollapseAssumption) :
  ∃ deltaInv : Coeffs, mulRq claim.invDelta deltaInv = oneRq := by
  exact p20InvertibilityWitness_mulRq_of_goldilocks_operand_assumptions_of_assumption
    (hInv := ctx.hLowNormInvertibility)
    (hP20 := hP20)
    (hDeltaEq := hDeltaEq)
    (hA := hA)
    (hB := hB)
    (hRaw := hRaw)
    (hCollapse := hCollapse)

theorem p20ForClaim_invertibilityWitness_mulRq_of_operand_norm_assumptions_via_universal_blockers
  {ctx : PSContext} {claim : PSClaim} {aDelta bDelta : Coeffs}
  {BA BB : Nat}
  (hP20 : p20ForClaim ctx claim)
  (hDeltaEq : claim.invDelta = mulRq aDelta bDelta)
  (hA : normInfCoeffs aDelta ≤ BA)
  (hB : normInfCoeffs bDelta ≤ BB)
  (hMulUniv : schoolbookMulUniversalBound)
  (hAddTri : schoolbookAddTriangleBound)
  (hSubTri : schoolbookSubTriangleBound)
  (hBLt :
    ((D * D) * (BA * BB)) + ((D * D) * (BA * BB)) + ((D * D) * (BA * BB))
      < bInvApprox) :
  ∃ deltaInv : Coeffs, mulRq claim.invDelta deltaInv = oneRq := by
  exact p20InvertibilityWitness_mulRq_of_operand_norm_assumptions_via_universal_blockers_of_assumption
    (hInv := ctx.hLowNormInvertibility)
    (hP20 := hP20)
    (hDeltaEq := hDeltaEq)
    (hA := hA)
    (hB := hB)
    (hMulUniv := hMulUniv)
    (hAddTri := hAddTri)
    (hSubTri := hSubTri)
    (hBLt := hBLt)

theorem p20ForClaim_invertibilityWitness_mulRq_of_operand_norm_assumptions_via_universal_blockers_tight
  {ctx : PSContext} {claim : PSClaim} {aDelta bDelta : Coeffs}
  {BA BB : Nat}
  (hP20 : p20ForClaim ctx claim)
  (hDeltaEq : claim.invDelta = mulRq aDelta bDelta)
  (hA : normInfCoeffs aDelta ≤ BA)
  (hB : normInfCoeffs bDelta ≤ BB)
  (hMulUniv : schoolbookMulUniversalBound)
  (hAddTri : schoolbookAddTriangleBound)
  (hSubTri : schoolbookSubTriangleBound)
  (hBLt :
    (D * (BA * BB)) + (D * (BA * BB)) + (D * (BA * BB))
      < bInvApprox) :
  ∃ deltaInv : Coeffs, mulRq claim.invDelta deltaInv = oneRq := by
  exact p20InvertibilityWitness_mulRq_of_operand_norm_assumptions_via_universal_blockers_tight_of_assumption
    (hInv := ctx.hLowNormInvertibility)
    (hP20 := hP20)
    (hDeltaEq := hDeltaEq)
    (hA := hA)
    (hB := hB)
    (hMulUniv := hMulUniv)
    (hAddTri := hAddTri)
    (hSubTri := hSubTri)
    (hBLt := hBLt)

/--
Triangle-bundle variant of
`p20ForClaim_invertibilityWitness_mulRq_of_operand_norm_assumptions_via_universal_blockers_tight`.
-/
theorem p20ForClaim_invertibilityWitness_mulRq_of_operand_norm_assumptions_via_universal_mul_and_triangles_tight
  {ctx : PSContext} {claim : PSClaim} {aDelta bDelta : Coeffs}
  {BA BB : Nat}
  (hP20 : p20ForClaim ctx claim)
  (hDeltaEq : claim.invDelta = mulRq aDelta bDelta)
  (hA : normInfCoeffs aDelta ≤ BA)
  (hB : normInfCoeffs bDelta ≤ BB)
  (hMulUniv : schoolbookMulUniversalBound)
  (hTri : schoolbookTriangleBounds)
  (hBLt :
    (D * (BA * BB)) + (D * (BA * BB)) + (D * (BA * BB))
      < bInvApprox) :
  ∃ deltaInv : Coeffs, mulRq claim.invDelta deltaInv = oneRq := by
  exact p20InvertibilityWitness_mulRq_of_operand_norm_assumptions_via_universal_mul_and_triangles_tight_of_assumption
    (hInv := ctx.hLowNormInvertibility)
    (hP20 := hP20)
    (hDeltaEq := hDeltaEq)
    (hA := hA)
    (hB := hB)
    (hMulUniv := hMulUniv)
    (hTri := hTri)
    (hBLt := hBLt)

/--
Add-only variant of
`p20ForClaim_invertibilityWitness_mulRq_of_operand_norm_assumptions_via_universal_mul_and_triangles_tight`.
-/
theorem p20ForClaim_invertibilityWitness_mulRq_of_operand_norm_assumptions_via_universal_mul_and_add_tight
  {ctx : PSContext} {claim : PSClaim} {aDelta bDelta : Coeffs}
  {BA BB : Nat}
  (hP20 : p20ForClaim ctx claim)
  (hDeltaEq : claim.invDelta = mulRq aDelta bDelta)
  (hA : normInfCoeffs aDelta ≤ BA)
  (hB : normInfCoeffs bDelta ≤ BB)
  (hMulUniv : schoolbookMulUniversalBound)
  (hAddTri : schoolbookAddTriangleBound)
  (hBLt :
    (D * (BA * BB)) + (D * (BA * BB)) + (D * (BA * BB))
      < bInvApprox) :
  ∃ deltaInv : Coeffs, mulRq claim.invDelta deltaInv = oneRq := by
  exact p20InvertibilityWitness_mulRq_of_operand_norm_assumptions_via_universal_mul_and_add_tight_of_assumption
    (hInv := ctx.hLowNormInvertibility)
    (hP20 := hP20)
    (hDeltaEq := hDeltaEq)
    (hA := hA)
    (hB := hB)
    (hMulUniv := hMulUniv)
    (hAddTri := hAddTri)
    (hBLt := hBLt)

theorem p20ForClaim_invertibilityWitness_mulRq_of_operand_norm_assumptions_via_centeredRep_mul_and_add_tight
  {ctx : PSContext} {claim : PSClaim} {aDelta bDelta : Coeffs}
  {BA BB : Nat}
  (hP20 : p20ForClaim ctx claim)
  (hDeltaEq : claim.invDelta = mulRq aDelta bDelta)
  (hA : normInfCoeffs aDelta ≤ BA)
  (hB : normInfCoeffs bDelta ≤ BB)
  (hMulRep :
    ∀ x y : F, Int.natAbs (centeredRep (x * y))
      ≤ Int.natAbs (centeredRep x) * Int.natAbs (centeredRep y))
  (hAddRep :
    ∀ x y : F, Int.natAbs (centeredRep (x + y))
      ≤ Int.natAbs (centeredRep x) + Int.natAbs (centeredRep y))
  (hBLt :
    (D * (BA * BB)) + (D * (BA * BB)) + (D * (BA * BB))
      < bInvApprox) :
  ∃ deltaInv : Coeffs, mulRq claim.invDelta deltaInv = oneRq := by
  exact p20InvertibilityWitness_mulRq_of_operand_norm_assumptions_via_centeredRep_mul_and_add_tight_of_assumption
    (hInv := ctx.hLowNormInvertibility)
    (hP20 := hP20)
    (hDeltaEq := hDeltaEq)
    (hA := hA)
    (hB := hB)
    (hMulRep := hMulRep)
    (hAddRep := hAddRep)
    (hBLt := hBLt)

theorem p20ForClaim_invertibilityWitness_mulRq_of_operand_norm_assumptions_via_centeredRep_mul_tight
  {ctx : PSContext} {claim : PSClaim} {aDelta bDelta : Coeffs}
  {BA BB : Nat}
  (hP20 : p20ForClaim ctx claim)
  (hDeltaEq : claim.invDelta = mulRq aDelta bDelta)
  (hA : normInfCoeffs aDelta ≤ BA)
  (hB : normInfCoeffs bDelta ≤ BB)
  (hMulRep :
    ∀ x y : F, Int.natAbs (centeredRep (x * y))
      ≤ Int.natAbs (centeredRep x) * Int.natAbs (centeredRep y))
  (hBLt :
    (D * (BA * BB)) + (D * (BA * BB)) + (D * (BA * BB))
      < bInvApprox) :
  ∃ deltaInv : Coeffs, mulRq claim.invDelta deltaInv = oneRq := by
  exact p20ForClaim_invertibilityWitness_mulRq_of_operand_norm_assumptions_via_centeredRep_mul_and_add_tight
    (hP20 := hP20)
    (hDeltaEq := hDeltaEq)
    (hA := hA)
    (hB := hB)
    (hMulRep := hMulRep)
    (hAddRep := centeredRepAddTriangleBound_theorem)
    (hBLt := hBLt)

/-- Bundle wrapper for centered-representation mul/add blockers (tight path). -/
theorem p20ForClaim_invertibilityWitness_mulRq_of_operand_norm_assumptions_via_centeredRepMulAddBounds_tight
  {ctx : PSContext} {claim : PSClaim} {aDelta bDelta : Coeffs}
  {BA BB : Nat}
  (hP20 : p20ForClaim ctx claim)
  (hDeltaEq : claim.invDelta = mulRq aDelta bDelta)
  (hA : normInfCoeffs aDelta ≤ BA)
  (hB : normInfCoeffs bDelta ≤ BB)
  (hRep : centeredRepMulAddBounds)
  (hBLt :
    (D * (BA * BB)) + (D * (BA * BB)) + (D * (BA * BB))
      < bInvApprox) :
  ∃ deltaInv : Coeffs, mulRq claim.invDelta deltaInv = oneRq := by
  exact p20ForClaim_invertibilityWitness_mulRq_of_operand_norm_assumptions_via_centeredRep_mul_and_add_tight
    (hP20 := hP20)
    (hDeltaEq := hDeltaEq)
    (hA := hA)
    (hB := hB)
    (hMulRep := centeredRepMulAddBounds_mul hRep)
    (hAddRep := centeredRepMulAddBounds_add hRep)
    (hBLt := hBLt)

/-- Assumption-free native invertibility witness extraction for `p20ForClaim`. -/
theorem p20ForClaim_invertibilityWitness_mulRq_of_operand_norm_assumptions_native
  {ctx : PSContext} {claim : PSClaim} {aDelta bDelta : Coeffs}
  {BA BB : Nat}
  (hP20 : p20ForClaim ctx claim)
  (hDeltaEq : claim.invDelta = mulRq aDelta bDelta)
  (hA : normInfCoeffs aDelta ≤ BA)
  (hB : normInfCoeffs bDelta ≤ BB)
  (hBLt :
    ((D * D) * (BA * BB)) + ((D * D) * (BA * BB)) + ((D * D) * (BA * BB))
      < bInvApprox) :
  ∃ deltaInv : Coeffs, mulRq claim.invDelta deltaInv = oneRq := by
  exact p20InvertibilityWitness_mulRq_of_operand_norm_assumptions_native_of_assumption
    (hInv := ctx.hLowNormInvertibility)
    (hP20 := hP20)
    (hDeltaEq := hDeltaEq)
    (hA := hA)
    (hB := hB)
    (hBLt := hBLt)

/-- Assumption-free native-tight invertibility witness extraction for `p20ForClaim`. -/
theorem p20ForClaim_invertibilityWitness_mulRq_of_operand_norm_assumptions_native_tight
  {ctx : PSContext} {claim : PSClaim} {aDelta bDelta : Coeffs}
  {BA BB : Nat}
  (hP20 : p20ForClaim ctx claim)
  (hDeltaEq : claim.invDelta = mulRq aDelta bDelta)
  (hA : normInfCoeffs aDelta ≤ BA)
  (hB : normInfCoeffs bDelta ≤ BB)
  (hBLt :
    (D * (BA * BB)) + (D * (BA * BB)) + (D * (BA * BB))
      < bInvApprox) :
  ∃ deltaInv : Coeffs, mulRq claim.invDelta deltaInv = oneRq := by
  exact p20InvertibilityWitness_mulRq_of_operand_norm_assumptions_native_tight_of_assumption
    (hInv := ctx.hLowNormInvertibility)
    (hP20 := hP20)
    (hDeltaEq := hDeltaEq)
    (hA := hA)
    (hB := hB)
    (hBLt := hBLt)

theorem p20ForClaim_invertibilityWitness_mulRq_of_operand_norm_assumptions_via_universal_blockers_and_raw
  {ctx : PSContext} {claim : PSClaim} {aDelta bDelta : Coeffs}
  {BA BB BRaw : Nat}
  (hP20 : p20ForClaim ctx claim)
  (hDeltaEq : claim.invDelta = mulRq aDelta bDelta)
  (hA : normInfCoeffs aDelta ≤ BA)
  (hB : normInfCoeffs bDelta ≤ BB)
  (hRawFromOperands : mulRqRawNormBoundFromOperands BA BB BRaw)
  (hAddTri : schoolbookAddTriangleBound)
  (hSubTri : schoolbookSubTriangleBound)
  (hBLt : BRaw + BRaw + BRaw < bInvApprox) :
  ∃ deltaInv : Coeffs, mulRq claim.invDelta deltaInv = oneRq := by
  exact p20InvertibilityWitness_mulRq_of_operand_norm_assumptions_via_universal_blockers_and_raw_of_assumption
    (hInv := ctx.hLowNormInvertibility)
    (hP20 := hP20)
    (hDeltaEq := hDeltaEq)
    (hA := hA)
    (hB := hB)
    (hRawFromOperands := hRawFromOperands)
    (hAddTri := hAddTri)
    (hSubTri := hSubTri)
    (hBLt := hBLt)

theorem p20ForClaim_invertibilityWitness_mulRq_of_operand_norm_assumptions_via_universal_blockers_and_rawCoeff
  {ctx : PSContext} {claim : PSClaim} {aDelta bDelta : Coeffs}
  {BA BB BRaw : Nat}
  (hP20 : p20ForClaim ctx claim)
  (hDeltaEq : claim.invDelta = mulRq aDelta bDelta)
  (hA : normInfCoeffs aDelta ≤ BA)
  (hB : normInfCoeffs bDelta ≤ BB)
  (hRawCoeffFromOperands : mulRqRawCoeffBoundFromOperands BA BB BRaw)
  (hAddTri : schoolbookAddTriangleBound)
  (hSubTri : schoolbookSubTriangleBound)
  (hBLt : BRaw + BRaw + BRaw < bInvApprox) :
  ∃ deltaInv : Coeffs, mulRq claim.invDelta deltaInv = oneRq := by
  exact p20ForClaim_invertibilityWitness_mulRq_of_operand_norm_assumptions_via_universal_blockers_and_raw
    (hP20 := hP20)
    (hDeltaEq := hDeltaEq)
    (hA := hA)
    (hB := hB)
    (hRawFromOperands := mulRqRawNormBoundFromOperands_of_rawCoeff hRawCoeffFromOperands)
    (hAddTri := hAddTri)
    (hSubTri := hSubTri)
    (hBLt := hBLt)

theorem p20ForClaim_invertibilityWitness_mulRq_of_operand_norm_assumptions_via_universal_mul_and_add_and_raw
  {ctx : PSContext} {claim : PSClaim} {aDelta bDelta : Coeffs}
  {BA BB BRaw : Nat}
  (hP20 : p20ForClaim ctx claim)
  (hDeltaEq : claim.invDelta = mulRq aDelta bDelta)
  (hA : normInfCoeffs aDelta ≤ BA)
  (hB : normInfCoeffs bDelta ≤ BB)
  (hRawFromOperands : mulRqRawNormBoundFromOperands BA BB BRaw)
  (hAddTri : schoolbookAddTriangleBound)
  (hBLt : BRaw + BRaw + BRaw < bInvApprox) :
  ∃ deltaInv : Coeffs, mulRq claim.invDelta deltaInv = oneRq := by
  exact p20InvertibilityWitness_mulRq_of_operand_norm_assumptions_via_universal_mul_and_add_and_raw_of_assumption
    (hInv := ctx.hLowNormInvertibility)
    (hP20 := hP20)
    (hDeltaEq := hDeltaEq)
    (hA := hA)
    (hB := hB)
    (hRawFromOperands := hRawFromOperands)
    (hAddTri := hAddTri)
    (hBLt := hBLt)

theorem p20ForClaim_invertibilityWitness_mulRq_of_operand_norm_assumptions_via_universal_mul_and_add_and_rawCoeff
  {ctx : PSContext} {claim : PSClaim} {aDelta bDelta : Coeffs}
  {BA BB BRaw : Nat}
  (hP20 : p20ForClaim ctx claim)
  (hDeltaEq : claim.invDelta = mulRq aDelta bDelta)
  (hA : normInfCoeffs aDelta ≤ BA)
  (hB : normInfCoeffs bDelta ≤ BB)
  (hRawCoeffFromOperands : mulRqRawCoeffBoundFromOperands BA BB BRaw)
  (hAddTri : schoolbookAddTriangleBound)
  (hBLt : BRaw + BRaw + BRaw < bInvApprox) :
  ∃ deltaInv : Coeffs, mulRq claim.invDelta deltaInv = oneRq := by
  exact p20ForClaim_invertibilityWitness_mulRq_of_operand_norm_assumptions_via_universal_mul_and_add_and_raw
    (hP20 := hP20)
    (hDeltaEq := hDeltaEq)
    (hA := hA)
    (hB := hB)
    (hRawFromOperands := mulRqRawNormBoundFromOperands_of_rawCoeff hRawCoeffFromOperands)
    (hAddTri := hAddTri)
    (hBLt := hBLt)

theorem p20ForClaim_invertibilityWitness_mulRq_of_operand_norm_assumptions_via_centeredRep_mul_and_add_and_raw
  {ctx : PSContext} {claim : PSClaim} {aDelta bDelta : Coeffs}
  {BA BB BRaw : Nat}
  (hP20 : p20ForClaim ctx claim)
  (hDeltaEq : claim.invDelta = mulRq aDelta bDelta)
  (hA : normInfCoeffs aDelta ≤ BA)
  (hB : normInfCoeffs bDelta ≤ BB)
  (hRawFromOperands : mulRqRawNormBoundFromOperands BA BB BRaw)
  (hAddRep :
    ∀ x y : F, Int.natAbs (centeredRep (x + y))
      ≤ Int.natAbs (centeredRep x) + Int.natAbs (centeredRep y))
  (hBLt : BRaw + BRaw + BRaw < bInvApprox) :
  ∃ deltaInv : Coeffs, mulRq claim.invDelta deltaInv = oneRq := by
  exact p20InvertibilityWitness_mulRq_of_operand_norm_assumptions_via_centeredRep_mul_and_add_and_raw_of_assumption
    (hInv := ctx.hLowNormInvertibility)
    (hP20 := hP20)
    (hDeltaEq := hDeltaEq)
    (hA := hA)
    (hB := hB)
    (hRawFromOperands := hRawFromOperands)
    (hAddRep := hAddRep)
    (hBLt := hBLt)

theorem p20ForClaim_invertibilityWitness_mulRq_of_operand_norm_assumptions_via_centeredRep_mul_and_add_and_rawCoeff
  {ctx : PSContext} {claim : PSClaim} {aDelta bDelta : Coeffs}
  {BA BB BRaw : Nat}
  (hP20 : p20ForClaim ctx claim)
  (hDeltaEq : claim.invDelta = mulRq aDelta bDelta)
  (hA : normInfCoeffs aDelta ≤ BA)
  (hB : normInfCoeffs bDelta ≤ BB)
  (hRawCoeffFromOperands : mulRqRawCoeffBoundFromOperands BA BB BRaw)
  (hAddRep :
    ∀ x y : F, Int.natAbs (centeredRep (x + y))
      ≤ Int.natAbs (centeredRep x) + Int.natAbs (centeredRep y))
  (hBLt : BRaw + BRaw + BRaw < bInvApprox) :
  ∃ deltaInv : Coeffs, mulRq claim.invDelta deltaInv = oneRq := by
  exact p20ForClaim_invertibilityWitness_mulRq_of_operand_norm_assumptions_via_centeredRep_mul_and_add_and_raw
    (hP20 := hP20)
    (hDeltaEq := hDeltaEq)
    (hA := hA)
    (hB := hB)
    (hRawFromOperands := mulRqRawNormBoundFromOperands_of_rawCoeff hRawCoeffFromOperands)
    (hAddRep := hAddRep)
    (hBLt := hBLt)

/-- Bundle wrapper for centered-representation blockers (raw-bound path). -/
theorem p20ForClaim_invertibilityWitness_mulRq_of_operand_norm_assumptions_via_centeredRepMulAddBounds_and_raw
  {ctx : PSContext} {claim : PSClaim} {aDelta bDelta : Coeffs}
  {BA BB BRaw : Nat}
  (hP20 : p20ForClaim ctx claim)
  (hDeltaEq : claim.invDelta = mulRq aDelta bDelta)
  (hA : normInfCoeffs aDelta ≤ BA)
  (hB : normInfCoeffs bDelta ≤ BB)
  (hRawFromOperands : mulRqRawNormBoundFromOperands BA BB BRaw)
  (hRep : centeredRepMulAddBounds)
  (hBLt : BRaw + BRaw + BRaw < bInvApprox) :
  ∃ deltaInv : Coeffs, mulRq claim.invDelta deltaInv = oneRq := by
  exact p20ForClaim_invertibilityWitness_mulRq_of_operand_norm_assumptions_via_centeredRep_mul_and_add_and_raw
    (hP20 := hP20)
    (hDeltaEq := hDeltaEq)
    (hA := hA)
    (hB := hB)
    (hRawFromOperands := hRawFromOperands)
    (hAddRep := centeredRepMulAddBounds_add hRep)
    (hBLt := hBLt)

/-- Bundle wrapper for centered-representation blockers (raw-coeff path). -/
theorem p20ForClaim_invertibilityWitness_mulRq_of_operand_norm_assumptions_via_centeredRepMulAddBounds_and_rawCoeff
  {ctx : PSContext} {claim : PSClaim} {aDelta bDelta : Coeffs}
  {BA BB BRaw : Nat}
  (hP20 : p20ForClaim ctx claim)
  (hDeltaEq : claim.invDelta = mulRq aDelta bDelta)
  (hA : normInfCoeffs aDelta ≤ BA)
  (hB : normInfCoeffs bDelta ≤ BB)
  (hRawCoeffFromOperands : mulRqRawCoeffBoundFromOperands BA BB BRaw)
  (hRep : centeredRepMulAddBounds)
  (hBLt : BRaw + BRaw + BRaw < bInvApprox) :
  ∃ deltaInv : Coeffs, mulRq claim.invDelta deltaInv = oneRq := by
  exact p20ForClaim_invertibilityWitness_mulRq_of_operand_norm_assumptions_via_centeredRepMulAddBounds_and_raw
    (hP20 := hP20)
    (hDeltaEq := hDeltaEq)
    (hA := hA)
    (hB := hB)
    (hRawFromOperands := mulRqRawNormBoundFromOperands_of_rawCoeff hRawCoeffFromOperands)
    (hRep := hRep)
    (hBLt := hBLt)

theorem p20ForClaim_invertibilityWitness_mulRq_of_goldilocks_operand_assumptions_inRange
  {ctx : PSContext} {claim : PSClaim} {aDelta bDelta : Coeffs}
  (hP20 : p20ForClaim ctx claim)
  (hDeltaEq : claim.invDelta = mulRq aDelta bDelta)
  (hA : normInfCoeffs aDelta ≤ Parameters.Goldilocks.B)
  (hB : normInfCoeffs bDelta ≤ Parameters.Goldilocks.B)
  (hRawInRange : GoldilocksRawInRangeBoundAssumption)
  (hCollapse : GoldilocksRawCollapseAssumption) :
  ∃ deltaInv : Coeffs, mulRq claim.invDelta deltaInv = oneRq := by
  exact p20ForClaim_invertibilityWitness_mulRq_of_goldilocks_operand_assumptions
    (hP20 := hP20)
    (hDeltaEq := hDeltaEq)
    (hA := hA)
    (hB := hB)
    (hRaw := goldilocksRawNormBoundAssumption_of_inRange hRawInRange)
    (hCollapse := hCollapse)

theorem p20ForClaim_invertibilityWitness_mulRq_of_goldilocks_operand_fieldOp_assumptions
  {ctx : PSContext} {claim : PSClaim} {aDelta bDelta : Coeffs}
  (hP20 : p20ForClaim ctx claim)
  (hDeltaEq : claim.invDelta = mulRq aDelta bDelta)
  (hA : normInfCoeffs aDelta ≤ Parameters.Goldilocks.B)
  (hB : normInfCoeffs bDelta ≤ Parameters.Goldilocks.B)
  (hRaw : GoldilocksRawNormBoundAssumption)
  (hFieldOps : GoldilocksFieldOpCollapseAssumption) :
  ∃ deltaInv : Coeffs, mulRq claim.invDelta deltaInv = oneRq := by
  exact p20ForClaim_invertibilityWitness_mulRq_of_goldilocks_operand_assumptions
    (hP20 := hP20)
    (hDeltaEq := hDeltaEq)
    (hA := hA)
    (hB := hB)
    (hRaw := hRaw)
    (hCollapse := goldilocksRawCollapseAssumption_of_fieldOp hFieldOps)

theorem p20ForClaim_invertibilityWitness_mulRq_of_goldilocks_operand_fieldOp_assumptions_inRange
  {ctx : PSContext} {claim : PSClaim} {aDelta bDelta : Coeffs}
  (hP20 : p20ForClaim ctx claim)
  (hDeltaEq : claim.invDelta = mulRq aDelta bDelta)
  (hA : normInfCoeffs aDelta ≤ Parameters.Goldilocks.B)
  (hB : normInfCoeffs bDelta ≤ Parameters.Goldilocks.B)
  (hRawInRange : GoldilocksRawInRangeBoundAssumption)
  (hFieldOps : GoldilocksFieldOpCollapseAssumption) :
  ∃ deltaInv : Coeffs, mulRq claim.invDelta deltaInv = oneRq := by
  exact p20ForClaim_invertibilityWitness_mulRq_of_goldilocks_operand_fieldOp_assumptions
    (hP20 := hP20)
    (hDeltaEq := hDeltaEq)
    (hA := hA)
    (hB := hB)
    (hRaw := goldilocksRawNormBoundAssumption_of_inRange hRawInRange)
    (hFieldOps := hFieldOps)

theorem p20ForClaim_invertibilityWitness_mulRq_of_goldilocks_operand_rawCoeff_assumptions
  {ctx : PSContext} {claim : PSClaim} {aDelta bDelta : Coeffs}
  (hP20 : p20ForClaim ctx claim)
  (hDeltaEq : claim.invDelta = mulRq aDelta bDelta)
  (hA : normInfCoeffs aDelta ≤ Parameters.Goldilocks.B)
  (hB : normInfCoeffs bDelta ≤ Parameters.Goldilocks.B)
  (hRawCoeff : GoldilocksRawCoeffBoundAssumption)
  (hCollapse : GoldilocksRawCollapseAssumption) :
  ∃ deltaInv : Coeffs, mulRq claim.invDelta deltaInv = oneRq := by
  exact p20ForClaim_invertibilityWitness_mulRq_of_goldilocks_operand_assumptions
    (hP20 := hP20)
    (hDeltaEq := hDeltaEq)
    (hA := hA)
    (hB := hB)
    (hRaw := goldilocksRawNormBoundAssumption_of_rawCoeff hRawCoeff)
    (hCollapse := hCollapse)

theorem p20ForClaim_invertibilityWitness_mulRq_of_goldilocks_operand_rawCoeff_fieldOp_assumptions
  {ctx : PSContext} {claim : PSClaim} {aDelta bDelta : Coeffs}
  (hP20 : p20ForClaim ctx claim)
  (hDeltaEq : claim.invDelta = mulRq aDelta bDelta)
  (hA : normInfCoeffs aDelta ≤ Parameters.Goldilocks.B)
  (hB : normInfCoeffs bDelta ≤ Parameters.Goldilocks.B)
  (hRawCoeff : GoldilocksRawCoeffBoundAssumption)
  (hFieldOps : GoldilocksFieldOpCollapseAssumption) :
  ∃ deltaInv : Coeffs, mulRq claim.invDelta deltaInv = oneRq := by
  exact p20ForClaim_invertibilityWitness_mulRq_of_goldilocks_operand_fieldOp_assumptions
    (hP20 := hP20)
    (hDeltaEq := hDeltaEq)
    (hA := hA)
    (hB := hB)
    (hRaw := goldilocksRawNormBoundAssumption_of_rawCoeff hRawCoeff)
    (hFieldOps := hFieldOps)


end SuperNeo
