import SuperNeo.InvertibilityAxioms.Window

/-! Invertibility witness extraction wrappers from window/precondition assumptions. -/

namespace SuperNeo

theorem invertible_of_norm_bounds_of_assumption
  (hInv : LowNormInvertibilityAssumption)
  {a : Coeffs}
  (hPos : 0 < normInfCoeffs a)
  (hLt : normInfCoeffs a < bInvApprox) :
  ∃ b : Coeffs, mulRq a b = oneRq := by
  exact hInv a hPos hLt

theorem invertible_of_withinInvertibilityWindow_of_assumption
  (hInv : LowNormInvertibilityAssumption)
  {a : Coeffs}
  (hWin : withinInvertibilityWindow a = true) :
  ∃ b : Coeffs, mulRq a b = oneRq := by
  rcases withinInvertibilityWindow_sound hWin with ⟨hPos, hLt⟩
  exact invertible_of_norm_bounds_of_assumption hInv hPos hLt

theorem invertible_of_norm_le_of_lt_of_assumption
  (hInv : LowNormInvertibilityAssumption)
  {a : Coeffs} {B : Nat}
  (hPos : 0 < normInfCoeffs a)
  (hLe : normInfCoeffs a ≤ B)
  (hBLt : B < bInvApprox) :
  ∃ b : Coeffs, mulRq a b = oneRq := by
  exact invertible_of_withinInvertibilityWindow_of_assumption hInv
    (withinInvertibilityWindow_of_norm_le_of_lt hPos hLe hBLt)

theorem invertible_of_allChallenge_nonzero_of_assumption
  (hInv : LowNormInvertibilityAssumption)
  {a : Coeffs}
  (hAll : AllChallengeCoeffs a)
  (hPos : 0 < normInfCoeffs a) :
  ∃ b : Coeffs, mulRq a b = oneRq := by
  exact invertible_of_withinInvertibilityWindow_of_assumption hInv
    (withinInvertibilityWindow_of_allChallenge hAll hPos)

theorem invertible_of_allChallenge_sub_nonzero_of_assumption
  (hInv : LowNormInvertibilityAssumption)
  {a b : Coeffs}
  (hSize : a.size = b.size)
  (hAllA : AllChallengeCoeffs a)
  (hAllB : AllChallengeCoeffs b)
  (hPos : 0 < normInfCoeffs (coeffSub a b)) :
  ∃ c : Coeffs, mulRq (coeffSub a b) c = oneRq := by
  exact invertible_of_withinInvertibilityWindow_of_assumption hInv
    (withinInvertibilityWindow_of_allChallenge_sub hSize hAllA hAllB hPos)

theorem invertible_mulRq_of_rawCoeffInRangeBound_of_assumption
  (hInv : LowNormInvertibilityAssumption)
  {a b : Coeffs} {BRaw B : Nat}
  (hPos : 0 < normInfCoeffs (mulRq a b))
  (hRawInRange : ∀ t, t < 2 * D - 1 → normInfF (mulRqRawCoeffSpec a b t) ≤ BRaw)
  (hAddSub : ∀ x y z, normInfF x ≤ BRaw → normInfF y ≤ BRaw → normInfF z ≤ BRaw → normInfF (x + y - z) ≤ B)
  (hSub : ∀ x y, normInfF x ≤ BRaw → normInfF y ≤ BRaw → normInfF (x - y) ≤ B)
  (hBLt : B < bInvApprox) :
  ∃ c : Coeffs, mulRq (mulRq a b) c = oneRq := by
  exact invertible_of_withinInvertibilityWindow_of_assumption hInv
    (withinInvertibilityWindow_mulRq_of_rawCoeffInRangeBound
      (a := a) (b := b) (BRaw := BRaw) (B := B)
      hPos hRawInRange hAddSub hSub hBLt)

theorem invertible_mulRq_of_rawCoeffsNorm_of_assumption
  (hInv : LowNormInvertibilityAssumption)
  {a b : Coeffs} {BRaw B : Nat}
  (hPos : 0 < normInfCoeffs (mulRq a b))
  (hRawCoeffs : normInfCoeffs (mulRqRawCoeffs a b) ≤ BRaw)
  (hAddSub : ∀ x y z, normInfF x ≤ BRaw → normInfF y ≤ BRaw → normInfF z ≤ BRaw → normInfF (x + y - z) ≤ B)
  (hSub : ∀ x y, normInfF x ≤ BRaw → normInfF y ≤ BRaw → normInfF (x - y) ≤ B)
  (hBLt : B < bInvApprox) :
  ∃ c : Coeffs, mulRq (mulRq a b) c = oneRq := by
  exact invertible_of_withinInvertibilityWindow_of_assumption hInv
    (withinInvertibilityWindow_mulRq_of_rawCoeffsNorm
      (a := a) (b := b) (BRaw := BRaw) (B := B)
      hPos hRawCoeffs hAddSub hSub hBLt)

theorem invertible_mulRq_of_norm_bounds_via_rawCoeffInRange_of_assumption
  (hInv : LowNormInvertibilityAssumption)
  {a b : Coeffs} {BA BB BRaw B : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hPos : 0 < normInfCoeffs (mulRq a b))
  (hRawFromNormInRange :
    ∀ t, t < 2 * D - 1 →
      normInfCoeffs a ≤ BA →
      normInfCoeffs b ≤ BB →
      normInfF (mulRqRawCoeffSpec a b t) ≤ BRaw)
  (hAddSub : ∀ x y z, normInfF x ≤ BRaw → normInfF y ≤ BRaw → normInfF z ≤ BRaw → normInfF (x + y - z) ≤ B)
  (hSub : ∀ x y, normInfF x ≤ BRaw → normInfF y ≤ BRaw → normInfF (x - y) ≤ B)
  (hBLt : B < bInvApprox) :
  ∃ c : Coeffs, mulRq (mulRq a b) c = oneRq := by
  exact invertible_of_withinInvertibilityWindow_of_assumption hInv
    (withinInvertibilityWindow_mulRq_of_norm_bounds_via_rawCoeffInRange
      (a := a) (b := b) (BA := BA) (BB := BB) (BRaw := BRaw) (B := B)
      hA hB hPos hRawFromNormInRange hAddSub hSub hBLt)

theorem invertible_mulRq_of_norm_bounds_via_rawCoeffsNorm_of_assumption
  (hInv : LowNormInvertibilityAssumption)
  {a b : Coeffs} {BA BB BRaw B : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hPos : 0 < normInfCoeffs (mulRq a b))
  (hRawCoeffsFromNorm :
    normInfCoeffs a ≤ BA →
      normInfCoeffs b ≤ BB →
      normInfCoeffs (mulRqRawCoeffs a b) ≤ BRaw)
  (hAddSub : ∀ x y z, normInfF x ≤ BRaw → normInfF y ≤ BRaw → normInfF z ≤ BRaw → normInfF (x + y - z) ≤ B)
  (hSub : ∀ x y, normInfF x ≤ BRaw → normInfF y ≤ BRaw → normInfF (x - y) ≤ B)
  (hBLt : B < bInvApprox) :
  ∃ c : Coeffs, mulRq (mulRq a b) c = oneRq := by
  exact invertible_of_withinInvertibilityWindow_of_assumption hInv
    (withinInvertibilityWindow_mulRq_of_norm_bounds_via_rawCoeffsNorm
      (a := a) (b := b) (BA := BA) (BB := BB) (BRaw := BRaw) (B := B)
      hA hB hPos hRawCoeffsFromNorm hAddSub hSub hBLt)

theorem invertible_mulRq_of_operand_norm_assumptions_of_assumption
  (hInv : LowNormInvertibilityAssumption)
  {a b : Coeffs} {BA BB BRaw B : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hPos : 0 < normInfCoeffs (mulRq a b))
  (hRawFromOperands : mulRqRawNormBoundFromOperands BA BB BRaw)
  (hAddSub : rawAddSubCollapseBound BRaw B)
  (hSub : rawSubCollapseBound BRaw B)
  (hBLt : B < bInvApprox) :
  ∃ c : Coeffs, mulRq (mulRq a b) c = oneRq := by
  exact invertible_of_withinInvertibilityWindow_of_assumption hInv
    (withinInvertibilityWindow_mulRq_of_operand_norm_assumptions
      (a := a) (b := b) (BA := BA) (BB := BB) (BRaw := BRaw) (B := B)
      hA hB hPos hRawFromOperands hAddSub hSub hBLt)

theorem invertible_mulRq_of_operand_norm_assumptions_inRange_of_assumption
  (hInv : LowNormInvertibilityAssumption)
  {a b : Coeffs} {BA BB BRaw B : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hPos : 0 < normInfCoeffs (mulRq a b))
  (hRawInRangeFromOperands : mulRqRawInRangeBoundFromOperands BA BB BRaw)
  (hAddSub : rawAddSubCollapseBound BRaw B)
  (hSub : rawSubCollapseBound BRaw B)
  (hBLt : B < bInvApprox) :
  ∃ c : Coeffs, mulRq (mulRq a b) c = oneRq := by
  exact invertible_of_withinInvertibilityWindow_of_assumption hInv
    (withinInvertibilityWindow_mulRq_of_operand_norm_assumptions_inRange
      (a := a) (b := b) (BA := BA) (BB := BB) (BRaw := BRaw) (B := B)
      hA hB hPos hRawInRangeFromOperands hAddSub hSub hBLt)

/--
Field-op collapse variant of invertibility extraction for `mulRq a b`,
keeping the trusted boundary explicit (`LowNormInvertibilityAssumption`).
-/
theorem invertible_mulRq_of_operand_norm_assumptions_fieldOp_of_assumption
  (hInv : LowNormInvertibilityAssumption)
  {a b : Coeffs} {BA BB B : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hPos : 0 < normInfCoeffs (mulRq a b))
  (hRawFromOperands : mulRqRawNormBoundFromOperands BA BB B)
  (hOps : rawFieldOpCollapseBound B B)
  (hBLt : B < bInvApprox) :
  ∃ c : Coeffs, mulRq (mulRq a b) c = oneRq := by
  rcases hOps with ⟨hAdd, hSub⟩
  exact invertible_mulRq_of_operand_norm_assumptions_of_assumption
    (hInv := hInv)
    (a := a) (b := b)
    (BA := BA) (BB := BB) (BRaw := B) (B := B)
    hA hB hPos hRawFromOperands
    (rawAddSubCollapseBound_of_add_and_sub_same (BRaw := B) hAdd hSub)
    hSub
    hBLt

theorem invertible_mulRq_of_operand_norm_assumptions_fieldOp_inRange_of_assumption
  (hInv : LowNormInvertibilityAssumption)
  {a b : Coeffs} {BA BB B : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hPos : 0 < normInfCoeffs (mulRq a b))
  (hRawInRangeFromOperands : mulRqRawInRangeBoundFromOperands BA BB B)
  (hOps : rawFieldOpCollapseBound B B)
  (hBLt : B < bInvApprox) :
  ∃ c : Coeffs, mulRq (mulRq a b) c = oneRq := by
  rcases hOps with ⟨hAdd, hSub⟩
  exact invertible_mulRq_of_operand_norm_assumptions_inRange_of_assumption
    (hInv := hInv)
    (a := a) (b := b)
    (BA := BA) (BB := BB) (BRaw := B) (B := B)
    hA hB hPos hRawInRangeFromOperands
    (rawAddSubCollapseBound_of_add_and_sub_same (BRaw := B) hAdd hSub)
    hSub
    hBLt

theorem invertible_mulRq_of_operand_norm_assumptions_rawCoeff_of_assumption
  (hInv : LowNormInvertibilityAssumption)
  {a b : Coeffs} {BA BB BRaw B : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hPos : 0 < normInfCoeffs (mulRq a b))
  (hRawCoeffFromOperands : mulRqRawCoeffBoundFromOperands BA BB BRaw)
  (hAddSub : rawAddSubCollapseBound BRaw B)
  (hSub : rawSubCollapseBound BRaw B)
  (hBLt : B < bInvApprox) :
  ∃ c : Coeffs, mulRq (mulRq a b) c = oneRq := by
  exact invertible_of_withinInvertibilityWindow_of_assumption hInv
    (withinInvertibilityWindow_mulRq_of_operand_norm_assumptions_rawCoeff
      (a := a) (b := b) (BA := BA) (BB := BB) (BRaw := BRaw) (B := B)
      hA hB hPos hRawCoeffFromOperands hAddSub hSub hBLt)

theorem invertible_mulRq_of_operand_norm_assumptions_rawCoeff_fieldOp_of_assumption
  (hInv : LowNormInvertibilityAssumption)
  {a b : Coeffs} {BA BB B : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hPos : 0 < normInfCoeffs (mulRq a b))
  (hRawCoeffFromOperands : mulRqRawCoeffBoundFromOperands BA BB B)
  (hOps : rawFieldOpCollapseBound B B)
  (hBLt : B < bInvApprox) :
  ∃ c : Coeffs, mulRq (mulRq a b) c = oneRq := by
  exact invertible_of_withinInvertibilityWindow_of_assumption hInv
    (withinInvertibilityWindow_mulRq_of_operand_norm_assumptions_rawCoeff_fieldOp
      (a := a) (b := b) (BA := BA) (BB := BB) (B := B)
      hA hB hPos hRawCoeffFromOperands hOps hBLt)

/--
Schoolbook-term wrapper for invertibility extraction of `mulRq a b`, keeping the
trusted invertibility boundary explicit.
-/
theorem invertible_mulRq_of_operand_norm_assumptions_via_schoolbook_of_assumption
  (hInv : LowNormInvertibilityAssumption)
  {a b : Coeffs} {BA BB BTerm BRaw B : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hPos : 0 < normInfCoeffs (mulRq a b))
  (hMul : ∀ x y : F, normInfF x ≤ BA → normInfF y ≤ BB → normInfF (x * y) ≤ BTerm)
  (hAdd : ∀ x y : F, normInfF x ≤ BRaw → normInfF y ≤ BTerm → normInfF (x + y) ≤ BRaw)
  (hZero : normInfF (0 : F) ≤ BRaw)
  (hAddSub : rawAddSubCollapseBound BRaw B)
  (hSub : rawSubCollapseBound BRaw B)
  (hBLt : B < bInvApprox) :
  ∃ c : Coeffs, mulRq (mulRq a b) c = oneRq := by
  exact invertible_of_withinInvertibilityWindow_of_assumption hInv
    (withinInvertibilityWindow_mulRq_of_operand_norm_assumptions_via_schoolbook
      (a := a) (b := b) (BA := BA) (BB := BB) (BTerm := BTerm) (BRaw := BRaw) (B := B)
      hA hB hPos hMul hAdd hZero hAddSub hSub hBLt)

/--
Field-op-collapse variant of
`invertible_mulRq_of_operand_norm_assumptions_via_schoolbook_of_assumption`.
-/
theorem invertible_mulRq_of_operand_norm_assumptions_via_schoolbook_fieldOp_of_assumption
  (hInv : LowNormInvertibilityAssumption)
  {a b : Coeffs} {BA BB BTerm B : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hPos : 0 < normInfCoeffs (mulRq a b))
  (hMul : ∀ x y : F, normInfF x ≤ BA → normInfF y ≤ BB → normInfF (x * y) ≤ BTerm)
  (hAdd : ∀ x y : F, normInfF x ≤ B → normInfF y ≤ BTerm → normInfF (x + y) ≤ B)
  (hZero : normInfF (0 : F) ≤ B)
  (hOps : rawFieldOpCollapseBound B B)
  (hBLt : B < bInvApprox) :
  ∃ c : Coeffs, mulRq (mulRq a b) c = oneRq := by
  exact invertible_of_withinInvertibilityWindow_of_assumption hInv
    (withinInvertibilityWindow_mulRq_of_operand_norm_assumptions_via_schoolbook_fieldOp
      (a := a) (b := b) (BA := BA) (BB := BB) (BTerm := BTerm) (B := B)
      hA hB hPos hMul hAdd hZero hOps hBLt)

theorem invertible_mulRq_of_operand_norm_assumptions_via_schoolbook_sum_of_assumption
  (hInv : LowNormInvertibilityAssumption)
  {a b : Coeffs} {BA BB BTerm B : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hPos : 0 < normInfCoeffs (mulRq a b))
  (hMul : ∀ x y : F, normInfF x ≤ BA → normInfF y ≤ BB → normInfF (x * y) ≤ BTerm)
  (hAddTri : ∀ x y : F, normInfF (x + y) ≤ normInfF x + normInfF y)
  (hAddSub : rawAddSubCollapseBound ((D * D) * BTerm) B)
  (hSub : rawSubCollapseBound ((D * D) * BTerm) B)
  (hBLt : B < bInvApprox) :
  ∃ c : Coeffs, mulRq (mulRq a b) c = oneRq := by
  exact invertible_of_withinInvertibilityWindow_of_assumption hInv
    (withinInvertibilityWindow_mulRq_of_operand_norm_assumptions_via_schoolbook_sum
      (a := a) (b := b) (BA := BA) (BB := BB) (BTerm := BTerm) (B := B)
      hA hB hPos hMul hAddTri hAddSub hSub hBLt)

theorem invertible_mulRq_of_operand_norm_assumptions_via_schoolbook_sum_fieldOp_of_assumption
  (hInv : LowNormInvertibilityAssumption)
  {a b : Coeffs} {BA BB BTerm : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hPos : 0 < normInfCoeffs (mulRq a b))
  (hMul : ∀ x y : F, normInfF x ≤ BA → normInfF y ≤ BB → normInfF (x * y) ≤ BTerm)
  (hAddTri : ∀ x y : F, normInfF (x + y) ≤ normInfF x + normInfF y)
  (hOps : rawFieldOpCollapseBound ((D * D) * BTerm) ((D * D) * BTerm))
  (hBLt : ((D * D) * BTerm) < bInvApprox) :
  ∃ c : Coeffs, mulRq (mulRq a b) c = oneRq := by
  exact invertible_of_withinInvertibilityWindow_of_assumption hInv
    (withinInvertibilityWindow_mulRq_of_operand_norm_assumptions_via_schoolbook_sum_fieldOp
      (a := a) (b := b) (BA := BA) (BB := BB) (BTerm := BTerm)
      hA hB hPos hMul hAddTri hOps hBLt)

theorem invertible_mulRq_of_operand_norm_assumptions_via_schoolbook_of_term_le_of_assumption
  (hInv : LowNormInvertibilityAssumption)
  {a b : Coeffs} {BA BB BTerm BRaw B : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hPos : 0 < normInfCoeffs (mulRq a b))
  (hMul : ∀ x y : F, normInfF x ≤ BA → normInfF y ≤ BB → normInfF (x * y) ≤ BTerm)
  (hTermLe : BTerm ≤ BRaw)
  (hAddCollapse : rawAddCollapseBound BRaw BRaw)
  (hAddSub : rawAddSubCollapseBound BRaw B)
  (hSub : rawSubCollapseBound BRaw B)
  (hBLt : B < bInvApprox) :
  ∃ c : Coeffs, mulRq (mulRq a b) c = oneRq := by
  exact invertible_of_withinInvertibilityWindow_of_assumption hInv
    (withinInvertibilityWindow_mulRq_of_operand_norm_assumptions_via_schoolbook_of_term_le
      (a := a) (b := b) (BA := BA) (BB := BB) (BTerm := BTerm) (BRaw := BRaw) (B := B)
      hA hB hPos hMul hTermLe hAddCollapse hAddSub hSub hBLt)

theorem invertible_mulRq_of_operand_norm_assumptions_via_schoolbook_sameBound_of_assumption
  (hInv : LowNormInvertibilityAssumption)
  {a b : Coeffs} {BA BB BRaw B : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hPos : 0 < normInfCoeffs (mulRq a b))
  (hMul : ∀ x y : F, normInfF x ≤ BA → normInfF y ≤ BB → normInfF (x * y) ≤ BRaw)
  (hAddCollapse : rawAddCollapseBound BRaw BRaw)
  (hAddSub : rawAddSubCollapseBound BRaw B)
  (hSub : rawSubCollapseBound BRaw B)
  (hBLt : B < bInvApprox) :
  ∃ c : Coeffs, mulRq (mulRq a b) c = oneRq := by
  exact invertible_mulRq_of_operand_norm_assumptions_via_schoolbook_of_term_le_of_assumption
    (hInv := hInv)
    (a := a) (b := b)
    (BA := BA) (BB := BB) (BTerm := BRaw) (BRaw := BRaw) (B := B)
    hA hB hPos hMul (Nat.le_refl BRaw) hAddCollapse hAddSub hSub hBLt

theorem invertible_mulRq_of_operand_norm_assumptions_via_schoolbook_sameBound_fieldOp_of_assumption
  (hInv : LowNormInvertibilityAssumption)
  {a b : Coeffs} {BA BB B : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hPos : 0 < normInfCoeffs (mulRq a b))
  (hMul : ∀ x y : F, normInfF x ≤ BA → normInfF y ≤ BB → normInfF (x * y) ≤ B)
  (hOps : rawFieldOpCollapseBound B B)
  (hBLt : B < bInvApprox) :
  ∃ c : Coeffs, mulRq (mulRq a b) c = oneRq := by
  exact invertible_of_withinInvertibilityWindow_of_assumption hInv
    (withinInvertibilityWindow_mulRq_of_operand_norm_assumptions_via_schoolbook_sameBound_fieldOp
      (a := a) (b := b) (BA := BA) (BB := BB) (B := B)
      hA hB hPos hMul hOps hBLt)

theorem invertible_mulRq_of_operand_norm_assumptions_via_universal_blockers_of_assumption
  (hInv : LowNormInvertibilityAssumption)
  {a b : Coeffs} {BA BB : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hPos : 0 < normInfCoeffs (mulRq a b))
  (hMulUniv : schoolbookMulUniversalBound)
  (hAddTri : schoolbookAddTriangleBound)
  (hSubTri : schoolbookSubTriangleBound)
  (hBLt :
    ((D * D) * (BA * BB)) + ((D * D) * (BA * BB)) + ((D * D) * (BA * BB))
      < bInvApprox) :
  ∃ c : Coeffs, mulRq (mulRq a b) c = oneRq := by
  exact invertible_of_withinInvertibilityWindow_of_assumption hInv
    (withinInvertibilityWindow_mulRq_of_operand_norm_assumptions_via_universal_blockers
      (a := a) (b := b) (BA := BA) (BB := BB)
      hA hB hPos hMulUniv hAddTri hSubTri hBLt)

theorem invertible_mulRq_of_operand_norm_assumptions_via_universal_blockers_tight_of_assumption
  (hInv : LowNormInvertibilityAssumption)
  {a b : Coeffs} {BA BB : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hPos : 0 < normInfCoeffs (mulRq a b))
  (hMulUniv : schoolbookMulUniversalBound)
  (hAddTri : schoolbookAddTriangleBound)
  (hSubTri : schoolbookSubTriangleBound)
  (hBLt :
    (D * (BA * BB)) + (D * (BA * BB)) + (D * (BA * BB))
      < bInvApprox) :
  ∃ c : Coeffs, mulRq (mulRq a b) c = oneRq := by
  exact invertible_of_withinInvertibilityWindow_of_assumption hInv
    (withinInvertibilityWindow_mulRq_of_operand_norm_assumptions_via_universal_blockers_tight
      (a := a) (b := b) (BA := BA) (BB := BB)
      hA hB hPos hMulUniv hAddTri hSubTri hBLt)

/--
Triangle-bundle variant of
`invertible_mulRq_of_operand_norm_assumptions_via_universal_blockers_tight_of_assumption`.
-/
theorem invertible_mulRq_of_operand_norm_assumptions_via_universal_mul_and_triangles_tight_of_assumption
  (hInv : LowNormInvertibilityAssumption)
  {a b : Coeffs} {BA BB : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hPos : 0 < normInfCoeffs (mulRq a b))
  (hMulUniv : schoolbookMulUniversalBound)
  (hTri : schoolbookTriangleBounds)
  (hBLt :
    (D * (BA * BB)) + (D * (BA * BB)) + (D * (BA * BB))
      < bInvApprox) :
  ∃ c : Coeffs, mulRq (mulRq a b) c = oneRq := by
  exact invertible_of_withinInvertibilityWindow_of_assumption hInv
    (withinInvertibilityWindow_mulRq_of_operand_norm_assumptions_via_universal_mul_and_triangles_tight
      (a := a) (b := b) (BA := BA) (BB := BB)
      hA hB hPos hMulUniv hTri hBLt)

theorem invertible_mulRq_of_operand_norm_assumptions_via_universal_mul_and_add_tight_of_assumption
  (hInv : LowNormInvertibilityAssumption)
  {a b : Coeffs} {BA BB : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hPos : 0 < normInfCoeffs (mulRq a b))
  (hMulUniv : schoolbookMulUniversalBound)
  (hAddTri : schoolbookAddTriangleBound)
  (hBLt :
    (D * (BA * BB)) + (D * (BA * BB)) + (D * (BA * BB))
      < bInvApprox) :
  ∃ c : Coeffs, mulRq (mulRq a b) c = oneRq := by
  exact invertible_mulRq_of_operand_norm_assumptions_via_universal_mul_and_triangles_tight_of_assumption
    (hInv := hInv) (a := a) (b := b) (BA := BA) (BB := BB)
    hA hB hPos hMulUniv (schoolbookTriangleBounds_of_add hAddTri) hBLt

theorem invertible_mulRq_of_operand_norm_assumptions_via_centeredRep_mul_and_add_tight_of_assumption
  (hInv : LowNormInvertibilityAssumption)
  {a b : Coeffs} {BA BB : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hPos : 0 < normInfCoeffs (mulRq a b))
  (hMulRep :
    ∀ x y : F, Int.natAbs (centeredRep (x * y))
      ≤ Int.natAbs (centeredRep x) * Int.natAbs (centeredRep y))
  (hAddRep :
    ∀ x y : F, Int.natAbs (centeredRep (x + y))
      ≤ Int.natAbs (centeredRep x) + Int.natAbs (centeredRep y))
  (hBLt :
    (D * (BA * BB)) + (D * (BA * BB)) + (D * (BA * BB))
      < bInvApprox) :
  ∃ c : Coeffs, mulRq (mulRq a b) c = oneRq := by
  exact invertible_mulRq_of_operand_norm_assumptions_via_universal_mul_and_add_tight_of_assumption
    (hInv := hInv) (a := a) (b := b) (BA := BA) (BB := BB)
    hA hB hPos
    (schoolbookMulUniversalBound_of_centeredRep hMulRep)
    (schoolbookAddTriangleBound_of_centeredRep hAddRep)
    hBLt

theorem invertible_mulRq_of_operand_norm_assumptions_via_centeredRep_mul_tight_of_assumption
  (hInv : LowNormInvertibilityAssumption)
  {a b : Coeffs} {BA BB : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hPos : 0 < normInfCoeffs (mulRq a b))
  (hMulRep :
    ∀ x y : F, Int.natAbs (centeredRep (x * y))
      ≤ Int.natAbs (centeredRep x) * Int.natAbs (centeredRep y))
  (hBLt :
    (D * (BA * BB)) + (D * (BA * BB)) + (D * (BA * BB))
      < bInvApprox) :
  ∃ c : Coeffs, mulRq (mulRq a b) c = oneRq := by
  exact invertible_mulRq_of_operand_norm_assumptions_via_centeredRep_mul_and_add_tight_of_assumption
    (hInv := hInv) (a := a) (b := b) (BA := BA) (BB := BB)
    hA hB hPos hMulRep centeredRepAddTriangleBound_theorem hBLt

theorem invertible_mulRq_of_operand_norm_assumptions_via_centeredRepMulAddBounds_tight_of_assumption
  (hInv : LowNormInvertibilityAssumption)
  {a b : Coeffs} {BA BB : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hPos : 0 < normInfCoeffs (mulRq a b))
  (hRep : centeredRepMulAddBounds)
  (hBLt :
    (D * (BA * BB)) + (D * (BA * BB)) + (D * (BA * BB))
      < bInvApprox) :
  ∃ c : Coeffs, mulRq (mulRq a b) c = oneRq := by
  exact invertible_mulRq_of_operand_norm_assumptions_via_centeredRep_mul_and_add_tight_of_assumption
    (hInv := hInv) (a := a) (b := b) (BA := BA) (BB := BB)
    hA hB hPos
    (centeredRepMulAddBounds_mul hRep)
    (centeredRepMulAddBounds_add hRep)
    hBLt

/-- Assumption-free native invertibility wrapper (`D^2` schoolbook path). -/
theorem invertible_mulRq_of_operand_norm_assumptions_native_of_assumption
  (hInv : LowNormInvertibilityAssumption)
  {a b : Coeffs} {BA BB : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hPos : 0 < normInfCoeffs (mulRq a b))
  (hBLt :
    ((D * D) * (BA * BB)) + ((D * D) * (BA * BB)) + ((D * D) * (BA * BB))
      < bInvApprox) :
  ∃ c : Coeffs, mulRq (mulRq a b) c = oneRq := by
  exact invertible_of_withinInvertibilityWindow_of_assumption hInv
    (withinInvertibilityWindow_mulRq_of_operand_norm_assumptions_native
      (a := a) (b := b) (BA := BA) (BB := BB)
      hA hB hPos hBLt)

/-- Assumption-free native-tight invertibility wrapper (`3 * D * BA * BB` path). -/
theorem invertible_mulRq_of_operand_norm_assumptions_native_tight_of_assumption
  (hInv : LowNormInvertibilityAssumption)
  {a b : Coeffs} {BA BB : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hPos : 0 < normInfCoeffs (mulRq a b))
  (hBLt :
    (D * (BA * BB)) + (D * (BA * BB)) + (D * (BA * BB))
      < bInvApprox) :
  ∃ c : Coeffs, mulRq (mulRq a b) c = oneRq := by
  exact invertible_of_withinInvertibilityWindow_of_assumption hInv
    (withinInvertibilityWindow_mulRq_of_operand_norm_assumptions_native_tight
      (a := a) (b := b) (BA := BA) (BB := BB)
      hA hB hPos hBLt)

/-- Assumption-free native invertibility wrapper with externally supplied raw bounds. -/
theorem invertible_mulRq_of_operand_norm_assumptions_native_and_raw_of_assumption
  (hInv : LowNormInvertibilityAssumption)
  {a b : Coeffs} {BA BB BRaw : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hPos : 0 < normInfCoeffs (mulRq a b))
  (hRawFromOperands : mulRqRawNormBoundFromOperands BA BB BRaw)
  (hBLt : BRaw + BRaw + BRaw < bInvApprox) :
  ∃ c : Coeffs, mulRq (mulRq a b) c = oneRq := by
  exact invertible_of_withinInvertibilityWindow_of_assumption hInv
    (withinInvertibilityWindow_mulRq_of_operand_norm_assumptions_native_and_raw
      (a := a) (b := b) (BA := BA) (BB := BB) (BRaw := BRaw)
      hA hB hPos hRawFromOperands hBLt)

/-- Raw-coeff native invertibility variant of `...native_and_raw...`. -/
theorem invertible_mulRq_of_operand_norm_assumptions_native_and_rawCoeff_of_assumption
  (hInv : LowNormInvertibilityAssumption)
  {a b : Coeffs} {BA BB BRaw : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hPos : 0 < normInfCoeffs (mulRq a b))
  (hRawCoeffFromOperands : mulRqRawCoeffBoundFromOperands BA BB BRaw)
  (hBLt : BRaw + BRaw + BRaw < bInvApprox) :
  ∃ c : Coeffs, mulRq (mulRq a b) c = oneRq := by
  exact invertible_mulRq_of_operand_norm_assumptions_native_and_raw_of_assumption
    (hInv := hInv) (a := a) (b := b) (BA := BA) (BB := BB) (BRaw := BRaw)
    hA hB hPos
    (mulRqRawNormBoundFromOperands_of_rawCoeff hRawCoeffFromOperands)
    hBLt

theorem invertible_mulRq_of_operand_norm_assumptions_via_universal_blockers_and_raw_of_assumption
  (hInv : LowNormInvertibilityAssumption)
  {a b : Coeffs} {BA BB BRaw : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hPos : 0 < normInfCoeffs (mulRq a b))
  (hRawFromOperands : mulRqRawNormBoundFromOperands BA BB BRaw)
  (hAddTri : schoolbookAddTriangleBound)
  (hSubTri : schoolbookSubTriangleBound)
  (hBLt : BRaw + BRaw + BRaw < bInvApprox) :
  ∃ c : Coeffs, mulRq (mulRq a b) c = oneRq := by
  exact invertible_of_withinInvertibilityWindow_of_assumption hInv
    (withinInvertibilityWindow_mulRq_of_operand_norm_assumptions_via_universal_blockers_and_raw
      (a := a) (b := b) (BA := BA) (BB := BB) (BRaw := BRaw)
      hA hB hPos hRawFromOperands hAddTri hSubTri hBLt)

theorem invertible_mulRq_of_operand_norm_assumptions_via_universal_blockers_and_rawCoeff_of_assumption
  (hInv : LowNormInvertibilityAssumption)
  {a b : Coeffs} {BA BB BRaw : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hPos : 0 < normInfCoeffs (mulRq a b))
  (hRawCoeffFromOperands : mulRqRawCoeffBoundFromOperands BA BB BRaw)
  (hAddTri : schoolbookAddTriangleBound)
  (hSubTri : schoolbookSubTriangleBound)
  (hBLt : BRaw + BRaw + BRaw < bInvApprox) :
  ∃ c : Coeffs, mulRq (mulRq a b) c = oneRq := by
  exact invertible_mulRq_of_operand_norm_assumptions_via_universal_blockers_and_raw_of_assumption
    (hInv := hInv) (a := a) (b := b) (BA := BA) (BB := BB) (BRaw := BRaw)
    hA hB hPos
    (mulRqRawNormBoundFromOperands_of_rawCoeff hRawCoeffFromOperands)
    hAddTri hSubTri hBLt

theorem invertible_mulRq_of_operand_norm_assumptions_via_universal_mul_and_add_and_raw_of_assumption
  (hInv : LowNormInvertibilityAssumption)
  {a b : Coeffs} {BA BB BRaw : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hPos : 0 < normInfCoeffs (mulRq a b))
  (hRawFromOperands : mulRqRawNormBoundFromOperands BA BB BRaw)
  (hAddTri : schoolbookAddTriangleBound)
  (hBLt : BRaw + BRaw + BRaw < bInvApprox) :
  ∃ c : Coeffs, mulRq (mulRq a b) c = oneRq := by
  exact invertible_mulRq_of_operand_norm_assumptions_via_universal_blockers_and_raw_of_assumption
    (hInv := hInv) (a := a) (b := b) (BA := BA) (BB := BB) (BRaw := BRaw)
    hA hB hPos hRawFromOperands hAddTri (schoolbookSubTriangleBound_of_add hAddTri) hBLt

theorem invertible_mulRq_of_operand_norm_assumptions_via_universal_mul_and_add_and_rawCoeff_of_assumption
  (hInv : LowNormInvertibilityAssumption)
  {a b : Coeffs} {BA BB BRaw : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hPos : 0 < normInfCoeffs (mulRq a b))
  (hRawCoeffFromOperands : mulRqRawCoeffBoundFromOperands BA BB BRaw)
  (hAddTri : schoolbookAddTriangleBound)
  (hBLt : BRaw + BRaw + BRaw < bInvApprox) :
  ∃ c : Coeffs, mulRq (mulRq a b) c = oneRq := by
  exact invertible_mulRq_of_operand_norm_assumptions_via_universal_mul_and_add_and_raw_of_assumption
    (hInv := hInv) (a := a) (b := b) (BA := BA) (BB := BB) (BRaw := BRaw)
    hA hB hPos
    (mulRqRawNormBoundFromOperands_of_rawCoeff hRawCoeffFromOperands)
    hAddTri hBLt

theorem invertible_mulRq_of_operand_norm_assumptions_via_centeredRep_mul_and_add_and_raw_of_assumption
  (hInv : LowNormInvertibilityAssumption)
  {a b : Coeffs} {BA BB BRaw : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hPos : 0 < normInfCoeffs (mulRq a b))
  (hRawFromOperands : mulRqRawNormBoundFromOperands BA BB BRaw)
  (hAddRep :
    ∀ x y : F, Int.natAbs (centeredRep (x + y))
      ≤ Int.natAbs (centeredRep x) + Int.natAbs (centeredRep y))
  (hBLt : BRaw + BRaw + BRaw < bInvApprox) :
  ∃ c : Coeffs, mulRq (mulRq a b) c = oneRq := by
  exact invertible_mulRq_of_operand_norm_assumptions_via_universal_mul_and_add_and_raw_of_assumption
    (hInv := hInv) (a := a) (b := b) (BA := BA) (BB := BB) (BRaw := BRaw)
    hA hB hPos hRawFromOperands
    (schoolbookAddTriangleBound_of_centeredRep hAddRep)
    hBLt

theorem invertible_mulRq_of_operand_norm_assumptions_via_centeredRep_mul_and_add_and_rawCoeff_of_assumption
  (hInv : LowNormInvertibilityAssumption)
  {a b : Coeffs} {BA BB BRaw : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hPos : 0 < normInfCoeffs (mulRq a b))
  (hRawCoeffFromOperands : mulRqRawCoeffBoundFromOperands BA BB BRaw)
  (hAddRep :
    ∀ x y : F, Int.natAbs (centeredRep (x + y))
      ≤ Int.natAbs (centeredRep x) + Int.natAbs (centeredRep y))
  (hBLt : BRaw + BRaw + BRaw < bInvApprox) :
  ∃ c : Coeffs, mulRq (mulRq a b) c = oneRq := by
  exact invertible_mulRq_of_operand_norm_assumptions_via_centeredRep_mul_and_add_and_raw_of_assumption
    (hInv := hInv) (a := a) (b := b) (BA := BA) (BB := BB) (BRaw := BRaw)
    hA hB hPos
    (mulRqRawNormBoundFromOperands_of_rawCoeff hRawCoeffFromOperands)
    hAddRep hBLt

theorem invertible_mulRq_of_goldilocks_operand_assumptions_of_assumption
  (hInv : LowNormInvertibilityAssumption)
  {a b : Coeffs}
  (hA : normInfCoeffs a ≤ Parameters.Goldilocks.B)
  (hB : normInfCoeffs b ≤ Parameters.Goldilocks.B)
  (hPos : 0 < normInfCoeffs (mulRq a b))
  (hRaw : GoldilocksRawNormBoundAssumption)
  (hCollapse : GoldilocksRawCollapseAssumption) :
  ∃ c : Coeffs, mulRq (mulRq a b) c = oneRq := by
  exact invertible_of_withinInvertibilityWindow_of_assumption hInv
    (withinInvertibilityWindow_mulRq_of_goldilocks_operand_assumptions
      (a := a) (b := b) hA hB hPos hRaw hCollapse)

theorem invertible_mulRq_of_goldilocks_operand_assumptions_inRange_of_assumption
  (hInv : LowNormInvertibilityAssumption)
  {a b : Coeffs}
  (hA : normInfCoeffs a ≤ Parameters.Goldilocks.B)
  (hB : normInfCoeffs b ≤ Parameters.Goldilocks.B)
  (hPos : 0 < normInfCoeffs (mulRq a b))
  (hRawInRange : GoldilocksRawInRangeBoundAssumption)
  (hCollapse : GoldilocksRawCollapseAssumption) :
  ∃ c : Coeffs, mulRq (mulRq a b) c = oneRq := by
  exact invertible_mulRq_of_goldilocks_operand_assumptions_of_assumption
    (a := a) (b := b)
    hInv hA hB hPos
    (goldilocksRawNormBoundAssumption_of_inRange hRawInRange)
    hCollapse

theorem invertible_mulRq_of_goldilocks_operand_fieldOp_assumptions_of_assumption
  (hInv : LowNormInvertibilityAssumption)
  {a b : Coeffs}
  (hA : normInfCoeffs a ≤ Parameters.Goldilocks.B)
  (hB : normInfCoeffs b ≤ Parameters.Goldilocks.B)
  (hPos : 0 < normInfCoeffs (mulRq a b))
  (hRaw : GoldilocksRawNormBoundAssumption)
  (hFieldOps : GoldilocksFieldOpCollapseAssumption) :
  ∃ c : Coeffs, mulRq (mulRq a b) c = oneRq := by
  exact invertible_mulRq_of_goldilocks_operand_assumptions_of_assumption
    (a := a) (b := b)
    hInv hA hB hPos hRaw
    (goldilocksRawCollapseAssumption_of_fieldOp hFieldOps)

theorem invertible_mulRq_of_goldilocks_operand_fieldOp_assumptions_inRange_of_assumption
  (hInv : LowNormInvertibilityAssumption)
  {a b : Coeffs}
  (hA : normInfCoeffs a ≤ Parameters.Goldilocks.B)
  (hB : normInfCoeffs b ≤ Parameters.Goldilocks.B)
  (hPos : 0 < normInfCoeffs (mulRq a b))
  (hRawInRange : GoldilocksRawInRangeBoundAssumption)
  (hFieldOps : GoldilocksFieldOpCollapseAssumption) :
  ∃ c : Coeffs, mulRq (mulRq a b) c = oneRq := by
  exact invertible_mulRq_of_goldilocks_operand_assumptions_inRange_of_assumption
    (a := a) (b := b)
    hInv hA hB hPos hRawInRange
    (goldilocksRawCollapseAssumption_of_fieldOp hFieldOps)

theorem invertible_mulRq_of_goldilocks_operand_rawCoeff_assumptions_of_assumption
  (hInv : LowNormInvertibilityAssumption)
  {a b : Coeffs}
  (hA : normInfCoeffs a ≤ Parameters.Goldilocks.B)
  (hB : normInfCoeffs b ≤ Parameters.Goldilocks.B)
  (hPos : 0 < normInfCoeffs (mulRq a b))
  (hRawCoeff : GoldilocksRawCoeffBoundAssumption)
  (hCollapse : GoldilocksRawCollapseAssumption) :
  ∃ c : Coeffs, mulRq (mulRq a b) c = oneRq := by
  exact invertible_mulRq_of_goldilocks_operand_assumptions_of_assumption
    (a := a) (b := b)
    hInv hA hB hPos
    (goldilocksRawNormBoundAssumption_of_rawCoeff hRawCoeff)
    hCollapse

theorem invertible_mulRq_of_goldilocks_operand_rawCoeff_fieldOp_assumptions_of_assumption
  (hInv : LowNormInvertibilityAssumption)
  {a b : Coeffs}
  (hA : normInfCoeffs a ≤ Parameters.Goldilocks.B)
  (hB : normInfCoeffs b ≤ Parameters.Goldilocks.B)
  (hPos : 0 < normInfCoeffs (mulRq a b))
  (hRawCoeff : GoldilocksRawCoeffBoundAssumption)
  (hFieldOps : GoldilocksFieldOpCollapseAssumption) :
  ∃ c : Coeffs, mulRq (mulRq a b) c = oneRq := by
  exact invertible_mulRq_of_goldilocks_operand_rawCoeff_assumptions_of_assumption
    (a := a) (b := b)
    hInv hA hB hPos hRawCoeff
    (goldilocksRawCollapseAssumption_of_fieldOp hFieldOps)


end SuperNeo
