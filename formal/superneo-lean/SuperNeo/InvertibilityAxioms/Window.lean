import SuperNeo.Parameters
import SuperNeo.Norm
import SuperNeo.Ring

/-! Invertibility-window lemmas and trusted low-norm invertibility boundary. -/


namespace SuperNeo

open F

/-- Appendix B.2 concrete estimate (Goldilocks / eta=81): b_inv ≈ 2.5e9. -/
def bInvApprox : Nat := 2500000000

/-- Coefficient difference bound for C with coeffs in [-2,-1,0,1,2]. -/
def challengeCoeffMaxDiff : Nat := 4

theorem challengeCoeffMaxDiff_eq_four : challengeCoeffMaxDiff = 4 := rfl

theorem four_lt_bInvApprox : 4 < bInvApprox := by
  unfold bInvApprox
  decide

def oneRq : Coeffs :=
  (Array.replicate d (0 : F)).set! 0 (1 : F)

theorem oneRq_size : oneRq.size = d := by
  unfold oneRq
  simp

theorem oneRq_hasRingDegreeShape : hasRingDegreeShape oneRq := by
  unfold hasRingDegreeShape
  simpa [D_eq_d] using oneRq_size

theorem ct_oneRq : ct oneRq = (1 : F) := by
  unfold oneRq ct
  simp [d]

def withinInvertibilityWindow (a : Coeffs) : Bool :=
  decide (0 < normInfCoeffs a ∧ normInfCoeffs a < bInvApprox)

/-- Assumption boundary used by later reductions/proofs (Theorem 8 interface). -/
def LowNormInvertibilityAssumption : Prop :=
  ∀ a : Coeffs, 0 < normInfCoeffs a → normInfCoeffs a < bInvApprox → ∃ b : Coeffs, mulRq a b = oneRq

/-- Concrete precondition checks for B.2 parameterization. -/
def invertibilityPreconditionsSanity : Bool :=
  decide
    (Parameters.Goldilocks.b = 2 ∧ Parameters.Goldilocks.k = 14 ∧
      challengeCoeffMaxDiff < bInvApprox ∧ Parameters.Goldilocks.B < bInvApprox)

def invertibilityPreconditionsProp : Prop :=
  Parameters.Goldilocks.b = 2 ∧ Parameters.Goldilocks.k = 14 ∧
    challengeCoeffMaxDiff < bInvApprox ∧ Parameters.Goldilocks.B < bInvApprox

theorem invertibilityPreconditionsSanity_sound
  (hOk : invertibilityPreconditionsSanity = true) :
  invertibilityPreconditionsProp := by
  unfold invertibilityPreconditionsProp
  unfold invertibilityPreconditionsSanity at hOk
  exact decide_eq_true_eq.mp hOk

theorem invertibilityPreconditions_from_constants : invertibilityPreconditionsProp := by
  unfold invertibilityPreconditionsProp
  refine ⟨Parameters.Goldilocks.b_eq_2, Parameters.Goldilocks.k_eq_14, ?_, ?_⟩
  · exact Nat.lt_of_le_of_lt (by simpa [challengeCoeffMaxDiff_eq_four]) four_lt_bInvApprox
  ·
    have hB : Parameters.Goldilocks.B = 16384 := Parameters.Goldilocks.B_eq_16384
    simpa [hB, bInvApprox] using (show 16384 < bInvApprox by decide)

theorem challengeCoeffMaxDiff_lt_bInvApprox : challengeCoeffMaxDiff < bInvApprox := by
  rcases invertibilityPreconditions_from_constants with ⟨_, _, hDiff, _⟩
  exact hDiff

theorem goldilocksB_lt_bInvApprox : Parameters.Goldilocks.B < bInvApprox := by
  rcases invertibilityPreconditions_from_constants with ⟨_, _, _, hB⟩
  exact hB

/--
Concrete operand-bound assumption shape at the Goldilocks proof bound `B`.
This is a non-coarse (sub-`halfQ`) regime used to thread P5 bounds into P16.
-/
def GoldilocksRawNormBoundAssumption : Prop :=
  mulRqRawNormBoundFromOperands Parameters.Goldilocks.B Parameters.Goldilocks.B Parameters.Goldilocks.B

/--
In-range raw coefficient variant of the Goldilocks operand-bound assumption.
-/
def GoldilocksRawInRangeBoundAssumption : Prop :=
  mulRqRawInRangeBoundFromOperands Parameters.Goldilocks.B Parameters.Goldilocks.B Parameters.Goldilocks.B

/--
All-index raw coefficient accessor variant of the Goldilocks operand-bound assumption.
-/
def GoldilocksRawCoeffBoundAssumption : Prop :=
  mulRqRawCoeffBoundFromOperands Parameters.Goldilocks.B Parameters.Goldilocks.B Parameters.Goldilocks.B

/--
Concrete collapse assumptions for `mulRqCoeffSpec` reduction steps at Goldilocks bound `B`.
-/
def GoldilocksRawCollapseAssumption : Prop :=
  rawAddSubCollapseBound Parameters.Goldilocks.B Parameters.Goldilocks.B ∧
  rawSubCollapseBound Parameters.Goldilocks.B Parameters.Goldilocks.B

/--
Alternative Goldilocks collapse surface: separate `x+y` and `x-y` bounds at `B`.
This can be collapsed into `GoldilocksRawCollapseAssumption` via
`rawAddSubCollapseBound_of_add_and_sub_same`.
-/
def GoldilocksFieldOpCollapseAssumption : Prop :=
  rawAddCollapseBound Parameters.Goldilocks.B Parameters.Goldilocks.B ∧
  rawSubCollapseBound Parameters.Goldilocks.B Parameters.Goldilocks.B

theorem goldilocksRawCollapseAssumption_of_fieldOp
  (hOps : GoldilocksFieldOpCollapseAssumption) :
  GoldilocksRawCollapseAssumption := by
  rcases hOps with ⟨hAdd, hSub⟩
  exact ⟨rawAddSubCollapseBound_of_add_and_sub_same hAdd hSub, hSub⟩

theorem goldilocksFieldOpCollapseAssumption_of_rawCollapse
  (hCollapse : GoldilocksRawCollapseAssumption) :
  GoldilocksFieldOpCollapseAssumption := by
  rcases hCollapse with ⟨hAddSub, hSub⟩
  exact rawFieldOpCollapseBound_of_addSub_and_sub hAddSub hSub

theorem goldilocksCollapseAssumption_iff_fieldOp :
  GoldilocksRawCollapseAssumption ↔ GoldilocksFieldOpCollapseAssumption := by
  constructor
  · exact goldilocksFieldOpCollapseAssumption_of_rawCollapse
  · exact goldilocksRawCollapseAssumption_of_fieldOp

theorem goldilocksRawNormBoundAssumption_of_inRange
  (hRawInRange : GoldilocksRawInRangeBoundAssumption) :
  GoldilocksRawNormBoundAssumption := by
  exact mulRqRawNormBoundFromOperands_of_inRange hRawInRange

theorem goldilocksRawCoeffBoundAssumption_of_inRange
  (hRawInRange : GoldilocksRawInRangeBoundAssumption) :
  GoldilocksRawCoeffBoundAssumption := by
  exact mulRqRawCoeffBoundFromOperands_of_inRange hRawInRange

theorem goldilocksRawInRangeBoundAssumption_of_norm
  (hRaw : GoldilocksRawNormBoundAssumption) :
  GoldilocksRawInRangeBoundAssumption := by
  exact mulRqRawInRangeBoundFromOperands_of_norm hRaw

theorem goldilocksRawInRangeBoundAssumption_of_rawCoeff
  (hRawCoeff : GoldilocksRawCoeffBoundAssumption) :
  GoldilocksRawInRangeBoundAssumption := by
  exact mulRqRawInRangeBoundFromOperands_of_rawCoeff hRawCoeff

theorem goldilocksRawNormBoundAssumption_of_rawCoeff
  (hRawCoeff : GoldilocksRawCoeffBoundAssumption) :
  GoldilocksRawNormBoundAssumption := by
  exact mulRqRawNormBoundFromOperands_of_rawCoeff hRawCoeff

theorem goldilocksRawNormBoundAssumption_iff_inRange :
  GoldilocksRawNormBoundAssumption ↔ GoldilocksRawInRangeBoundAssumption := by
  exact mulRqRawNormBoundFromOperands_iff_inRange

theorem goldilocksRawCoeffBoundAssumption_iff_inRange :
  GoldilocksRawCoeffBoundAssumption ↔ GoldilocksRawInRangeBoundAssumption := by
  exact mulRqRawCoeffBoundFromOperands_iff_inRange

theorem challengeCoeff_sub_norm_bound
  {x y : F}
  (hx : IsChallengeCoeff x)
  (hy : IsChallengeCoeff y) :
  normInfF (x - y) ≤ challengeCoeffMaxDiff := by
  simpa [challengeCoeffMaxDiff_eq_four] using
    (normInfF_sub_le_four_of_isChallengeCoeff hx hy)

theorem normInfCoeffs_lt_bInvApprox_of_allChallenge
  {a : Coeffs}
  (hAll : AllChallengeCoeffs a) :
  normInfCoeffs a < bInvApprox := by
  have hLe4 : normInfCoeffs a ≤ 4 := normInfCoeffs_le_four_of_allChallenge hAll
  exact Nat.lt_of_le_of_lt hLe4 four_lt_bInvApprox

theorem normInfCoeffs_sub_lt_bInvApprox_of_allChallenge
  {a b : Coeffs}
  (hSize : a.size = b.size)
  (hAllA : AllChallengeCoeffs a)
  (hAllB : AllChallengeCoeffs b) :
  normInfCoeffs (coeffSub a b) < bInvApprox := by
  have hLe4 : normInfCoeffs (coeffSub a b) ≤ 4 :=
    normInfCoeffs_le_four_of_allChallenge_sub hSize hAllA hAllB
  exact Nat.lt_of_le_of_lt hLe4 four_lt_bInvApprox

theorem invertibilityPreconditionsSanity_true : invertibilityPreconditionsSanity = true := by
  unfold invertibilityPreconditionsSanity
  exact decide_eq_true invertibilityPreconditions_from_constants

theorem withinInvertibilityWindow_sound
  {a : Coeffs}
  (hOk : withinInvertibilityWindow a = true) :
  0 < normInfCoeffs a ∧ normInfCoeffs a < bInvApprox := by
  unfold withinInvertibilityWindow at hOk
  exact decide_eq_true_eq.mp hOk

theorem withinInvertibilityWindow_complete
  {a : Coeffs}
  (h : 0 < normInfCoeffs a ∧ normInfCoeffs a < bInvApprox) :
  withinInvertibilityWindow a = true := by
  unfold withinInvertibilityWindow
  exact decide_eq_true h

theorem withinInvertibilityWindow_iff_prop
  {a : Coeffs} :
  withinInvertibilityWindow a = true ↔
    (0 < normInfCoeffs a ∧ normInfCoeffs a < bInvApprox) := by
  constructor
  · intro hOk
    exact withinInvertibilityWindow_sound (a := a) hOk
  · intro h
    exact withinInvertibilityWindow_complete (a := a) h

theorem normInfCoeffs_lt_bInvApprox_of_le
  {a : Coeffs} {B : Nat}
  (hLe : normInfCoeffs a ≤ B)
  (hBLt : B < bInvApprox) :
  normInfCoeffs a < bInvApprox := by
  exact Nat.lt_of_le_of_lt hLe hBLt

theorem withinInvertibilityWindow_of_norm_le_of_lt
  {a : Coeffs} {B : Nat}
  (hPos : 0 < normInfCoeffs a)
  (hLe : normInfCoeffs a ≤ B)
  (hBLt : B < bInvApprox) :
  withinInvertibilityWindow a = true := by
  exact withinInvertibilityWindow_complete
    ⟨hPos, normInfCoeffs_lt_bInvApprox_of_le hLe hBLt⟩

theorem withinInvertibilityWindow_mulRq_of_norm_le_of_lt
  {a b : Coeffs} {B : Nat}
  (hPos : 0 < normInfCoeffs (mulRq a b))
  (hLe : normInfCoeffs (mulRq a b) ≤ B)
  (hBLt : B < bInvApprox) :
  withinInvertibilityWindow (mulRq a b) = true := by
  exact withinInvertibilityWindow_of_norm_le_of_lt hPos hLe hBLt

theorem withinInvertibilityWindow_mulRq_of_rawCoeffInRangeBound
  {a b : Coeffs} {BRaw B : Nat}
  (hPos : 0 < normInfCoeffs (mulRq a b))
  (hRawInRange : ∀ t, t < 2 * D - 1 → normInfF (mulRqRawCoeffSpec a b t) ≤ BRaw)
  (hAddSub : ∀ x y z, normInfF x ≤ BRaw → normInfF y ≤ BRaw → normInfF z ≤ BRaw → normInfF (x + y - z) ≤ B)
  (hSub : ∀ x y, normInfF x ≤ BRaw → normInfF y ≤ BRaw → normInfF (x - y) ≤ B)
  (hBLt : B < bInvApprox) :
  withinInvertibilityWindow (mulRq a b) = true := by
  exact withinInvertibilityWindow_mulRq_of_norm_le_of_lt hPos
    (normInfCoeffs_mulRq_le_of_rawCoeffInRangeBound
      (a := a) (b := b) (BRaw := BRaw) (B := B)
      hRawInRange hAddSub hSub)
    hBLt

theorem withinInvertibilityWindow_mulRq_of_rawCoeffsNorm
  {a b : Coeffs} {BRaw B : Nat}
  (hPos : 0 < normInfCoeffs (mulRq a b))
  (hRawCoeffs : normInfCoeffs (mulRqRawCoeffs a b) ≤ BRaw)
  (hAddSub : ∀ x y z, normInfF x ≤ BRaw → normInfF y ≤ BRaw → normInfF z ≤ BRaw → normInfF (x + y - z) ≤ B)
  (hSub : ∀ x y, normInfF x ≤ BRaw → normInfF y ≤ BRaw → normInfF (x - y) ≤ B)
  (hBLt : B < bInvApprox) :
  withinInvertibilityWindow (mulRq a b) = true := by
  exact withinInvertibilityWindow_mulRq_of_norm_le_of_lt hPos
    (normInfCoeffs_mulRq_le_of_rawCoeffsNorm
      (a := a) (b := b) (BRaw := BRaw) (B := B)
      hRawCoeffs hAddSub hSub)
    hBLt

theorem withinInvertibilityWindow_mulRq_of_norm_bounds_via_rawCoeffInRange
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
  withinInvertibilityWindow (mulRq a b) = true := by
  exact withinInvertibilityWindow_mulRq_of_norm_le_of_lt hPos
    (normInfCoeffs_mulRq_le_of_norm_bounds_via_rawCoeff_inRange
      (a := a) (b := b) (BA := BA) (BB := BB) (BRaw := BRaw) (B := B)
      hA hB hRawFromNormInRange hAddSub hSub)
    hBLt

theorem withinInvertibilityWindow_mulRq_of_norm_bounds_via_rawCoeffsNorm
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
  withinInvertibilityWindow (mulRq a b) = true := by
  exact withinInvertibilityWindow_mulRq_of_norm_le_of_lt hPos
    (normInfCoeffs_mulRq_le_of_norm_bounds_via_rawCoeffsNorm
      (a := a) (b := b) (BA := BA) (BB := BB) (BRaw := BRaw) (B := B)
      hA hB hRawCoeffsFromNorm hAddSub hSub)
    hBLt

theorem withinInvertibilityWindow_mulRq_of_operand_norm_assumptions
  {a b : Coeffs} {BA BB BRaw B : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hPos : 0 < normInfCoeffs (mulRq a b))
  (hRawFromOperands : mulRqRawNormBoundFromOperands BA BB BRaw)
  (hAddSub : rawAddSubCollapseBound BRaw B)
  (hSub : rawSubCollapseBound BRaw B)
  (hBLt : B < bInvApprox) :
  withinInvertibilityWindow (mulRq a b) = true := by
  exact withinInvertibilityWindow_mulRq_of_norm_le_of_lt hPos
    (normInfCoeffs_mulRq_le_of_operand_norm_assumptions
      (a := a) (b := b) (BA := BA) (BB := BB) (BRaw := BRaw) (B := B)
      hA hB hRawFromOperands hAddSub hSub)
    hBLt

theorem withinInvertibilityWindow_mulRq_of_operand_norm_assumptions_inRange
  {a b : Coeffs} {BA BB BRaw B : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hPos : 0 < normInfCoeffs (mulRq a b))
  (hRawInRangeFromOperands : mulRqRawInRangeBoundFromOperands BA BB BRaw)
  (hAddSub : rawAddSubCollapseBound BRaw B)
  (hSub : rawSubCollapseBound BRaw B)
  (hBLt : B < bInvApprox) :
  withinInvertibilityWindow (mulRq a b) = true := by
  exact withinInvertibilityWindow_mulRq_of_norm_le_of_lt hPos
    (normInfCoeffs_mulRq_le_of_operand_norm_assumptions_inRange
      (a := a) (b := b) (BA := BA) (BB := BB) (BRaw := BRaw) (B := B)
      hA hB hRawInRangeFromOperands hAddSub hSub)
    hBLt

/--
Field-op collapse variant of the P5->P16 window handoff.
This avoids manual construction of `rawAddSubCollapseBound` when the caller already has
separate `x+y` and `x-y` bounds at the same `B`.
-/
theorem withinInvertibilityWindow_mulRq_of_operand_norm_assumptions_fieldOp
  {a b : Coeffs} {BA BB B : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hPos : 0 < normInfCoeffs (mulRq a b))
  (hRawFromOperands : mulRqRawNormBoundFromOperands BA BB B)
  (hOps : rawFieldOpCollapseBound B B)
  (hBLt : B < bInvApprox) :
  withinInvertibilityWindow (mulRq a b) = true := by
  rcases hOps with ⟨hAdd, hSub⟩
  exact withinInvertibilityWindow_mulRq_of_operand_norm_assumptions
    (a := a) (b := b)
    (BA := BA) (BB := BB) (BRaw := B) (B := B)
    hA hB hPos hRawFromOperands
    (rawAddSubCollapseBound_of_add_and_sub_same (BRaw := B) hAdd hSub)
    hSub
    hBLt

theorem withinInvertibilityWindow_mulRq_of_operand_norm_assumptions_fieldOp_inRange
  {a b : Coeffs} {BA BB B : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hPos : 0 < normInfCoeffs (mulRq a b))
  (hRawInRangeFromOperands : mulRqRawInRangeBoundFromOperands BA BB B)
  (hOps : rawFieldOpCollapseBound B B)
  (hBLt : B < bInvApprox) :
  withinInvertibilityWindow (mulRq a b) = true := by
  rcases hOps with ⟨hAdd, hSub⟩
  exact withinInvertibilityWindow_mulRq_of_operand_norm_assumptions_inRange
    (a := a) (b := b)
    (BA := BA) (BB := BB) (BRaw := B) (B := B)
    hA hB hPos hRawInRangeFromOperands
    (rawAddSubCollapseBound_of_add_and_sub_same (BRaw := B) hAdd hSub)
    hSub
    hBLt

theorem withinInvertibilityWindow_mulRq_of_operand_norm_assumptions_rawCoeff
  {a b : Coeffs} {BA BB BRaw B : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hPos : 0 < normInfCoeffs (mulRq a b))
  (hRawCoeffFromOperands : mulRqRawCoeffBoundFromOperands BA BB BRaw)
  (hAddSub : rawAddSubCollapseBound BRaw B)
  (hSub : rawSubCollapseBound BRaw B)
  (hBLt : B < bInvApprox) :
  withinInvertibilityWindow (mulRq a b) = true := by
  exact withinInvertibilityWindow_mulRq_of_norm_le_of_lt hPos
    (normInfCoeffs_mulRq_le_of_operand_norm_assumptions_rawCoeff
      (a := a) (b := b) (BA := BA) (BB := BB) (BRaw := BRaw) (B := B)
      hA hB hRawCoeffFromOperands hAddSub hSub)
    hBLt

theorem withinInvertibilityWindow_mulRq_of_operand_norm_assumptions_rawCoeff_fieldOp
  {a b : Coeffs} {BA BB B : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hPos : 0 < normInfCoeffs (mulRq a b))
  (hRawCoeffFromOperands : mulRqRawCoeffBoundFromOperands BA BB B)
  (hOps : rawFieldOpCollapseBound B B)
  (hBLt : B < bInvApprox) :
  withinInvertibilityWindow (mulRq a b) = true := by
  exact withinInvertibilityWindow_mulRq_of_norm_le_of_lt hPos
    (normInfCoeffs_mulRq_le_of_operand_norm_assumptions_rawCoeff_fieldOp
      (a := a) (b := b) (BA := BA) (BB := BB) (B := B)
      hA hB hRawCoeffFromOperands hOps)
    hBLt

/--
Schoolbook-term wrapper for the P5 -> P16 window handoff.
This packages the raw-coefficient assumption from theorem-native multiplication/addition
bounds on schoolbook terms.
-/
theorem withinInvertibilityWindow_mulRq_of_operand_norm_assumptions_via_schoolbook
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
  withinInvertibilityWindow (mulRq a b) = true := by
  exact withinInvertibilityWindow_mulRq_of_operand_norm_assumptions_rawCoeff
    (a := a) (b := b) (BA := BA) (BB := BB) (BRaw := BRaw) (B := B)
    hA hB hPos
    (mulRqRawCoeffBoundFromOperands_of_schoolbook_term_assumptions
      (hMul := hMul) (hAdd := hAdd) (hZero := hZero))
    hAddSub hSub hBLt

/--
Field-op-collapse variant of
`withinInvertibilityWindow_mulRq_of_operand_norm_assumptions_via_schoolbook`.
-/
theorem withinInvertibilityWindow_mulRq_of_operand_norm_assumptions_via_schoolbook_fieldOp
  {a b : Coeffs} {BA BB BTerm B : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hPos : 0 < normInfCoeffs (mulRq a b))
  (hMul : ∀ x y : F, normInfF x ≤ BA → normInfF y ≤ BB → normInfF (x * y) ≤ BTerm)
  (hAdd : ∀ x y : F, normInfF x ≤ B → normInfF y ≤ BTerm → normInfF (x + y) ≤ B)
  (hZero : normInfF (0 : F) ≤ B)
  (hOps : rawFieldOpCollapseBound B B)
  (hBLt : B < bInvApprox) :
  withinInvertibilityWindow (mulRq a b) = true := by
  exact withinInvertibilityWindow_mulRq_of_operand_norm_assumptions_rawCoeff_fieldOp
    (a := a) (b := b) (BA := BA) (BB := BB) (B := B)
    hA hB hPos
    (mulRqRawCoeffBoundFromOperands_of_schoolbook_term_assumptions
      (hMul := hMul) (hAdd := hAdd) (hZero := hZero))
    hOps hBLt

/--
Sum-style schoolbook wrapper for the P5 -> P16 window handoff.
Uses only per-term multiplication bounds + triangle addition, with derived
raw bound `((D * D) * BTerm)`.
-/
theorem withinInvertibilityWindow_mulRq_of_operand_norm_assumptions_via_schoolbook_sum
  {a b : Coeffs} {BA BB BTerm B : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hPos : 0 < normInfCoeffs (mulRq a b))
  (hMul : ∀ x y : F, normInfF x ≤ BA → normInfF y ≤ BB → normInfF (x * y) ≤ BTerm)
  (hAddTri : ∀ x y : F, normInfF (x + y) ≤ normInfF x + normInfF y)
  (hAddSub : rawAddSubCollapseBound ((D * D) * BTerm) B)
  (hSub : rawSubCollapseBound ((D * D) * BTerm) B)
  (hBLt : B < bInvApprox) :
  withinInvertibilityWindow (mulRq a b) = true := by
  exact withinInvertibilityWindow_mulRq_of_operand_norm_assumptions_rawCoeff
    (a := a) (b := b) (BA := BA) (BB := BB) (BRaw := (D * D) * BTerm) (B := B)
    hA hB hPos
    (mulRqRawCoeffBoundFromOperands_of_schoolbook_term_assumptions_sum
      (hMul := hMul) (hAddTri := hAddTri))
    hAddSub hSub hBLt

theorem withinInvertibilityWindow_mulRq_of_operand_norm_assumptions_via_schoolbook_sum_fieldOp
  {a b : Coeffs} {BA BB BTerm : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hPos : 0 < normInfCoeffs (mulRq a b))
  (hMul : ∀ x y : F, normInfF x ≤ BA → normInfF y ≤ BB → normInfF (x * y) ≤ BTerm)
  (hAddTri : ∀ x y : F, normInfF (x + y) ≤ normInfF x + normInfF y)
  (hOps : rawFieldOpCollapseBound ((D * D) * BTerm) ((D * D) * BTerm))
  (hBLt : ((D * D) * BTerm) < bInvApprox) :
  withinInvertibilityWindow (mulRq a b) = true := by
  exact withinInvertibilityWindow_mulRq_of_operand_norm_assumptions_rawCoeff_fieldOp
    (a := a) (b := b) (BA := BA) (BB := BB) (B := (D * D) * BTerm)
    hA hB hPos
    (mulRqRawCoeffBoundFromOperands_of_schoolbook_term_assumptions_sum
      (hMul := hMul) (hAddTri := hAddTri))
    hOps hBLt

theorem withinInvertibilityWindow_mulRq_of_operand_norm_assumptions_via_schoolbook_of_term_le
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
  withinInvertibilityWindow (mulRq a b) = true := by
  exact withinInvertibilityWindow_mulRq_of_operand_norm_assumptions_rawCoeff
    (a := a) (b := b) (BA := BA) (BB := BB) (BRaw := BRaw) (B := B)
    hA hB hPos
    (mulRqRawCoeffBoundFromOperands_of_schoolbook_term_assumptions_of_term_le
      (hMul := hMul) (hTermLe := hTermLe) (hAddCollapse := hAddCollapse))
    hAddSub hSub hBLt

theorem withinInvertibilityWindow_mulRq_of_operand_norm_assumptions_via_schoolbook_sameBound
  {a b : Coeffs} {BA BB BRaw B : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hPos : 0 < normInfCoeffs (mulRq a b))
  (hMul : ∀ x y : F, normInfF x ≤ BA → normInfF y ≤ BB → normInfF (x * y) ≤ BRaw)
  (hAddCollapse : rawAddCollapseBound BRaw BRaw)
  (hAddSub : rawAddSubCollapseBound BRaw B)
  (hSub : rawSubCollapseBound BRaw B)
  (hBLt : B < bInvApprox) :
  withinInvertibilityWindow (mulRq a b) = true := by
  exact withinInvertibilityWindow_mulRq_of_operand_norm_assumptions_via_schoolbook_of_term_le
    (a := a) (b := b) (BA := BA) (BB := BB) (BTerm := BRaw) (BRaw := BRaw) (B := B)
    hA hB hPos hMul (Nat.le_refl BRaw) hAddCollapse hAddSub hSub hBLt

theorem withinInvertibilityWindow_mulRq_of_operand_norm_assumptions_via_schoolbook_sameBound_fieldOp
  {a b : Coeffs} {BA BB B : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hPos : 0 < normInfCoeffs (mulRq a b))
  (hMul : ∀ x y : F, normInfF x ≤ BA → normInfF y ≤ BB → normInfF (x * y) ≤ B)
  (hOps : rawFieldOpCollapseBound B B)
  (hBLt : B < bInvApprox) :
  withinInvertibilityWindow (mulRq a b) = true := by
  exact withinInvertibilityWindow_mulRq_of_operand_norm_assumptions_rawCoeff_fieldOp
    (a := a) (b := b) (BA := BA) (BB := BB) (B := B)
    hA hB hPos
    (mulRqRawCoeffBoundFromOperands_of_schoolbook_term_assumptions_sameBound
      (hMul := hMul) (hAddCollapse := hOps.1))
    hOps hBLt

/--
Universal-blocker wrapper for the non-coarse P5 path into the P16 window check.
-/
theorem withinInvertibilityWindow_mulRq_of_operand_norm_assumptions_via_universal_blockers
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
  withinInvertibilityWindow (mulRq a b) = true := by
  exact withinInvertibilityWindow_mulRq_of_norm_le_of_lt hPos
    (normInfCoeffs_mulRq_le_of_universal_blockers
      (a := a) (b := b) (BA := BA) (BB := BB)
      hA hB hMulUniv hAddTri hSubTri)
    hBLt

theorem withinInvertibilityWindow_mulRq_of_operand_norm_assumptions_via_universal_blockers_tight
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
  withinInvertibilityWindow (mulRq a b) = true := by
  exact withinInvertibilityWindow_mulRq_of_norm_le_of_lt hPos
    (normInfCoeffs_mulRq_le_of_universal_blockers_tight
      (a := a) (b := b) (BA := BA) (BB := BB)
      hA hB hMulUniv hAddTri hSubTri)
    hBLt

/--
Triangle-bundle variant of
`withinInvertibilityWindow_mulRq_of_operand_norm_assumptions_via_universal_blockers_tight`.
-/
theorem withinInvertibilityWindow_mulRq_of_operand_norm_assumptions_via_universal_mul_and_triangles_tight
  {a b : Coeffs} {BA BB : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hPos : 0 < normInfCoeffs (mulRq a b))
  (hMulUniv : schoolbookMulUniversalBound)
  (hTri : schoolbookTriangleBounds)
  (hBLt :
    (D * (BA * BB)) + (D * (BA * BB)) + (D * (BA * BB))
      < bInvApprox) :
  withinInvertibilityWindow (mulRq a b) = true := by
  exact withinInvertibilityWindow_mulRq_of_norm_le_of_lt hPos
    (normInfCoeffs_mulRq_le_of_universal_mul_and_triangles_tight
      (a := a) (b := b) (BA := BA) (BB := BB)
      hA hB hMulUniv hTri)
    hBLt

theorem withinInvertibilityWindow_mulRq_of_operand_norm_assumptions_via_universal_mul_and_add_tight
  {a b : Coeffs} {BA BB : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hPos : 0 < normInfCoeffs (mulRq a b))
  (hMulUniv : schoolbookMulUniversalBound)
  (hAddTri : schoolbookAddTriangleBound)
  (hBLt :
    (D * (BA * BB)) + (D * (BA * BB)) + (D * (BA * BB))
      < bInvApprox) :
  withinInvertibilityWindow (mulRq a b) = true := by
  exact withinInvertibilityWindow_mulRq_of_operand_norm_assumptions_via_universal_mul_and_triangles_tight
    (a := a) (b := b) (BA := BA) (BB := BB)
    hA hB hPos hMulUniv (schoolbookTriangleBounds_of_add hAddTri) hBLt

theorem withinInvertibilityWindow_mulRq_of_operand_norm_assumptions_via_centeredRep_mul_and_add_tight
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
  withinInvertibilityWindow (mulRq a b) = true := by
  exact withinInvertibilityWindow_mulRq_of_operand_norm_assumptions_via_universal_mul_and_add_tight
    (a := a) (b := b) (BA := BA) (BB := BB)
    hA hB hPos
    (schoolbookMulUniversalBound_of_centeredRep hMulRep)
    (schoolbookAddTriangleBound_of_centeredRep hAddRep)
    hBLt

theorem withinInvertibilityWindow_mulRq_of_operand_norm_assumptions_via_centeredRep_mul_tight
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
  withinInvertibilityWindow (mulRq a b) = true := by
  exact withinInvertibilityWindow_mulRq_of_operand_norm_assumptions_via_centeredRep_mul_and_add_tight
    (a := a) (b := b) (BA := BA) (BB := BB)
    hA hB hPos hMulRep centeredRepAddTriangleBound_theorem hBLt

theorem withinInvertibilityWindow_mulRq_of_operand_norm_assumptions_via_centeredRepMulAddBounds_tight
  {a b : Coeffs} {BA BB : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hPos : 0 < normInfCoeffs (mulRq a b))
  (hRep : centeredRepMulAddBounds)
  (hBLt :
    (D * (BA * BB)) + (D * (BA * BB)) + (D * (BA * BB))
      < bInvApprox) :
  withinInvertibilityWindow (mulRq a b) = true := by
  exact withinInvertibilityWindow_mulRq_of_operand_norm_assumptions_via_centeredRep_mul_and_add_tight
    (a := a) (b := b) (BA := BA) (BB := BB)
    hA hB hPos
    (centeredRepMulAddBounds_mul hRep)
    (centeredRepMulAddBounds_add hRep)
    hBLt

/--
Assumption-free native P16 wrapper using the proved universal blocker theorems.
-/
theorem withinInvertibilityWindow_mulRq_of_operand_norm_assumptions_native
  {a b : Coeffs} {BA BB : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hPos : 0 < normInfCoeffs (mulRq a b))
  (hBLt :
    ((D * D) * (BA * BB)) + ((D * D) * (BA * BB)) + ((D * D) * (BA * BB))
      < bInvApprox) :
  withinInvertibilityWindow (mulRq a b) = true := by
  exact withinInvertibilityWindow_mulRq_of_operand_norm_assumptions_via_universal_blockers
    (a := a) (b := b) (BA := BA) (BB := BB)
    hA hB hPos
    schoolbookMulUniversalBound_theorem
    (schoolbookTriangleBounds_add schoolbookTriangleBounds_theorem)
    (schoolbookTriangleBounds_sub schoolbookTriangleBounds_theorem)
    hBLt

/--
Assumption-free native-tight P16 wrapper (`3 * D * BA * BB` path).
-/
theorem withinInvertibilityWindow_mulRq_of_operand_norm_assumptions_native_tight
  {a b : Coeffs} {BA BB : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hPos : 0 < normInfCoeffs (mulRq a b))
  (hBLt :
    (D * (BA * BB)) + (D * (BA * BB)) + (D * (BA * BB))
      < bInvApprox) :
  withinInvertibilityWindow (mulRq a b) = true := by
  exact withinInvertibilityWindow_mulRq_of_operand_norm_assumptions_via_universal_mul_and_triangles_tight
    (a := a) (b := b) (BA := BA) (BB := BB)
    hA hB hPos
    schoolbookMulUniversalBound_theorem
    schoolbookTriangleBounds_theorem
    hBLt

/--
Assumption-free native P16 wrapper for externally supplied raw schoolbook bounds.
-/
theorem withinInvertibilityWindow_mulRq_of_operand_norm_assumptions_native_and_raw
  {a b : Coeffs} {BA BB BRaw : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hPos : 0 < normInfCoeffs (mulRq a b))
  (hRawFromOperands : mulRqRawNormBoundFromOperands BA BB BRaw)
  (hBLt : BRaw + BRaw + BRaw < bInvApprox) :
  withinInvertibilityWindow (mulRq a b) = true := by
  exact withinInvertibilityWindow_mulRq_of_norm_le_of_lt hPos
    (normInfCoeffs_mulRq_le_native_and_raw
      (a := a) (b := b) (BA := BA) (BB := BB) (BRaw := BRaw)
      hA hB hRawFromOperands)
    hBLt

/--
Raw-coeff native variant of `withinInvertibilityWindow_mulRq_of_operand_norm_assumptions_native_and_raw`.
-/
theorem withinInvertibilityWindow_mulRq_of_operand_norm_assumptions_native_and_rawCoeff
  {a b : Coeffs} {BA BB BRaw : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hPos : 0 < normInfCoeffs (mulRq a b))
  (hRawCoeffFromOperands : mulRqRawCoeffBoundFromOperands BA BB BRaw)
  (hBLt : BRaw + BRaw + BRaw < bInvApprox) :
  withinInvertibilityWindow (mulRq a b) = true := by
  exact withinInvertibilityWindow_mulRq_of_operand_norm_assumptions_native_and_raw
    (a := a) (b := b) (BA := BA) (BB := BB) (BRaw := BRaw)
    hA hB hPos
    (mulRqRawNormBoundFromOperands_of_rawCoeff hRawCoeffFromOperands)
    hBLt

theorem withinInvertibilityWindow_mulRq_of_operand_norm_assumptions_via_universal_blockers_and_raw
  {a b : Coeffs} {BA BB BRaw : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hPos : 0 < normInfCoeffs (mulRq a b))
  (hRawFromOperands : mulRqRawNormBoundFromOperands BA BB BRaw)
  (hAddTri : schoolbookAddTriangleBound)
  (hSubTri : schoolbookSubTriangleBound)
  (hBLt : BRaw + BRaw + BRaw < bInvApprox) :
  withinInvertibilityWindow (mulRq a b) = true := by
  exact withinInvertibilityWindow_mulRq_of_norm_le_of_lt hPos
    (normInfCoeffs_mulRq_le_of_universal_blockers_and_raw
      (a := a) (b := b) (BA := BA) (BB := BB) (BRaw := BRaw)
      hA hB hRawFromOperands hAddTri hSubTri)
    hBLt

theorem withinInvertibilityWindow_mulRq_of_operand_norm_assumptions_via_universal_blockers_and_rawCoeff
  {a b : Coeffs} {BA BB BRaw : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hPos : 0 < normInfCoeffs (mulRq a b))
  (hRawCoeffFromOperands : mulRqRawCoeffBoundFromOperands BA BB BRaw)
  (hAddTri : schoolbookAddTriangleBound)
  (hSubTri : schoolbookSubTriangleBound)
  (hBLt : BRaw + BRaw + BRaw < bInvApprox) :
  withinInvertibilityWindow (mulRq a b) = true := by
  exact withinInvertibilityWindow_mulRq_of_operand_norm_assumptions_via_universal_blockers_and_raw
    (a := a) (b := b) (BA := BA) (BB := BB) (BRaw := BRaw)
    hA hB hPos
    (mulRqRawNormBoundFromOperands_of_rawCoeff hRawCoeffFromOperands)
    hAddTri hSubTri hBLt

theorem withinInvertibilityWindow_mulRq_of_operand_norm_assumptions_via_universal_mul_and_add_and_raw
  {a b : Coeffs} {BA BB BRaw : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hPos : 0 < normInfCoeffs (mulRq a b))
  (hRawFromOperands : mulRqRawNormBoundFromOperands BA BB BRaw)
  (hAddTri : schoolbookAddTriangleBound)
  (hBLt : BRaw + BRaw + BRaw < bInvApprox) :
  withinInvertibilityWindow (mulRq a b) = true := by
  exact withinInvertibilityWindow_mulRq_of_operand_norm_assumptions_via_universal_blockers_and_raw
    (a := a) (b := b) (BA := BA) (BB := BB) (BRaw := BRaw)
    hA hB hPos hRawFromOperands hAddTri (schoolbookSubTriangleBound_of_add hAddTri) hBLt

theorem withinInvertibilityWindow_mulRq_of_operand_norm_assumptions_via_universal_mul_and_add_and_rawCoeff
  {a b : Coeffs} {BA BB BRaw : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hPos : 0 < normInfCoeffs (mulRq a b))
  (hRawCoeffFromOperands : mulRqRawCoeffBoundFromOperands BA BB BRaw)
  (hAddTri : schoolbookAddTriangleBound)
  (hBLt : BRaw + BRaw + BRaw < bInvApprox) :
  withinInvertibilityWindow (mulRq a b) = true := by
  exact withinInvertibilityWindow_mulRq_of_operand_norm_assumptions_via_universal_mul_and_add_and_raw
    (a := a) (b := b) (BA := BA) (BB := BB) (BRaw := BRaw)
    hA hB hPos
    (mulRqRawNormBoundFromOperands_of_rawCoeff hRawCoeffFromOperands)
    hAddTri hBLt

theorem withinInvertibilityWindow_mulRq_of_operand_norm_assumptions_via_centeredRep_mul_and_add_and_raw
  {a b : Coeffs} {BA BB BRaw : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hPos : 0 < normInfCoeffs (mulRq a b))
  (hRawFromOperands : mulRqRawNormBoundFromOperands BA BB BRaw)
  (hAddRep :
    ∀ x y : F, Int.natAbs (centeredRep (x + y))
      ≤ Int.natAbs (centeredRep x) + Int.natAbs (centeredRep y))
  (hBLt : BRaw + BRaw + BRaw < bInvApprox) :
  withinInvertibilityWindow (mulRq a b) = true := by
  exact withinInvertibilityWindow_mulRq_of_operand_norm_assumptions_via_universal_mul_and_add_and_raw
    (a := a) (b := b) (BA := BA) (BB := BB) (BRaw := BRaw)
    hA hB hPos hRawFromOperands
    (schoolbookAddTriangleBound_of_centeredRep hAddRep)
    hBLt

theorem withinInvertibilityWindow_mulRq_of_operand_norm_assumptions_via_centeredRep_mul_and_add_and_rawCoeff
  {a b : Coeffs} {BA BB BRaw : Nat}
  (hA : normInfCoeffs a ≤ BA)
  (hB : normInfCoeffs b ≤ BB)
  (hPos : 0 < normInfCoeffs (mulRq a b))
  (hRawCoeffFromOperands : mulRqRawCoeffBoundFromOperands BA BB BRaw)
  (hAddRep :
    ∀ x y : F, Int.natAbs (centeredRep (x + y))
      ≤ Int.natAbs (centeredRep x) + Int.natAbs (centeredRep y))
  (hBLt : BRaw + BRaw + BRaw < bInvApprox) :
  withinInvertibilityWindow (mulRq a b) = true := by
  exact withinInvertibilityWindow_mulRq_of_operand_norm_assumptions_via_centeredRep_mul_and_add_and_raw
    (a := a) (b := b) (BA := BA) (BB := BB) (BRaw := BRaw)
    hA hB hPos
    (mulRqRawNormBoundFromOperands_of_rawCoeff hRawCoeffFromOperands)
    hAddRep hBLt

theorem withinInvertibilityWindow_mulRq_of_goldilocks_operand_assumptions
  {a b : Coeffs}
  (hA : normInfCoeffs a ≤ Parameters.Goldilocks.B)
  (hB : normInfCoeffs b ≤ Parameters.Goldilocks.B)
  (hPos : 0 < normInfCoeffs (mulRq a b))
  (hRaw : GoldilocksRawNormBoundAssumption)
  (hCollapse : GoldilocksRawCollapseAssumption) :
  withinInvertibilityWindow (mulRq a b) = true := by
  rcases hCollapse with ⟨hAddSub, hSub⟩
  exact withinInvertibilityWindow_mulRq_of_operand_norm_assumptions
    (a := a) (b := b)
    (BA := Parameters.Goldilocks.B)
    (BB := Parameters.Goldilocks.B)
    (BRaw := Parameters.Goldilocks.B)
    (B := Parameters.Goldilocks.B)
    hA hB hPos hRaw hAddSub hSub goldilocksB_lt_bInvApprox

theorem withinInvertibilityWindow_mulRq_of_goldilocks_operand_assumptions_inRange
  {a b : Coeffs}
  (hA : normInfCoeffs a ≤ Parameters.Goldilocks.B)
  (hB : normInfCoeffs b ≤ Parameters.Goldilocks.B)
  (hPos : 0 < normInfCoeffs (mulRq a b))
  (hRawInRange : GoldilocksRawInRangeBoundAssumption)
  (hCollapse : GoldilocksRawCollapseAssumption) :
  withinInvertibilityWindow (mulRq a b) = true := by
  rcases hCollapse with ⟨hAddSub, hSub⟩
  exact withinInvertibilityWindow_mulRq_of_operand_norm_assumptions_inRange
    (a := a) (b := b)
    (BA := Parameters.Goldilocks.B)
    (BB := Parameters.Goldilocks.B)
    (BRaw := Parameters.Goldilocks.B)
    (B := Parameters.Goldilocks.B)
    hA hB hPos hRawInRange hAddSub hSub goldilocksB_lt_bInvApprox

theorem withinInvertibilityWindow_mulRq_of_goldilocks_operand_fieldOp_assumptions
  {a b : Coeffs}
  (hA : normInfCoeffs a ≤ Parameters.Goldilocks.B)
  (hB : normInfCoeffs b ≤ Parameters.Goldilocks.B)
  (hPos : 0 < normInfCoeffs (mulRq a b))
  (hRaw : GoldilocksRawNormBoundAssumption)
  (hFieldOps : GoldilocksFieldOpCollapseAssumption) :
  withinInvertibilityWindow (mulRq a b) = true := by
  exact withinInvertibilityWindow_mulRq_of_goldilocks_operand_assumptions
    (a := a) (b := b) hA hB hPos hRaw
    (goldilocksRawCollapseAssumption_of_fieldOp hFieldOps)

theorem withinInvertibilityWindow_mulRq_of_goldilocks_operand_fieldOp_assumptions_inRange
  {a b : Coeffs}
  (hA : normInfCoeffs a ≤ Parameters.Goldilocks.B)
  (hB : normInfCoeffs b ≤ Parameters.Goldilocks.B)
  (hPos : 0 < normInfCoeffs (mulRq a b))
  (hRawInRange : GoldilocksRawInRangeBoundAssumption)
  (hFieldOps : GoldilocksFieldOpCollapseAssumption) :
  withinInvertibilityWindow (mulRq a b) = true := by
  exact withinInvertibilityWindow_mulRq_of_goldilocks_operand_assumptions_inRange
    (a := a) (b := b) hA hB hPos hRawInRange
    (goldilocksRawCollapseAssumption_of_fieldOp hFieldOps)

theorem withinInvertibilityWindow_mulRq_of_goldilocks_operand_rawCoeff_assumptions
  {a b : Coeffs}
  (hA : normInfCoeffs a ≤ Parameters.Goldilocks.B)
  (hB : normInfCoeffs b ≤ Parameters.Goldilocks.B)
  (hPos : 0 < normInfCoeffs (mulRq a b))
  (hRawCoeff : GoldilocksRawCoeffBoundAssumption)
  (hCollapse : GoldilocksRawCollapseAssumption) :
  withinInvertibilityWindow (mulRq a b) = true := by
  exact withinInvertibilityWindow_mulRq_of_goldilocks_operand_assumptions
    (a := a) (b := b)
    hA hB hPos
    (goldilocksRawNormBoundAssumption_of_rawCoeff hRawCoeff)
    hCollapse

theorem withinInvertibilityWindow_mulRq_of_goldilocks_operand_rawCoeff_fieldOp_assumptions
  {a b : Coeffs}
  (hA : normInfCoeffs a ≤ Parameters.Goldilocks.B)
  (hB : normInfCoeffs b ≤ Parameters.Goldilocks.B)
  (hPos : 0 < normInfCoeffs (mulRq a b))
  (hRawCoeff : GoldilocksRawCoeffBoundAssumption)
  (hFieldOps : GoldilocksFieldOpCollapseAssumption) :
  withinInvertibilityWindow (mulRq a b) = true := by
  exact withinInvertibilityWindow_mulRq_of_goldilocks_operand_rawCoeff_assumptions
    (a := a) (b := b) hA hB hPos hRawCoeff
    (goldilocksRawCollapseAssumption_of_fieldOp hFieldOps)

theorem withinInvertibilityWindow_of_allChallenge
  {a : Coeffs}
  (hAll : AllChallengeCoeffs a)
  (hPos : 0 < normInfCoeffs a) :
  withinInvertibilityWindow a = true := by
  exact withinInvertibilityWindow_complete
    ⟨hPos, normInfCoeffs_lt_bInvApprox_of_allChallenge hAll⟩

theorem withinInvertibilityWindow_of_allChallenge_sub
  {a b : Coeffs}
  (hSize : a.size = b.size)
  (hAllA : AllChallengeCoeffs a)
  (hAllB : AllChallengeCoeffs b)
  (hPos : 0 < normInfCoeffs (coeffSub a b)) :
  withinInvertibilityWindow (coeffSub a b) = true := by
  exact withinInvertibilityWindow_complete
    ⟨hPos, normInfCoeffs_sub_lt_bInvApprox_of_allChallenge hSize hAllA hAllB⟩


end SuperNeo
