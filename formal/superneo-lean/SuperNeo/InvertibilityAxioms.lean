import SuperNeo.Parameters
import SuperNeo.Norm
import SuperNeo.Ring

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

axiom lowNormInvertibility : LowNormInvertibilityAssumption

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
  unfold invertibilityPreconditionsProp challengeCoeffMaxDiff bInvApprox
  constructor
  · exact rfl
  constructor
  · exact rfl
  constructor
  · decide
  ·
    have hB : Parameters.Goldilocks.B = 16384 := Parameters.Goldilocks.B_eq_16384
    simp [hB]

theorem challengeCoeffMaxDiff_lt_bInvApprox : challengeCoeffMaxDiff < bInvApprox := by
  simpa [challengeCoeffMaxDiff_eq_four] using four_lt_bInvApprox

theorem goldilocksB_lt_bInvApprox : Parameters.Goldilocks.B < bInvApprox := by
  have hB : Parameters.Goldilocks.B = 16384 := Parameters.Goldilocks.B_eq_16384
  simp [hB, bInvApprox]

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

theorem invertible_of_norm_bounds
  {a : Coeffs}
  (hPos : 0 < normInfCoeffs a)
  (hLt : normInfCoeffs a < bInvApprox) :
  ∃ b : Coeffs, mulRq a b = oneRq := by
  exact lowNormInvertibility a hPos hLt

theorem invertible_of_withinInvertibilityWindow
  {a : Coeffs}
  (hWin : withinInvertibilityWindow a = true) :
  ∃ b : Coeffs, mulRq a b = oneRq := by
  rcases withinInvertibilityWindow_sound hWin with ⟨hPos, hLt⟩
  exact invertible_of_norm_bounds hPos hLt

theorem invertible_of_allChallenge_nonzero
  {a : Coeffs}
  (hAll : AllChallengeCoeffs a)
  (hPos : 0 < normInfCoeffs a) :
  ∃ b : Coeffs, mulRq a b = oneRq := by
  exact invertible_of_withinInvertibilityWindow
    (withinInvertibilityWindow_of_allChallenge hAll hPos)

theorem invertible_of_allChallenge_sub_nonzero
  {a b : Coeffs}
  (hSize : a.size = b.size)
  (hAllA : AllChallengeCoeffs a)
  (hAllB : AllChallengeCoeffs b)
  (hPos : 0 < normInfCoeffs (coeffSub a b)) :
  ∃ c : Coeffs, mulRq (coeffSub a b) c = oneRq := by
  exact invertible_of_withinInvertibilityWindow
    (withinInvertibilityWindow_of_allChallenge_sub hSize hAllA hAllB hPos)

end SuperNeo
