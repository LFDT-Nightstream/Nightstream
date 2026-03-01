import SuperNeo.Ring

namespace SuperNeo

/-- Infinity norm on field elements via centered representatives. -/
def normInfF (x : F) : Nat :=
  F.centeredAbs x

/-- Infinity norm on coefficient arrays. -/
def normInfCoeffs (a : Coeffs) : Nat :=
  a.foldl (fun acc x => Nat.max acc (normInfF x)) 0

/-- Alias used by protocol-facing statements. -/
def maxRhoNorm (a : Coeffs) : Nat :=
  normInfCoeffs a

@[simp] theorem normInfF_zero : normInfF (0 : F) = 0 := by
  have hVal : (0 : F).val = 0 := by
    change (F.ofNat 0).val = 0
    simp [F.ofNat]
  have hRep : F.centeredRep (0 : F) = Int.ofNat (0 : F).val := by
    apply F.centeredRep_eq_of_le_halfQ
    simp [hVal]
  simp [normInfF, F.centeredAbs, hRep, hVal]

@[simp] theorem normInfCoeffs_empty : normInfCoeffs (#[] : Coeffs) = 0 := by
  simp [normInfCoeffs]

theorem normInfCoeffs_nonneg (a : Coeffs) : 0 ≤ normInfCoeffs a :=
  Nat.zero_le _

theorem maxRhoNorm_nonneg (a : Coeffs) : 0 ≤ maxRhoNorm a :=
  normInfCoeffs_nonneg a

/-- Assumption bundle: vector-add norm bound from operand norms. -/
def vecAddNormBoundFromOperands (BA BB B : Nat) : Prop :=
  ∀ a b : Coeffs, a.size = b.size →
    normInfCoeffs a ≤ BA →
    normInfCoeffs b ≤ BB →
    normInfCoeffs (vecAdd a b) ≤ B

/-- Assumption bundle: vector-scale norm bound from operand norms. -/
def vecScaleNormBoundFromOperands (BS BA B : Nat) : Prop :=
  ∀ s : F, ∀ a : Coeffs,
    normInfF s ≤ BS →
    normInfCoeffs a ≤ BA →
    normInfCoeffs (vecScale s a) ≤ B

/-- Assumption bundle: ring multiplication norm bound from operand norms. -/
def mulRqNormBoundFromOperands (BA BB B : Nat) : Prop :=
  ∀ a b : Coeffs,
    normInfCoeffs a ≤ BA →
    normInfCoeffs b ≤ BB →
    normInfCoeffs (mulRq a b) ≤ B

/-- Assumption bundle: subtraction norm bound from operand norms. -/
def coeffSubNormBoundFromOperands (BA BB B : Nat) : Prop :=
  ∀ a b : Coeffs, a.size = b.size →
    normInfCoeffs a ≤ BA →
    normInfCoeffs b ≤ BB →
    normInfCoeffs (vecAdd a (vecScale (-1) b)) ≤ B

def AllChallengeCoeffs (a : Coeffs) : Prop :=
  ∀ i : Fin a.size, normInfF a[i] ≤ 2

theorem allChallengeCoeffs_empty : AllChallengeCoeffs (#[] : Coeffs) := by
  intro i
  exact False.elim (Nat.not_lt_zero _ i.2)

end SuperNeo
