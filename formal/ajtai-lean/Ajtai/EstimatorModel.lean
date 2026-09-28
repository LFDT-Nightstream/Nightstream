import Ajtai.LogInterval
import Ajtai.Parameters
import Mathlib.Analysis.Real.Pi.Bounds

/-!
Executable log-domain model for the rank-two protocol-binding SIS estimate.

The selected policy is ADPS16 quantum Core-SVP (`0.265 β`), 128 post-union
bits, and seven verifier-key setup targets conservatively rounded to eight.
This proves the arithmetic inside that model; it does not prove Module-SIS
hardness or the structured-matrix heuristic.
-/

namespace Ajtai.EstimatorModel

open Ajtai.LogInterval
open Ajtai.Parameters
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.ShiftedTernaryCompiler
open Nightstream.SuperNeo.Concrete

def piLowerNumerator : Nat := 314159265358979323846
def piUpperNumerator : Nat := 314159265358979323847
def decimalDenominator : Nat := 10 ^ 20

def lnTwoLower : ℚ :=
  logRatioLower (1 / 3) precisionTerms

def lnTwoUpper : ℚ :=
  logRatioUpper (1 / 3) precisionTerms

def lnPiLower : ℚ :=
  lnRatLower piLowerNumerator decimalDenominator

def lnPiUpper : ℚ :=
  lnRatUpper piUpperNumerator decimalDenominator

theorem ln_two_bounds :
    ((lnTwoLower : ℚ) : ℝ) ≤ Real.log 2 ∧
      Real.log 2 ≤ ((lnTwoUpper : ℚ) : ℝ) := by
  have bounds :=
    log_ratio_bounds (1 / 3) precisionTerms (by norm_num) (by norm_num)
  norm_num at bounds ⊢
  exact bounds

theorem ln_pi_bounds :
    ((lnPiLower : ℚ) : ℝ) ≤ Real.log Real.pi ∧
      Real.log Real.pi ≤ ((lnPiUpper : ℚ) : ℝ) := by
  have lowerBounds :
      ((lnRatLower piLowerNumerator decimalDenominator : ℚ) : ℝ) ≤
        Real.log
          ((piLowerNumerator : ℝ) / (decimalDenominator : ℝ)) :=
    (ln_rat_bounds (by decide) (by decide)).1
  have upperBounds :
      Real.log
          ((piUpperNumerator : ℝ) / (decimalDenominator : ℝ)) ≤
        ((lnRatUpper piUpperNumerator decimalDenominator : ℚ) : ℝ) :=
    (ln_rat_bounds (by decide) (by decide)).2
  have lowerValue :
      (piLowerNumerator : ℝ) / (decimalDenominator : ℝ) < Real.pi := by
    have valueEq :
        (piLowerNumerator : ℝ) / (decimalDenominator : ℝ) =
          3.14159265358979323846 := by
      norm_num [piLowerNumerator, decimalDenominator]
    rw [valueEq]
    exact Real.pi_gt_d20
  have upperValue :
      Real.pi < (piUpperNumerator : ℝ) / (decimalDenominator : ℝ) := by
    have valueEq :
        (piUpperNumerator : ℝ) / (decimalDenominator : ℝ) =
          3.14159265358979323847 := by
      norm_num [piUpperNumerator, decimalDenominator]
    rw [valueEq]
    exact Real.pi_lt_d20
  constructor
  · exact lowerBounds.trans
      (Real.log_le_log
        (by norm_num [piLowerNumerator, decimalDenominator]) lowerValue.le)
  · exact (Real.log_le_log Real.pi_pos upperValue.le).trans upperBounds

def lnDeltaLower (beta : Nat) : ℚ :=
  (lnNatLower beta - lnTwoUpper - lnPiUpper - 1 +
      (lnPiLower + lnNatLower beta) / beta) /
    (2 * ((beta - 1 : Nat) : ℚ))

def lnDeltaUpper (beta : Nat) : ℚ :=
  (lnNatUpper beta - lnTwoLower - lnPiLower - 1 +
      (lnPiUpper + lnNatUpper beta) / beta) /
    (2 * ((beta - 1 : Nat) : ℚ))

/-- Chen's root-Hermite expression used by the pinned estimator for `β > 40`,
written in the log domain. -/
noncomputable def chenLogDelta (beta : Nat) : ℝ :=
  (Real.log (beta : ℝ) - Real.log 2 - Real.log Real.pi - 1 +
      (Real.log Real.pi + Real.log (beta : ℝ)) / beta) /
    (2 * ((beta - 1 : Nat) : ℝ))

theorem ln_delta_bounds {beta : Nat} (betaGtOne : 1 < beta) :
    ((lnDeltaLower beta : ℚ) : ℝ) ≤ chenLogDelta beta ∧
      chenLogDelta beta ≤ ((lnDeltaUpper beta : ℚ) : ℝ) := by
  have betaBounds :=
    ln_nat_bounds (Nat.ne_of_gt (Nat.lt_trans Nat.zero_lt_one betaGtOne))
  have twoBounds := ln_two_bounds
  have piBounds := ln_pi_bounds
  have betaPosReal : (0 : ℝ) < beta := by positivity
  have betaMinusOnePos : 0 < beta - 1 := by omega
  have denominatorPosReal :
      (0 : ℝ) < 2 * ((beta - 1 : Nat) : ℝ) := by
    positivity
  constructor
  · unfold lnDeltaLower chenLogDelta
    push_cast
    have ratioLe :
        (((lnPiLower : ℚ) : ℝ) + ((lnNatLower beta : ℚ) : ℝ)) / beta ≤
          (Real.log Real.pi + Real.log (beta : ℝ)) / beta := by
      exact div_le_div_of_nonneg_right
        (add_le_add piBounds.1 betaBounds.1) betaPosReal.le
    exact (div_le_div_iff_of_pos_right denominatorPosReal).2 (by linarith)
  · unfold lnDeltaUpper chenLogDelta
    push_cast
    have ratioLe :
        (Real.log Real.pi + Real.log (beta : ℝ)) / beta ≤
          (((lnPiUpper : ℚ) : ℝ) + ((lnNatUpper beta : ℚ) : ℝ)) / beta := by
      exact div_le_div_of_nonneg_right
        (add_le_add piBounds.2 betaBounds.2) betaPosReal.le
    exact (div_le_div_iff_of_pos_right denominatorPosReal).2 (by linarith)

def targetSecurityBits : Nat := 128
def setupAttackTargets : Nat := 7

def ceilLogTwo (n : Nat) : Nat :=
  if n ≤ 1 then 0 else (n - 1).log2 + 1

def quantumCostNumerator : Nat := 265
def quantumCostDenominator : Nat := 1_000

def requiredRawBits : Nat :=
  targetSecurityBits + ceilLogTwo setupAttackTargets

def minimumAcceptedBeta : Nat :=
  (quantumCostDenominator * requiredRawBits +
      quantumCostNumerator - 1) /
    quantumCostNumerator

def rejectedBeta : Nat :=
  minimumAcceptedBeta - 1

def betaMeetsCostTarget (beta : Nat) : Prop :=
  quantumCostDenominator * requiredRawBits ≤
    quantumCostNumerator * beta

theorem selected_policy_values :
    ceilLogTwo setupAttackTargets = 3 ∧
      requiredRawBits = 131 ∧
      minimumAcceptedBeta = 495 ∧
      rejectedBeta = 494 := by
  decide

theorem beta_meets_target_iff (beta : Nat) :
    betaMeetsCostTarget beta ↔ minimumAcceptedBeta ≤ beta := by
  simp only [betaMeetsCostTarget]
  rw [selected_policy_values.2.1, selected_policy_values.2.2.1]
  simp only [quantumCostDenominator, quantumCostNumerator]
  omega

def sisModulus : Nat := goldilocksP

def latticeRankDimension : Nat :=
  protocolBindingRank * ringDegree

def coefficientColumns (ringColumns : Nat) : Nat :=
  ringColumns * ringDegree

def lnVolumeLower : ℚ :=
  latticeRankDimension * lnNatLower sisModulus

def lnVolumeUpper : ℚ :=
  latticeRankDimension * lnNatUpper sisModulus

def lnLengthBoundLower (ringColumns : Nat) : ℚ :=
  lnTwoLower + lnNatLower (coefficientColumns ringColumns) / 2

def lnLengthBoundUpper (ringColumns : Nat) : ℚ :=
  lnTwoUpper + lnNatUpper (coefficientColumns ringColumns) / 2

def secureQuadraticLower (ringColumns dimension : Nat) : ℚ :=
  lnDeltaLower rejectedBeta * dimension * (dimension - 1) -
    lnLengthBoundUpper ringColumns * dimension +
    lnVolumeLower

def attackQuadraticUpper (ringColumns dimension : Nat) : ℚ :=
  lnDeltaUpper rejectedBeta * dimension * (dimension - 1) -
    lnLengthBoundLower ringColumns * dimension +
    lnVolumeUpper

def selectedAttackDimension : Nat := 1_182

def CertifiedSecure (ringColumns : Nat) : Prop :=
  0 < lnDeltaLower rejectedBeta ∧
    2 * lnDeltaUpper rejectedBeta * (selectedAttackDimension - 1) <
      lnLengthBoundLower ringColumns ∧
    lnLengthBoundUpper ringColumns ≤
      2 * lnDeltaLower rejectedBeta * selectedAttackDimension ∧
    0 < secureQuadraticLower ringColumns selectedAttackDimension

def CertifiedAttack (ringColumns dimension : Nat) : Prop :=
  attackQuadraticUpper ringColumns dimension < 0

instance (ringColumns : Nat) : Decidable (CertifiedSecure ringColumns) := by
  unfold CertifiedSecure
  infer_instance

instance (ringColumns dimension : Nat) :
    Decidable (CertifiedAttack ringColumns dimension) := by
  unfold CertifiedAttack
  infer_instance

noncomputable def exactLogVolume : ℝ :=
  latticeRankDimension * Real.log (sisModulus : ℝ)

/-- `log(2√m)` for `m = 54 * ringColumns`. -/
noncomputable def exactLogLengthBound (ringColumns : Nat) : ℝ :=
  Real.log 2 + Real.log (coefficientColumns ringColumns : ℝ) / 2

noncomputable def requiredLogDelta
    (ringColumns dimension : Nat) : ℝ :=
  (exactLogLengthBound ringColumns * dimension - exactLogVolume) /
    ((dimension : ℝ) * ((dimension : ℝ) - 1))

noncomputable def quadratic
    (t b volume : ℝ) (dimension : Nat) : ℝ :=
  t * dimension * ((dimension : ℝ) - 1) -
    b * dimension + volume

theorem quadratic_difference
    (t b volume : ℝ) (left right : Nat) :
    quadratic t b volume left - quadratic t b volume right =
      ((left : ℝ) - right) *
        (t * ((left : ℝ) + right - 1) - b) := by
  unfold quadratic
  ring

theorem quadratic_pos_of_discrete_min
    {t b volume : ℝ} {pivot : Nat}
    (tPos : 0 < t)
    (decreasingBefore : 2 * t * ((pivot : ℝ) - 1) < b)
    (increasingAfter : b ≤ 2 * t * pivot)
    (minimumPos : 0 < quadratic t b volume pivot) :
    ∀ dimension : Nat, 0 < quadratic t b volume dimension := by
  intro dimension
  by_cases below : dimension < pivot
  · have dimensionLe : (dimension : ℝ) ≤ (pivot : ℝ) - 1 := by
      have castIntegerGap : (dimension : ℝ) + 1 ≤ pivot := by
        exact_mod_cast (show dimension + 1 ≤ pivot by omega)
      linarith
    have sumLe :
        (dimension : ℝ) + pivot - 1 ≤ 2 * ((pivot : ℝ) - 1) := by
      linarith
    have scaledLe :
        t * ((dimension : ℝ) + pivot - 1) ≤
          t * (2 * ((pivot : ℝ) - 1)) :=
      mul_le_mul_of_nonneg_left sumLe tPos.le
    have factorPos :
        0 < b - t * ((dimension : ℝ) + pivot - 1) := by
      nlinarith
    have distancePos : 0 < (pivot : ℝ) - dimension := by
      have castBelow : (dimension : ℝ) < pivot := by exact_mod_cast below
      linarith
    have productPos :
        0 < ((pivot : ℝ) - dimension) *
          (b - t * ((dimension : ℝ) + pivot - 1)) :=
      mul_pos distancePos factorPos
    have productEq :
        ((pivot : ℝ) - dimension) *
            (b - t * ((dimension : ℝ) + pivot - 1)) =
          quadratic t b volume dimension -
            quadratic t b volume pivot := by
      rw [quadratic_difference]
      ring
    rw [productEq] at productPos
    linarith
  · have pivotLe : pivot ≤ dimension := Nat.le_of_not_gt below
    by_cases equal : pivot = dimension
    · simpa [equal] using minimumPos
    · have pivotLt : pivot < dimension := Nat.lt_of_le_of_ne pivotLe equal
      have sumGe :
          2 * (pivot : ℝ) ≤ (dimension : ℝ) + pivot - 1 := by
        have castGap : (pivot : ℝ) + 1 ≤ dimension := by
          exact_mod_cast (show pivot + 1 ≤ dimension by omega)
        linarith
      have scaledGe :
          t * (2 * (pivot : ℝ)) ≤
            t * ((dimension : ℝ) + pivot - 1) :=
        mul_le_mul_of_nonneg_left sumGe tPos.le
      have factorNonneg :
          0 ≤ t * ((dimension : ℝ) + pivot - 1) - b := by
        nlinarith
      have distanceNonneg : 0 ≤ (dimension : ℝ) - pivot := by
        have castLe : (pivot : ℝ) ≤ dimension := by exact_mod_cast pivotLe
        linarith
      have productNonneg :
          0 ≤ ((dimension : ℝ) - pivot) *
            (t * ((dimension : ℝ) + pivot - 1) - b) :=
        mul_nonneg distanceNonneg factorNonneg
      rw [← quadratic_difference t b volume dimension pivot] at productNonneg
      linarith

theorem exact_log_volume_bounds :
    ((lnVolumeLower : ℚ) : ℝ) ≤ exactLogVolume ∧
      exactLogVolume ≤ ((lnVolumeUpper : ℚ) : ℝ) := by
  have modulusBounds := ln_nat_bounds (show sisModulus ≠ 0 by decide)
  unfold lnVolumeLower lnVolumeUpper exactLogVolume
  push_cast
  constructor
  · exact mul_le_mul_of_nonneg_left modulusBounds.1 (by positivity)
  · exact mul_le_mul_of_nonneg_left modulusBounds.2 (by positivity)

theorem exact_log_length_bound_bounds
    {ringColumns : Nat} (positive : 0 < ringColumns) :
    ((lnLengthBoundLower ringColumns : ℚ) : ℝ) ≤
        exactLogLengthBound ringColumns ∧
      exactLogLengthBound ringColumns ≤
        ((lnLengthBoundUpper ringColumns : ℚ) : ℝ) := by
  have columnNonzero : coefficientColumns ringColumns ≠ 0 := by
    unfold coefficientColumns ringDegree
    omega
  have columnBounds := ln_nat_bounds columnNonzero
  have twoBounds := ln_two_bounds
  unfold lnLengthBoundLower lnLengthBoundUpper exactLogLengthBound
  push_cast
  constructor <;> linarith

/-- Every lattice dimension requires at least `minimumAcceptedBeta` under the
selected Chen root-Hermite threshold. -/
def WidthAccepted (ringColumns : Nat) : Prop :=
  0 < ringColumns ∧
    ∀ dimension : Nat, 2 ≤ dimension →
      requiredLogDelta ringColumns dimension <
        chenLogDelta rejectedBeta

def WidthRejectedBy
    (ringColumns dimension : Nat) : Prop :=
  2 ≤ dimension ∧
    chenLogDelta rejectedBeta ≤
      requiredLogDelta ringColumns dimension

theorem required_log_delta_lt_of_quadratic_pos
    {ringColumns dimension : Nat}
    (dimensionAtLeastTwo : 2 ≤ dimension)
    (positive :
      0 < quadratic
        (chenLogDelta rejectedBeta)
        (exactLogLengthBound ringColumns)
        exactLogVolume
        dimension) :
    requiredLogDelta ringColumns dimension <
      chenLogDelta rejectedBeta := by
  have dimensionPos : (0 : ℝ) < dimension := by positivity
  have dimensionMinusOnePos : (0 : ℝ) < (dimension : ℝ) - 1 := by
    have castLt : (1 : ℝ) < dimension := by
      exact_mod_cast (show 1 < dimension by omega)
    linarith
  have denominatorPos :
      (0 : ℝ) < (dimension : ℝ) * ((dimension : ℝ) - 1) :=
    mul_pos dimensionPos dimensionMinusOnePos
  unfold requiredLogDelta
  apply (div_lt_iff₀ denominatorPos).2
  unfold quadratic at positive
  nlinarith

theorem required_log_delta_ge_of_quadratic_nonpos
    {ringColumns dimension : Nat}
    (dimensionAtLeastTwo : 2 ≤ dimension)
    (nonpositive :
      quadratic
        (chenLogDelta rejectedBeta)
        (exactLogLengthBound ringColumns)
        exactLogVolume
        dimension ≤ 0) :
    chenLogDelta rejectedBeta ≤
      requiredLogDelta ringColumns dimension := by
  have dimensionPos : (0 : ℝ) < dimension := by positivity
  have dimensionMinusOnePos : (0 : ℝ) < (dimension : ℝ) - 1 := by
    have castLt : (1 : ℝ) < dimension := by
      exact_mod_cast (show 1 < dimension by omega)
    linarith
  have denominatorPos :
      (0 : ℝ) < (dimension : ℝ) * ((dimension : ℝ) - 1) :=
    mul_pos dimensionPos dimensionMinusOnePos
  unfold requiredLogDelta
  apply (le_div_iff₀ denominatorPos).2
  unfold quadratic at nonpositive
  nlinarith

theorem certified_secure_boundary :
    CertifiedSecure 50_371 := by
  native_decide

theorem certified_attack_after_boundary :
    CertifiedAttack 50_372 selectedAttackDimension := by
  native_decide

theorem width_50371_accepted :
    WidthAccepted 50_371 := by
  have deltaBounds :=
    ln_delta_bounds
      (beta := rejectedBeta)
      (show 1 < rejectedBeta by native_decide)
  have lengthBounds :=
    exact_log_length_bound_bounds (ringColumns := 50_371) (by norm_num)
  have volumeBounds := exact_log_volume_bounds
  have certificate := certified_secure_boundary
  unfold CertifiedSecure at certificate
  rcases certificate with
    ⟨deltaLowerPos, leftCertificate, rightCertificate, minimumCertificate⟩
  have deltaLowerPosReal :
      (0 : ℝ) < ((lnDeltaLower rejectedBeta : ℚ) : ℝ) := by
    exact_mod_cast deltaLowerPos
  have leftCertificateReal :
      2 * ((lnDeltaUpper rejectedBeta : ℚ) : ℝ) *
          (selectedAttackDimension - 1 : Nat) <
        ((lnLengthBoundLower 50_371 : ℚ) : ℝ) := by
    exact_mod_cast leftCertificate
  have rightCertificateReal :
      ((lnLengthBoundUpper 50_371 : ℚ) : ℝ) ≤
        2 * ((lnDeltaLower rejectedBeta : ℚ) : ℝ) *
          selectedAttackDimension := by
    exact_mod_cast rightCertificate
  have minimumCertificateReal :
      (0 : ℝ) <
        ((secureQuadraticLower
          50_371 selectedAttackDimension : ℚ) : ℝ) := by
    exact_mod_cast minimumCertificate
  have deltaPos : 0 < chenLogDelta rejectedBeta := by
    linarith [deltaBounds.1]
  have decreasing :
      2 * chenLogDelta rejectedBeta *
          ((selectedAttackDimension : ℝ) - 1) <
        exactLogLengthBound 50_371 := by
    norm_num [selectedAttackDimension] at leftCertificateReal ⊢
    nlinarith [deltaBounds.2, lengthBounds.1]
  have increasing :
      exactLogLengthBound 50_371 ≤
        2 * chenLogDelta rejectedBeta * selectedAttackDimension := by
    norm_num [selectedAttackDimension] at rightCertificateReal ⊢
    nlinarith [deltaBounds.1, lengthBounds.2]
  have minimumPositive :
      0 < quadratic
        (chenLogDelta rejectedBeta)
        (exactLogLengthBound 50_371)
        exactLogVolume
        selectedAttackDimension := by
    unfold secureQuadraticLower at minimumCertificateReal
    unfold quadratic
    norm_num [selectedAttackDimension] at minimumCertificateReal ⊢
    nlinarith [deltaBounds.1, lengthBounds.2, volumeBounds.1]
  constructor
  · norm_num
  · intro dimension dimensionAtLeastTwo
    apply required_log_delta_lt_of_quadratic_pos dimensionAtLeastTwo
    exact quadratic_pos_of_discrete_min
      deltaPos decreasing increasing minimumPositive dimension

theorem boundary_next_quadratic_negative :
    quadratic
      (chenLogDelta rejectedBeta)
      (exactLogLengthBound 50_372)
      exactLogVolume
      selectedAttackDimension < 0 := by
  have deltaBounds :=
    ln_delta_bounds
      (beta := rejectedBeta)
      (show 1 < rejectedBeta by native_decide)
  have lengthBounds :=
    exact_log_length_bound_bounds (ringColumns := 50_372) (by norm_num)
  have volumeBounds := exact_log_volume_bounds
  have certificate := certified_attack_after_boundary
  unfold CertifiedAttack at certificate
  have certificateReal :
      ((attackQuadraticUpper
        50_372 selectedAttackDimension : ℚ) : ℝ) < 0 := by
    exact_mod_cast certificate
  unfold attackQuadraticUpper at certificateReal
  unfold quadratic
  norm_num [selectedAttackDimension] at certificateReal ⊢
  nlinarith [deltaBounds.2, lengthBounds.1, volumeBounds.2]

theorem width_50372_rejected_by_selected_dimension :
    WidthRejectedBy 50_372 selectedAttackDimension := by
  constructor
  · norm_num [selectedAttackDimension]
  · apply required_log_delta_ge_of_quadratic_nonpos
      (by norm_num [selectedAttackDimension])
    exact boundary_next_quadratic_negative.le

theorem exact_log_length_bound_mono
    {smaller larger : Nat}
    (smallerPositive : 0 < smaller)
    (ordered : smaller ≤ larger) :
    exactLogLengthBound smaller ≤ exactLogLengthBound larger := by
  have smallerColumnsPositive : 0 < coefficientColumns smaller := by
    unfold coefficientColumns ringDegree
    omega
  have columnsOrdered :
      coefficientColumns smaller ≤ coefficientColumns larger := by
    simpa [coefficientColumns] using
      Nat.mul_le_mul_right ringDegree ordered
  have castOrdered :
      (coefficientColumns smaller : ℝ) ≤
        coefficientColumns larger := by
    exact_mod_cast columnsOrdered
  have logOrdered :
      Real.log (coefficientColumns smaller : ℝ) ≤
        Real.log (coefficientColumns larger : ℝ) :=
    Real.log_le_log (by positivity) castOrdered
  unfold exactLogLengthBound
  linarith

/-- Reducing the message width cannot weaken an accepted rank-two instance.
The rank, modulus, norm model, and attack target remain unchanged. -/
theorem width_accepted_of_le
    {smaller larger : Nat}
    (smallerPositive : 0 < smaller)
    (ordered : smaller ≤ larger)
    (accepted : WidthAccepted larger) :
    WidthAccepted smaller := by
  have lengthMono :
      exactLogLengthBound smaller ≤ exactLogLengthBound larger :=
    exact_log_length_bound_mono smallerPositive ordered
  constructor
  · exact smallerPositive
  · intro dimension dimensionAtLeastTwo
    have dimensionPositive : (0 : ℝ) < dimension := by positivity
    have dimensionMinusOnePositive :
        (0 : ℝ) < (dimension : ℝ) - 1 := by
      have castLt : (1 : ℝ) < dimension := by
        exact_mod_cast (show 1 < dimension by omega)
      linarith
    have denominatorPositive :
        (0 : ℝ) < (dimension : ℝ) * ((dimension : ℝ) - 1) :=
      mul_pos dimensionPositive dimensionMinusOnePositive
    have requiredMono :
        requiredLogDelta smaller dimension ≤
          requiredLogDelta larger dimension := by
      unfold requiredLogDelta
      apply (div_le_div_iff_of_pos_right denominatorPositive).2
      nlinarith
    exact requiredMono.trans_lt
      (accepted.2 dimension dimensionAtLeastTwo)

theorem compact_primary_width_accepted :
    WidthAccepted 738 :=
  width_accepted_of_le (by decide) (by decide) width_50371_accepted

theorem no_width_above_50371_is_accepted
    {ringColumns : Nat} (above : 50_371 < ringColumns) :
    ¬ WidthAccepted ringColumns := by
  have boundaryNextLe : 50_372 ≤ ringColumns := by omega
  have lengthMono :
      exactLogLengthBound 50_372 ≤
        exactLogLengthBound ringColumns :=
    exact_log_length_bound_mono (by norm_num) boundaryNextLe
  have quadraticNonpositive :
      quadratic
        (chenLogDelta rejectedBeta)
        (exactLogLengthBound ringColumns)
        exactLogVolume
        selectedAttackDimension ≤ 0 := by
    have negative := boundary_next_quadratic_negative
    unfold quadratic at negative ⊢
    norm_num [selectedAttackDimension] at negative ⊢
    nlinarith
  intro accepted
  have acceptedAtDimension :=
    accepted.2 selectedAttackDimension
      (by norm_num [selectedAttackDimension])
  have rejectedAtDimension :=
    required_log_delta_ge_of_quadratic_nonpos
      (ringColumns := ringColumns)
      (dimension := selectedAttackDimension)
      (by norm_num [selectedAttackDimension])
      quadraticNonpositive
  linarith

def boundarySearchAccepts (ringColumns : Nat) : Bool :=
  decide (0 < secureQuadraticLower ringColumns selectedAttackDimension)

def boundarySearchStep (state : Nat × Nat) : Nat × Nat :=
  let lower := state.1
  let upper := state.2
  if upper ≤ lower then
    state
  else
    let middle := (lower + upper + 1) / 2
    if boundarySearchAccepts middle then
      (middle, upper)
    else
      (lower, middle - 1)

def boundarySearchRounds : Nat := 16
def boundarySearchUpper : Nat := 2 ^ boundarySearchRounds - 1

/-- Width computed by the executable rational-certificate search. -/
def computedMaxRingColumns : Nat :=
  ((List.range boundarySearchRounds).foldl
    (fun state _ => boundarySearchStep state)
    (1, boundarySearchUpper)).1

theorem computedMaxRingColumns_eq :
    computedMaxRingColumns = 50_371 := by
  native_decide

theorem computedMaxRingColumns_is_largest :
    WidthAccepted computedMaxRingColumns ∧
      ∀ ringColumns,
        computedMaxRingColumns < ringColumns →
          ¬ WidthAccepted ringColumns := by
  rw [computedMaxRingColumns_eq]
  exact ⟨width_50371_accepted, fun _ => no_width_above_50371_is_accepted⟩

def computedMaxSourceFields : Nat :=
  maxPackedFields digitCount ringDegree computedMaxRingColumns

theorem computedMaxSourceFields_eq :
    computedMaxSourceFields = 66_342 := by
  native_decide

theorem computedBoundary_fits :
    requiredRingColumns computedMaxSourceFields =
      computedMaxRingColumns := by
  native_decide

theorem computedBoundary_next_does_not_fit :
    requiredRingColumns (computedMaxSourceFields + 1) =
      computedMaxRingColumns + 1 := by
  native_decide

theorem sourceFields_fit_iff (sourceFields : Nat) :
    requiredRingColumns sourceFields ≤ computedMaxRingColumns ↔
      sourceFields ≤ computedMaxSourceFields := by
  rw [computedMaxRingColumns_eq, computedMaxSourceFields_eq]
  simp only [requiredRingColumns, digitCount, ringDegree]
  omega

end Ajtai.EstimatorModel
