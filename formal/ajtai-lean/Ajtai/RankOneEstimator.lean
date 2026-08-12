import Ajtai.EstimatorModel

/-!
Contract: rank-one security arithmetic for the Nebula V2 short compact-token
map.

Owns the same pinned ADPS16 quantum Core-SVP and Chen-delta model as
`Ajtai.EstimatorModel`, with lattice rank `1 * 54`, exact width 82, norm-two
Euclidean length bound for signed-unit opening differences, and the same
131-bit raw target that leaves 128 bits after seven setup roles.

Does not prove Module-SIS hardness, the structured-ring heuristic, ChaCha8
pseudorandomness, or a probability reduction. It certifies only the arithmetic
inside the named estimator model.
-/

set_option autoImplicit false

namespace Ajtai.RankOneEstimator

open Ajtai.EstimatorModel
open Ajtai.LogInterval
open Ajtai.Parameters
open Nightstream.Implementation.R1CS
open Nightstream.SuperNeo.Concrete

def rank : Nat := 1
def ringColumns : Nat := 82
def latticeRankDimension : Nat := rank * ringDegree
def selectedAttackDimension : Nat := 714

def lnVolumeLower : ℚ :=
  latticeRankDimension * lnNatLower sisModulus

def lnVolumeUpper : ℚ :=
  latticeRankDimension * lnNatUpper sisModulus

noncomputable def exactLogVolume : ℝ :=
  latticeRankDimension * Real.log (sisModulus : ℝ)

def secureQuadraticLower (dimension : Nat) : ℚ :=
  lnDeltaLower rejectedBeta * dimension * (dimension - 1) -
    lnLengthBoundUpper ringColumns * dimension +
    lnVolumeLower

def CertifiedSecure : Prop :=
  0 < lnDeltaLower rejectedBeta ∧
    2 * lnDeltaUpper rejectedBeta * (selectedAttackDimension - 1) <
      lnLengthBoundLower ringColumns ∧
    lnLengthBoundUpper ringColumns ≤
      2 * lnDeltaLower rejectedBeta * selectedAttackDimension ∧
    0 < secureQuadraticLower selectedAttackDimension

instance : Decidable CertifiedSecure := by
  unfold CertifiedSecure
  infer_instance

noncomputable def requiredLogDelta (dimension : Nat) : ℝ :=
  (exactLogLengthBound ringColumns * dimension - exactLogVolume) /
    ((dimension : ℝ) * ((dimension : ℝ) - 1))

def WidthAccepted : Prop :=
  ∀ dimension : Nat, 2 ≤ dimension →
    requiredLogDelta dimension < chenLogDelta rejectedBeta

theorem exact_log_volume_bounds :
    ((lnVolumeLower : ℚ) : ℝ) ≤ exactLogVolume ∧
      exactLogVolume ≤ ((lnVolumeUpper : ℚ) : ℝ) := by
  have modulusBounds := ln_nat_bounds (show sisModulus ≠ 0 by decide)
  unfold lnVolumeLower lnVolumeUpper exactLogVolume
  push_cast
  constructor
  · exact mul_le_mul_of_nonneg_left modulusBounds.1 (by positivity)
  · exact mul_le_mul_of_nonneg_left modulusBounds.2 (by positivity)

theorem required_log_delta_lt_of_quadratic_pos
    {dimension : Nat}
    (dimensionAtLeastTwo : 2 ≤ dimension)
    (positive :
      0 < quadratic
        (chenLogDelta rejectedBeta)
        (exactLogLengthBound ringColumns)
        exactLogVolume dimension) :
    requiredLogDelta dimension < chenLogDelta rejectedBeta := by
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

/-- Executable rational certificate at the discrete minimum. Its axiom audit
records Lean's native-decide compiler trust boundary. -/
theorem certified_secure : CertifiedSecure := by
  native_decide

/-- The exact rank-one 1-by-82 ring map meets the selected 131-bit raw
Core-SVP threshold in the pinned estimator model. -/
theorem width_accepted : WidthAccepted := by
  have deltaBounds :=
    ln_delta_bounds
      (beta := rejectedBeta)
      (show 1 < rejectedBeta by decide)
  have lengthBounds :=
    exact_log_length_bound_bounds
      (ringColumns := ringColumns) (by decide)
  have volumeBounds := exact_log_volume_bounds
  have certificate := certified_secure
  unfold CertifiedSecure at certificate
  rcases certificate with
    ⟨deltaLowerPos, leftCertificate, rightCertificate,
      minimumCertificate⟩
  have deltaLowerPosReal :
      (0 : ℝ) < ((lnDeltaLower rejectedBeta : ℚ) : ℝ) := by
    exact_mod_cast deltaLowerPos
  have leftCertificateReal :
      2 * ((lnDeltaUpper rejectedBeta : ℚ) : ℝ) *
          (selectedAttackDimension - 1 : Nat) <
        ((lnLengthBoundLower ringColumns : ℚ) : ℝ) := by
    exact_mod_cast leftCertificate
  have rightCertificateReal :
      ((lnLengthBoundUpper ringColumns : ℚ) : ℝ) ≤
        2 * ((lnDeltaLower rejectedBeta : ℚ) : ℝ) *
          selectedAttackDimension := by
    exact_mod_cast rightCertificate
  have minimumCertificateReal :
      (0 : ℝ) <
        ((secureQuadraticLower selectedAttackDimension : ℚ) : ℝ) := by
    exact_mod_cast minimumCertificate
  have deltaPos : 0 < chenLogDelta rejectedBeta := by
    linarith [deltaBounds.1]
  have decreasing :
      2 * chenLogDelta rejectedBeta *
          ((selectedAttackDimension : ℝ) - 1) <
        exactLogLengthBound ringColumns := by
    norm_num [selectedAttackDimension] at leftCertificateReal ⊢
    nlinarith [deltaBounds.2, lengthBounds.1]
  have increasing :
      exactLogLengthBound ringColumns ≤
        2 * chenLogDelta rejectedBeta * selectedAttackDimension := by
    norm_num [selectedAttackDimension] at rightCertificateReal ⊢
    nlinarith [deltaBounds.1, lengthBounds.2]
  have minimumPositive :
      0 < quadratic
        (chenLogDelta rejectedBeta)
        (exactLogLengthBound ringColumns)
        exactLogVolume selectedAttackDimension := by
    unfold secureQuadraticLower at minimumCertificateReal
    unfold quadratic
    norm_num [selectedAttackDimension] at minimumCertificateReal ⊢
    nlinarith [deltaBounds.1, lengthBounds.2, volumeBounds.1]
  intro dimension dimensionAtLeastTwo
  apply required_log_delta_lt_of_quadratic_pos dimensionAtLeastTwo
  exact quadratic_pos_of_discrete_min
    deltaPos decreasing increasing minimumPositive dimension

theorem selected_dimensions :
    rank = 1 ∧ ringColumns = 82 ∧
      latticeRankDimension = 54 ∧
      selectedAttackDimension = 714 := by
  decide

theorem security_policy :
    targetSecurityBits = 128 ∧
      setupAttackTargets = 7 ∧
      requiredRawBits = 131 ∧
      minimumAcceptedBeta = 495 := by
  decide

end Ajtai.RankOneEstimator
