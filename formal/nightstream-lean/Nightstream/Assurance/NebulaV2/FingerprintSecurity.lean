import Mathlib.Algebra.MvPolynomial.SchwartzZippel
import Nightstream.Implementation.NebulaV2.ConcreteField
import Nightstream.Protocol.NebulaV2.IdealAcceptance
import Nightstream.Protocol.NebulaV2.IdealFingerprint

/-!
Contract: exact public-coin security of the two-repetition V2 memory
fingerprint over the selected SuperNeo extension field.

Owns the complete uniform challenge space, the exact independent product
space for two repetitions, the Schwartz--Zippel bound, the V2 degree
specialization, and the integer security floor used by the planning budget.

Does not own the Fiat--Shamir transform, Poseidon2, adaptive oracle
programming, transcript collisions, or the final union of all protocol terms.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Assurance.NebulaV2.FingerprintSecurity

open Finset Fintype
open Nightstream.Implementation.NebulaV2.ConcreteField
open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.Fingerprint
open Nightstream.Protocol.NebulaV2.IdealAcceptance
open Nightstream.Protocol.NebulaV2.IdealFingerprint

/-- One fingerprint repetition samples both extension-field coordinates
uniformly. -/
noncomputable def challengePoints :
    Finset (Fin 2 → ChallengeField) :=
  piFinset fun _ => Finset.univ

noncomputable def zeroPoints
    (polynomial : MvPolynomial (Fin 2) ChallengeField) :
    Finset (Fin 2 → ChallengeField) :=
  challengePoints.filter fun point => polynomial.eval point = 0

@[simp]
theorem mem_zeroPoints_iff
    (polynomial : MvPolynomial (Fin 2) ChallengeField)
    (point : Fin 2 → ChallengeField) :
    point ∈ zeroPoints polynomial ↔ polynomial.eval point = 0 := by
  simp [zeroPoints, challengePoints]

/-- Exact probability of one uniform public-coin repetition. -/
noncomputable def singleProbability
    (polynomial : MvPolynomial (Fin 2) ChallengeField) : ℚ≥0 :=
  (zeroPoints polynomial).card / Fintype.card ChallengeField ^ 2

/-- The two repetitions use the Cartesian product, so the definition records
independence rather than assuming a squared bound. -/
noncomputable def repeatedChallengePoints :
    Finset ((Fin 2 → ChallengeField) × (Fin 2 → ChallengeField)) :=
  challengePoints ×ˢ challengePoints

noncomputable def acceptingRepeatedPoints
    (polynomial : MvPolynomial (Fin 2) ChallengeField) :
    Finset ((Fin 2 → ChallengeField) × (Fin 2 → ChallengeField)) :=
  zeroPoints polynomial ×ˢ zeroPoints polynomial

/-- Put the two exact V2 challenge pairs into the independent public-coin
sample representation. -/
def repeatedPoint
    {FieldType : Type}
    (challenges : Fin 2 → ChallengePair FieldType) :
    (Fin 2 → FieldType) × (Fin 2 → FieldType) :=
  ((challenges 0).point, (challenges 1).point)

/-- The accepting points counted by `repeatedProbability` are exactly the
points that make the ideal V2 fingerprint check accept. -/
theorem accepts_iff_repeatedPoint_mem
    {initial final : Snapshot} {accesses : List Access}
    (check : Check encode initial accesses final) :
    check.Accepts ↔
      repeatedPoint check.challenges ∈
        acceptingRepeatedPoints check.polynomial := by
  unfold Check.Accepts repeatedPoint acceptingRepeatedPoints evaluate
  simp only [Finset.mem_product, mem_zeroPoints_iff]
  constructor
  · intro accepted
    exact ⟨accepted 0, accepted 1⟩
  · rintro ⟨atZero, atOne⟩ repetition
    fin_cases repetition
    · exact atZero
    · exact atOne

noncomputable def repeatedProbability
    (polynomial : MvPolynomial (Fin 2) ChallengeField) : ℚ≥0 :=
  (acceptingRepeatedPoints polynomial).card /
    (Fintype.card ChallengeField ^ 2) ^ 2

theorem challengePoints_card :
    challengePoints.card = Fintype.card ChallengeField ^ 2 := by
  simp [challengePoints]

theorem repeatedProbability_eq_square
    (polynomial : MvPolynomial (Fin 2) ChallengeField) :
    repeatedProbability polynomial = singleProbability polynomial ^ 2 := by
  unfold repeatedProbability acceptingRepeatedPoints singleProbability
  rw [Finset.card_product]
  simpa only [Nat.cast_mul, Nat.cast_pow, pow_two] using
    (mul_div_mul_comm
      (a := (↑(zeroPoints polynomial).card : ℚ≥0))
      (b := (↑(zeroPoints polynomial).card : ℚ≥0))
      (c := (↑(Fintype.card ChallengeField) : ℚ≥0) *
        ↑(Fintype.card ChallengeField))
      (d := (↑(Fintype.card ChallengeField) : ℚ≥0) *
        ↑(Fintype.card ChallengeField)))

/-- Exact Schwartz--Zippel bound for one complete challenge pair. -/
theorem singleProbability_le
    {polynomial : MvPolynomial (Fin 2) ChallengeField}
    (nonzero : polynomial ≠ 0) :
    singleProbability polynomial ≤
      polynomial.totalDegree / Fintype.card ChallengeField := by
  unfold singleProbability zeroPoints challengePoints
  simpa only [Finset.card_univ] using
    (MvPolynomial.schwartz_zippel_totalDegree nonzero
      (Finset.univ : Finset ChallengeField))

/-- Independent repetitions square the exact single-repetition bound. -/
theorem repeatedProbability_le
    {polynomial : MvPolynomial (Fin 2) ChallengeField}
    (nonzero : polynomial ≠ 0) :
    repeatedProbability polynomial ≤
      (polynomial.totalDegree / Fintype.card ChallengeField : ℚ≥0) ^ 2 := by
  rw [repeatedProbability_eq_square]
  have single := singleProbability_le nonzero
  exact pow_le_pow_left₀ zero_le single 2

theorem repeatedProbability_le_profile
    {polynomial : MvPolynomial (Fin 2) ChallengeField}
    (nonzero : polynomial ≠ 0)
    (degreeBound : polynomial.totalDegree ≤ maxSegmentFactors) :
    repeatedProbability polynomial ≤
      (maxSegmentFactors / Fintype.card ChallengeField : ℚ≥0) ^ 2 := by
  calc
    repeatedProbability polynomial ≤
        (polynomial.totalDegree /
          Fintype.card ChallengeField : ℚ≥0) ^ 2 :=
      repeatedProbability_le nonzero
    _ ≤ (maxSegmentFactors /
          Fintype.card ChallengeField : ℚ≥0) ^ 2 := by
      gcongr

/-- Concrete bridge for the exact polynomial in an ideal segment check. -/
theorem unbalanced_check_probability_le_profile
    {initial final : Snapshot} {accesses : List Access}
    (check : Check encode initial accesses final)
    (unbalanced :
      ¬ Memory.Balanced initial.tuples accesses final.tuples)
    (accessBound : accesses.length ≤ 63 * 1088) :
    repeatedProbability check.polynomial ≤
      (maxSegmentFactors / Fintype.card ChallengeField : ℚ≥0) ^ 2 := by
  have unequal : check.left ≠ check.right := by
    intro equal
    exact unbalanced (check.balance_of_bounded_eq equal)
  exact repeatedProbability_le_profile
    (boundedDifference_ne_zero encode encode_injective_below_goldilocks unequal)
    (check.degree_le_maxSegmentFactors accessBound)

/-- Every fingerprint failure exposed by a raw V2 segment has the concrete
profile degree bound because `SegmentCheck` now owns the fixed-port capacity
constraint. This remains a public-coin bound. -/
theorem segment_fingerprint_failure_probability_le_profile
    {Profile Plan Commitment Digest : Type}
    {schema : FullClaim.Schema}
    {config : Config ChallengeField Profile Plan Commitment Digest}
    {bundleComponent :
      schema.CommitmentBundle → CommitmentBundle.Component → Commitment}
    {verify : FullVerifier schema Digest ChallengeField}
    {segmentIndex timestampIn timestampOut : Nat}
    {initial final : Snapshot} {accesses : List Access}
    (segment : SegmentCheck config schema bundleComponent verify segmentIndex
      initial timestampIn accesses final timestampOut)
    (failure : EvaluationFailure segment.fingerprint) :
    repeatedProbability segment.fingerprint.polynomial ≤
      (maxSegmentFactors / Fintype.card ChallengeField : ℚ≥0) ^ 2 :=
  repeatedProbability_le_profile failure.polynomialNonzero
    (segment.fingerprint.degree_le_maxSegmentFactors segment.accessBound)

/-- Query and segment factors used by the current conservative planning
bound. This theorem is integer arithmetic, not a Fiat--Shamir theorem. -/
def planningLoss : Nat := 64 * 2 ^ 29

/-- Rational form of the two-repetition fingerprint term after the planning
loss. This is still a planning expression, not a Fiat--Shamir reduction. -/
def planningFingerprintBound : Rat :=
  (planningLoss : Rat) * (maxSegmentFactors : Rat) ^ 2 /
    (goldilocksModulus : Rat) ^ 4

def fingerprintSecurityTarget (bits : Nat) : Rat :=
  (1 : Rat) / (((2 : Nat) ^ bits : Nat) : Rat)

/-- Factored numerator for the lifetime-scaled two-repetition term. The
factorization keeps the arithmetic certificate small enough for kernel
checking. -/
theorem scaledFingerprintNumerator_eq (bits : Nat) :
    maxSegmentFactors ^ 2 * planningLoss * 2 ^ bits =
      ((2 ^ 12 * 2 ^ 35) * 2 ^ bits) * 2159 ^ 2 := by
  rw [maxSegmentFactors_eq]
  unfold planningLoss
  rw [show 138176 = 2 ^ 6 * 2159 by norm_num,
    show 64 = 2 ^ 6 by norm_num, mul_pow]
  rw [show (2 ^ 6) ^ 2 = 2 ^ 12 by
    rw [← pow_mul]]
  ac_rfl

theorem planning_fingerprint_bits_at_least_186 :
    maxSegmentFactors ^ 2 * planningLoss * 2 ^ 186 ≤
      goldilocksModulus ^ 4 := by
  have modulusLower : 255 * 2 ^ 56 ≤ goldilocksModulus := by
    norm_num [goldilocksModulus]
  have small : 512 * 2159 ^ 2 ≤ 255 ^ 4 := by norm_num
  calc
    maxSegmentFactors ^ 2 * planningLoss * 2 ^ 186 =
        2 ^ 224 * (512 * 2159 ^ 2) := by
      rw [scaledFingerprintNumerator_eq]
      rw [← pow_add, ← pow_add]
      norm_num [pow_add]
    _ ≤ 2 ^ 224 * 255 ^ 4 := Nat.mul_le_mul_left _ small
    _ = (255 * 2 ^ 56) ^ 4 := by
      rw [mul_pow, ← pow_mul]
      norm_num
    _ ≤ goldilocksModulus ^ 4 := Nat.pow_le_pow_left modulusLower 4

theorem planning_fingerprint_bits_not_187 :
    ¬ maxSegmentFactors ^ 2 * planningLoss * 2 ^ 187 ≤
      goldilocksModulus ^ 4 := by
  intro alleged
  have modulusUpper : goldilocksModulus < 2 ^ 64 := by
    norm_num [goldilocksModulus]
  have fourthPowerUpper : goldilocksModulus ^ 4 < 2 ^ 256 := by
    calc
      goldilocksModulus ^ 4 < (2 ^ 64) ^ 4 :=
        Nat.pow_lt_pow_left modulusUpper (by norm_num)
      _ = 2 ^ 256 := by rw [← pow_mul]
  have small : 2 ^ 22 ≤ 2159 ^ 2 := by norm_num
  have numeratorLower :
      2 ^ 256 ≤ maxSegmentFactors ^ 2 * planningLoss * 2 ^ 187 := by
    rw [scaledFingerprintNumerator_eq]
    calc
      2 ^ 256 = ((2 ^ 12 * 2 ^ 35) * 2 ^ 187) * 2 ^ 22 := by
        rw [← pow_add, ← pow_add, ← pow_add]
      _ ≤ ((2 ^ 12 * 2 ^ 35) * 2 ^ 187) * 2159 ^ 2 :=
        Nat.mul_le_mul_left _ small
  exact (Nat.not_lt_of_ge alleged) (fourthPowerUpper.trans_le numeratorLower)

end Nightstream.Assurance.NebulaV2.FingerprintSecurity
