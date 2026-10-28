import NightstreamFPrime.Spec.Algebra

/-!
Goldilocks/Phi81 instantiation of the production `Pi_RLC` strong sampling set.

Owns: the semantic coefficient-to-Goldilocks map; the production challenge
set as the image of all 54-coordinate five-symbol vectors; pointwise
difference bounds; the `LowNormInvertibility` statement (the `z = 3` case of
SuperNeo Theorem 4); and the derivation of pairwise invertibility from it.

Does not own: the proof of `LowNormInvertibility`
(`Phi81StrongSet/LowNormInvertibility.lean`, which needs Mathlib and is kept
out of this Mathlib-free module); quotient-ring multiplication refinement to
Rust; Fiat-Shamir sampling; R1CS rows; row removal; or cost totals.
-/

namespace NightstreamFPrime.Spec.Phi81StrongSet

def alphabetSize : Nat := 5

def coefficientCount : Nat := 54

abbrev Coefficient := Fin alphabetSize
abbrev Scalar := Fin coefficientCount → Coefficient

def centeredRepresentative (coefficient : Coefficient) : Nat :=
  if coefficient.val < 2 then
    goldilocksModulus - (2 - coefficient.val)
  else
    coefficient.val - 2

theorem centeredRepresentative_lt (coefficient : Coefficient) :
    centeredRepresentative coefficient < goldilocksModulus := by
  have coefficientLt : coefficient.val < 5 := by
    have upper := coefficient.isLt
    change coefficient.val < 5 at upper
    exact upper
  unfold centeredRepresentative goldilocksModulus
  split <;> omega

def embedCoefficient (coefficient : Coefficient) : F :=
  ⟨centeredRepresentative coefficient, centeredRepresentative_lt coefficient⟩

theorem embedCoefficient_values :
    (List.ofFn fun coefficient : Fin alphabetSize =>
      (embedCoefficient coefficient).val) =
      [goldilocksModulus - 2, goldilocksModulus - 1, 0, 1, 2] := by
  decide

theorem embedCoefficient_injective : Function.Injective embedCoefficient := by
  intro left right equal
  apply Fin.ext
  have representativeEqual :
      centeredRepresentative left = centeredRepresentative right :=
    congrArg Fin.val equal
  have leftLt := left.isLt
  have rightLt := right.isLt
  change left.val < 5 at leftLt
  change right.val < 5 at rightLt
  unfold centeredRepresentative goldilocksModulus at representativeEqual
  split at representativeEqual <;>
    split at representativeEqual <;> omega

theorem coefficientCount_eq_ringDegree : coefficientCount = ringDegree := by
  rfl

def scalarPosition (position : Fin ringDegree) : Fin coefficientCount :=
  Fin.cast coefficientCount_eq_ringDegree.symm position

def ringPosition (position : Fin coefficientCount) : Fin ringDegree :=
  Fin.cast coefficientCount_eq_ringDegree position

def embedScalar (scalar : Scalar) : RingF :=
  fun position => embedCoefficient (scalar (scalarPosition position))

theorem embedScalar_injective : Function.Injective embedScalar := by
  intro left right equal
  funext position
  apply embedCoefficient_injective
  have atPosition := congrFun equal (ringPosition position)
  simpa [embedScalar, scalarPosition, ringPosition] using atPosition

def ringFSub (left right : RingF) : RingF :=
  fun position => left position - right position

def RingFInvertible (value : RingF) : Prop :=
  exists inverse,
    ringFMul value inverse = ringFOne /\
      ringFMul inverse value = ringFOne

def PointwiseNormAtMost (bound : Nat) (value : RingF) : Prop :=
  forall position, centeredMagnitude (value position) ≤ bound

theorem coefficientDifference_norm_le_four (left right : Coefficient) :
    centeredMagnitude (embedCoefficient left - embedCoefficient right) ≤ 4 := by
  revert left right
  decide

theorem embeddedDifference_normAtMostFour (left right : Scalar) :
    PointwiseNormAtMost 4 (ringFSub (embedScalar left) (embedScalar right)) := by
  intro position
  exact coefficientDifference_norm_le_four
    (left (scalarPosition position)) (right (scalarPosition position))

theorem ringFSub_eq_zero_iff (left right : RingF) :
    ringFSub left right = ringFZero ↔ left = right := by
  constructor
  · intro differenceZero
    funext position
    have atPosition := congrFun differenceZero position
    change left position - right position = 0 at atPosition
    exact Lean.Grind.AddCommGroup.sub_eq_zero_iff.mp atPosition
  · rintro rfl
    funext position
    exact Fin.sub_self

theorem embeddedDifference_nonzero
    {left right : Scalar} (different : left ≠ right) :
    ringFSub (embedScalar left) (embedScalar right) ≠ ringFZero := by
  intro differenceZero
  apply different
  apply embedScalar_injective
  exact (ringFSub_eq_zero_iff _ _).mp differenceZero

/-- Exact integer form of `4 < b_inv = sqrt(q / 3)` (SuperNeo Theorem 4, `z = 3`). -/
theorem differenceBound_below_goldilocks : 3 * 4 ^ 2 < goldilocksModulus := by
  decide

/-- SuperNeo Theorem 4 for `z = 3`: nonzero values with centered coefficients
at most `bound`, where `3 * bound^2 < q`, are invertible. -/
structure LowNormInvertibility : Prop where
  invertible_of_bound :
    forall (value : RingF) (bound : Nat),
      value ≠ ringFZero ->
      PointwiseNormAtMost bound value ->
      3 * bound ^ 2 < goldilocksModulus ->
      RingFInvertible value

def ProductionMember (value : RingF) : Prop :=
  exists scalar : Scalar, value = embedScalar scalar

theorem embedScalar_member (scalar : Scalar) :
    ProductionMember (embedScalar scalar) :=
  ⟨scalar, rfl⟩

def outsideChallenge : RingF := fun _ => (3 : F)

theorem embedCoefficient_ne_three (coefficient : Coefficient) :
    embedCoefficient coefficient ≠ (3 : F) := by
  revert coefficient
  decide

theorem outsideChallenge_not_member :
    ¬ ProductionMember outsideChallenge := by
  rintro ⟨scalar, equal⟩
  let position : Fin ringDegree := ⟨0, by decide⟩
  have atPosition := congrFun equal position
  exact embedCoefficient_ne_three
    (scalar (scalarPosition position)) (by
      simpa [outsideChallenge, embedScalar] using atPosition.symm)

/-- Pairwise Definition-6 consequence needed by extraction: distinct valid
challenges have an invertible difference. -/
def StrongSamplingSet : Prop :=
  forall {left right : RingF},
    ProductionMember left ->
    ProductionMember right ->
    left ≠ right ->
    RingFInvertible (ringFSub left right)

theorem productionSet_strong
    (theorem8 : LowNormInvertibility) : StrongSamplingSet := by
  intro left right leftMember rightMember different
  obtain ⟨leftScalar, rfl⟩ := leftMember
  obtain ⟨rightScalar, rfl⟩ := rightMember
  apply theorem8.invertible_of_bound _ 4
  · intro differenceZero
    exact different ((ringFSub_eq_zero_iff _ _).mp differenceZero)
  · exact embeddedDifference_normAtMostFour leftScalar rightScalar
  · exact differenceBound_below_goldilocks

end NightstreamFPrime.Spec.Phi81StrongSet
