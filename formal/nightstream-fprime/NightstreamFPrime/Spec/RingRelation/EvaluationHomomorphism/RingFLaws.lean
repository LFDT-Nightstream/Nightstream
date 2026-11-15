import NightstreamFPrime.Spec.RingRelation.EvaluationHomomorphism.CarrierAction

/-! SuperNeo v1.2 Theorem 11. -/

namespace NightstreamFPrime.Spec.RingRelation.EvaluationHomomorphism.RingFLaws

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.MatrixCoefficientSource


def basis (degree : Nat) : RingF :=
  ringFMonomial degree 1

def monomialReduce (degree : Nat) : RingF :=
  let residue := degree % (2 * ringDegree)
  if residue < ringDegree then
    basis residue
  else
    CarrierAction.ringFScale (-1) (basis (residue - ringDegree))

private theorem ringFCoeff_monomial
    (degree : Nat) (coefficient : F) (degreeLt : degree < ringDegree)
    (index : Nat) :
    ringFCoeff (ringFMonomial degree coefficient) index =
      if index = degree then coefficient else 0 := by
  unfold ringFCoeff ringFMonomial
  by_cases indexLt : index < ringDegree
  · rw [dite_eq_left indexLt]
  · rw [dite_eq_right indexLt]
    have notEqual : index ≠ degree := by omega
    simp [notEqual]

private theorem foldl_absent_oneHot
    (indices : List Nat) (selected : Nat) (value initial : F)
    (absent : selected ∉ indices) :
    indices.foldl
        (fun accumulated index =>
          accumulated + if index = selected then value else 0)
        initial = initial := by
  induction indices generalizing initial with
  | nil => rfl
  | cons index indices inductionHypothesis =>
      have indexNe : index ≠ selected := by
        intro equal
        apply absent
        simp [equal]
      have absentTail : selected ∉ indices := by
        intro member
        exact absent (by simp [member])
      rw [List.foldl_cons, ite_eq_right indexNe, Fin.add_zero]
      exact inductionHypothesis initial absentTail

private theorem foldl_oneHot
    (indices : List Nat) (selected : Nat) (value : F)
    (nodup : indices.Nodup) (member : selected ∈ indices) :
    indices.foldl
        (fun accumulated index =>
          accumulated + if index = selected then value else 0)
        0 = value := by
  induction indices with
  | nil => simp at member
  | cons index indices inductionHypothesis =>
      simp only [List.foldl_cons]
      by_cases indexEq : index = selected
      · subst index
        rw [ite_eq_left rfl, Fin.zero_add]
        exact foldl_absent_oneHot indices selected value value
          (List.nodup_cons.mp nodup).1
      · have memberTail : selected ∈ indices := by
          simpa [Ne.symm indexEq] using member
        rw [ite_eq_right indexEq, Fin.add_zero]
        exact inductionHypothesis (List.nodup_cons.mp nodup).2 memberTail

private theorem foldl_keep
    (indices : List Nat) (initial : F) :
    indices.foldl (fun accumulated _ => accumulated) initial = initial := by
  induction indices generalizing initial with
  | nil => rfl
  | cons _ indices inductionHypothesis =>
      exact inductionHypothesis initial

theorem rawMulCoeffF_monomial
    (leftDegree rightDegree : Nat)
    (leftCoefficient rightCoefficient : F)
    (leftLt : leftDegree < ringDegree)
    (rightLt : rightDegree < ringDegree)
    (degree : Nat) :
    rawMulCoeffF
        (ringFMonomial leftDegree leftCoefficient)
        (ringFMonomial rightDegree rightCoefficient) degree =
      if degree = leftDegree + rightDegree then
        leftCoefficient * rightCoefficient
      else
        0 := by
  unfold rawMulCoeffF
  by_cases degreeEq : degree = leftDegree + rightDegree
  · have stepEquality :
        (fun accumulated index =>
          if index <= degree ∧ degree - index < ringDegree then
            accumulated +
              ringFCoeff (ringFMonomial leftDegree leftCoefficient) index *
                ringFCoeff (ringFMonomial rightDegree rightCoefficient)
                  (degree - index)
          else accumulated) =
        (fun accumulated index =>
          accumulated + if index = leftDegree then
            leftCoefficient * rightCoefficient
          else 0) := by
      funext accumulated index
      by_cases indexEq : index = leftDegree
      · subst index
        have active : leftDegree <= degree ∧
            degree - leftDegree < ringDegree := by
          constructor <;> omega
        have subtract : degree - leftDegree = rightDegree := by omega
        simp [active, ringFCoeff_monomial, leftLt, rightLt, subtract]
      · by_cases active : index <= degree ∧ degree - index < ringDegree
        · rw [ite_eq_left active,
            ringFCoeff_monomial leftDegree leftCoefficient leftLt index,
            ite_eq_right indexEq, Fin.zero_mul, Fin.add_zero, ite_eq_right indexEq,
            Fin.add_zero]
        · rw [ite_eq_right active, ite_eq_right indexEq, Fin.add_zero]
    rw [stepEquality, ite_eq_left degreeEq]
    exact foldl_oneHot (List.range ringDegree) leftDegree
      (leftCoefficient * rightCoefficient) List.nodup_range
      (by simpa using leftLt)
  · have stepEquality :
        (fun accumulated index =>
          if index <= degree ∧ degree - index < ringDegree then
            accumulated +
              ringFCoeff (ringFMonomial leftDegree leftCoefficient) index *
                ringFCoeff (ringFMonomial rightDegree rightCoefficient)
                  (degree - index)
          else accumulated) =
        (fun accumulated _ => accumulated) := by
      funext accumulated index
      by_cases active : index <= degree ∧ degree - index < ringDegree
      · by_cases indexEq : index = leftDegree
        · have rightNe : degree - index ≠ rightDegree := by omega
          rw [ite_eq_left active,
            ringFCoeff_monomial leftDegree leftCoefficient leftLt index,
            ringFCoeff_monomial rightDegree rightCoefficient rightLt
              (degree - index),
            ite_eq_right rightNe, Fin.mul_zero, Fin.add_zero]
        · rw [ite_eq_left active,
            ringFCoeff_monomial leftDegree leftCoefficient leftLt index,
            ite_eq_right indexEq, Fin.zero_mul, Fin.add_zero]
      · rw [ite_eq_right active]
    rw [stepEquality, ite_eq_right degreeEq]
    exact foldl_keep (List.range ringDegree) 0

private def delta (output degree : Nat) : F :=
  if output = degree then 1 else 0

private theorem ringFMul_basis_coefficient
    (left right output : Fin ringDegree) :
    ringFMul (basis left.val) (basis right.val) output =
      delta output.val (left.val + right.val) -
        delta (output.val + ringDegree) (left.val + right.val) := by
  simp only [ringFMul, basis,
    rawMulCoeffF_monomial left.val right.val 1 1 left.isLt right.isLt,
    Fin.one_mul, delta]

private theorem negOne_mul (value : F) :
    (-1 : F) * value = -value := by
  calc
    (-1 : F) * value = -(1 * value) := Lean.Grind.Fin.neg_mul 1 value
    _ = -value := by rw [Fin.one_mul]

private theorem reduction_low
    (sumDegree output : Nat)
    (sumLt : sumDegree < ringDegree) :
    delta output sumDegree - delta (output + ringDegree) sumDegree =
      delta output sumDegree := by
  have foldedNe : output + ringDegree ≠ sumDegree := by omega
  have foldedDelta : delta (output + ringDegree) sumDegree = 0 := by
    simp only [delta, ite_eq_right foldedNe]
  have zeroSelf : (0 : F) - 0 = 0 := Fin.sub_self
  have negZero : -(0 : F) = 0 := by
    simpa only [Fin.sub_eq_add_neg, Fin.zero_add] using zeroSelf
  rw [foldedDelta, Fin.sub_eq_add_neg, negZero, Fin.add_zero]

private theorem reduction_high
    (sumDegree output : Nat)
    (sumGe : ringDegree <= sumDegree)
    (outputLt : output < ringDegree) :
    delta output sumDegree - delta (output + ringDegree) sumDegree =
      (-1 : F) * delta output (sumDegree - ringDegree) := by
  have baseNe : output ≠ sumDegree := by omega
  have baseDelta : delta output sumDegree = 0 := by
    simp only [delta, ite_eq_right baseNe]
  have foldedIff : output + ringDegree = sumDegree ↔
      output = sumDegree - ringDegree := by omega
  have foldedDelta : delta (output + ringDegree) sumDegree =
      delta output (sumDegree - ringDegree) := by
    simp only [delta, foldedIff]
  rw [baseDelta, foldedDelta, negOne_mul, Fin.sub_eq_add_neg, Fin.zero_add]

theorem ringFMul_basis_basis
    (left right : Fin ringDegree) :
    ringFMul (basis left.val) (basis right.val) =
      monomialReduce (left.val + right.val) := by
  funext output
  rw [ringFMul_basis_coefficient]
  have modEq : (left.val + right.val) % (2 * ringDegree) =
      left.val + right.val :=
    Nat.mod_eq_of_lt (by have := left.isLt; have := right.isLt; omega)
  by_cases low : left.val + right.val < ringDegree
  · have normalForm :
        monomialReduce (left.val + right.val) output =
          delta output.val (left.val + right.val) := by
      simp [monomialReduce, modEq, low, basis, delta, ringFMonomial]
    rw [normalForm]
    exact reduction_low (left.val + right.val) output.val low
  · have normalForm :
        monomialReduce (left.val + right.val) output =
          (-1 : F) * delta output.val
            (left.val + right.val - ringDegree) := by
      simp [monomialReduce, modEq, low, basis, delta,
        CarrierAction.ringFScale, ringFMonomial]
    rw [normalForm]
    exact reduction_high (left.val + right.val) output.val
      (Nat.le_of_not_gt low) output.isLt

private theorem ringFScale_negOne_negOne (value : RingF) :
    CarrierAction.ringFScale (-1) (CarrierAction.ringFScale (-1) value) =
      value := by
  funext output
  simp only [CarrierAction.ringFScale, negOne_mul,
    Lean.Grind.AddCommGroup.neg_neg]

/-- `X ^ ringDegree = -1`. -/
theorem monomialReduce_add_ringDegree (degree : Nat) :
    monomialReduce (degree + ringDegree) =
      CarrierAction.ringFScale (-1) (monomialReduce degree) := by
  have residueLt : degree % (2 * ringDegree) < 2 * ringDegree :=
    Nat.mod_lt _ (by decide)
  by_cases low : degree % (2 * ringDegree) < ringDegree
  · have shifted : (degree + ringDegree) % (2 * ringDegree) =
        degree % (2 * ringDegree) + ringDegree := by
      simp only [ringDegree] at low ⊢
      omega
    have high : ¬ degree % (2 * ringDegree) + ringDegree < ringDegree := by
      omega
    simp only [monomialReduce, shifted, ite_eq_right high, ite_eq_left low,
      Nat.add_sub_cancel]
  · have shifted : (degree + ringDegree) % (2 * ringDegree) =
        degree % (2 * ringDegree) - ringDegree := by
      simp only [ringDegree] at low ⊢
      omega
    have reduced : degree % (2 * ringDegree) - ringDegree < ringDegree := by
      omega
    simp only [monomialReduce, shifted, ite_eq_left reduced,
      ite_eq_right low, ringFScale_negOne_negOne]

private theorem monomialReduce_eq_of_modEq
    (leftDegree rightDegree : Nat)
    (modEq : leftDegree % (2 * ringDegree) = rightDegree % (2 * ringDegree)) :
    monomialReduce leftDegree = monomialReduce rightDegree := by
  unfold monomialReduce
  rw [modEq]

theorem ringFMul_basis_monomialReduce
    (left : Fin ringDegree) (degree : Nat) :
    ringFMul (basis left.val) (monomialReduce degree) =
      monomialReduce (left.val + degree) := by
  have residueLt : degree % (2 * ringDegree) < 2 * ringDegree :=
    Nat.mod_lt _ (by decide)
  have sumReduce :
      monomialReduce (left.val + degree % (2 * ringDegree)) =
        monomialReduce (left.val + degree) := by
    apply monomialReduce_eq_of_modEq
    rw [Nat.add_mod_mod]
  by_cases low : degree % (2 * ringDegree) < ringDegree
  · have reduced :
        monomialReduce degree = basis (degree % (2 * ringDegree)) := by
      simp only [monomialReduce, ite_eq_left low]
    rw [reduced,
      ringFMul_basis_basis left ⟨degree % (2 * ringDegree), low⟩, sumReduce]
  · have highLt : degree % (2 * ringDegree) - ringDegree < ringDegree := by
      omega
    have reduced :
        monomialReduce degree =
          CarrierAction.ringFScale (-1)
            (basis (degree % (2 * ringDegree) - ringDegree)) := by
      simp only [monomialReduce, ite_eq_right low]
    have total :
        left.val + (degree % (2 * ringDegree) - ringDegree) + ringDegree =
          left.val + degree % (2 * ringDegree) := by
      omega
    rw [reduced, CarrierAction.ringFMul_scale_right,
      ringFMul_basis_basis left
        ⟨degree % (2 * ringDegree) - ringDegree, highLt⟩,
      ← monomialReduce_add_ringDegree, total, sumReduce]

theorem ringFMul_leftActionComm_basis
    (left middle right : Fin ringDegree) :
    ringFMul (basis left.val)
        (ringFMul (basis middle.val) (basis right.val)) =
      ringFMul (basis middle.val)
        (ringFMul (basis left.val) (basis right.val)) := by
  rw [ringFMul_basis_basis middle right,
    ringFMul_basis_monomialReduce left,
    ringFMul_basis_basis left right,
    ringFMul_basis_monomialReduce middle]
  congr 1
  omega

private def ringFSumRange : Nat -> (Nat -> RingF) -> RingF
  | 0, _ => ringFZero
  | count + 1, term =>
      ringFAdd (ringFSumRange count term) (term count)

private theorem ringFSumRange_apply
    (count : Nat) (term : Nat -> RingF) (output : Fin ringDegree) :
    ringFSumRange count term output =
      sumRange ConcreteCarrier.baseOps count (fun index => term index output) := by
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      simp only [ringFSumRange, ringFAdd, sumRange, ConcreteCarrier.baseOps,
        inductionHypothesis]

private def basisTerm (value : RingF) (index : Nat) : RingF :=
  if indexLt : index < ringDegree then
    CarrierAction.ringFScale (value ⟨index, indexLt⟩) (basis index)
  else
    ringFZero

private def basisExpansion (value : RingF) : RingF :=
  ringFSumRange ringDegree (basisTerm value)

private theorem basisExpansion_eq (value : RingF) :
    basisExpansion value = value := by
  funext output
  unfold basisExpansion
  rw [ringFSumRange_apply]
  calc
    sumRange ConcreteCarrier.baseOps ringDegree
        (fun index => basisTerm value index output) =
      sumRange ConcreteCarrier.baseOps ringDegree
        (fun index => if index = output.val then value output else 0) := by
      apply sumRange_congr
      intro index indexLt
      unfold basisTerm CarrierAction.ringFScale basis ringFMonomial
      rw [dite_eq_left indexLt]
      by_cases equal : index = output.val
      · rw [ite_eq_left equal, ite_eq_left equal.symm, Fin.mul_one]
        apply congrArg value
        apply Fin.ext
        exact equal
      · rw [ite_eq_right equal, ite_eq_right (Ne.symm equal), Fin.mul_zero]
    _ = value output := by
      simpa using!
        (sumRange_select ConcreteCarrier.baseOps ConcreteCarrier.baseLaws
          ringDegree output.val (fun _ => value output) output.isLt)

private theorem ringFScale_zero (value : RingF) :
    CarrierAction.ringFScale 0 value = ringFZero := by
  funext output
  exact Fin.zero_mul (value output)

theorem ringFMul_zero_left (right : RingF) :
    ringFMul ringFZero right = ringFZero := by
  calc
    ringFMul ringFZero right =
        ringFMul (CarrierAction.ringFScale 0 ringFZero) right := by
          rw [ringFScale_zero]
    _ = CarrierAction.ringFScale 0 (ringFMul ringFZero right) :=
      CarrierAction.ringFMul_scale_left 0 ringFZero right
    _ = ringFZero := ringFScale_zero _

theorem ringF_linear_eq_of_basis
    (leftMap rightMap : RingF -> RingF)
    (leftZero : leftMap ringFZero = ringFZero)
    (rightZero : rightMap ringFZero = ringFZero)
    (leftAdd : forall left right,
      leftMap (ringFAdd left right) =
        ringFAdd (leftMap left) (leftMap right))
    (rightAdd : forall left right,
      rightMap (ringFAdd left right) =
        ringFAdd (rightMap left) (rightMap right))
    (leftScale : forall scalar value,
      leftMap (CarrierAction.ringFScale scalar value) =
        CarrierAction.ringFScale scalar (leftMap value))
    (rightScale : forall scalar value,
      rightMap (CarrierAction.ringFScale scalar value) =
        CarrierAction.ringFScale scalar (rightMap value))
    (onBasis : forall index : Fin ringDegree,
      leftMap (basis index.val) = rightMap (basis index.val))
    (value : RingF) :
    leftMap value = rightMap value := by
  have prefixEqual : forall count, count <= ringDegree ->
      leftMap (ringFSumRange count (basisTerm value)) =
        rightMap (ringFSumRange count (basisTerm value)) := by
    intro count countLe
    induction count with
    | zero =>
        rw [ringFSumRange, leftZero, rightZero]
    | succ count inductionHypothesis =>
        have countLt : count < ringDegree := by omega
        rw [ringFSumRange, leftAdd, rightAdd,
          inductionHypothesis (by omega)]
        congr 1
        unfold basisTerm
        rw [dite_eq_left countLt, leftScale, rightScale,
          onBasis ⟨count, countLt⟩]
  calc
    leftMap value = leftMap (basisExpansion value) := by
      rw [basisExpansion_eq]
    _ = rightMap (basisExpansion value) := by
      exact prefixEqual ringDegree (Nat.le_refl ringDegree)
    _ = rightMap value := by rw [basisExpansion_eq]

private theorem ringF_bilinear_eq_of_basis
    (leftMap rightMap : RingF -> RingF -> RingF)
    (leftZeroLeft : forall right, leftMap ringFZero right = ringFZero)
    (rightZeroLeft : forall right, rightMap ringFZero right = ringFZero)
    (leftZeroRight : forall left, leftMap left ringFZero = ringFZero)
    (rightZeroRight : forall left, rightMap left ringFZero = ringFZero)
    (leftAddLeft : forall left₁ left₂ right,
      leftMap (ringFAdd left₁ left₂) right =
        ringFAdd (leftMap left₁ right) (leftMap left₂ right))
    (rightAddLeft : forall left₁ left₂ right,
      rightMap (ringFAdd left₁ left₂) right =
        ringFAdd (rightMap left₁ right) (rightMap left₂ right))
    (leftAddRight : forall left right₁ right₂,
      leftMap left (ringFAdd right₁ right₂) =
        ringFAdd (leftMap left right₁) (leftMap left right₂))
    (rightAddRight : forall left right₁ right₂,
      rightMap left (ringFAdd right₁ right₂) =
        ringFAdd (rightMap left right₁) (rightMap left right₂))
    (leftScaleLeft : forall scalar left right,
      leftMap (CarrierAction.ringFScale scalar left) right =
        CarrierAction.ringFScale scalar (leftMap left right))
    (rightScaleLeft : forall scalar left right,
      rightMap (CarrierAction.ringFScale scalar left) right =
        CarrierAction.ringFScale scalar (rightMap left right))
    (leftScaleRight : forall scalar left right,
      leftMap left (CarrierAction.ringFScale scalar right) =
        CarrierAction.ringFScale scalar (leftMap left right))
    (rightScaleRight : forall scalar left right,
      rightMap left (CarrierAction.ringFScale scalar right) =
        CarrierAction.ringFScale scalar (rightMap left right))
    (onBasis : forall left right : Fin ringDegree,
      leftMap (basis left.val) (basis right.val) =
        rightMap (basis left.val) (basis right.val))
    (left right : RingF) :
    leftMap left right = rightMap left right := by
  have leftBasis : forall index : Fin ringDegree,
      leftMap (basis index.val) right =
        rightMap (basis index.val) right := by
    intro index
    exact ringF_linear_eq_of_basis
      (fun value => leftMap (basis index.val) value)
      (fun value => rightMap (basis index.val) value)
      (leftZeroRight _) (rightZeroRight _)
      (leftAddRight _) (rightAddRight _)
      (fun scalar value =>
        leftScaleRight scalar (basis index.val) value)
      (fun scalar value =>
        rightScaleRight scalar (basis index.val) value)
      (onBasis index) right
  exact ringF_linear_eq_of_basis
    (fun value => leftMap value right)
    (fun value => rightMap value right)
    (leftZeroLeft right) (rightZeroLeft right)
    (fun left₁ left₂ => leftAddLeft left₁ left₂ right)
    (fun left₁ left₂ => rightAddLeft left₁ left₂ right)
    (fun scalar value => leftScaleLeft scalar value right)
    (fun scalar value => rightScaleLeft scalar value right)
    leftBasis left

theorem ringFMul_one_left (value : RingF) :
    ringFMul ringFOne value = value := by
  apply ringF_linear_eq_of_basis
      (fun right => ringFMul ringFOne right) (fun right => right)
  · exact CarrierAction.ringFMul_zero_right ringFOne
  · rfl
  · intro left right
    exact CarrierAction.ringFMul_add_right ringFOne left right
  · intro _ _
    rfl
  · intro scalar right
    exact CarrierAction.ringFMul_scale_right ringFOne scalar right
  · intro _ _
    rfl
  · intro index
    let zero : Fin ringDegree := ⟨0, by decide⟩
    have residue : index.val % (2 * ringDegree) = index.val :=
      Nat.mod_eq_of_lt (by have := index.isLt; omega)
    change ringFMul (basis zero.val) (basis index.val) = basis index.val
    rw [ringFMul_basis_basis zero index]
    simp only [zero, Nat.zero_add]
    simp [monomialReduce, residue, index.isLt]

theorem kernelImage_constant (block : RingF) :
    CarrierAction.kernelImage RingCoefficientKernel.constant block = block := by
  rw [CarrierAction.kernelImage_eq_ringFMul,
    RingCoefficientKernel.barBasis_constant_eq_one, ringFMul_one_left]

theorem ringFMul_leftActionComm (left middle right : RingF) :
    ringFMul left (ringFMul middle right) =
      ringFMul middle (ringFMul left right) := by
  apply ringF_linear_eq_of_basis
    (fun value => ringFMul left (ringFMul middle value))
    (fun value => ringFMul middle (ringFMul left value))
  · rw [CarrierAction.ringFMul_zero_right,
      CarrierAction.ringFMul_zero_right]
  · rw [CarrierAction.ringFMul_zero_right,
      CarrierAction.ringFMul_zero_right]
  · intro value₁ value₂
    rw [CarrierAction.ringFMul_add_right,
      CarrierAction.ringFMul_add_right]
  · intro value₁ value₂
    rw [CarrierAction.ringFMul_add_right,
      CarrierAction.ringFMul_add_right]
  · intro scalar value
    rw [CarrierAction.ringFMul_scale_right,
      CarrierAction.ringFMul_scale_right]
  · intro scalar value
    rw [CarrierAction.ringFMul_scale_right,
      CarrierAction.ringFMul_scale_right]
  · intro rightBasis
    apply ringF_bilinear_eq_of_basis
      (fun leftValue middleValue =>
        ringFMul leftValue (ringFMul middleValue (basis rightBasis.val)))
      (fun leftValue middleValue =>
        ringFMul middleValue (ringFMul leftValue (basis rightBasis.val)))
    · intro middleValue
      rw [ringFMul_zero_left]
    · intro middleValue
      rw [ringFMul_zero_left, CarrierAction.ringFMul_zero_right]
    · intro leftValue
      rw [ringFMul_zero_left, CarrierAction.ringFMul_zero_right]
    · intro leftValue
      rw [ringFMul_zero_left]
    · intro left₁ left₂ middleValue
      rw [CarrierAction.ringFMul_add_left]
    · intro left₁ left₂ middleValue
      rw [CarrierAction.ringFMul_add_left,
        CarrierAction.ringFMul_add_right]
    · intro leftValue middle₁ middle₂
      rw [CarrierAction.ringFMul_add_left,
        CarrierAction.ringFMul_add_right]
    · intro leftValue middle₁ middle₂
      rw [CarrierAction.ringFMul_add_left]
    · intro scalar leftValue middleValue
      rw [CarrierAction.ringFMul_scale_left]
    · intro scalar leftValue middleValue
      rw [CarrierAction.ringFMul_scale_left,
        CarrierAction.ringFMul_scale_right]
    · intro scalar leftValue middleValue
      rw [CarrierAction.ringFMul_scale_left,
        CarrierAction.ringFMul_scale_right]
    · intro scalar leftValue middleValue
      rw [CarrierAction.ringFMul_scale_left]
    · intro leftBasis middleBasis
      exact ringFMul_leftActionComm_basis
        leftBasis middleBasis rightBasis

theorem ringFMul_comm (left right : RingF) :
    ringFMul left right = ringFMul right left := by
  apply ringF_bilinear_eq_of_basis
    (fun leftValue rightValue => ringFMul leftValue rightValue)
    (fun leftValue rightValue => ringFMul rightValue leftValue)
  · exact ringFMul_zero_left
  · intro rightValue
    exact CarrierAction.ringFMul_zero_right rightValue
  · exact CarrierAction.ringFMul_zero_right
  · exact ringFMul_zero_left
  · exact CarrierAction.ringFMul_add_left
  · intro left₁ left₂ rightValue
    exact CarrierAction.ringFMul_add_right rightValue left₁ left₂
  · exact CarrierAction.ringFMul_add_right
  · intro leftValue right₁ right₂
    exact CarrierAction.ringFMul_add_left right₁ right₂ leftValue
  · exact CarrierAction.ringFMul_scale_left
  · intro scalar leftValue rightValue
    exact CarrierAction.ringFMul_scale_right rightValue scalar leftValue
  · intro scalar leftValue rightValue
    exact CarrierAction.ringFMul_scale_right leftValue scalar rightValue
  · intro scalar leftValue rightValue
    exact CarrierAction.ringFMul_scale_left scalar rightValue leftValue
  · intro leftBasis rightBasis
    rw [ringFMul_basis_basis, ringFMul_basis_basis, Nat.add_comm]

theorem ringFMul_one_right (value : RingF) :
    ringFMul value ringFOne = value := by
  rw [ringFMul_comm, ringFMul_one_left]

theorem ringFMul_assoc (left middle right : RingF) :
    ringFMul (ringFMul left middle) right =
      ringFMul left (ringFMul middle right) := by
  calc
    ringFMul (ringFMul left middle) right =
        ringFMul right (ringFMul left middle) :=
      ringFMul_comm _ _
    _ = ringFMul left (ringFMul right middle) :=
      (ringFMul_leftActionComm left right middle).symm
    _ = ringFMul left (ringFMul middle right) := by
      rw [ringFMul_comm right middle]

theorem ringFMul_barBasis_productOrder
    (row : Fin ringDegree) (challenge block : RingF) :
    ringFMul (RingCoefficientKernel.barBasis row)
        (ringFMul challenge block) =
      ringFMul challenge
        (ringFMul (RingCoefficientKernel.barBasis row) block) :=
  ringFMul_leftActionComm
    (RingCoefficientKernel.barBasis row) challenge block

end NightstreamFPrime.Spec.RingRelation.EvaluationHomomorphism.RingFLaws
