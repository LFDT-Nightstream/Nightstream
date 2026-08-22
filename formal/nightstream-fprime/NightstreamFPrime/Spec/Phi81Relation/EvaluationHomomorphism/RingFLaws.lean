import NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.CarrierAction

/-! Provenance: copied from `formal/nightstream-lean/Nightstream/SuperNeo/Concrete/Phi81Relation/EvaluationHomomorphism/RingFLaws.lean`
at commit `fb7a8a99aefbb8ebb5474681ecf80f1b95a1b7a2`; namespaces renamed, otherwise unchanged. -/

/-!
Symbolic Phi81 multiplication laws needed by the `Pi_RLC` carrier bridge.

Protocol: SuperNeo Theorem 5, concrete assignment-side ring action.
Phase: monomial normal form and product-order compatibility.
Constraint family: semantic coefficient algebra only; this file emits no rows.

Owns: a symbolic schoolbook-convolution theorem for two monomials; the
canonical `X^81 = 1`, `X^54 = -X^27 - 1` monomial normal form; and the
smallest product-order law needed to commute the fixed bar action with an
arbitrary challenge action.

Does not own: coefficientwise `RingF -> RingK` embedding, Boolean MLE,
commitments, transcripts, Rust/R1CS refinement, row removal, or counts.

Emits constraints: no.

Authority boundary: all products are the executable `Concrete.ringFMul`.
Finite arithmetic may close bounded index side conditions, but no exhaustive
check unfolds the full 54-lane schoolbook multiplier.

| Stage path | Mathematical obligation | Proof shape | Emits constraints |
|---|---|---|---|
| `nifs.pi_rlc.verify.evaluation_hom.ring_f.raw_monomial` | raw convolution has exactly one live coefficient | symbolic fold / one-hot selection | no |
| `nifs.pi_rlc.verify.evaluation_hom.ring_f.normal_form` | executable Phi81 reduction agrees with the canonical monomial image | symbolic coefficient reduction | no |
| `nifs.pi_ccs.output.identity.constant_row` | the constant bar basis acts as the exact quotient-ring unit | finite-basis linear lift | no |
| `nifs.pi_rlc.verify.evaluation_hom.ring_f.bar_commutation` | `bar * (rho * z) = rho * (bar * z)` | finite-basis lift from normal forms | no |
-/

namespace NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.RingFLaws

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.MatrixCoefficientSource


/-- The coefficient basis element `X^degree`. Degrees outside the canonical
range denote the zero vector, exactly as `ringFMonomial` does. -/
def basis (degree : Nat) : RingF :=
  ringFMonomial degree 1

/-- Canonical image of `X^degree` modulo `X^54 + X^27 + 1`, using the
derived period `X^81 = 1`. -/
def monomialReduce (degree : Nat) : RingF :=
  let residue := degree % 81
  if residue < ringDegree then
    basis residue
  else
    ringFAdd
      (CarrierAction.ringFScale (-1) (basis (residue - ringDegree)))
      (CarrierAction.ringFScale (-1)
        (basis (residue - ringMiddleDegree)))

private theorem ringFCoeff_monomial
    (degree : Nat) (coefficient : F) (degreeLt : degree < ringDegree)
    (index : Nat) :
    ringFCoeff (ringFMonomial degree coefficient) index =
      if index = degree then coefficient else 0 := by
  unfold ringFCoeff ringFMonomial
  by_cases indexLt : index < ringDegree
  · rw [dif_pos indexLt]
  · rw [dif_neg indexLt]
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
      rw [List.foldl_cons, if_neg indexNe, Fin.add_zero]
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
        rw [if_pos rfl, Fin.zero_add]
        exact foldl_absent_oneHot indices selected value value
          (List.nodup_cons.mp nodup).1
      · have memberTail : selected ∈ indices := by
          simpa [Ne.symm indexEq] using member
        rw [if_neg indexEq, Fin.add_zero]
        exact inductionHypothesis (List.nodup_cons.mp nodup).2 memberTail

private theorem foldl_keep
    (indices : List Nat) (initial : F) :
    indices.foldl (fun accumulated _ => accumulated) initial = initial := by
  induction indices generalizing initial with
  | nil => rfl
  | cons _ indices inductionHypothesis =>
      exact inductionHypothesis initial

/-- Raw schoolbook convolution of two canonical monomials has one live
coefficient, at the sum of their degrees. This proof is symbolic in the
field coefficients and does not enumerate the Goldilocks carrier. -/
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
        · rw [if_pos active,
            ringFCoeff_monomial leftDegree leftCoefficient leftLt index,
            if_neg indexEq, Fin.zero_mul, Fin.add_zero, if_neg indexEq,
            Fin.add_zero]
        · rw [if_neg active, if_neg indexEq, Fin.add_zero]
    rw [stepEquality, if_pos degreeEq]
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
          rw [if_pos active,
            ringFCoeff_monomial leftDegree leftCoefficient leftLt index,
            ringFCoeff_monomial rightDegree rightCoefficient rightLt
              (degree - index),
            if_neg rightNe, Fin.mul_zero, Fin.add_zero]
        · rw [if_pos active,
            ringFCoeff_monomial leftDegree leftCoefficient leftLt index,
            if_neg indexEq, Fin.zero_mul, Fin.add_zero]
      · rw [if_neg active]
    rw [stepEquality, if_neg degreeEq]
    exact foldl_keep (List.range ringDegree) 0

private def delta (output degree : Nat) : F :=
  if output = degree then 1 else 0

private theorem fin_neg_zero : -(0 : F) = 0 := by
  have zeroSelf : (0 : F) - 0 = 0 := Fin.sub_self
  simpa only [Fin.sub_eq_add_neg, Fin.zero_add] using zeroSelf

private theorem ringFMul_basis_coefficient
    (left right output : Fin ringDegree) :
    ringFMul (basis left.val) (basis right.val) output =
      delta output.val (left.val + right.val) -
        (if output.val < ringMiddleDegree then
          delta (output.val + ringDegree) (left.val + right.val)
        else
          delta (output.val + ringMiddleDegree) (left.val + right.val)) +
        (if output.val + 81 <= 106 then
          delta (output.val + 81) (left.val + right.val)
        else
          0) := by
  simp only [ringFMul, basis,
    rawMulCoeffF_monomial left.val right.val 1 1 left.isLt right.isLt,
    Fin.one_mul, delta]

private theorem reduction_low
    (sumDegree output : Nat)
    (sumLt : sumDegree < 54)
    (outputLt : output < 54) :
    delta output sumDegree -
        (if output < 27 then
          delta (output + 54) sumDegree
        else
          delta (output + 27) sumDegree) +
        (if output + 81 <= 106 then delta (output + 81) sumDegree else 0) =
      delta output sumDegree := by
  have twiceNe : output + 81 ≠ sumDegree := by omega
  by_cases outputLow : output < 27
  · have foldedNe : output + 54 ≠ sumDegree := by omega
    simp [delta, outputLow, foldedNe, twiceNe, Fin.sub_eq_add_neg,
      fin_neg_zero, Fin.add_zero]
  · have outputGe27 : 27 <= output := by
      exact Nat.le_of_not_gt outputLow
    have foldedNe : output + 27 ≠ sumDegree := by omega
    simp [delta, outputLow, foldedNe, twiceNe, Fin.sub_eq_add_neg,
      fin_neg_zero, Fin.add_zero]

private theorem reduction_middle
    (sumDegree output : Nat)
    (sumGe : 54 <= sumDegree)
    (sumLt : sumDegree < 81)
    (outputLt : output < 54) :
    delta output sumDegree -
        (if output < 27 then
          delta (output + 54) sumDegree
        else
          delta (output + 27) sumDegree) +
        (if output + 81 <= 106 then delta (output + 81) sumDegree else 0) =
      (-1 : F) * delta output (sumDegree - 54) +
        (-1 : F) * delta output (sumDegree - 27) := by
  have baseNe : output ≠ sumDegree := by omega
  have twiceNe : output + 81 ≠ sumDegree := by omega
  have baseDelta : delta output sumDegree = 0 := by
    simp only [delta, if_neg baseNe]
  have twiceDelta : delta (output + 81) sumDegree = 0 := by
    simp only [delta, if_neg twiceNe]
  by_cases outputLow : output < 27
  · have secondNe : output ≠ sumDegree - 27 := by omega
    have foldedIff : output + 54 = sumDegree ↔
        output = sumDegree - 54 := by omega
    by_cases folded : output + 54 = sumDegree
    · have target : output = sumDegree - 54 := foldedIff.mp folded
      have foldedDelta : delta (output + 54) sumDegree = 1 := by
        simp only [delta, if_pos folded]
      have firstDelta : delta output (sumDegree - 54) = 1 := by
        simp only [delta, if_pos target]
      have secondDelta : delta output (sumDegree - 27) = 0 := by
        simp only [delta, if_neg secondNe]
      by_cases twiceEnabled : output + 81 <= 106 <;>
        simp only [outputLow, if_pos, baseDelta, foldedDelta, twiceEnabled,
          twiceDelta, firstDelta, secondDelta, Fin.sub_eq_add_neg,
          Fin.zero_add, Fin.add_zero, Fin.mul_one, Fin.mul_zero] <;>
        simp
    · have target : output ≠ sumDegree - 54 := by
        exact fun equal => folded (foldedIff.mpr equal)
      have foldedDelta : delta (output + 54) sumDegree = 0 := by
        simp only [delta, if_neg folded]
      have firstDelta : delta output (sumDegree - 54) = 0 := by
        simp only [delta, if_neg target]
      have secondDelta : delta output (sumDegree - 27) = 0 := by
        simp only [delta, if_neg secondNe]
      by_cases twiceEnabled : output + 81 <= 106 <;>
        simp only [outputLow, if_pos, baseDelta, foldedDelta, twiceEnabled,
          twiceDelta, firstDelta, secondDelta, Fin.sub_eq_add_neg,
          Fin.zero_add, Fin.add_zero, Fin.mul_zero, fin_neg_zero] <;>
        simp
  · have outputGe27 : 27 <= output := by
      exact Nat.le_of_not_gt outputLow
    have firstNe : output ≠ sumDegree - 54 := by omega
    have foldedIff : output + 27 = sumDegree ↔
        output = sumDegree - 27 := by omega
    by_cases folded : output + 27 = sumDegree
    · have target : output = sumDegree - 27 := foldedIff.mp folded
      have foldedDelta : delta (output + 27) sumDegree = 1 := by
        simp only [delta, if_pos folded]
      have firstDelta : delta output (sumDegree - 54) = 0 := by
        simp only [delta, if_neg firstNe]
      have secondDelta : delta output (sumDegree - 27) = 1 := by
        simp only [delta, if_pos target]
      by_cases twiceEnabled : output + 81 <= 106 <;>
        simp only [outputLow, if_false, baseDelta, foldedDelta, twiceEnabled,
          if_pos, twiceDelta, firstDelta, secondDelta,
          Fin.sub_eq_add_neg, Fin.zero_add, Fin.add_zero, Fin.mul_one,
          Fin.mul_zero] <;>
        simp
    · have target : output ≠ sumDegree - 27 := by
        exact fun equal => folded (foldedIff.mpr equal)
      have foldedDelta : delta (output + 27) sumDegree = 0 := by
        simp only [delta, if_neg folded]
      have firstDelta : delta output (sumDegree - 54) = 0 := by
        simp only [delta, if_neg firstNe]
      have secondDelta : delta output (sumDegree - 27) = 0 := by
        simp only [delta, if_neg target]
      by_cases twiceEnabled : output + 81 <= 106 <;>
        simp only [outputLow, if_false, baseDelta, foldedDelta, twiceEnabled,
          if_pos, twiceDelta, firstDelta, secondDelta,
          Fin.sub_eq_add_neg, Fin.add_zero, Fin.mul_zero,
          fin_neg_zero] <;>
        simp

private theorem reduction_high
    (sumDegree output : Nat)
    (sumGe : 81 <= sumDegree)
    (sumLe : sumDegree <= 106)
    (outputLt : output < 54) :
    delta output sumDegree -
        (if output < 27 then
          delta (output + 54) sumDegree
        else
          delta (output + 27) sumDegree) +
        (if output + 81 <= 106 then delta (output + 81) sumDegree else 0) =
      delta output (sumDegree - 81) := by
  have baseNe : output ≠ sumDegree := by omega
  have baseDelta : delta output sumDegree = 0 := by
    simp only [delta, if_neg baseNe]
  by_cases outputLow : output < 27
  · have foldedNe : output + 54 ≠ sumDegree := by omega
    have foldedDelta : delta (output + 54) sumDegree = 0 := by
      simp only [delta, if_neg foldedNe]
    have twiceIff : output + 81 = sumDegree ↔
        output = sumDegree - 81 := by omega
    by_cases twice : output + 81 = sumDegree
    · have target : output = sumDegree - 81 := twiceIff.mp twice
      have hasTwice : output + 81 <= 106 := by omega
      have twiceDelta : delta (output + 81) sumDegree = 1 := by
        simp only [delta, if_pos twice]
      have targetDelta : delta output (sumDegree - 81) = 1 := by
        simp only [delta, if_pos target]
      simp only [outputLow, if_pos, baseDelta, foldedDelta, hasTwice,
        twiceDelta, targetDelta, Fin.sub_eq_add_neg, fin_neg_zero,
        Fin.zero_add, Fin.add_zero]
    · have target : output ≠ sumDegree - 81 := by
        exact fun equal => twice (twiceIff.mpr equal)
      have twiceDelta : delta (output + 81) sumDegree = 0 := by
        simp only [delta, if_neg twice]
      have targetDelta : delta output (sumDegree - 81) = 0 := by
        simp only [delta, if_neg target]
      by_cases hasTwice : output + 81 <= 106 <;>
        simp only [outputLow, if_pos, baseDelta, foldedDelta, hasTwice,
          twiceDelta, targetDelta, Fin.sub_eq_add_neg, fin_neg_zero,
          Fin.zero_add, Fin.add_zero] <;>
        simp
  · have foldedNe : output + 27 ≠ sumDegree := by omega
    have foldedDelta : delta (output + 27) sumDegree = 0 := by
      simp only [delta, if_neg foldedNe]
    have twiceIff : output + 81 = sumDegree ↔
        output = sumDegree - 81 := by omega
    by_cases twice : output + 81 = sumDegree
    · have target : output = sumDegree - 81 := twiceIff.mp twice
      have hasTwice : output + 81 <= 106 := by omega
      have twiceDelta : delta (output + 81) sumDegree = 1 := by
        simp only [delta, if_pos twice]
      have targetDelta : delta output (sumDegree - 81) = 1 := by
        simp only [delta, if_pos target]
      simp only [outputLow, if_false, baseDelta, foldedDelta, hasTwice,
        if_pos, twiceDelta, targetDelta, Fin.sub_eq_add_neg, fin_neg_zero,
        Fin.zero_add, Fin.add_zero]
    · have target : output ≠ sumDegree - 81 := by
        exact fun equal => twice (twiceIff.mpr equal)
      have twiceDelta : delta (output + 81) sumDegree = 0 := by
        simp only [delta, if_neg twice]
      have targetDelta : delta output (sumDegree - 81) = 0 := by
        simp only [delta, if_neg target]
      by_cases hasTwice : output + 81 <= 106 <;>
        simp only [outputLow, if_false, baseDelta, foldedDelta, hasTwice,
          if_pos, twiceDelta, targetDelta, Fin.sub_eq_add_neg,
          fin_neg_zero, Fin.add_zero] <;>
        simp

/-- Multiplication of two coefficient bases is exactly the canonical Phi81
monomial normal form. The executable convolution is eliminated first by
`rawMulCoeffF_monomial`; only bounded natural-number reduction remains. -/
theorem ringFMul_basis_basis
    (left right : Fin ringDegree) :
    ringFMul (basis left.val) (basis right.val) =
      monomialReduce (left.val + right.val) := by
  funext output
  rw [ringFMul_basis_coefficient]
  have leftLt54 : left.val < 54 := by
    change left.val < ringDegree
    exact left.isLt
  have rightLt54 : right.val < 54 := by
    change right.val < ringDegree
    exact right.isLt
  have sumLe106 : left.val + right.val <= 106 := by omega
  by_cases low : left.val + right.val < ringDegree
  · have modEq : (left.val + right.val) % 81 =
        left.val + right.val := Nat.mod_eq_of_lt (by
          simp only [ringDegree] at low
          omega)
    have normalForm :
        monomialReduce (left.val + right.val) output =
          delta output.val (left.val + right.val) := by
      simp [monomialReduce, modEq, low, basis, delta, ringFMonomial]
    rw [normalForm]
    exact reduction_low (left.val + right.val) output.val low output.isLt
  · by_cases middle : left.val + right.val < 81
    · have sumGe : ringDegree <= left.val + right.val :=
        Nat.le_of_not_gt low
      have modEq : (left.val + right.val) % 81 =
          left.val + right.val := Nat.mod_eq_of_lt middle
      have normalForm :
          monomialReduce (left.val + right.val) output =
            (-1 : F) * delta output.val
                (left.val + right.val - ringDegree) +
              (-1 : F) * delta output.val
                (left.val + right.val - ringMiddleDegree) := by
        simp [monomialReduce, modEq, low, basis, delta, ringFAdd,
          CarrierAction.ringFScale, ringFMonomial]
      rw [normalForm]
      exact reduction_middle (left.val + right.val) output.val
        sumGe middle output.isLt
    · have sumGe : 81 <= left.val + right.val := Nat.le_of_not_gt middle
      have modEq : (left.val + right.val) % 81 =
          left.val + right.val - 81 := by omega
      have residueLt : left.val + right.val - 81 < ringDegree := by
        simp only [ringDegree]
        omega
      have normalForm :
          monomialReduce (left.val + right.val) output =
            delta output.val (left.val + right.val - 81) := by
        simp [monomialReduce, modEq, residueLt, basis, delta, ringFMonomial]
      rw [normalForm]
      exact reduction_high (left.val + right.val) output.val
        sumGe sumLe106 output.isLt

/-! ## Canonical-reduction recurrence -/

private theorem negOne_mul (value : F) :
    (-1 : F) * value = -value := by
  calc
    (-1 : F) * value = -(1 * value) := Lean.Grind.Fin.neg_mul 1 value
    _ = -value := by rw [Fin.one_mul]

private theorem reduction_cancel_left (left right : F) :
    (-1 : F) * right +
        (-1 : F) * ((-1 : F) * left + (-1 : F) * right) = left := by
  simp only [negOne_mul, Lean.Grind.AddCommGroup.neg_add,
    Lean.Grind.AddCommGroup.neg_neg]
  letI : Std.Associative (fun (a b : F) => a + b) :=
    ⟨ConcreteCarrier.baseLaws.add_assoc⟩
  letI : Std.Commutative (fun (a b : F) => a + b) :=
    ⟨ConcreteCarrier.baseLaws.add_comm⟩
  have cancel : right + -right = 0 := by
    calc
      right + -right = -right + right := Lean.Grind.Fin.add_comm _ _
      _ = 0 := Lean.Grind.Fin.neg_add_cancel right
  calc
    -right + (left + right) = left + (right + -right) := by ac_rfl
    _ = left + 0 := by rw [cancel]
    _ = left := ConcreteCarrier.baseLaws.add_zero left

private theorem reduction_cancel_right (left right : F) :
    (-1 : F) * ((-1 : F) * left + (-1 : F) * right) +
        (-1 : F) * left = right := by
  simp only [negOne_mul, Lean.Grind.AddCommGroup.neg_add,
    Lean.Grind.AddCommGroup.neg_neg]
  letI : Std.Associative (fun (a b : F) => a + b) :=
    ⟨ConcreteCarrier.baseLaws.add_assoc⟩
  letI : Std.Commutative (fun (a b : F) => a + b) :=
    ⟨ConcreteCarrier.baseLaws.add_comm⟩
  have cancel : left + -left = 0 := by
    calc
      left + -left = -left + left := Lean.Grind.Fin.add_comm _ _
      _ = 0 := Lean.Grind.Fin.neg_add_cancel left
  calc
    (left + right) + -left = right + (left + -left) := by ac_rfl
    _ = right + 0 := by rw [cancel]
    _ = right := ConcreteCarrier.baseLaws.add_zero right

/-- The canonical monomial image satisfies the defining Phi81 recurrence
`X^(n+54) = -X^n - X^(n+27)` throughout the only residue range needed by
one further coefficient-basis multiplication. -/
theorem monomialReduce_recurrence
    (degree : Nat) (degreeLt : degree < 80) :
    monomialReduce (degree + 54) =
      ringFAdd
        (CarrierAction.ringFScale (-1) (monomialReduce degree))
        (CarrierAction.ringFScale (-1) (monomialReduce (degree + 27))) := by
  by_cases degreeLt27 : degree < 27
  · have modDegree : degree % 81 = degree := Nat.mod_eq_of_lt (by omega)
    have modPlus27 : (degree + 27) % 81 = degree + 27 :=
      Nat.mod_eq_of_lt (by omega)
    have modPlus54 : (degree + 54) % 81 = degree + 54 :=
      Nat.mod_eq_of_lt (by omega)
    have degreeLt54 : degree < 54 := by omega
    have plus27Lt54 : degree + 27 < 54 := by omega
    have plus54Ge54 : ¬ degree + 54 < 54 := by omega
    funext output
    simp [monomialReduce, modDegree, modPlus27, modPlus54, degreeLt54,
      plus27Lt54, plus54Ge54, basis, ringFAdd, CarrierAction.ringFScale,
      ringFMonomial, ringDegree, ringMiddleDegree]
  · have degreeGe27 : 27 <= degree := Nat.le_of_not_gt degreeLt27
    by_cases degreeLt54 : degree < 54
    · have modDegree : degree % 81 = degree := Nat.mod_eq_of_lt (by omega)
      have modPlus27 : (degree + 27) % 81 = degree + 27 :=
        Nat.mod_eq_of_lt (by omega)
      have modPlus54 : (degree + 54) % 81 = degree - 27 := by omega
      have plus27Ge54 : ¬ degree + 27 < 54 := by omega
      have residueLt54 : degree - 27 < 54 := by omega
      funext output
      simp only [monomialReduce, modDegree, modPlus27, modPlus54,
        if_pos degreeLt54, if_neg plus27Ge54, if_pos residueLt54,
        ringFAdd, CarrierAction.ringFScale, ringDegree, ringMiddleDegree]
      have sub54 : degree + 27 - 54 = degree - 27 := by omega
      have sub27 : degree + 27 - 27 = degree := by omega
      simpa only [sub54, sub27] using
        (reduction_cancel_left
          (basis (degree - 27) output) (basis degree output)).symm
    · have degreeGe54 : 54 <= degree := Nat.le_of_not_gt degreeLt54
      have modDegree : degree % 81 = degree := Nat.mod_eq_of_lt (by omega)
      have modPlus27 : (degree + 27) % 81 = degree - 54 := by omega
      have modPlus54 : (degree + 54) % 81 = degree - 27 := by omega
      have residue27Lt54 : degree - 54 < 54 := by omega
      have residue54Lt54 : degree - 27 < 54 := by omega
      funext output
      simp only [monomialReduce, modDegree, modPlus27, modPlus54,
        if_neg degreeLt54, if_pos residue27Lt54, if_pos residue54Lt54,
        ringFAdd, CarrierAction.ringFScale, ringDegree, ringMiddleDegree]
      exact (reduction_cancel_right
        (basis (degree - 54) output) (basis (degree - 27) output)).symm

private theorem monomialReduce_eq_of_modEq
    (leftDegree rightDegree : Nat)
    (modEq : leftDegree % 81 = rightDegree % 81) :
    monomialReduce leftDegree = monomialReduce rightDegree := by
  unfold monomialReduce
  rw [modEq]

/-- Multiplying an already reduced monomial image by one coefficient basis
adds the basis degree before canonical reduction. -/
theorem ringFMul_basis_monomialReduce_residue
    (left : Fin ringDegree) (residue : Fin 81) :
    ringFMul (basis left.val) (monomialReduce residue.val) =
      monomialReduce (left.val + residue.val) := by
  by_cases residueLt54 : residue.val < ringDegree
  · have residueMod : residue.val % 81 = residue.val :=
      Nat.mod_eq_of_lt residue.isLt
    have reduced : monomialReduce residue.val = basis residue.val := by
      simp only [monomialReduce, residueMod, if_pos residueLt54]
    rw [reduced]
    exact ringFMul_basis_basis left ⟨residue.val, residueLt54⟩
  · have residueGe54 : ringDegree <= residue.val :=
      Nat.le_of_not_gt residueLt54
    have firstLt : residue.val - ringDegree < ringDegree := by
      simp only [ringDegree] at residueGe54 ⊢
      omega
    have secondLt : residue.val - ringMiddleDegree < ringDegree := by
      simp only [ringDegree, ringMiddleDegree] at residueGe54 ⊢
      omega
    have residueMod : residue.val % 81 = residue.val :=
      Nat.mod_eq_of_lt residue.isLt
    have reduced :
        monomialReduce residue.val =
          ringFAdd
            (CarrierAction.ringFScale (-1)
              (basis (residue.val - ringDegree)))
            (CarrierAction.ringFScale (-1)
              (basis (residue.val - ringMiddleDegree))) := by
      simp only [monomialReduce, residueMod, if_neg residueLt54]
    rw [reduced, CarrierAction.ringFMul_add_right,
      CarrierAction.ringFMul_scale_right,
      CarrierAction.ringFMul_scale_right,
      ringFMul_basis_basis left ⟨residue.val - ringDegree, firstLt⟩,
      ringFMul_basis_basis left
        ⟨residue.val - ringMiddleDegree, secondLt⟩]
    let degree := left.val + residue.val - ringDegree
    have degreeLt : degree < 80 := by
      simp only [degree, ringDegree] at ⊢
      have leftLt54 : left.val < 54 := by
        change left.val < ringDegree
        exact left.isLt
      have residueLt81 : residue.val < 81 := residue.isLt
      omega
    have recurrence := monomialReduce_recurrence degree degreeLt
    have firstDegree :
        left.val + (residue.val - ringDegree) = degree := by
      simp only [degree, ringDegree] at ⊢
      simp only [ringDegree] at residueGe54
      omega
    have secondDegree :
        left.val + (residue.val - ringMiddleDegree) = degree + 27 := by
      simp only [degree, ringDegree, ringMiddleDegree] at ⊢
      simp only [ringDegree] at residueGe54
      omega
    have totalDegree : degree + 54 = left.val + residue.val := by
      simp only [degree, ringDegree] at ⊢
      simp only [ringDegree] at residueGe54
      omega
    calc
      ringFAdd
          (CarrierAction.ringFScale (-1)
            (monomialReduce
              (left.val + (residue.val - ringDegree))))
          (CarrierAction.ringFScale (-1)
            (monomialReduce
              (left.val + (residue.val - ringMiddleDegree)))) =
          ringFAdd
            (CarrierAction.ringFScale (-1) (monomialReduce degree))
            (CarrierAction.ringFScale (-1)
              (monomialReduce (degree + 27))) := by
            rw [firstDegree, secondDegree]
      _ = monomialReduce (degree + 54) := recurrence.symm
      _ = monomialReduce (left.val + residue.val) := by rw [totalDegree]

/-- The previous action law for an arbitrary exponent; only its residue
modulo the derived period `X^81 = 1` affects the canonical image. -/
theorem ringFMul_basis_monomialReduce
    (left : Fin ringDegree) (degree : Nat) :
    ringFMul (basis left.val) (monomialReduce degree) =
      monomialReduce (left.val + degree) := by
  let residue : Fin 81 := ⟨degree % 81, Nat.mod_lt _ (by decide)⟩
  have inputReduce : monomialReduce degree = monomialReduce residue.val := by
    apply monomialReduce_eq_of_modEq
    simp only [residue, Nat.mod_mod]
  calc
    ringFMul (basis left.val) (monomialReduce degree) =
        ringFMul (basis left.val) (monomialReduce residue.val) := by
          rw [inputReduce]
    _ = monomialReduce (left.val + residue.val) :=
      ringFMul_basis_monomialReduce_residue left residue
    _ = monomialReduce (left.val + degree) := by
      apply monomialReduce_eq_of_modEq
      simp [residue, Nat.add_mod]

/-- Coefficient-basis left actions commute on a coefficient-basis input.
This is the finite leaf later lifted by bilinearity; no carrier values are
enumerated. -/
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

/-! ## Finite-basis lift -/

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
      rw [dif_pos indexLt]
      by_cases equal : index = output.val
      · rw [if_pos equal, if_pos equal.symm, Fin.mul_one]
        apply congrArg value
        apply Fin.ext
        exact equal
      · rw [if_neg equal, if_neg (Ne.symm equal), Fin.mul_zero]
    _ = value output := by
      simpa using
        (sumRange_select ConcreteCarrier.baseOps ConcreteCarrier.baseLaws
          ringDegree output.val (fun _ => value output) output.isLt)

private theorem ringFScale_zero (value : RingF) :
    CarrierAction.ringFScale 0 value = ringFZero := by
  funext output
  exact Fin.zero_mul (value output)

/-- Multiplication by the additive identity on the left is zero. -/
theorem ringFMul_zero_left (right : RingF) :
    ringFMul ringFZero right = ringFZero := by
  calc
    ringFMul ringFZero right =
        ringFMul (CarrierAction.ringFScale 0 ringFZero) right := by
          rw [ringFScale_zero]
    _ = CarrierAction.ringFScale 0 (ringFMul ringFZero right) :=
      CarrierAction.ringFMul_scale_left 0 ringFZero right
    _ = ringFZero := ringFScale_zero _

/-- Two base-field-linear maps on `RingF` are equal when they agree on all
54 coefficient basis elements. -/
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
        rw [dif_pos countLt, leftScale, rightScale,
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

/-- The executable quotient multiplication has the expected left unit. The
proof is a symbolic finite-basis lift from `ringFMul_basis_basis`; it does not
unfold the multiplier for an arbitrary 54-lane value. -/
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
    have indexLt81 : index.val < 81 := by
      have := index.isLt
      simp only [ringDegree] at this
      omega
    have residue : index.val % 81 = index.val :=
      Nat.mod_eq_of_lt indexLt81
    change ringFMul (basis zero.val) (basis index.val) = basis index.val
    rw [ringFMul_basis_basis zero index]
    simp only [zero, Nat.zero_add]
    simp [monomialReduce, residue, index.isLt]

/-- The constant coefficient row of the canonical Phi81 kernel returns the
authoritative assignment block unchanged. This is an algebraic identity only;
an identity-matrix opening must still prove that it selects this row. -/
theorem kernelImage_constant (block : RingF) :
    CarrierAction.kernelImage Phi81CoefficientKernel.constant block = block := by
  rw [CarrierAction.kernelImage_eq_ringFMul,
    Phi81CoefficientKernel.barBasis_constant_eq_one, ringFMul_one_left]

/-- Exact left-action commutation for the executable Phi81 quotient-ring
multiplication. This is the strongest algebraic statement needed to reconcile
the two product orders in the concrete `Pi_RLC` carrier bridge. -/
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

/-- The executable Phi81 quotient multiplication is commutative. The proof
uses the same finite coefficient-basis normal form as the product-order law. -/
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

/-- The executable Phi81 quotient multiplication has the expected right
unit. -/
theorem ringFMul_one_right (value : RingF) :
    ringFMul value ringFOne = value := by
  rw [ringFMul_comm, ringFMul_one_left]

/-- The executable Phi81 quotient multiplication is associative. This
follows from commutativity and the already proved commutation of left actions. -/
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

/-- The exact product-order bridge required by the concrete `Pi_RLC`
evaluation-homomorphism field. The fixed bar image and the sampled challenge
act in either order on every complete assignment block. -/
theorem ringFMul_barBasis_productOrder
    (row : Fin ringDegree) (challenge block : RingF) :
    ringFMul (Phi81CoefficientKernel.barBasis row)
        (ringFMul challenge block) =
      ringFMul challenge
        (ringFMul (Phi81CoefficientKernel.barBasis row) block) :=
  ringFMul_leftActionComm
    (Phi81CoefficientKernel.barBasis row) challenge block

end NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.RingFLaws
