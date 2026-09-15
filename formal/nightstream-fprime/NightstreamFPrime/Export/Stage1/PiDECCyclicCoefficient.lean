import NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.RingFLaws
import Mathlib.Tactic.SplitIfs

set_option autoImplicit false

/-! Exact cyclic coefficients for the signed PiDEC product kernel. -/

namespace NightstreamFPrime.Export.Stage1.PiDECCyclicCoefficient

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism

/-- The 81 coefficients `[A0, A1 - A0, -A1]`, with zero padding. -/
def cycle (left : RingF) (index : Nat) : F :=
  if index < 27 then
    ringFCoeff left index
  else if index < 54 then
    ringFCoeff left index - ringFCoeff left (index - 27)
  else
    -ringFCoeff left (index - 27)

/-- One input lane's exact contribution to a quotient-ring output lane. -/
def coefficient (left : RingF) (output input : Fin ringDegree) : F :=
  if output.val < 27 then
    cycle left ((output.val + 81 - input.val) % 81)
  else
    -cycle left ((output.val + 27 + 81 - input.val) % 81)

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

private theorem raw_monomial_left
    (left : RingF) (input : Fin ringDegree) (degree : Nat) :
    rawMulCoeffF (ringFMonomial input.val 1) left degree =
      if input.val ≤ degree ∧ degree - input.val < ringDegree then
        ringFCoeff left (degree - input.val)
      else 0 := by
  unfold rawMulCoeffF
  have stepEquality :
      (fun accumulated index =>
        if index ≤ degree ∧ degree - index < ringDegree then
          accumulated +
            ringFCoeff (ringFMonomial input.val 1) index *
              ringFCoeff left (degree - index)
        else accumulated) =
      (fun accumulated index =>
        accumulated + if index = input.val then
          if input.val ≤ degree ∧ degree - input.val < ringDegree then
            ringFCoeff left (degree - input.val)
          else 0
        else 0) := by
    funext accumulated index
    rw [ringFCoeff_monomial input.val 1 input.isLt index]
    by_cases equal : index = input.val
    · subst index
      by_cases active : input.val ≤ degree ∧ degree - input.val < ringDegree <;>
        simp [active, Fin.one_mul, Fin.add_zero]
    · by_cases active : index ≤ degree ∧ degree - index < ringDegree <;>
        simp [active, equal, Fin.zero_mul, Fin.add_zero]
  rw [stepEquality]
  exact foldl_oneHot (List.range ringDegree) input.val
    (if input.val ≤ degree ∧ degree - input.val < ringDegree then
      ringFCoeff left (degree - input.val) else 0)
    List.nodup_range (by simpa using input.isLt)

/-- The cyclic kernel agrees with the public quotient-ring coefficient for
every key and every input and output lane. -/
theorem coefficient_eq_rightCoefficient
    (left : RingF) (output input : Fin ringDegree) :
    coefficient left output input =
      CarrierAction.rightCoefficient left output input := by
  have outputLt : output.val < 54 := by
    simpa only [ringDegree] using output.isLt
  have inputLt : input.val < 54 := by
    simpa only [ringDegree] using input.isLt
  unfold coefficient CarrierAction.rightCoefficient
  rw [RingFLaws.ringFMul_comm left (ringFMonomial input.val 1)]
  unfold ringFMul
  simp only [raw_monomial_left, ringDegree, ringMiddleDegree]
  by_cases low : output.val < 27
  · by_cases before : input.val ≤ output.val
    · have residue : (output.val + 81 - input.val) % 81 =
          output.val - input.val := by omega
      simp (disch := omega) [cycle, residue, low, before,
        Fin.sub_eq_add_neg, Fin.zero_add, Fin.add_zero,
        Lean.Grind.AddCommGroup.neg_zero] <;> split_ifs <;> first | omega |
          simp_all only [Fin.sub_eq_add_neg, Fin.zero_add, Fin.add_zero,
            Lean.Grind.AddCommGroup.neg_zero, Lean.Grind.AddCommGroup.neg_neg,
            Lean.Grind.AddCommGroup.neg_add, Lean.Grind.Fin.add_comm]
    · by_cases near : input.val ≤ output.val + 27
      · have residue : (output.val + 81 - input.val) % 81 =
            output.val + 81 - input.val := by omega
        have subtract : output.val + 81 - input.val - 27 =
            output.val + 54 - input.val := by omega
        simp (disch := omega) [cycle, residue, subtract, low, before, near,
          Fin.sub_eq_add_neg, Fin.zero_add, Fin.add_zero,
          Lean.Grind.AddCommGroup.neg_zero] <;> split_ifs <;> first | omega |
          simp_all only [Fin.sub_eq_add_neg, Fin.zero_add, Fin.add_zero,
            Lean.Grind.AddCommGroup.neg_zero, Lean.Grind.AddCommGroup.neg_neg,
            Lean.Grind.AddCommGroup.neg_add, Lean.Grind.Fin.add_comm]
      · have residue : (output.val + 81 - input.val) % 81 =
            output.val + 81 - input.val := by omega
        have subtract : output.val + 81 - input.val - 27 =
            output.val + 54 - input.val := by omega
        simp (disch := omega) [cycle, residue, subtract, low, before, near,
          Fin.sub_eq_add_neg, Fin.zero_add, Fin.add_zero,
          Lean.Grind.AddCommGroup.neg_zero, Lean.Grind.Fin.add_comm] <;> split_ifs <;> first | omega |
          simp_all only [Fin.sub_eq_add_neg, Fin.zero_add, Fin.add_zero,
            Lean.Grind.AddCommGroup.neg_zero, Lean.Grind.AddCommGroup.neg_neg,
            Lean.Grind.AddCommGroup.neg_add, Lean.Grind.Fin.add_comm]
  · have residue : (output.val + 27 + 81 - input.val) % 81 =
        output.val + 27 - input.val := by omega
    by_cases before : input.val + 27 ≤ output.val
    · have subtract : output.val + 27 - input.val - 27 =
          output.val - input.val := by omega
      simp (disch := omega) [cycle, residue, subtract, low, before,
        Fin.sub_eq_add_neg, Fin.zero_add, Fin.add_zero,
        Lean.Grind.AddCommGroup.neg_zero, Lean.Grind.AddCommGroup.neg_neg] <;> split_ifs <;> first | omega |
          simp_all only [Fin.sub_eq_add_neg, Fin.zero_add, Fin.add_zero,
            Lean.Grind.AddCommGroup.neg_zero, Lean.Grind.AddCommGroup.neg_neg,
            Lean.Grind.AddCommGroup.neg_add, Lean.Grind.Fin.add_comm]
    · by_cases near : input.val ≤ output.val
      · have subtract : output.val + 27 - input.val - 27 =
            output.val - input.val := by omega
        simp (disch := omega) [cycle, residue, subtract, low, before, near,
          Fin.sub_eq_add_neg, Fin.zero_add, Fin.add_zero,
          Lean.Grind.AddCommGroup.neg_zero, Lean.Grind.AddCommGroup.neg_neg,
          Lean.Grind.AddCommGroup.neg_add, Lean.Grind.Fin.add_comm] <;> split_ifs <;> first | omega |
          simp_all only [Fin.sub_eq_add_neg, Fin.zero_add, Fin.add_zero,
            Lean.Grind.AddCommGroup.neg_zero, Lean.Grind.AddCommGroup.neg_neg,
            Lean.Grind.AddCommGroup.neg_add, Lean.Grind.Fin.add_comm]
      · simp (disch := omega) [cycle, residue, low, before, near,
          Fin.sub_eq_add_neg, Fin.zero_add, Fin.add_zero,
          Lean.Grind.AddCommGroup.neg_zero] <;> split_ifs <;> first | omega |
          simp_all only [Fin.sub_eq_add_neg, Fin.zero_add, Fin.add_zero,
            Lean.Grind.AddCommGroup.neg_zero, Lean.Grind.AddCommGroup.neg_neg,
            Lean.Grind.AddCommGroup.neg_add, Lean.Grind.Fin.add_comm]

end NightstreamFPrime.Export.Stage1.PiDECCyclicCoefficient
