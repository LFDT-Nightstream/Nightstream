import NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.CarrierAction
import NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.Norm.Centered

namespace NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.Norm.Product

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism
open NightstreamFPrime.Spec.Phi81StrongSet


def supportActive (degree index : Nat) : Prop :=
  index <= degree /\ degree - index < ringDegree

instance supportActiveDecidable (degree index : Nat) :
    Decidable (supportActive degree index) := by
  unfold supportActive
  infer_instance

def supportCount (degree : Nat) : Nat :=
  ((List.range ringDegree).filter fun index =>
    decide (supportActive degree index)).length

def rawTerm (left right : RingF) (degree index : Nat) : F :=
  if supportActive degree index then
    ringFCoeff left index * ringFCoeff right (degree - index)
  else
    0

def fieldListSum : List Nat -> (Nat -> F) -> F
  | [], _ => 0
  | index :: rest, term => term index + fieldListSum rest term

private theorem field_zero_add (value : F) : (0 : F) + value = value := by
  calc
    (0 : F) + value = value + 0 := Lean.Grind.Fin.add_comm _ _
    _ = value := Lean.Grind.Fin.add_zero _

private theorem foldl_eq_add_fieldListSum
    (indices : List Nat) (term : Nat -> F) (initial : F) :
    indices.foldl (fun accumulated index => accumulated + term index) initial =
      initial + fieldListSum indices term := by
  induction indices generalizing initial with
  | nil => exact (Lean.Grind.Fin.add_zero initial).symm
  | cons index indices inductionHypothesis =>
      rw [List.foldl_cons, inductionHypothesis]
      exact Lean.Grind.Fin.add_assoc _ _ _

theorem rawMulCoeffF_eq_fieldListSum
    (left right : RingF) (degree : Nat) :
    rawMulCoeffF left right degree =
      fieldListSum (List.range ringDegree) (rawTerm left right degree) := by
  unfold rawMulCoeffF
  have step :
      (fun accumulated index =>
        if index <= degree /\ degree - index < ringDegree then
          accumulated +
            ringFCoeff left index * ringFCoeff right (degree - index)
        else
          accumulated) =
      (fun accumulated index =>
        accumulated + rawTerm left right degree index) := by
    funext accumulated index
    by_cases active : supportActive degree index
    · have active' : index <= degree /\ degree - index < ringDegree := by
        simpa [supportActive] using active
      simp [rawTerm, active, active']
    · have inactive' : ¬ (index <= degree /\
          degree - index < ringDegree) := by
        simpa [supportActive] using active
      simp [rawTerm, active, inactive']
  rw [step, foldl_eq_add_fieldListSum]
  exact field_zero_add _

private theorem fieldListSum_le_activeCount
    (indices : List Nat) (active : Nat -> Prop) [DecidablePred active]
    (term : Nat -> F) (bound : Nat)
    (termBound : forall index, index ∈ indices -> active index ->
      centeredMagnitude (term index) <= bound)
    (inactiveZero : forall index, index ∈ indices -> ¬ active index ->
      term index = 0) :
    centeredMagnitude (fieldListSum indices term) <=
      ((indices.filter fun index => decide (active index)).length) * bound := by
  induction indices with
  | nil => simp [fieldListSum, Centered.centeredMagnitude_zero]
  | cons index rest inductionHypothesis =>
      by_cases headActive : active index
      · have headBound : centeredMagnitude (term index) <= bound :=
          termBound index (by simp) headActive
        have tailBound : centeredMagnitude (fieldListSum rest term) <=
            ((rest.filter fun item => decide (active item)).length) * bound := by
          apply inductionHypothesis
          · intro item member itemActive
            exact termBound item (by simp [member]) itemActive
          · intro item member itemInactive
            exact inactiveZero item (by simp [member]) itemInactive
        calc
          centeredMagnitude (fieldListSum (index :: rest) term) =
              centeredMagnitude (term index + fieldListSum rest term) := rfl
          _ <= centeredMagnitude (term index) +
              centeredMagnitude (fieldListSum rest term) :=
            Centered.centeredMagnitude_add_le _ _
          _ <= bound +
              ((rest.filter fun item => decide (active item)).length) * bound :=
            Nat.add_le_add headBound tailBound
          _ = (((index :: rest).filter fun item =>
              decide (active item)).length) * bound := by
            simp [headActive, Nat.add_mul, Nat.add_comm]
      · have headZero : term index = 0 :=
          inactiveZero index (by simp) headActive
        have tailBound : centeredMagnitude (fieldListSum rest term) <=
            ((rest.filter fun item => decide (active item)).length) * bound := by
          apply inductionHypothesis
          · intro item member itemActive
            exact termBound item (by simp [member]) itemActive
          · intro item member itemInactive
            exact inactiveZero item (by simp [member]) itemInactive
        calc
          centeredMagnitude (fieldListSum (index :: rest) term) =
              centeredMagnitude (fieldListSum rest term) := by
            simp only [fieldListSum, headZero, field_zero_add]
          _ <= ((rest.filter fun item => decide (active item)).length) * bound :=
            tailBound
          _ = (((index :: rest).filter fun item =>
              decide (active item)).length) * bound := by
            simp [headActive]

theorem rawTerm_le_twice_bound
    {bound : Nat} (challenge : RingF) (block : RingF)
    (challengeMember : ProductionMember challenge)
    (blockBounded : forall lane, centeredMagnitude (block lane) <= bound)
    (degree index : Nat) (indexLt : index < ringDegree) :
    centeredMagnitude (rawTerm challenge block degree index) <= 2 * bound := by
  by_cases active : supportActive degree index
  · have rightIndexLt : degree - index < ringDegree := active.2
    let challengeLane : Fin ringDegree := ⟨index, indexLt⟩
    let blockLane : Fin ringDegree := ⟨degree - index, rightIndexLt⟩
    obtain ⟨scalar, rfl⟩ := challengeMember
    have productBound := Centered.embedCoefficient_mul_le_two
      (scalar (Phi81StrongSet.scalarPosition challengeLane)) (block blockLane)
    calc
      centeredMagnitude
          (rawTerm (Phi81StrongSet.embedScalar scalar) block degree index) =
          centeredMagnitude
            (Phi81StrongSet.embedCoefficient
              (scalar (Phi81StrongSet.scalarPosition challengeLane)) *
                block blockLane) := by
        simp [rawTerm, active, ringFCoeff, indexLt,
          rightIndexLt, Phi81StrongSet.embedScalar, challengeLane, blockLane]
      _ <= 2 * centeredMagnitude (block blockLane) := productBound
      _ <= 2 * bound := Nat.mul_le_mul_left 2 (blockBounded blockLane)
  · simp [rawTerm, active, Centered.centeredMagnitude_zero]

theorem rawMulCoeffF_le_support_of_bounded
    {bound : Nat} (challenge : RingF) (block : RingF)
    (challengeMember : ProductionMember challenge)
    (blockBounded : forall lane, centeredMagnitude (block lane) <= bound)
    (degree : Nat) :
    centeredMagnitude (rawMulCoeffF challenge block degree) <=
      supportCount degree * (2 * bound) := by
  rw [rawMulCoeffF_eq_fieldListSum]
  apply fieldListSum_le_activeCount
      (active := supportActive degree) (bound := 2 * bound)
  · intro index member active
    have indexLt : index < ringDegree := List.mem_range.mp member
    exact rawTerm_le_twice_bound challenge block challengeMember blockBounded degree
      index indexLt
  · intro index member inactive
    simp [rawTerm, inactive]

theorem rawTerm_le_two
    (challenge : RingF) (block : RingF)
    (challengeMember : ProductionMember challenge)
    (blockFresh : forall lane, centeredMagnitude (block lane) < 2)
    (degree index : Nat) (indexLt : index < ringDegree) :
    centeredMagnitude (rawTerm challenge block degree index) <= 2 := by
  simpa using rawTerm_le_twice_bound (bound := 1) challenge block challengeMember
    (fun lane => Nat.le_of_lt_succ (blockFresh lane)) degree index indexLt

theorem rawMulCoeffF_le_support
    (challenge : RingF) (block : RingF)
    (challengeMember : ProductionMember challenge)
    (blockFresh : forall lane, centeredMagnitude (block lane) < 2)
    (degree : Nat) :
    centeredMagnitude (rawMulCoeffF challenge block degree) <= supportCount degree * 2 := by
  simpa using rawMulCoeffF_le_support_of_bounded (bound := 1) challenge block challengeMember
    (fun lane => Nat.le_of_lt_succ (blockFresh lane)) degree

def foldedDegree (output : Nat) : Nat :=
  if output < ringMiddleDegree then output + ringDegree
  else output + ringMiddleDegree

def twiceEnabled (output : Nat) : Bool :=
  decide (output + 81 <= 106)

def totalSupport (output : Nat) : Nat :=
  supportCount output + supportCount (foldedDegree output) +
    if twiceEnabled output then supportCount (output + 81) else 0

theorem totalSupport_le_two_degrees :
    forall output : Fin ringDegree,
      totalSupport output.val <= 2 * ringDegree := by
  simp [Fin.forall_fin_succ, totalSupport, supportCount, supportActive,
    foldedDegree, twiceEnabled, ringDegree, ringMiddleDegree]
  decide

def foldedRaw (challenge block : RingF) (output : Nat) : F :=
  if output < ringMiddleDegree then
    rawMulCoeffF challenge block (output + ringDegree)
  else
    rawMulCoeffF challenge block (output + ringMiddleDegree)

def twiceRaw (challenge block : RingF) (output : Nat) : F :=
  if output + 81 <= 106 then
    rawMulCoeffF challenge block (output + 81)
  else
    0

def reducedValue (challenge block : RingF) (output : Nat) : F :=
  rawMulCoeffF challenge block output - foldedRaw challenge block output +
    twiceRaw challenge block output

theorem ringFMul_eq_reducedValue
    (challenge block : RingF) (output : Fin ringDegree) :
    ringFMul challenge block output =
      reducedValue challenge block output.val := by
  rfl

theorem ringFMul_le_expansion_of_bounded
    {bound : Nat} (challenge : RingF) (block : RingF)
    (challengeMember : ProductionMember challenge)
    (blockBounded : forall lane, centeredMagnitude (block lane) <= bound)
    (output : Fin ringDegree) :
    centeredMagnitude (ringFMul challenge block output) <= 216 * bound := by
  let folded := foldedDegree output.val
  have baseBound := rawMulCoeffF_le_support_of_bounded challenge block challengeMember
    blockBounded output.val
  have foldedBound := rawMulCoeffF_le_support_of_bounded challenge block challengeMember
    blockBounded folded
  have foldedRawEq : foldedRaw challenge block output.val =
      rawMulCoeffF challenge block folded := by
    by_cases low : output.val < ringMiddleDegree
    · simp [foldedRaw, foldedDegree, folded, low]
    · simp [foldedRaw, foldedDegree, folded, low]
  have twiceBound : centeredMagnitude (twiceRaw challenge block output.val) <=
      (if twiceEnabled output.val then supportCount (output.val + 81) else 0) * (2 * bound) := by
    by_cases enabled : output.val + 81 <= 106
    · have rawBound := rawMulCoeffF_le_support_of_bounded challenge block challengeMember
        blockBounded (output.val + 81)
      simpa [twiceRaw, twiceEnabled, enabled] using rawBound
    · simp [twiceRaw, twiceEnabled, enabled,
        Centered.centeredMagnitude_zero]
  have reductionTriangle :
      centeredMagnitude (reducedValue challenge block output.val) <=
        centeredMagnitude (rawMulCoeffF challenge block output.val) +
          centeredMagnitude (rawMulCoeffF challenge block folded) +
            centeredMagnitude (twiceRaw challenge block output.val) := by
    unfold reducedValue
    rw [foldedRawEq]
    exact Nat.le_trans
      (Centered.centeredMagnitude_add_le
        (rawMulCoeffF challenge block output.val -
          rawMulCoeffF challenge block folded)
        (twiceRaw challenge block output.val))
      (Nat.add_le_add_right
        (Centered.centeredMagnitude_sub_le
          (rawMulCoeffF challenge block output.val)
          (rawMulCoeffF challenge block folded)) _)
  have supportBound := totalSupport_le_two_degrees output
  have expandedBound : totalSupport output.val * (2 * bound) <= 216 * bound := by
    calc
      totalSupport output.val * (2 * bound) <= (2 * ringDegree) * (2 * bound) :=
        Nat.mul_le_mul_right (2 * bound) supportBound
      _ = 216 * bound := by simp [ringDegree, ← Nat.mul_assoc]
  rw [ringFMul_eq_reducedValue]
  calc
    centeredMagnitude (reducedValue challenge block output.val) <=
        centeredMagnitude (rawMulCoeffF challenge block output.val) +
          centeredMagnitude (rawMulCoeffF challenge block folded) +
            centeredMagnitude (twiceRaw challenge block output.val) :=
      reductionTriangle
    _ <= supportCount output.val * (2 * bound) + supportCount folded * (2 * bound) +
          (if twiceEnabled output.val then
            supportCount (output.val + 81) else 0) * (2 * bound) :=
      Nat.add_le_add (Nat.add_le_add baseBound foldedBound) twiceBound
    _ = totalSupport output.val * (2 * bound) := by
      simp [totalSupport, folded, Nat.add_mul]
    _ <= 216 * bound := expandedBound

theorem ringFMul_le_expansion
    (challenge : RingF) (block : RingF)
    (challengeMember : ProductionMember challenge)
    (blockFresh : forall lane, centeredMagnitude (block lane) < 2)
    (output : Fin ringDegree) :
    centeredMagnitude (ringFMul challenge block output) <= 216 := by
  simpa using ringFMul_le_expansion_of_bounded (bound := 1) challenge block challengeMember
    (fun lane => Nat.le_of_lt_succ (blockFresh lane)) output

end NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.Norm.Product
