import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.Phi81RingAction
import Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.PiRLCFinite

/-!
Contract: coordinate refinement for the concrete `RingF` action on `RingK`.

The selected physical `Pi_RLC` program emits two independent base-ring
actions for each extension-valued evaluation.  This module proves that those
two actions are exactly the low and high coordinates of the independently
defined `PiRLCFinite.combineEvaluation`.

It owns no rows, columns, transcript values, or source authority.
-/

set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 2000000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.RingKBaseActionCoordinates

open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism

/-- Low base-ring coordinate of an extension-ring value. -/
def low (value : RingK) : RingF :=
  fun lane => (value lane).c0

/-- High base-ring coordinate of an extension-ring value. -/
def high (value : RingK) : RingF :=
  fun lane => (value lane).c1

private theorem ringKCoeff_low
    (value : RingK) (degree : Nat) :
    (ringKCoeff value degree).c0 = ringFCoeff (low value) degree := by
  unfold ringKCoeff ringFCoeff low
  split <;> rfl

private theorem ringKCoeff_high
    (value : RingK) (degree : Nat) :
    (ringKCoeff value degree).c1 = ringFCoeff (high value) degree := by
  unfold ringKCoeff ringFCoeff high
  split <;> rfl

private theorem ringKCoeff_embed_low
    (value : RingF) (degree : Nat) :
    (ringKCoeff (RingKAction.embedChallenge value) degree).c0 =
      ringFCoeff value degree := by
  unfold ringKCoeff ringFCoeff RingKAction.embedChallenge
  split <;> rfl

private theorem ringKCoeff_embed_high
    (value : RingF) (degree : Nat) :
    (ringKCoeff (RingKAction.embedChallenge value) degree).c1 = 0 := by
  unfold ringKCoeff RingKAction.embedChallenge
  split <;> rfl

private theorem ringFCoeff_low_embed
    (value : RingF) (degree : Nat) :
    ringFCoeff (low (RingKAction.embedChallenge value)) degree =
      ringFCoeff value degree := by
  unfold ringFCoeff low RingKAction.embedChallenge
  split <;> rfl

private theorem ringFCoeff_high_embed
    (value : RingF) (degree : Nat) :
    ringFCoeff (high (RingKAction.embedChallenge value)) degree = 0 := by
  unfold ringFCoeff high RingKAction.embedChallenge
  split <;> rfl

private theorem foldl_raw_low
    (indices : List Nat)
    (challenge : RingF)
    (value : RingK)
    (degree : Nat)
    (initial : K) :
    (indices.foldl
        (fun accumulated index =>
          if index <= degree ∧ degree - index < ringDegree then
            K.add accumulated
              (K.mul
                (ringKCoeff (RingKAction.embedChallenge challenge) index)
                (ringKCoeff value (degree - index)))
          else accumulated)
        initial).c0 =
      indices.foldl
        (fun accumulated index =>
          if index <= degree ∧ degree - index < ringDegree then
            accumulated +
              ringFCoeff challenge index *
                ringFCoeff (low value) (degree - index)
          else accumulated)
        initial.c0 := by
  induction indices generalizing initial with
  | nil =>
      rfl
  | cons index tail inductionHypothesis =>
      simp only [List.foldl_cons]
      by_cases active :
          index <= degree ∧ degree - index < ringDegree
      · simp only [if_pos active]
        rw [inductionHypothesis]
        simp only [K.add, K.mul, ringKCoeff_embed_low,
          ringKCoeff_embed_high, ringKCoeff_low,
          ringFCoeff_low_embed]
        simp only [Fin.mul_zero, Fin.zero_mul, Fin.add_zero]
      · simp only [if_neg active]
        exact inductionHypothesis initial

private theorem foldl_raw_high
    (indices : List Nat)
    (challenge : RingF)
    (value : RingK)
    (degree : Nat)
    (initial : K) :
    (indices.foldl
        (fun accumulated index =>
          if index <= degree ∧ degree - index < ringDegree then
            K.add accumulated
              (K.mul
                (ringKCoeff (RingKAction.embedChallenge challenge) index)
                (ringKCoeff value (degree - index)))
          else accumulated)
        initial).c1 =
      indices.foldl
        (fun accumulated index =>
          if index <= degree ∧ degree - index < ringDegree then
            accumulated +
              ringFCoeff challenge index *
                ringFCoeff (high value) (degree - index)
          else accumulated)
        initial.c1 := by
  induction indices generalizing initial with
  | nil =>
      rfl
  | cons index tail inductionHypothesis =>
      simp only [List.foldl_cons]
      by_cases active :
          index <= degree ∧ degree - index < ringDegree
      · simp only [if_pos active]
        rw [inductionHypothesis]
        simp only [K.add, K.mul, ringKCoeff_embed_low,
          ringKCoeff_embed_high, ringKCoeff_high,
          ringFCoeff_high_embed]
        simp only [Fin.mul_zero, Fin.zero_mul, Fin.add_zero]
      · simp only [if_neg active]
        exact inductionHypothesis initial

theorem rawMulCoeffK_low
    (challenge : RingF) (value : RingK) (degree : Nat) :
    (rawMulCoeffK (RingKAction.embedChallenge challenge) value degree).c0 =
      rawMulCoeffF challenge (low value) degree := by
  unfold rawMulCoeffK rawMulCoeffF
  exact foldl_raw_low (List.range ringDegree) challenge value degree K.zero

theorem rawMulCoeffK_high
    (challenge : RingF) (value : RingK) (degree : Nat) :
    (rawMulCoeffK (RingKAction.embedChallenge challenge) value degree).c1 =
      rawMulCoeffF challenge (high value) degree := by
  unfold rawMulCoeffK rawMulCoeffF
  exact foldl_raw_high (List.range ringDegree) challenge value degree K.zero

/-- Low coordinate of the concrete extension-ring action. -/
theorem ringKMul_low
    (challenge : RingF) (value : RingK) (output : Fin ringDegree) :
    (ringKMul (RingKAction.embedChallenge challenge) value output).c0 =
      ringFMul challenge (low value) output := by
  unfold ringKMul ringFMul
  by_cases foldedLow : output.val < ringMiddleDegree
  · by_cases hasTwice : output.val + 81 <= 106
    · simp only [if_pos foldedLow, if_pos hasTwice, K.add, K.sub,
        rawMulCoeffK_low]
    · simp only [if_pos foldedLow, if_neg hasTwice, K.add, K.sub,
        K.zero, rawMulCoeffK_low]
  · by_cases hasTwice : output.val + 81 <= 106
    · simp only [if_neg foldedLow, if_pos hasTwice, K.add, K.sub,
        rawMulCoeffK_low]
    · simp only [if_neg foldedLow, if_neg hasTwice, K.add, K.sub,
        K.zero, rawMulCoeffK_low]

/-- High coordinate of the concrete extension-ring action. -/
theorem ringKMul_high
    (challenge : RingF) (value : RingK) (output : Fin ringDegree) :
    (ringKMul (RingKAction.embedChallenge challenge) value output).c1 =
      ringFMul challenge (high value) output := by
  unfold ringKMul ringFMul
  by_cases foldedLow : output.val < ringMiddleDegree
  · by_cases hasTwice : output.val + 81 <= 106
    · simp only [if_pos foldedLow, if_pos hasTwice, K.add, K.sub,
        rawMulCoeffK_high]
    · simp only [if_pos foldedLow, if_neg hasTwice, K.add, K.sub,
        K.zero, rawMulCoeffK_high]
  · by_cases hasTwice : output.val + 81 <= 106
    · simp only [if_neg foldedLow, if_pos hasTwice, K.add, K.sub,
        rawMulCoeffK_high]
    · simp only [if_neg foldedLow, if_neg hasTwice, K.add, K.sub,
        K.zero, rawMulCoeffK_high]

/-- Low coordinate of the exact finite `Pi_RLC` evaluation fold. -/
theorem combineEvaluation_low
    {count : Nat}
    (challenges : Fin count → RingF)
    (items : Fin count → RingK)
    (lane : Fin ringDegree) :
    (PiRLCFinite.combineEvaluation challenges items lane).c0 =
      Phi81RingAction.combine challenges (fun source => low (items source))
        lane := by
  induction count with
  | zero =>
      rfl
  | succ count inductionHypothesis =>
      simp only [PiRLCFinite.combineEvaluation, Phi81RingAction.combine,
        ringKAdd, K.add]
      rw [ringKMul_low]
      rw [inductionHypothesis
        (fun source => challenges source.succ)
        (fun source => items source.succ)]
      rfl

/-- High coordinate of the exact finite `Pi_RLC` evaluation fold. -/
theorem combineEvaluation_high
    {count : Nat}
    (challenges : Fin count → RingF)
    (items : Fin count → RingK)
    (lane : Fin ringDegree) :
    (PiRLCFinite.combineEvaluation challenges items lane).c1 =
      Phi81RingAction.combine challenges (fun source => high (items source))
        lane := by
  induction count with
  | zero =>
      rfl
  | succ count inductionHypothesis =>
      simp only [PiRLCFinite.combineEvaluation, Phi81RingAction.combine,
        ringKAdd, K.add]
      rw [ringKMul_high]
      rw [inductionHypothesis
        (fun source => challenges source.succ)
        (fun source => items source.succ)]
      rfl

/-- Low coordinate of one canonical matrix entry in the array-level fold. -/
theorem combineEvaluations_getD_low
    {shape : Shape} {count : Nat}
    (challenges : Fin count → RingF)
    (items : Fin count → Array RingK)
    (matrix : Fin shape.matrixCount)
    (lane : Fin ringDegree) :
    ((PiRLCFinite.combineEvaluations (shape := shape) challenges items).getD
        matrix.val BaseLinear.evaluationZero lane).c0 =
      Phi81RingAction.combine challenges
        (fun source =>
          low ((items source).getD matrix.val BaseLinear.evaluationZero))
        lane := by
  have matrixLt :
      matrix.val <
        (PiRLCFinite.combineEvaluations
          (shape := shape) challenges items).size := by
    simp [PiRLCFinite.combineEvaluations]
  rw [Array.getD_eq_getD_getElem?,
    Array.getElem?_eq_getElem matrixLt]
  simp only [PiRLCFinite.combineEvaluations, Array.getElem_ofFn,
    Option.getD_some]
  exact combineEvaluation_low challenges
    (fun source =>
      (items source).getD matrix.val BaseLinear.evaluationZero) lane

/-- High coordinate of one canonical matrix entry in the array-level fold. -/
theorem combineEvaluations_getD_high
    {shape : Shape} {count : Nat}
    (challenges : Fin count → RingF)
    (items : Fin count → Array RingK)
    (matrix : Fin shape.matrixCount)
    (lane : Fin ringDegree) :
    ((PiRLCFinite.combineEvaluations (shape := shape) challenges items).getD
        matrix.val BaseLinear.evaluationZero lane).c1 =
      Phi81RingAction.combine challenges
        (fun source =>
          high ((items source).getD matrix.val BaseLinear.evaluationZero))
        lane := by
  have matrixLt :
      matrix.val <
        (PiRLCFinite.combineEvaluations
          (shape := shape) challenges items).size := by
    simp [PiRLCFinite.combineEvaluations]
  rw [Array.getD_eq_getD_getElem?,
    Array.getElem?_eq_getElem matrixLt]
  simp only [PiRLCFinite.combineEvaluations, Array.getElem_ofFn,
    Option.getD_some]
  exact combineEvaluation_high challenges
    (fun source =>
      (items source).getD matrix.val BaseLinear.evaluationZero) lane

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.RingKBaseActionCoordinates
