import Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.Embedding
import Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.RingFLaws

/-!
Embedded-`RingF` module laws for the executable Phi81 `RingK` carrier.

Protocol: SuperNeo `Pi_RLC`, evaluation branch.
Phase: algebra used by coordinate-fork extraction.
Constraint family: semantic coefficient algebra only; this file emits no rows.

Owns: the two base-ring component projections of `RingK`; exact componentwise
refinement of multiplication by an embedded `RingF` value; and the complete
module laws for that action on `RingK`.

Does not own: commitments, public-input projection, transcript sampling,
Ajtai binding, Rust/R1CS refinement, row removal, or counts.

Emits constraints: no.

| Obligation | Local owner | Emits constraints? | Authority source |
|---|---|---|---|
| Embedded `RingF` action on `RingK` | component lemmas and module laws | no | Executable `ringKMul` and proved `RingF` laws |

Authority boundary: multiplication is the executable `ringKMul`. The proof
expands its finite convolution and Phi81 reduction. Associativity of the
embedded action is derived from the independently proved `RingF` quotient
ring laws; it is not supplied by a caller.
-/

namespace Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.RingKModule

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.RingFLaws

/-- Base component of every quadratic-extension coefficient. -/
def component0 (value : RingK) : RingF :=
  fun lane => (value lane).c0

/-- Quadratic component of every quadratic-extension coefficient. -/
def component1 (value : RingK) : RingF :=
  fun lane => (value lane).c1

/-- A `RingK` value is determined by its two `RingF` components. -/
theorem ext_components {left right : RingK}
    (component0Eq : component0 left = component0 right)
    (component1Eq : component1 left = component1 right) :
    left = right := by
  funext lane
  have c0 := congrFun component0Eq lane
  have c1 := congrFun component1Eq lane
  cases leftValue : left lane with
  | mk left0 left1 =>
      cases rightValue : right lane with
      | mk right0 right1 =>
          simp only [component0, leftValue, rightValue] at c0
          simp only [component1, leftValue, rightValue] at c1
          cases c0
          cases c1
          rfl

@[simp] theorem component0_zero : component0 ringKZero = ringFZero := by
  rfl

@[simp] theorem component1_zero : component1 ringKZero = ringFZero := by
  rfl

@[simp] theorem component0_add (left right : RingK) :
    component0 (ringKAdd left right) =
      ringFAdd (component0 left) (component0 right) := by
  rfl

@[simp] theorem component1_add (left right : RingK) :
    component1 (ringKAdd left right) =
      ringFAdd (component1 left) (component1 right) := by
  rfl

private theorem ringKCoeff_component0 (value : RingK) (degree : Nat) :
    (ringKCoeff value degree).c0 = ringFCoeff (component0 value) degree := by
  unfold ringKCoeff ringFCoeff component0
  split <;> rfl

private theorem ringKCoeff_component1 (value : RingK) (degree : Nat) :
    (ringKCoeff value degree).c1 = ringFCoeff (component1 value) degree := by
  unfold ringKCoeff ringFCoeff component1
  split <;> rfl

private theorem embeddedMul_component0 (left : F) (right : K) :
    (K.mul (K.embed left) right).c0 = left * right.c0 := by
  unfold K.mul K.embed
  change left * right.c0 + 0 * right.c1 = left * right.c0
  rw [Fin.zero_mul, Fin.add_zero]

private theorem embeddedMul_component1 (left : F) (right : K) :
    (K.mul (K.embed left) right).c1 = left * right.c1 := by
  unfold K.mul K.embed
  change left * right.c1 + 0 * right.c0 = left * right.c1
  rw [Fin.zero_mul, Fin.add_zero]

private theorem ringKCoeff_embedChallenge
    (value : RingF) (degree : Nat) :
    ringKCoeff (RingKAction.embedChallenge value) degree =
      K.embed (ringFCoeff value degree) := by
  unfold ringKCoeff ringFCoeff RingKAction.embedChallenge
  split <;> rfl

private theorem foldl_raw_component0
    (indices : List Nat) (left : RingF) (right : RingK)
    (degree : Nat) (initial : K) :
    (indices.foldl
      (fun accumulated index =>
        if index <= degree /\ degree - index < ringDegree then
          K.add accumulated
            (K.mul
              (ringKCoeff (RingKAction.embedChallenge left) index)
              (ringKCoeff right (degree - index)))
        else accumulated)
      initial).c0 =
      indices.foldl
        (fun accumulated index =>
          if index <= degree /\ degree - index < ringDegree then
            accumulated +
              ringFCoeff left index *
                ringFCoeff (component0 right) (degree - index)
          else accumulated)
        initial.c0 := by
  induction indices generalizing initial with
  | nil => rfl
  | cons index indices inductionHypothesis =>
      simp only [List.foldl_cons]
      by_cases active : index <= degree /\ degree - index < ringDegree
      · simp only [if_pos active]
        simpa only [K.add, ringKCoeff_embedChallenge,
          ringKCoeff_component0, embeddedMul_component0] using
          inductionHypothesis
            (K.add initial
              (K.mul
                (ringKCoeff (RingKAction.embedChallenge left) index)
                (ringKCoeff right (degree - index))))
      · simp only [if_neg active]
        exact inductionHypothesis initial

private theorem foldl_raw_component1
    (indices : List Nat) (left : RingF) (right : RingK)
    (degree : Nat) (initial : K) :
    (indices.foldl
      (fun accumulated index =>
        if index <= degree /\ degree - index < ringDegree then
          K.add accumulated
            (K.mul
              (ringKCoeff (RingKAction.embedChallenge left) index)
              (ringKCoeff right (degree - index)))
        else accumulated)
      initial).c1 =
      indices.foldl
        (fun accumulated index =>
          if index <= degree /\ degree - index < ringDegree then
            accumulated +
              ringFCoeff left index *
                ringFCoeff (component1 right) (degree - index)
          else accumulated)
        initial.c1 := by
  induction indices generalizing initial with
  | nil => rfl
  | cons index indices inductionHypothesis =>
      simp only [List.foldl_cons]
      by_cases active : index <= degree /\ degree - index < ringDegree
      · simp only [if_pos active]
        simpa only [K.add, ringKCoeff_embedChallenge,
          ringKCoeff_component1, embeddedMul_component1] using
          inductionHypothesis
            (K.add initial
              (K.mul
                (ringKCoeff (RingKAction.embedChallenge left) index)
                (ringKCoeff right (degree - index))))
      · simp only [if_neg active]
        exact inductionHypothesis initial

private theorem rawMulCoeffK_component0
    (left : RingF) (right : RingK) (degree : Nat) :
    (rawMulCoeffK (RingKAction.embedChallenge left) right degree).c0 =
      rawMulCoeffF left (component0 right) degree := by
  unfold rawMulCoeffK rawMulCoeffF
  exact foldl_raw_component0 (List.range ringDegree) left right degree K.zero

private theorem rawMulCoeffK_component1
    (left : RingF) (right : RingK) (degree : Nat) :
    (rawMulCoeffK (RingKAction.embedChallenge left) right degree).c1 =
      rawMulCoeffF left (component1 right) degree := by
  unfold rawMulCoeffK rawMulCoeffF
  exact foldl_raw_component1 (List.range ringDegree) left right degree K.zero

/-- Multiplication by an embedded challenge acts on the base component by
the exact executable `RingF` quotient multiplication. -/
theorem action_component0 (scalar : RingF) (value : RingK) :
    component0
        (ringKMul (RingKAction.embedChallenge scalar) value) =
      ringFMul scalar (component0 value) := by
  funext output
  change (ringKMul (RingKAction.embedChallenge scalar) value output).c0 =
    ringFMul scalar (fun lane => (value lane).c0) output
  unfold ringKMul ringFMul
  by_cases foldedLow : output.val < ringMiddleDegree
  · by_cases hasTwice : output.val + 81 <= 106
    · simp only [if_pos foldedLow, if_pos hasTwice, K.add, K.sub,
        rawMulCoeffK_component0]
      rfl
    · simp only [if_pos foldedLow, if_neg hasTwice, K.add, K.sub,
        rawMulCoeffK_component0, K.zero]
      rfl
  · by_cases hasTwice : output.val + 81 <= 106
    · simp only [if_neg foldedLow, if_pos hasTwice, K.add, K.sub,
        rawMulCoeffK_component0]
      rfl
    · simp only [if_neg foldedLow, if_neg hasTwice, K.add, K.sub,
        rawMulCoeffK_component0, K.zero]
      rfl

/-- Multiplication by an embedded challenge acts on the quadratic component
by the same executable `RingF` quotient multiplication. -/
theorem action_component1 (scalar : RingF) (value : RingK) :
    component1
        (ringKMul (RingKAction.embedChallenge scalar) value) =
      ringFMul scalar (component1 value) := by
  funext output
  change (ringKMul (RingKAction.embedChallenge scalar) value output).c1 =
    ringFMul scalar (fun lane => (value lane).c1) output
  unfold ringKMul ringFMul
  by_cases foldedLow : output.val < ringMiddleDegree
  · by_cases hasTwice : output.val + 81 <= 106
    · simp only [if_pos foldedLow, if_pos hasTwice, K.add, K.sub,
        rawMulCoeffK_component1]
      rfl
    · simp only [if_pos foldedLow, if_neg hasTwice, K.add, K.sub,
        rawMulCoeffK_component1, K.zero]
      rfl
  · by_cases hasTwice : output.val + 81 <= 106
    · simp only [if_neg foldedLow, if_pos hasTwice, K.add, K.sub,
        rawMulCoeffK_component1]
      rfl
    · simp only [if_neg foldedLow, if_neg hasTwice, K.add, K.sub,
        rawMulCoeffK_component1, K.zero]
      rfl

/-- Canonical action of the challenge ring on one full evaluation ring. -/
def act (scalar : RingF) (value : RingK) : RingK :=
  ringKMul (RingKAction.embedChallenge scalar) value

/-- Embedded zero acts as zero on every full evaluation ring. -/
theorem zero_act (value : RingK) : act ringFZero value = ringKZero := by
  apply ext_components
  · unfold act
    rw [action_component0, component0_zero]
    exact ringFMul_zero_left _
  · unfold act
    rw [action_component1, component1_zero]
    exact ringFMul_zero_left _

/-- Addition of challenges is addition of their actions. -/
theorem add_act (left right : RingF) (value : RingK) :
    act (ringFAdd left right) value =
      ringKAdd (act left value) (act right value) := by
  apply ext_components
  · unfold act
    rw [action_component0, component0_add, action_component0,
      action_component0, CarrierAction.ringFMul_add_left]
  · unfold act
    rw [action_component1, component1_add, action_component1,
      action_component1, CarrierAction.ringFMul_add_left]

/-- The challenge-ring unit acts as the identity. -/
theorem one_act (value : RingK) : act ringFOne value = value := by
  apply ext_components
  · unfold act
    rw [action_component0, ringFMul_one_left]
  · unfold act
    rw [action_component1, ringFMul_one_left]

/-- Product action is nested action. This is the required embedded-scalar
associativity theorem; no global `RingK` associativity premise is used. -/
theorem mul_act (left right : RingF) (value : RingK) :
    act (ringFMul left right) value = act left (act right value) := by
  apply ext_components
  · unfold act
    rw [action_component0, action_component0, action_component0,
      ringFMul_assoc]
  · unfold act
    rw [action_component1, action_component1, action_component1,
      ringFMul_assoc]

/-- Every challenge maps the zero evaluation to zero. -/
theorem act_zero (scalar : RingF) : act scalar ringKZero = ringKZero :=
  RingKAction.ringKMul_right_zero _

/-- Every challenge action is additive in the evaluation. -/
theorem act_add (scalar : RingF) (left right : RingK) :
    act scalar (ringKAdd left right) =
      ringKAdd (act scalar left) (act scalar right) :=
  RingKAction.ringKMul_right_add _ _ _

end Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.RingKModule
